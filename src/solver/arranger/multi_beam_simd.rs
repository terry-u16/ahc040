use super::Arranger;
use crate::{
    beam::{self, BayesianBeamWidthSuggester},
    problem::{Dir, Input, Op},
    solver::{
        estimator::Sampler,
        simd::{horizontal_add, horizontal_or, round_u16, SimdRectSet, SIMD_WIDTH},
    },
    util::BitSetIterU128,
};
use itertools::{izip, Itertools};
use rand::Rng;
use std::arch::x86_64::*;

/// 多変量正規分布からN個の長方形を16インスタンス生成し、それぞれについて並行に操作を行うビームサーチ。
/// 考え方は粒子フィルタなどと同じで、非線形性を考慮するために多数のインスタンスでシミュレートする。
/// 内部的にAVX2を使用して高速化している。
pub(super) struct MultiBeamArrangerSimd {
    duration_sec: f64,
}

impl MultiBeamArrangerSimd {
    pub(super) fn new(duration_sec: f64) -> Self {
        Self { duration_sec }
    }
}

impl Arranger for MultiBeamArrangerSimd {
    fn arrange(
        &mut self,
        input: &Input,
        sampler: &mut impl Sampler,
        rng: &mut impl Rng,
    ) -> Vec<Op> {
        let since = std::time::Instant::now();

        let rects = sampler.sample(rng);
        let large_state = unsafe { LargeState::new(input.clone(), rects, rng) };
        let small_state = SmallState::default();
        let act_gen = ActGen;

        let remaining_time = self.duration_sec - since.elapsed().as_secs_f64();
        let mut beam = beam::BeamSearch::new(large_state, small_state, act_gen);
        let standard_beam_width = 2_000_000 / (input.rect_cnt() as usize).pow(2);
        let beam_width_suggester = BayesianBeamWidthSuggester::new(
            input.rect_cnt(),
            5,
            remaining_time,
            standard_beam_width,
            1,
            20000,
            1,
        );
        let deduplicator = beam::HashSingleDeduplicator::new();
        let (ops, _) = beam.run(input.rect_cnt(), beam_width_suggester, deduplicator);

        ops
    }
}

#[derive(Debug, Clone)]
struct LargeState {
    // 256bitのレジスタをunsigned 16bit integer x16で使う
    rects_h: Vec<__m256i>,
    rects_w: Vec<__m256i>,
    widths: __m256i,
    heights: __m256i,
    placements_x0: Vec<__m256i>,
    placements_x1: Vec<__m256i>,
    placements_y0: Vec<__m256i>,
    placements_y1: Vec<__m256i>,
    hash: u32,
    hash_base_x: Vec<__m256i>,
    hash_base_y: Vec<__m256i>,
    hash_base_rot: Vec<u32>,
    turn: usize,
    avaliable_base_left: u128,
    avaliable_base_up: u128,
    avaliable_collision_left: u128,
    avaliable_collision_up: u128,
}

impl LargeState {
    #[target_feature(enable = "avx,avx2")]
    unsafe fn new(input: Input, rects: SimdRectSet, rng: &mut impl rand::Rng) -> Self {
        let zero = unsafe { _mm256_setzero_si256() };
        let rects_h = rects
            .heights
            .iter()
            .map(|rect| unsafe { _mm256_loadu_si256(rect.as_ptr() as *const __m256i) })
            .collect_vec();
        let rects_w = rects
            .widths
            .iter()
            .map(|rect| unsafe { _mm256_loadu_si256(rect.as_ptr() as *const __m256i) })
            .collect_vec();
        let hash_base_x = (0..input.rect_cnt())
            .map(|_| {
                let v: [u16; SIMD_WIDTH] = [0; SIMD_WIDTH].map(|_| rng.gen());
                unsafe { _mm256_loadu_si256(v.as_ptr() as *const __m256i) }
            })
            .collect();
        let hash_base_y = (0..input.rect_cnt())
            .map(|_| {
                let v: [u16; SIMD_WIDTH] = [0; SIMD_WIDTH].map(|_| rng.gen());
                unsafe { _mm256_loadu_si256(v.as_ptr() as *const __m256i) }
            })
            .collect();
        let hash_base_rot = (0..input.rect_cnt()).map(|_| rng.gen()).collect();
        let placements = vec![zero; input.rect_cnt()];

        // 最初から幅を確保しておく
        let mut areas = [0; SIMD_WIDTH];

        for (h, w) in izip!(&rects.heights, &rects.widths) {
            for i in 0..SIMD_WIDTH {
                let w = w[i] as u64;
                let h = h[i] as u64;
                areas[i] += w * h;
            }
        }

        // 10%余裕を持たせる
        let default_width: [u16; 16] = areas.map(|a| (a as f64 * 1.1).sqrt() as u16);
        let default_width = unsafe { _mm256_loadu_si256(default_width.as_ptr() as *const __m256i) };

        Self {
            rects_h,
            rects_w,
            widths: default_width,
            heights: zero,
            placements_x0: placements.clone(),
            placements_x1: placements.clone(),
            placements_y0: placements.clone(),
            placements_y1: placements.clone(),
            hash: 0,
            hash_base_x,
            hash_base_y,
            hash_base_rot,
            turn: 0,
            avaliable_base_left: 0,
            avaliable_base_up: 0,
            avaliable_collision_left: 0,
            avaliable_collision_up: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct SmallState {
    placements_x0: __m256i,
    placements_x1: __m256i,
    placements_y0: __m256i,
    placements_y1: __m256i,
    old_widths: __m256i,
    old_heights: __m256i,
    new_widths: __m256i,
    new_heights: __m256i,
    hash: u32,
    hash_xor: u32,
    op: Op,
    score: i32,
    avaliable_base_left_xor: u128,
    avaliable_base_up_xor: u128,
    avaliable_collision_left_xor: u128,
    avaliable_collision_up_xor: u128,
}

impl SmallState {
    fn flip(mut self) -> Self {
        std::mem::swap(&mut self.placements_x0, &mut self.placements_y0);
        std::mem::swap(&mut self.placements_x1, &mut self.placements_y1);
        std::mem::swap(&mut self.old_widths, &mut self.old_heights);
        std::mem::swap(&mut self.new_widths, &mut self.new_heights);
        std::mem::swap(
            &mut self.avaliable_base_left_xor,
            &mut self.avaliable_base_up_xor,
        );
        std::mem::swap(
            &mut self.avaliable_collision_left_xor,
            &mut self.avaliable_collision_up_xor,
        );
        let dir = match self.op.dir() {
            Dir::Left => Dir::Up,
            Dir::Up => Dir::Left,
        };

        self.op = Op::new(self.op.rect_idx(), self.op.rotate(), dir, self.op.base());
        self
    }
}

impl Default for SmallState {
    fn default() -> Self {
        let zero = unsafe { _mm256_setzero_si256() };

        Self {
            placements_x0: zero,
            placements_x1: zero,
            placements_y0: zero,
            placements_y1: zero,
            old_widths: zero,
            old_heights: zero,
            new_widths: zero,
            new_heights: zero,
            hash: Default::default(),
            hash_xor: Default::default(),
            op: Default::default(),
            score: Default::default(),
            avaliable_base_left_xor: Default::default(),
            avaliable_base_up_xor: Default::default(),
            avaliable_collision_left_xor: Default::default(),
            avaliable_collision_up_xor: Default::default(),
        }
    }
}

impl beam::SmallState for SmallState {
    type Score = i32;
    type Hash = u32;
    type LargeState = LargeState;
    type Action = Op;

    fn raw_score(&self) -> Self::Score {
        self.score
    }

    fn hash(&self) -> Self::Hash {
        self.hash
    }

    fn apply(&self, state: &mut Self::LargeState) {
        state.placements_x0[state.turn] = self.placements_x0;
        state.placements_x1[state.turn] = self.placements_x1;
        state.placements_y0[state.turn] = self.placements_y0;
        state.placements_y1[state.turn] = self.placements_y1;
        state.widths = self.new_widths;
        state.heights = self.new_heights;
        state.hash ^= self.hash_xor;
        state.turn += 1;
        state.avaliable_base_left ^= self.avaliable_base_left_xor;
        state.avaliable_base_up ^= self.avaliable_base_up_xor;
        state.avaliable_collision_left ^= self.avaliable_collision_left_xor;
        state.avaliable_collision_up ^= self.avaliable_collision_up_xor;
    }

    fn rollback(&self, state: &mut Self::LargeState) {
        // placementsは削除しなくても良い
        state.widths = self.old_widths;
        state.heights = self.old_heights;
        state.hash ^= self.hash_xor;
        state.turn -= 1;
        state.avaliable_base_left ^= self.avaliable_base_left_xor;
        state.avaliable_base_up ^= self.avaliable_base_up_xor;
        state.avaliable_collision_left ^= self.avaliable_collision_left_xor;
        state.avaliable_collision_up ^= self.avaliable_collision_up_xor;
    }

    fn action(&self) -> Self::Action {
        self.op
    }
}

struct ActGen;

impl ActGen {
    fn gen_left_cand(
        &self,
        large_state: &LargeState,
        base: Option<usize>,
        rotate: bool,
        available_base_left_xor: u128,
        available_base_up_xor: u128,
        available_collision_left_xor: u128,
        available_collision_up_xor: u128,
    ) -> Option<SmallState> {
        let turn = large_state.turn;
        let rect_h = large_state.rects_h[turn];
        let rect_w = large_state.rects_w[turn];
        let placements_x0 = &large_state.placements_x0;
        let placements_x1 = &large_state.placements_x1;
        let placements_y0 = &large_state.placements_y0;
        let placements_y1 = &large_state.placements_y1;
        let heights = large_state.heights;
        let widths = large_state.widths;
        let hash = large_state.hash;
        let hash_base_x = &large_state.hash_base_x;
        let hash_base_y = &large_state.hash_base_y;
        let hash_base_rot = &large_state.hash_base_rot;
        let available_collision_left =
            large_state.avaliable_collision_left ^ available_collision_left_xor;

        unsafe {
            self.gen_cand(
                turn,
                base,
                rotate,
                rect_h,
                rect_w,
                placements_x0,
                placements_x1,
                placements_y0,
                placements_y1,
                heights,
                widths,
                hash,
                hash_base_x,
                hash_base_y,
                hash_base_rot,
                available_base_left_xor,
                available_base_up_xor,
                available_collision_left,
                available_collision_left_xor,
                available_collision_up_xor,
            )
        }
    }

    fn gen_up_cand(
        &self,
        large_state: &LargeState,
        base: Option<usize>,
        rotate: bool,
        available_base_left_xor: u128,
        available_base_up_xor: u128,
        avaliable_collision_left_xor: u128,
        avaliable_collision_up_xor: u128,
    ) -> Option<SmallState> {
        // 下からrectを置くので、水平・垂直を入れ替える
        let turn = large_state.turn;
        let rect_h = large_state.rects_w[turn];
        let rect_w = large_state.rects_h[turn];
        let placements_x0 = &large_state.placements_y0;
        let placements_x1 = &large_state.placements_y1;
        let placements_y0 = &large_state.placements_x0;
        let placements_y1 = &large_state.placements_x1;
        let heights = large_state.widths;
        let widths = large_state.heights;
        let hash = large_state.hash;
        let hash_base_x = &large_state.hash_base_y;
        let hash_base_y = &large_state.hash_base_x;
        let hash_base_rot = &large_state.hash_base_rot;
        let (available_base_left_xor, available_base_up_xor) =
            (available_base_up_xor, available_base_left_xor);
        let available_collision_left =
            large_state.avaliable_collision_up ^ avaliable_collision_up_xor;
        let (available_collision_left_xor, available_collision_up_xor) =
            (avaliable_collision_up_xor, avaliable_collision_left_xor);

        let state = unsafe {
            self.gen_cand(
                turn,
                base,
                rotate,
                rect_h,
                rect_w,
                placements_x0,
                placements_x1,
                placements_y0,
                placements_y1,
                heights,
                widths,
                hash,
                hash_base_x,
                hash_base_y,
                hash_base_rot,
                available_base_left_xor,
                available_base_up_xor,
                available_collision_left,
                available_collision_left_xor,
                available_collision_up_xor,
            )
        };

        state.map(|s| s.flip())
    }

    /// 右からrectを置く候補を生成する
    /// 下から置く場合はx, yをflipして呼び出し、返り値を再度flipすればよい
    #[target_feature(enable = "avx,avx2")]
    unsafe fn gen_cand(
        &self,
        turn: usize,
        base: Option<usize>,
        rotate: bool,
        mut rect_h: __m256i,
        mut rect_w: __m256i,
        placements_x0: &[__m256i],
        placements_x1: &[__m256i],
        placements_y0: &[__m256i],
        placements_y1: &[__m256i],
        heights: __m256i,
        widths: __m256i,
        hash: u32,
        hash_base_x: &[__m256i],
        hash_base_y: &[__m256i],
        hash_base_rot: &[u32],
        available_base_left_xor: u128,
        available_base_up_xor: u128,
        available_collision_left: u128,
        available_collision_left_xor: u128,
        available_collision_up_xor: u128,
    ) -> Option<SmallState> {
        if rotate {
            std::mem::swap(&mut rect_w, &mut rect_h);
        }

        let y0 = match base {
            Some(index) => placements_y1[index],
            None => _mm256_setzero_si256(),
        };
        let y1 = _mm256_add_epi16(y0, rect_h);

        // 長方形がどこに置かれるかを調べる
        let mut x0 = _mm256_setzero_si256();

        for rect_i in BitSetIterU128::new(available_collision_left) {
            // やっていることはジャッジコードと同じ
            //
            // let x = if y0.max(p_y0) < y1.min(p_y1) {
            //     p_x1
            // } else {
            //     0
            // };
            //
            // x0 = x0.max(x);
            let p_x1 = placements_x1[rect_i];
            let p_y0 = placements_y0[rect_i];
            let p_y1 = placements_y1[rect_i];
            let max_y0 = _mm256_max_epu16(y0, p_y0);
            let min_y1 = _mm256_min_epu16(y1, p_y1);

            let gt = _mm256_cmpgt_epi16(min_y1, max_y0);
            let x = _mm256_and_si256(p_x1, gt);
            x0 = _mm256_max_epu16(x0, x);
        }

        let x1 = _mm256_add_epi16(x0, rect_w);

        // 側面がピッタリくっついているかチェック
        let is_touching = match base {
            Some(base) => {
                // 側面が対象の長方形とmin(w0, w1) / 4以上隣接していることを要求する
                let p_x0 = placements_x0[base];
                let p_x1 = placements_x1[base];

                // 隣接長さを求める
                let max_x0 = _mm256_max_epu16(x0, p_x0);
                let min_x1 = _mm256_min_epu16(x1, p_x1);
                let touching_len = _mm256_subs_epu16(min_x1, max_x0);

                // min(配置した長方形の幅, 対象の長方形の幅) / 4を要求値とする
                let p_width = _mm256_sub_epi16(p_x1, p_x0);
                let min_edge = _mm256_min_epu16(rect_w, p_width);
                let required_len = _mm256_srli_epi16(min_edge, 2);

                // 要求値と比較
                let gt = _mm256_cmpgt_epi16(touching_len, required_len);
                gt
            }
            None => {
                // a == a を行うことで全bitが1になる
                _mm256_cmpeq_epi16(x0, x0)
            }
        };

        // 右ビットシフトすることで0 or 1を作り、合計を求める
        let is_touching = _mm256_srli_epi16(is_touching, 15);
        let is_touching_cnt = horizontal_add(is_touching) as usize;

        // ピッタリくっついていないものがあったらNG
        if is_touching_cnt < SIMD_WIDTH {
            return None;
        }

        // ハッシュ計算
        // 16bit * 16bit = 32bit を上位・下位16bitずつに分け、それぞれの和を取る
        let mul_hi = _mm256_mulhi_epu16(x0, hash_base_x[turn]);
        let mul_lo = _mm256_mullo_epi16(x0, hash_base_x[turn]);
        let x_hi_sum = horizontal_add(mul_hi) as u32;
        let x_lo_sum = horizontal_add(mul_lo) as u32;

        let mul_hi = _mm256_mulhi_epu16(y0, hash_base_y[turn]);
        let mul_lo = _mm256_mullo_epi16(y0, hash_base_y[turn]);
        let y_hi_sum = horizontal_add(mul_hi) as u32;
        let y_lo_sum = horizontal_add(mul_lo) as u32;

        // 適当に並べる
        // xとyの対称性がないとflip時に壊れるので注意
        let mut hash_xor = ((x_hi_sum ^ y_hi_sum) << 16) | (x_lo_sum ^ y_lo_sum);

        if rotate {
            hash_xor ^= hash_base_rot[turn];
        }

        let new_width = _mm256_max_epu16(x1, widths);
        let new_height = _mm256_max_epu16(y1, heights);

        // スコアを計算
        // スコアの昇順にソートした上で、減衰させながら和を取る
        // （期待値を最大化するよりは上振れを狙いたいため）
        let score = _mm256_add_epi16(new_height, new_width);
        let mut scores = [0u16; SIMD_WIDTH];
        _mm256_storeu_si256(scores.as_mut_ptr() as *mut __m256i, score);
        scores.sort_unstable();

        let mut mul = 1.0;
        let mut score = 0.0;
        const RATE: f64 = 0.9;

        for &x in scores.iter() {
            score -= x as f64 * mul;
            mul *= RATE;
        }

        let score = score.round() as i32;

        let hash = hash ^ hash_xor;
        let op = Op::new(turn, rotate, Dir::Left, base);

        let avaliable_base_left_xor = available_base_left_xor | (1 << turn);
        let avaliable_base_up_xor = available_base_up_xor | (1 << turn);
        let avaliable_collision_left_xor = available_collision_left_xor | (1 << turn);
        let avaliable_collision_up_xor = available_collision_up_xor | (1 << turn);

        Some(SmallState {
            placements_x0: x0,
            placements_x1: x1,
            placements_y0: y0,
            placements_y1: y1,
            old_widths: widths,
            old_heights: heights,
            new_widths: new_width,
            new_heights: new_height,
            hash,
            hash_xor,
            op,
            score,
            avaliable_base_left_xor,
            avaliable_base_up_xor,
            avaliable_collision_left_xor,
            avaliable_collision_up_xor,
        })
    }

    fn get_left_invalid_bases(large_state: &LargeState, turn: usize) -> u128 {
        if turn == 0 {
            return 0;
        }

        let prev_turn = turn - 1;
        let placements_x1 = &large_state.placements_x1[..prev_turn];
        let placements_y1 = &large_state.placements_y1[..prev_turn];

        let new_x1 = large_state.placements_x1[prev_turn];
        let new_y0 = large_state.placements_y0[prev_turn];
        let new_y1 = large_state.placements_y1[prev_turn];

        // prev_turnに置いたrectは必ずvalidなので対象外とする
        let available_flag = large_state.avaliable_base_left & !(1 << prev_turn);

        unsafe {
            Self::get_invalid_bases(
                placements_x1,
                placements_y1,
                new_x1,
                new_y0,
                new_y1,
                available_flag,
            )
        }
    }

    fn get_up_invalid_bases(large_state: &LargeState, turn: usize) -> u128 {
        if turn == 0 {
            return 0;
        }

        // x, yを反転させて呼び出す
        let prev_turn = turn - 1;
        let placements_x1 = &large_state.placements_y1[..prev_turn];
        let placements_y1 = &large_state.placements_x1[..prev_turn];

        let new_x1 = large_state.placements_y1[prev_turn];
        let new_y0 = large_state.placements_x0[prev_turn];
        let new_y1 = large_state.placements_x1[prev_turn];

        // prev_turnに置いたrectは必ずvalidなので対象外とする
        let available_flag = large_state.avaliable_base_up & !(1 << prev_turn);

        unsafe {
            Self::get_invalid_bases(
                placements_x1,
                placements_y1,
                new_x1,
                new_y0,
                new_y1,
                available_flag,
            )
        }
    }

    #[target_feature(enable = "avx,avx2")]
    unsafe fn get_invalid_bases(
        placements_x1: &[__m256i],
        placements_y1: &[__m256i],
        new_x1: __m256i,
        new_y0: __m256i,
        new_y1: __m256i,
        available_flag: u128,
    ) -> u128 {
        unsafe {
            const MIN_EDGE_LEN: u16 = round_u16(20000);
            let mut invalid_flag = 0;
            let min_edge_len = _mm256_set1_epi16(MIN_EDGE_LEN as i16);

            for rect_i in BitSetIterU128::new(available_flag) {
                let xi1 = placements_x1[rect_i];
                let yi1 = placements_y1[rect_i];

                // Yi1 + MIN_EDGE_LEN
                let yi1pl0 = _mm256_add_epi16(yi1, min_edge_len);

                // max(yi1, yj0) < min(yi1 + l0, yj1)
                let max = _mm256_max_epu16(yi1, new_y0);
                let min = _mm256_min_epu16(yi1pl0, new_y1);
                let pred_y = _mm256_cmpgt_epi16(min, max);

                // x_i1 ≦ x_j1
                // AVX2に整数の≦はないので、x_i1 > x_j1 を求めたあとにAND_NOTを取る
                let pred_x_not = _mm256_cmpgt_epi16(xi1, new_x1);

                // 新しく無効となった箱フラグ
                let invalid = _mm256_andnot_si256(pred_x_not, pred_y);

                // invalidなものが1つでもあったらピッタリくっつけられないのでNG
                let is_invalid = horizontal_or(invalid) > 0;
                invalid_flag |= (is_invalid as u128) << rect_i;
            }

            invalid_flag
        }
    }

    fn get_invalid_collisions_left(large_state: &LargeState, turn: usize) -> u128 {
        if turn == 0 {
            return 0;
        }

        let prev_turn = turn - 1;
        let placements_x1 = &large_state.placements_x1[..prev_turn];
        let placements_y0 = &large_state.placements_y0[..prev_turn];
        let placements_y1 = &large_state.placements_y1[..prev_turn];

        // prev_turnに置いたrectは必ずvalidなので対象外とする
        let collision_flag = large_state.avaliable_collision_left & !(1 << prev_turn);

        unsafe {
            Self::get_invalid_collisions(
                placements_x1,
                placements_y0,
                placements_y1,
                collision_flag,
            )
        }
    }

    fn get_invalid_collisions_up(large_state: &LargeState, turn: usize) -> u128 {
        if turn == 0 {
            return 0;
        }

        // x, yを反転させて呼び出す
        let prev_turn = turn - 1;
        let placements_x1 = &large_state.placements_y1[..prev_turn];
        let placements_y0 = &large_state.placements_x0[..prev_turn];
        let placements_y1 = &large_state.placements_x1[..prev_turn];

        // prev_turnに置いたrectは必ずvalidなので対象外とする
        let collision_flag = large_state.avaliable_collision_up & !(1 << prev_turn);

        unsafe {
            Self::get_invalid_collisions(
                placements_x1,
                placements_y0,
                placements_y1,
                collision_flag,
            )
        }
    }

    #[target_feature(enable = "avx,avx2")]
    unsafe fn get_invalid_collisions(
        placements_x1: &[__m256i],
        placements_y0: &[__m256i],
        placements_y1: &[__m256i],
        collision_flag: u128,
    ) -> u128 {
        unsafe {
            let mut invalid_flag = 0;

            for rect_i in BitSetIterU128::new(collision_flag) {
                let mut y_top = placements_y0[rect_i];
                let xi1 = placements_x1[rect_i];
                let yi1 = placements_y1[rect_i];
                let rect_next = rect_i + 1;

                for (&xj1, &yj0, &yj1) in izip!(
                    &placements_x1[rect_next..],
                    &placements_y0[rect_next..],
                    &placements_y1[rect_next..]
                ) {
                    // 上側から被っている領域
                    // top = max(top, y_j1 if x_i1 < x_j1 and y_j0 < y_i1 else 0)
                    let x_gt = _mm256_cmpgt_epi16(xj1, xi1);
                    let y_gt = _mm256_cmpgt_epi16(yi1, yj0);
                    let pred = _mm256_and_si256(x_gt, y_gt);
                    let y_top_j = _mm256_and_si256(yj1, pred);
                    y_top = _mm256_max_epi16(y_top, y_top_j);
                }

                // 右側面に露出している長さが0以上であることを要求する
                let valid = _mm256_cmpgt_epi16(yi1, y_top);

                // 全てinvalidだったら見る必要なし
                // これで弾かれず漏れる可能性もあるが、安全側なので許容する
                let is_invalid = horizontal_or(valid) == 0;
                invalid_flag |= (is_invalid as u128) << rect_i;
            }

            invalid_flag
        }
    }
}

impl beam::ActGen<SmallState> for ActGen {
    fn generate(
        &self,
        _small_state: &SmallState,
        large_state: &<SmallState as beam::SmallState>::LargeState,
        next_states: &mut Vec<SmallState>,
    ) {
        // 無効な候補の列挙
        let invalid_bases_left = Self::get_left_invalid_bases(large_state, large_state.turn);
        let invalid_bases_up = Self::get_up_invalid_bases(large_state, large_state.turn);
        let avaliable_bases_left = large_state.avaliable_base_left & !invalid_bases_left;
        let avaliable_bases_up = large_state.avaliable_base_up & !invalid_bases_up;

        // 見なくて良い衝突判定の列挙
        let invalid_collisions_left =
            Self::get_invalid_collisions_left(large_state, large_state.turn);
        let invalid_collisions_up = Self::get_invalid_collisions_up(large_state, large_state.turn);

        // 生成
        let rotates = [false, true];

        for rotate in rotates {
            // Left
            next_states.extend(self.gen_left_cand(
                &large_state,
                None,
                rotate,
                invalid_bases_left,
                invalid_bases_up,
                invalid_collisions_left,
                invalid_collisions_up,
            ));

            for i in BitSetIterU128::new(avaliable_bases_left) {
                next_states.extend(self.gen_left_cand(
                    &large_state,
                    Some(i),
                    rotate,
                    invalid_bases_left,
                    invalid_bases_up,
                    invalid_collisions_left,
                    invalid_collisions_up,
                ));
            }

            // Up
            next_states.extend(self.gen_up_cand(
                &large_state,
                None,
                rotate,
                invalid_bases_left,
                invalid_bases_up,
                invalid_collisions_left,
                invalid_collisions_up,
            ));

            for i in BitSetIterU128::new(avaliable_bases_up) {
                next_states.extend(self.gen_up_cand(
                    &large_state,
                    Some(i),
                    rotate,
                    invalid_bases_left,
                    invalid_bases_up,
                    invalid_collisions_left,
                    invalid_collisions_up,
                ));
            }
        }
    }
}
