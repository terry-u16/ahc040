use super::Arranger;
use crate::{
    beam::{self, BayesianBeamWidthSuggester},
    problem::{Dir, Input, Op},
    solver::{
        estimator::Sampler,
        simd::{horizontal_add, horizontal_or, round_u16, AlignedU16, SimdRectSet, AVX2_U16_W},
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
        let mut beam = beam::BeamSearch::new(act_gen);
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
        let (ops, _) = beam.run(
            large_state,
            small_state,
            input.rect_cnt(),
            beam_width_suggester,
            deduplicator,
        );

        ops
    }
}

#[derive(Debug, Clone)]
struct LargeState {
    // 256bitのレジスタをunsigned 16bit integer x16で使う
    rects_h: [AlignedU16; Input::MAX_RECT_CNT],
    rects_w: [AlignedU16; Input::MAX_RECT_CNT],
    widths: AlignedU16,
    heights: AlignedU16,
    placements_x0: [AlignedU16; Input::MAX_RECT_CNT],
    placements_x1: [AlignedU16; Input::MAX_RECT_CNT],
    placements_y0: [AlignedU16; Input::MAX_RECT_CNT],
    placements_y1: [AlignedU16; Input::MAX_RECT_CNT],
    hash: u32,
    hash_base_x: [AlignedU16; Input::MAX_RECT_CNT],
    hash_base_y: [AlignedU16; Input::MAX_RECT_CNT],
    hash_base_rot: [u32; Input::MAX_RECT_CNT],
    turn: usize,
    avaliable_base_left: u128,
    avaliable_base_up: u128,
}

impl LargeState {
    #[target_feature(enable = "avx,avx2")]
    unsafe fn new(input: Input, rects: SimdRectSet, rng: &mut impl rand::Rng) -> Self {
        let mut rects_h = [AlignedU16::ZERO; Input::MAX_RECT_CNT];
        let mut rects_w = [AlignedU16::ZERO; Input::MAX_RECT_CNT];

        for i in 0..input.rect_cnt() {
            rects_h[i] = AlignedU16(rects.heights[i]);
            rects_w[i] = AlignedU16(rects.widths[i]);
        }

        let hash_base_x = core::array::from_fn(|_| {
            let mut v = [0; AVX2_U16_W];
            rng.fill(&mut v);
            AlignedU16(v)
        });
        let hash_base_y = core::array::from_fn(|_| {
            let mut v = [0; AVX2_U16_W];
            rng.fill(&mut v);
            AlignedU16(v)
        });
        let mut hash_base_rot = [0; Input::MAX_RECT_CNT];
        rng.fill(&mut hash_base_rot);
        let placements = [AlignedU16::ZERO; Input::MAX_RECT_CNT];

        // 最初から幅を確保しておく
        let mut areas = [0; AVX2_U16_W];

        for (h, w) in izip!(&rects.heights, &rects.widths) {
            for i in 0..AVX2_U16_W {
                let w = w[i] as u64;
                let h = h[i] as u64;
                areas[i] += w * h;
            }
        }

        // 10%余裕を持たせる
        let default_width = AlignedU16(areas.map(|a| (a as f64 * 1.1).sqrt() as u16));

        Self {
            rects_h,
            rects_w,
            widths: default_width,
            heights: AlignedU16::ZERO,
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
        }
    }
}

#[derive(Debug, Clone)]
struct SmallState {
    placements_x0: AlignedU16,
    placements_x1: AlignedU16,
    placements_y0: AlignedU16,
    placements_y1: AlignedU16,
    old_widths: AlignedU16,
    old_heights: AlignedU16,
    new_widths: AlignedU16,
    new_heights: AlignedU16,
    hash: u32,
    hash_xor: u32,
    op: Op,
    score: i32,
    avaliable_base_left_xor: u128,
    avaliable_base_up_xor: u128,
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
        Self {
            placements_x0: AlignedU16::ZERO,
            placements_x1: AlignedU16::ZERO,
            placements_y0: AlignedU16::ZERO,
            placements_y1: AlignedU16::ZERO,
            old_widths: AlignedU16::ZERO,
            old_heights: AlignedU16::ZERO,
            new_widths: AlignedU16::ZERO,
            new_heights: AlignedU16::ZERO,
            hash: Default::default(),
            hash_xor: Default::default(),
            op: Default::default(),
            score: Default::default(),
            avaliable_base_left_xor: Default::default(),
            avaliable_base_up_xor: Default::default(),
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
    }

    fn rollback(&self, state: &mut Self::LargeState) {
        // placementsは削除しなくても良い
        state.widths = self.old_widths;
        state.heights = self.old_heights;
        state.hash ^= self.hash_xor;
        state.turn -= 1;
        state.avaliable_base_left ^= self.avaliable_base_left_xor;
        state.avaliable_base_up ^= self.avaliable_base_up_xor;
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
    ) -> Option<SmallState> {
        let turn = large_state.turn;
        let rect_h = large_state.rects_h[turn].into();
        let rect_w = large_state.rects_w[turn].into();
        let placements_x0 = &large_state.placements_x0[..turn];
        let placements_x1 = &large_state.placements_x1[..turn];
        let placements_y0 = &large_state.placements_y0[..turn];
        let placements_y1 = &large_state.placements_y1[..turn];
        let heights = large_state.heights.into();
        let widths = large_state.widths.into();
        let hash = large_state.hash;
        let hash_base_x = &large_state.hash_base_x;
        let hash_base_y = &large_state.hash_base_y;
        let hash_base_rot = &large_state.hash_base_rot;

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
    ) -> Option<SmallState> {
        // 下からrectを置くので、水平・垂直を入れ替える
        let turn = large_state.turn;
        let rect_h = large_state.rects_w[turn].into();
        let rect_w = large_state.rects_h[turn].into();
        let placements_x0 = &large_state.placements_y0[..turn];
        let placements_x1 = &large_state.placements_y1[..turn];
        let placements_y0 = &large_state.placements_x0[..turn];
        let placements_y1 = &large_state.placements_x1[..turn];
        let heights = large_state.widths.into();
        let widths = large_state.heights.into();
        let hash = large_state.hash;
        let hash_base_x = &large_state.hash_base_y;
        let hash_base_y = &large_state.hash_base_x;
        let hash_base_rot = &large_state.hash_base_rot;
        let (available_base_left_xor, available_base_up_xor) =
            (available_base_up_xor, available_base_left_xor);

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
        mut rect_h: AlignedU16,
        mut rect_w: AlignedU16,
        placements_x0: &[AlignedU16],
        placements_x1: &[AlignedU16],
        placements_y0: &[AlignedU16],
        placements_y1: &[AlignedU16],
        heights: AlignedU16,
        widths: AlignedU16,
        hash: u32,
        hash_base_x: &[AlignedU16],
        hash_base_y: &[AlignedU16],
        hash_base_rot: &[u32],
        available_base_left_xor: u128,
        available_base_up_xor: u128,
    ) -> Option<SmallState> {
        if rotate {
            std::mem::swap(&mut rect_w, &mut rect_h);
        }
        let rect_w = rect_w.load();
        let rect_h = rect_h.load();

        let y0 = match base {
            Some(index) => placements_y1[index].load(),
            None => _mm256_setzero_si256(),
        };
        let y1 = _mm256_add_epi16(y0, rect_h);

        // 長方形がどこに置かれるかを調べる
        let mut x0 = _mm256_setzero_si256();

        for (p_x1, p_y0, p_y1) in izip!(placements_x1, placements_y0, placements_y1) {
            // やっていることはジャッジコードと同じ
            //
            // let x = if y0.max(p_y0) < y1.min(p_y1) {
            //     p_x1
            // } else {
            //     0
            // };
            //
            // x0 = x0.max(x);
            let max_y0 = _mm256_max_epu16(y0, p_y0.load());
            let min_y1 = _mm256_min_epu16(y1, p_y1.load());

            let gt = _mm256_cmpgt_epi16(min_y1, max_y0);
            let x = _mm256_and_si256(p_x1.load(), gt);
            x0 = _mm256_max_epu16(x0, x);
        }

        let x1 = _mm256_add_epi16(x0, rect_w);

        // 側面がピッタリくっついているかチェック
        let is_touching = match base {
            Some(base) => {
                // 側面が対象の長方形とmin(w0, w1) / 4以上隣接していることを要求する
                let p_x0 = placements_x0[base].load();
                let p_x1 = placements_x1[base].load();

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
        if is_touching_cnt < AVX2_U16_W {
            return None;
        }

        // ハッシュ計算
        // 16bit * 16bit = 32bit を上位・下位16bitずつに分け、それぞれの和を取る
        let hash_base_x = hash_base_x[turn].load();
        let hash_base_y = hash_base_y[turn].load();

        let mul_hi = _mm256_mulhi_epu16(x0, hash_base_x);
        let mul_lo = _mm256_mullo_epi16(x0, hash_base_x);
        let x_hi_sum = horizontal_add(mul_hi) as u32;
        let x_lo_sum = horizontal_add(mul_lo) as u32;

        let mul_hi = _mm256_mulhi_epu16(y0, hash_base_y);
        let mul_lo = _mm256_mullo_epi16(y0, hash_base_y);
        let y_hi_sum = horizontal_add(mul_hi) as u32;
        let y_lo_sum = horizontal_add(mul_lo) as u32;

        // 適当に並べる
        // xとyの対称性がないとflip時に壊れるので注意
        let mut hash_xor = ((x_hi_sum ^ y_hi_sum) << 16) | (x_lo_sum ^ y_lo_sum);

        if rotate {
            hash_xor ^= hash_base_rot[turn];
        }

        let heights = heights.load();
        let widths = widths.load();
        let new_width = _mm256_max_epu16(x1, widths);
        let new_height = _mm256_max_epu16(y1, heights);

        // スコアを計算
        // スコアの昇順にソートした上で、減衰させながら和を取る
        // （期待値を最大化するよりは上振れを狙いたいため）
        let score = _mm256_add_epi16(new_height, new_width);
        let mut scores = [0u16; AVX2_U16_W];
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

        Some(SmallState {
            placements_x0: x0.into(),
            placements_x1: x1.into(),
            placements_y0: y0.into(),
            placements_y1: y1.into(),
            old_widths: widths.into(),
            old_heights: heights.into(),
            new_widths: new_width.into(),
            new_heights: new_height.into(),
            hash,
            hash_xor,
            op,
            score,
            avaliable_base_left_xor,
            avaliable_base_up_xor,
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
        placements_x1: &[AlignedU16],
        placements_y1: &[AlignedU16],
        new_x1: AlignedU16,
        new_y0: AlignedU16,
        new_y1: AlignedU16,
        available_flag: u128,
    ) -> u128 {
        unsafe {
            const MIN_EDGE_LEN: u16 = round_u16(20000);
            let mut invalid_flag = 0;
            let min_edge_len = _mm256_set1_epi16(MIN_EDGE_LEN as i16);
            let new_x1 = new_x1.load();
            let new_y0 = new_y0.load();
            let new_y1 = new_y1.load();

            for rect_i in BitSetIterU128::new(available_flag) {
                let xi1 = placements_x1[rect_i].load();
                let yi1 = placements_y1[rect_i].load();

                // Yi1 + MIN_EDGE_LEN
                let yi1pl0 = _mm256_add_epi16(yi1, min_edge_len);

                // 新しく無効となった箱フラグ
                let mut invalid = _mm256_setzero_si256();

                // max(yi1, yj0) < min(yi1 + l0, yj1)
                let max = _mm256_max_epu16(yi1, new_y0);
                let min = _mm256_min_epu16(yi1pl0, new_y1);
                let pred_y = _mm256_cmpgt_epi16(min, max);

                // x_i1 ≦ x_j1
                // AVX2に整数の≦はないので、x_i1 > x_j1 を求めたあとにAND_NOTを取る
                let pred_x_not = _mm256_cmpgt_epi16(xi1, new_x1);
                let pred = _mm256_andnot_si256(pred_x_not, pred_y);
                invalid = _mm256_or_si256(invalid, pred);

                // invalidなものが1つでもあったらピッタリくっつけられないのでNG
                let is_invalid = horizontal_or(invalid) > 0;
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
            ));

            for i in BitSetIterU128::new(avaliable_bases_left) {
                next_states.extend(self.gen_left_cand(
                    &large_state,
                    Some(i),
                    rotate,
                    invalid_bases_left,
                    invalid_bases_up,
                ));
            }

            // Up
            next_states.extend(self.gen_up_cand(
                &large_state,
                None,
                rotate,
                invalid_bases_left,
                invalid_bases_up,
            ));

            for i in BitSetIterU128::new(avaliable_bases_up) {
                next_states.extend(self.gen_up_cand(
                    &large_state,
                    Some(i),
                    rotate,
                    invalid_bases_left,
                    invalid_bases_up,
                ));
            }
        }
    }
}
