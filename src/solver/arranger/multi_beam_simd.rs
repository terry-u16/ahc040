use super::Arranger;
use crate::{
    beam::{self, BayesianBeamWidthSuggester},
    problem::{Dir, Input, Op, Rect},
    solver::estimator::Estimator,
};
use itertools::{izip, Itertools};
use rand::Rng;
use std::arch::x86_64::*;

const PARALLEL_CNT: usize = 16;

/// 多変量正規分布からN個の長方形を16インスタンス生成し、それぞれについて並行に操作を行うビームサーチ。
/// 考え方は粒子フィルタなどと同じで、非線形性を考慮するために多数のインスタンスでシミュレートする。
/// 内部的にAVX2を使用して高速化している。
pub(super) struct MultiBeamArrangerSimd<'a, R: Rng> {
    estimator: &'a Estimator,
    rng: &'a mut R,
    duration_sec: f64,
}

impl<'a, R: Rng> MultiBeamArrangerSimd<'a, R> {
    pub(super) fn new(estimator: &'a Estimator, rng: &'a mut R, duration_sec: f64) -> Self {
        Self {
            estimator,
            rng,
            duration_sec,
        }
    }
}

impl<R: Rng> Arranger for MultiBeamArrangerSimd<'_, R> {
    fn arrange(&mut self, input: &Input) -> Vec<Op> {
        let since = std::time::Instant::now();
        let sampler = self.estimator.get_sampler();

        let mut rects = vec![[Rect::default(); PARALLEL_CNT]; input.rect_cnt()];

        for i in 0..PARALLEL_CNT {
            let rects_i = sampler.sample(self.rng);

            for j in 0..input.rect_cnt() {
                rects[j][i] = rects_i[j];
            }
        }

        let large_state = unsafe { LargeState::new(input.clone(), rects, self.rng) };
        let small_state = SmallState::default();
        let act_gen = ActGen;

        let remaining_time = self.duration_sec - since.elapsed().as_secs_f64();
        let mut beam = beam::BeamSearch::new(large_state, small_state, act_gen);
        let standard_beam_width = 100_000_000 / (input.rect_cnt() as usize).pow(3);
        let beam_width_suggester = BayesianBeamWidthSuggester::new(
            input.rect_cnt(),
            5,
            remaining_time,
            standard_beam_width,
            1,
            10000,
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
}

impl LargeState {
    #[target_feature(enable = "avx,avx2")]
    unsafe fn new(
        input: Input,
        rects: Vec<[Rect; PARALLEL_CNT]>,
        rng: &mut impl rand::Rng,
    ) -> Self {
        let zero = unsafe { _mm256_setzero_si256() };
        let rects_h = rects
            .iter()
            .map(|rect| {
                let h = rect.map(|r| Self::round_u16(r.height()));
                unsafe { _mm256_loadu_si256(h.as_ptr() as *const __m256i) }
            })
            .collect_vec();
        let rects_w = rects
            .iter()
            .map(|rect| {
                let w = rect.map(|r| Self::round_u16(r.width()));
                unsafe { _mm256_loadu_si256(w.as_ptr() as *const __m256i) }
            })
            .collect_vec();
        let hash_base_x = (0..input.rect_cnt())
            .map(|_| {
                let v: [u16; PARALLEL_CNT] = [0; PARALLEL_CNT].map(|_| rng.gen());
                unsafe { _mm256_loadu_si256(v.as_ptr() as *const __m256i) }
            })
            .collect();
        let hash_base_y = (0..input.rect_cnt())
            .map(|_| {
                let v: [u16; PARALLEL_CNT] = [0; PARALLEL_CNT].map(|_| rng.gen());
                unsafe { _mm256_loadu_si256(v.as_ptr() as *const __m256i) }
            })
            .collect();
        let hash_base_rot = (0..input.rect_cnt()).map(|_| rng.gen()).collect();
        let placements = vec![zero; input.rect_cnt()];

        Self {
            rects_h,
            rects_w,
            widths: zero,
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
        }
    }

    const fn round_u16(x: u32) -> u16 {
        // 座標の最大値は2^22 = 4_194_304とする（さすがに大丈夫やろ……）
        // これを16bitに収めるためには、6bit右シフトすればよい（64単位で丸められる）
        (x >> 6) as u16
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
}

impl SmallState {
    fn flip(mut self) -> Self {
        std::mem::swap(&mut self.placements_x0, &mut self.placements_y0);
        std::mem::swap(&mut self.placements_x1, &mut self.placements_y1);
        std::mem::swap(&mut self.old_widths, &mut self.old_heights);
        std::mem::swap(&mut self.new_widths, &mut self.new_heights);
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
    }

    fn rollback(&self, state: &mut Self::LargeState) {
        // placementsは削除しなくても良い
        state.widths = self.old_widths;
        state.heights = self.old_heights;
        state.hash ^= self.hash_xor;
        state.turn -= 1;
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
            )
        }
    }

    fn gen_up_cand(
        &self,
        large_state: &LargeState,
        base: Option<usize>,
        rotate: bool,
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
            let max_y0 = _mm256_max_epu16(y0, *p_y0);
            let min_y1 = _mm256_min_epu16(y1, *p_y1);

            let gt = _mm256_cmpgt_epi16(min_y1, max_y0);
            let x = _mm256_and_si256(*p_x1, gt);
            x0 = _mm256_max_epu16(x0, x);
        }

        let x1 = _mm256_add_epi16(x0, rect_w);

        // 側面がピッタリくっついているかチェック
        let is_touching = match base {
            Some(base) => {
                // if x0.max(p.x0) < x1.min(p.x1) {
                //     1
                // } else {
                //     0
                // }
                let p_x0 = placements_x0[base];
                let p_x1 = placements_x1[base];
                let max_x0 = _mm256_max_epu16(x0, p_x0);
                let min_x1 = _mm256_min_epu16(x1, p_x1);
                let gt = _mm256_cmpgt_epi16(min_x1, max_x0);
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

        // ピッタリくっついているものが半分以下だったら中止
        if is_touching_cnt < PARALLEL_CNT / 2 {
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
        // width + height の小さいもの8つを取り、それらの和をスコアとする
        // （期待値を最大化するよりは上振れを狙いたいため）
        const SCORE_TAKE_CNT: usize = PARALLEL_CNT / 2;
        let score = _mm256_add_epi16(new_height, new_width);
        let mut scores = [0u16; PARALLEL_CNT];
        _mm256_storeu_si256(scores.as_mut_ptr() as *mut __m256i, score);
        scores.sort_unstable();
        let score = scores
            .iter()
            .take(SCORE_TAKE_CNT)
            .map(|&x| -(x as i32))
            .sum();

        let hash = hash ^ hash_xor;
        let op = Op::new(turn, rotate, Dir::Left, base);

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
        })
    }
}

unsafe fn horizontal_add(x: __m256i) -> i32 {
    let low = _mm256_extracti128_si256(x, 0);
    let high = _mm256_extracti128_si256(x, 1);
    let x = _mm_add_epi16(low, high);
    let x = _mm_hadd_epi16(x, x);
    let x = _mm_hadd_epi16(x, x);
    let x = _mm_extract_epi16(x, 0);
    x
}

impl beam::ActGen<SmallState> for ActGen {
    fn generate(
        &self,
        _small_state: &SmallState,
        large_state: &<SmallState as beam::SmallState>::LargeState,
        next_states: &mut Vec<SmallState>,
    ) {
        // 0ターン目に回転させると、ハッシュが異なるが全く同じ状態が2つできてしまう
        let rotates = if large_state.turn == 0 {
            vec![false]
        } else {
            vec![false, true]
        };

        for rotate in rotates {
            // Left
            next_states.extend(self.gen_left_cand(&large_state, None, rotate));

            for i in 0..large_state.turn {
                next_states.extend(self.gen_left_cand(&large_state, Some(i), rotate));
            }

            // Up
            next_states.extend(self.gen_up_cand(&large_state, None, rotate));

            for i in 0..large_state.turn {
                next_states.extend(self.gen_up_cand(&large_state, Some(i), rotate));
            }
        }
    }
}
