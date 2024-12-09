use crate::{
    beam::{self, BayesianBeamWidthSuggester},
    problem::{params::Params, Dir, Input, Op},
    solver::simd::*,
    util::BitSetIterU128,
};
use itertools::izip;
use ordered_float::OrderedFloat;
use std::arch::x86_64::*;

/// 多変量正規分布からN個の長方形を16インスタンス生成し、それぞれについて並行に操作を行うビームサーチ。
/// 考え方は粒子フィルタなどと同じで、非線形性を考慮するために多数のインスタンスでシミュレートする。
/// 内部的にAVX2を使用して高速化している。
pub struct MultiBeamArrangerSimd;

impl MultiBeamArrangerSimd {
    pub fn arrange(
        &mut self,
        input: &Input,
        end_turn: usize,
        rects: SimdRectSet,
        duration_sec: f64,
    ) -> Vec<Op> {
        let since = std::time::Instant::now();

        let large_state = unsafe { LargeState::new(input.clone(), rects) };
        let small_state = SmallState::default();
        let act_gen = ActGen::new();

        let remaining_time = (duration_sec - since.elapsed().as_secs_f64()).max(0.01);
        let mut beam = beam::BeamSearch::new(act_gen);
        let standard_beam_width = 100_000 / (input.rect_cnt() as usize);
        let beam_width_suggester = BayesianBeamWidthSuggester::new(
            input.rect_cnt(),
            5,
            remaining_time,
            standard_beam_width,
            1,
            30000,
            0,
        );
        let deduplicator = beam::NoOpDeduplicator;
        let (ops, _) = beam.run(
            large_state,
            small_state,
            end_turn,
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
    width_limit: AlignedU16,
    height_limit: AlignedU16,
    placements_x0: [AlignedU16; Input::MAX_RECT_CNT],
    placements_x1: [AlignedU16; Input::MAX_RECT_CNT],
    placements_y0: [AlignedU16; Input::MAX_RECT_CNT],
    placements_y1: [AlignedU16; Input::MAX_RECT_CNT],
    turn: usize,
    avaliable_base_up: u128,
    min_rect_size: AlignedU16,
}

impl LargeState {
    #[target_feature(enable = "avx,avx2")]
    unsafe fn new(input: Input, rects: SimdRectSet) -> Self {
        let mut rects_h = [AlignedU16::ZERO; Input::MAX_RECT_CNT];
        let mut rects_w = [AlignedU16::ZERO; Input::MAX_RECT_CNT];

        for i in 0..input.rect_cnt() {
            rects_h[i] = AlignedU16(rects.heights[i]);
            rects_w[i] = AlignedU16(rects.widths[i]);
        }

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

        // 少し余裕を持たせる
        let default_width =
            AlignedU16(areas.map(|a| (a as f64 * Params::get().borrow().width_buf).sqrt() as u16));

        let width_limit = default_width;
        let height_limit = AlignedU16([u16::MAX / 2; AVX2_U16_W]);

        let min_rect_size = unsafe {
            let mut min = _mm256_set1_epi16(i16::MAX);

            for size in rects_h[..input.rect_cnt()]
                .iter()
                .chain(rects_w[..input.rect_cnt()].iter())
            {
                min = _mm256_min_epu16(min, size.load());
            }

            let mut min_rect_size: [u16; AVX2_U16_W] = [0; AVX2_U16_W];
            _mm256_storeu_si256(min_rect_size.as_mut_ptr() as *mut __m256i, min);

            AlignedU16(min_rect_size)
        };

        Self {
            rects_h,
            rects_w,
            widths: default_width,
            heights: AlignedU16::ZERO,
            width_limit,
            height_limit,
            placements_x0: placements.clone(),
            placements_x1: placements.clone(),
            placements_y0: placements.clone(),
            placements_y1: placements.clone(),
            turn: 0,
            avaliable_base_up: 0,
            min_rect_size,
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
    op: Op,
    score: f32,
    avaliable_base_up_xor: u128,
}

impl SmallState {
    fn new(
        mut placements_x0: AlignedU16,
        mut placements_x1: AlignedU16,
        mut placements_y0: AlignedU16,
        mut placements_y1: AlignedU16,
        mut old_widths: AlignedU16,
        mut old_heights: AlignedU16,
        mut new_widths: AlignedU16,
        mut new_heights: AlignedU16,
        mut op: Op,
        avaliable_base_up_xor: u128,
        flip: bool,
    ) -> Self {
        if flip {
            std::mem::swap(&mut placements_x0, &mut placements_y0);
            std::mem::swap(&mut placements_x1, &mut placements_y1);
            std::mem::swap(&mut old_widths, &mut old_heights);
            std::mem::swap(&mut new_widths, &mut new_heights);
            let dir = match op.dir() {
                Dir::Left => Dir::Up,
                Dir::Up => Dir::Left,
            };

            op = Op::new(op.rect_idx(), op.rotate(), dir, op.base());
        }

        let score = unsafe { Self::calc_score(new_heights) };

        Self {
            placements_x0,
            placements_x1,
            placements_y0,
            placements_y1,
            old_widths,
            old_heights,
            new_widths,
            new_heights,
            op,
            score,
            avaliable_base_up_xor,
        }
    }

    #[target_feature(enable = "avx,avx2")]
    unsafe fn calc_score(height: AlignedU16) -> f32 {
        // スコアの昇順にソートした上で、減衰させながら和を取る
        // （期待値を最大化するよりは上振れを狙いたいため）
        thread_local!(static SCORE_MUL: [AlignedF32; 2] = {
            let score_mul = Params::get().borrow().parallel_score_mul;
            [
                AlignedF32(std::array::from_fn(|i| score_mul.powi(i as i32))),
                AlignedF32(std::array::from_fn(|i| {
                    score_mul.powi((i + AVX2_F32_W) as i32)
                })),
            ]
        });

        SCORE_MUL.with(|score_mul| {
            let score = bitonic_sort_u16(height.load());

            // u16 x 16 -> (u32 x 8) x 2 -> (f32 x 8) x 2
            let score_low = _mm256_castsi256_si128(score);
            let score_high = _mm256_extracti128_si256(score, 1);
            let score_u32_low = _mm256_cvtepu16_epi32(score_low);
            let score_u32_high = _mm256_cvtepu16_epi32(score_high);
            let score_f32_low = _mm256_cvtepi32_ps(score_u32_low);
            let score_f32_high = _mm256_cvtepi32_ps(score_u32_high);

            // 用意していた係数とかける
            let score_mul_low = score_mul[0].load();
            let score_mul_high = score_mul[1].load();
            let score_low = _mm256_mul_ps(score_f32_low, score_mul_low);
            let score_high = _mm256_mul_ps(score_f32_high, score_mul_high);
            let score = _mm256_add_ps(score_low, score_high);
            let score = horizontal_add_f32(score);
            -score
        })
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
            op: Default::default(),
            score: Default::default(),
            avaliable_base_up_xor: Default::default(),
        }
    }
}

impl beam::SmallState for SmallState {
    type Score = OrderedFloat<f32>;
    type Hash = u32;
    type LargeState = LargeState;
    type Action = Op;

    fn raw_score(&self) -> Self::Score {
        // 最終ターンは縦と横からスコアを再計算
        let score_h =
            unsafe { Self::calc_score(self.new_heights) + Self::calc_score(self.new_widths) };
        OrderedFloat(score_h)
    }

    fn beam_score(&self) -> Self::Score {
        OrderedFloat(self.score)
    }

    fn hash(&self) -> Self::Hash {
        0
    }

    fn apply(&self, state: &mut Self::LargeState) {
        state.placements_x0[state.turn] = self.placements_x0;
        state.placements_x1[state.turn] = self.placements_x1;
        state.placements_y0[state.turn] = self.placements_y0;
        state.placements_y1[state.turn] = self.placements_y1;
        state.widths = self.new_widths;
        state.heights = self.new_heights;
        state.turn += 1;
        state.avaliable_base_up ^= self.avaliable_base_up_xor;
    }

    fn rollback(&self, state: &mut Self::LargeState) {
        // placementsは削除しなくても良い
        state.widths = self.old_widths;
        state.heights = self.old_heights;
        state.turn -= 1;
        state.avaliable_base_up ^= self.avaliable_base_up_xor;
    }

    fn action(&self) -> Self::Action {
        self.op
    }
}

struct ActGen;

impl ActGen {
    fn new() -> Self {
        Self
    }

    // 以前は左向きのrectを置いたりしていたので、その名残で残っている
    fn gen_up_cand(
        &self,
        large_state: &LargeState,
        base: Option<usize>,
        rotate: bool,
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
        let width_limit = large_state.height_limit;
        let height_limit = large_state.width_limit;

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
                available_base_up_xor,
                width_limit,
                height_limit,
                true,
            )
        }
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
        available_base_up_xor: u128,
        width_limit: AlignedU16,
        height_limit: AlignedU16,
        flip: bool,
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

        // 1つでもリミット超えたらNG
        let width_limit = _mm256_cmpgt_epi16(x1, width_limit.load());
        let height_limit = _mm256_cmpgt_epi16(y1, height_limit.load());
        let over_limit = _mm256_or_si256(width_limit, height_limit);
        let over_limit = _mm256_movemask_epi8(over_limit);

        if over_limit != 0 {
            return None;
        }

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

        // ピッタリくっついているものが閾値未満だったらNG
        // 右シフトで0xffを0x01に変換してから足すことで個数をカウントできる
        let is_touching = _mm256_srli_epi16(is_touching, 15);
        let touching_count = horizontal_add_u16(is_touching) as usize;

        if touching_count < Params::get().borrow().touching_threshold {
            return None;
        }

        let heights = heights.load();
        let widths = widths.load();
        let new_width = _mm256_max_epu16(x1, widths);
        let new_height = _mm256_max_epu16(y1, heights);

        let op = Op::new(turn, rotate, Dir::Left, base);

        let avaliable_base_up_xor = available_base_up_xor | (1 << turn);

        Some(SmallState::new(
            x0.into(),
            x1.into(),
            y0.into(),
            y1.into(),
            widths.into(),
            heights.into(),
            new_width.into(),
            new_height.into(),
            op,
            avaliable_base_up_xor,
            flip,
        ))
    }

    fn get_up_invalid_bases(large_state: &LargeState, turn: usize) -> u128 {
        if turn == 0 {
            return 0;
        }

        // x, yを反転させて呼び出す
        let prev_turn = turn - 1;
        let placements_x1 = &large_state.placements_y1;
        let placements_y1 = &large_state.placements_x1;

        let new_x1 = large_state.placements_y1[prev_turn];
        let new_y0 = large_state.placements_x0[prev_turn];
        let new_y1 = large_state.placements_x1[prev_turn];
        let available_flag = large_state.avaliable_base_up;
        let width_limit = large_state.height_limit;
        let min_rect_size = large_state.min_rect_size;

        unsafe {
            Self::get_invalid_bases(
                placements_x1,
                placements_y1,
                new_x1,
                new_y0,
                new_y1,
                available_flag,
                width_limit,
                min_rect_size,
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
        width_limit: AlignedU16,
        min_rect_size: AlignedU16,
    ) -> u128 {
        unsafe {
            let mut invalid_flag = 0;
            let new_x1 = new_x1.load();
            let new_y0 = new_y0.load();
            let new_y1 = new_y1.load();
            let width_limit = width_limit.load();
            let min_rect_size = min_rect_size.load();

            for rect_i in BitSetIterU128::new(available_flag) {
                let xi1 = placements_x1[rect_i].load();
                let yi1 = placements_y1[rect_i].load();

                // 条件1: rect_iにピッタリくっつけられる可能性がないとダメ
                //        つまりrect_iの右側に邪魔者がいるとダメ
                // Yi1 + MIN_RECT_SIZE
                let yi1pl0 = _mm256_add_epi16(yi1, min_rect_size);

                // max(yi1, yj0) < min(yi1 + l0, yj1)
                let max = _mm256_max_epu16(yi1, new_y0);
                let min = _mm256_min_epu16(yi1pl0, new_y1);
                let pred_y = _mm256_cmpgt_epi16(min, max);

                // x_i1 < x_j1
                let invalid_base = _mm256_cmpgt_epi16(new_x1, xi1);

                // 条件2: 置いたときにwidth_limitを超えるとダメ
                let next_x = _mm256_add_epi16(new_x1, min_rect_size);
                let invalid_width_limit = _mm256_cmpgt_epi16(next_x, width_limit);

                // invalidなものが閾値を超えていたらNG
                let invalid = _mm256_or_si256(invalid_base, invalid_width_limit);
                let invalid = _mm256_and_si256(pred_y, invalid);
                let invalid = _mm256_srli_epi16(invalid, 15);
                let invalid_cnt = horizontal_add_u16(invalid) as usize;
                let is_invalid = invalid_cnt > Params::get().borrow().invalid_cnt_threshold;
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
        let invalid_bases_up = Self::get_up_invalid_bases(large_state, large_state.turn);
        let avaliable_bases_up = large_state.avaliable_base_up & !invalid_bases_up;

        // 生成
        let rotates = [false, true];

        for rotate in rotates {
            // Up
            next_states.extend(self.gen_up_cand(&large_state, None, rotate, invalid_bases_up));

            for i in BitSetIterU128::new(avaliable_bases_up) {
                next_states.extend(self.gen_up_cand(
                    &large_state,
                    Some(i),
                    rotate,
                    invalid_bases_up,
                ));
            }
        }
    }
}
