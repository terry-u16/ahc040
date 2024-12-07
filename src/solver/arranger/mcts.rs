use itertools::izip;
use rand::Rng;

use crate::{
    problem::{Dir, Input, Op},
    solver::simd::*,
    util::{BitSetIterU128, ChangeMinMax},
};
use core::f32;
use std::{arch::x86_64::*, u64};

use super::Arranger;

pub(super) struct MCTSArranger;

impl Arranger for MCTSArranger {
    fn arrange(
        &mut self,
        input: &Input,
        sampler: &mut impl super::Sampler,
        rng: &mut impl Rng,
        duration_sec: f64,
    ) -> Vec<Op> {
        let since = std::time::Instant::now();
        let rects = sampler.sample(rng);
        let mut best_score = f32::NEG_INFINITY;
        let mut best_ops = vec![];

        let state = unsafe { State::new(input.clone(), rects) };
        let mut root = Node::new(Action::default());
        let mut score_sum = 0.0;

        while since.elapsed().as_secs_f64() < duration_sec {
            let mut state = state.clone();
            let (score, _) = root.evaluate(&mut state, &mut best_score, &mut best_ops, rng);
            score_sum += score;
        }

        eprintln!("total_iter: {}", root.total_count);
        eprintln!("average_score: {}", score_sum / root.total_count as f32);

        best_ops.pop();
        best_ops.reverse();
        best_ops
    }
}

#[derive(Debug, Clone)]
struct Node {
    action: Action,
    node_score: AlignedF32,
    node_count: AlignedF32,
    total_count: usize,
    is_expanded: bool,
    valid_mask: AlignedF32,
    candidates: [Option<Box<Node>>; AVX2_F32_W],
}

impl Node {
    const EXPANSION_THRESHOLD: usize = 2;

    fn new(action: Action) -> Self {
        Self {
            action,
            node_score: AlignedF32::default(),
            node_count: AlignedF32::default(),
            total_count: 0,
            is_expanded: false,
            valid_mask: AlignedF32::default(),
            candidates: std::array::from_fn(|_| None),
        }
    }

    fn evaluate(
        &mut self,
        state: &mut State,
        best_score: &mut f32,
        best_ops: &mut Vec<Op>,
        rng: &mut impl Rng,
    ) -> (f32, bool) {
        // stateがactionによって変更されないうちに展開を行う（帰りがけにやるとダメ）
        if self.total_count == Self::EXPANSION_THRESHOLD {
            self.expand(state);
        }

        let (score, updated) = if state.turn == state.rect_count {
            // ゲーム終了
            self.evaluate_end_game(state, best_score, best_ops)
        } else if self.is_expanded {
            // 子ノードを選んで再帰的に評価
            let child_index = self.choose_next_child();
            let child = self.candidates[child_index].as_mut().unwrap();
            child.action.apply(state);
            let (score, updated) = child.evaluate(state, best_score, best_ops, rng);

            self.node_score.0[child_index] += score;
            self.node_count.0[child_index] += 1.0;
            (score, updated)
        } else {
            // Bottom-Left法でプレイアウト
            let action = state.gen_bottom_left(rng.gen_bool(0.5));
            action.apply(state);
            self.playout(state, best_score, best_ops, action.op, rng)
        };

        self.total_count += 1;

        if updated {
            best_ops.push(self.action.op);
        }

        (score, updated)
    }

    fn evaluate_end_game(
        &mut self,
        state: &mut State,
        best_score: &mut f32,
        best_ops: &mut Vec<Op>,
    ) -> (f32, bool) {
        let score = unsafe { state.calc_score() };
        eprintln!("{}", score);
        let score_updated = best_score.change_max(score);

        // スコアが更新されたら手順を更新
        if score_updated {
            best_ops.clear();
        }

        return (score, score_updated);
    }

    fn playout(
        &mut self,
        state: &mut State,
        best_score: &mut f32,
        best_ops: &mut Vec<Op>,
        op: Op,
        rng: &mut impl Rng,
    ) -> (f32, bool) {
        let (score, updated) = if state.turn == state.rect_count {
            let score = unsafe { state.calc_score() };
            let score_updated = best_score.change_max(score);

            // スコアが更新されたら手順をクリア
            if score_updated {
                best_ops.clear();
            }

            (score, score_updated)
        } else {
            let action = state.gen_bottom_left(rng.gen_bool(0.5));
            action.apply(state);
            self.playout(state, best_score, best_ops, action.op, rng)
        };

        // スコアが更新されたら手順を更新
        if updated {
            best_ops.push(op);
        }

        (score, updated)
    }

    fn choose_next_child(&self) -> usize {
        let mut best_ucb1 = f32::NEG_INFINITY;
        let mut best_index = 0;

        for i in 0..AVX2_F32_W {
            if self.valid_mask.0[i] == 0.0 {
                continue;
            }

            if self.node_count.0[i] == 0.0 {
                return i;
            }

            let count = self.node_count.0[i];
            let score = self.node_score.0[i];
            let total_count = self.total_count as f32;
            let inv_count = 1.0 / count;

            // TODO: UCB1-Tunedに変更
            let ucb1 = score * inv_count + (2.0 * total_count.ln() * inv_count).sqrt();

            if best_ucb1.change_max(ucb1) {
                best_index = i;
            }
        }

        best_index
    }

    fn expand(&mut self, state: &State) {
        self.is_expanded = true;
        let actions = state.gen_all_actions();

        // 全てのbitが1のf32を作る
        let valid_mask_v = f32::from_bits(!0);
        let mut candidates: [Option<Action>; 8] = [None; AVX2_F32_W];

        'outer: for action in actions {
            let mut worst_bottom_left = action.bottom_left_value();
            let mut worst_index = None;

            // 回転あり・なしで4個ずつ生成したい
            // 前4つを回転なし、後4つを回転ありとする
            let rotation_offset = action.op.rotate() as usize * AVX2_F32_W / 2;

            for i in 0..AVX2_F32_W / 2 {
                let index = rotation_offset + i;

                let bl = match candidates[index] {
                    Some(ref candidate) => {
                        let duplicated = unsafe { candidate.are_same_placement(&action) };

                        if duplicated {
                            continue 'outer;
                        }

                        candidate.bottom_left_value()
                    }
                    None => std::u64::MAX,
                };

                if worst_bottom_left.change_max(bl) {
                    worst_index = Some(index);
                }
            }

            if let Some(index) = worst_index {
                candidates[index] = Some(action);
                self.valid_mask.0[index] = valid_mask_v;
            }
        }

        self.candidates = candidates.map(|action| action.map(|action| Box::new(Node::new(action))));
    }
}

#[derive(Debug, Clone)]
struct State {
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
    rect_count: usize,
    avaliable_base_left: u128,
    avaliable_base_up: u128,
    min_rect_size: AlignedU16,
    score_coef: f32, // 1 / (√Σ(w_i * h_i) * PARALLEL_CNT) で正規化するための値
}

impl State {
    // スコア係数
    thread_local!(static SCORE_MUL: [AlignedF32; 2] = {
        // TODO: パラメータ調整
        const SCORE_MUL: f32 = 0.9;
        [
            AlignedF32(std::array::from_fn(|i| SCORE_MUL.powi(i as i32))),
            AlignedF32(std::array::from_fn(|i| {
                SCORE_MUL.powi((i + AVX2_F32_W) as i32)
            })),
        ]
    });

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

        // 10%余裕を持たせる
        // TODO: パラメータ調整
        let default_width = AlignedU16(areas.map(|a| (a as f64 * 1.1).sqrt() as u16));

        let width_limit = default_width;
        let height_limit = AlignedU16([u16::MAX / 2; AVX2_U16_W]);

        let wh_lower_bound =
            2.0 * areas.iter().map(|a| (*a as f32).sqrt()).sum::<f32>() / AVX2_U16_W as f32;
        let score_coef = Self::SCORE_MUL.with(|score_mul| {
            let sum1 = unsafe { horizontal_add_f32(score_mul[0].load()) };
            let sum2 = unsafe { horizontal_add_f32(score_mul[1].load()) };
            sum1 + sum2
        });

        let score_coef = 1.0 / (wh_lower_bound * score_coef);

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
            widths: AlignedU16::ZERO,
            heights: AlignedU16::ZERO,
            width_limit,
            height_limit,
            placements_x0: placements.clone(),
            placements_x1: placements.clone(),
            placements_y0: placements.clone(),
            placements_y1: placements.clone(),
            turn: 0,
            avaliable_base_left: 0,
            avaliable_base_up: 0,
            min_rect_size,
            rect_count: input.rect_cnt(),
            score_coef,
        }
    }

    #[target_feature(enable = "avx,avx2")]
    unsafe fn calc_score(&self) -> f32 {
        // スコアの昇順にソートした上で、減衰させながら和を取る
        // （期待値を最大化するよりは上振れを狙いたいため）
        Self::SCORE_MUL.with(|score_mul| {
            let height = self.heights.load();
            let width = self.widths.load();

            let sum = _mm256_add_epi16(height, width);
            let score = bitonic_sort_u16(sum);

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
            let normalized_score = score * self.score_coef;

            1.0 - normalized_score
        })
    }

    fn gen_all_actions(&self) -> Vec<Action> {
        // 無効な候補の列挙
        let left_xor = self.get_left_invalid_bases(self.turn);
        let up_xor = self.get_up_invalid_bases(self.turn);
        let left_cands = self.avaliable_base_left & !left_xor;
        let up_cands = self.avaliable_base_up & !up_xor;

        // 生成
        let rotates = [false, true];
        let mut actions = vec![];

        for rotate in rotates {
            actions.extend(self.gen_left_action(None, rotate, left_xor, up_xor));

            for base in BitSetIterU128::new(left_cands) {
                actions.extend(self.gen_left_action(Some(base), rotate, left_xor, up_xor));
            }

            actions.extend(self.gen_up_action(None, rotate, left_xor, up_xor));

            for base in BitSetIterU128::new(up_cands) {
                actions.extend(self.gen_up_action(Some(base), rotate, left_xor, up_xor));
            }
        }

        actions
    }

    fn gen_bottom_left(&self, rotate: bool) -> Action {
        // 無効な候補の列挙
        let left_xor = self.get_left_invalid_bases(self.turn);
        let up_xor = self.get_up_invalid_bases(self.turn);
        let left_cands = self.avaliable_base_left & !left_xor;
        let up_cands = self.avaliable_base_up & !up_xor;

        // Bottom-Leftを探す
        let mut best_score = u64::MAX;

        // 壁に寄せたUPは常にvalid
        let mut best_action = self.gen_up_action(None, rotate, left_xor, up_xor).unwrap();

        for action in BitSetIterU128::new(up_cands)
            .map(|base| self.gen_up_action(Some(base), rotate, left_xor, up_xor))
            .flatten()
        {
            let score = action.bottom_left_value();
            if best_score.change_min(score) {
                best_action = action;
            }
        }

        let action = self.gen_left_action(None, rotate, left_xor, up_xor);

        if let Some(action) = action {
            let score = action.bottom_left_value();
            if best_score.change_min(score) {
                best_action = action;
            }
        }

        for action in BitSetIterU128::new(left_cands)
            .map(|base| self.gen_left_action(Some(base), rotate, left_xor, up_xor))
            .flatten()
        {
            let score = action.bottom_left_value();
            if best_score.change_min(score) {
                best_action = action;
            }
        }

        best_action
    }

    fn gen_left_action(
        &self,
        base: Option<usize>,
        rotate: bool,
        available_base_left_xor: u128,
        available_base_up_xor: u128,
    ) -> Option<Action> {
        let turn = self.turn;
        let rect_h = self.rects_h[turn].into();
        let rect_w = self.rects_w[turn].into();
        let placements_x0 = &self.placements_x0[..turn];
        let placements_x1 = &self.placements_x1[..turn];
        let placements_y0 = &self.placements_y0[..turn];
        let placements_y1 = &self.placements_y1[..turn];
        let heights = self.heights.into();
        let widths = self.widths.into();
        let width_limit = self.width_limit;
        let height_limit = self.height_limit;

        unsafe {
            self.gen_action(
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
                available_base_left_xor,
                available_base_up_xor,
                width_limit,
                height_limit,
                false,
            )
        }
    }

    fn gen_up_action(
        &self,
        base: Option<usize>,
        rotate: bool,
        available_base_left_xor: u128,
        available_base_up_xor: u128,
    ) -> Option<Action> {
        // 下からrectを置くので、水平・垂直を入れ替える
        let turn = self.turn;
        let rect_h = self.rects_w[turn].into();
        let rect_w = self.rects_h[turn].into();
        let placements_x0 = &self.placements_y0[..turn];
        let placements_x1 = &self.placements_y1[..turn];
        let placements_y0 = &self.placements_x0[..turn];
        let placements_y1 = &self.placements_x1[..turn];
        let heights = self.widths.into();
        let widths = self.heights.into();
        let (available_base_left_xor, available_base_up_xor) =
            (available_base_up_xor, available_base_left_xor);
        let width_limit = self.height_limit;
        let height_limit = self.width_limit;

        unsafe {
            self.gen_action(
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
                available_base_left_xor,
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
    unsafe fn gen_action(
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
        available_base_left_xor: u128,
        available_base_up_xor: u128,
        width_limit: AlignedU16,
        height_limit: AlignedU16,
        flip: bool,
    ) -> Option<Action> {
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

        // リミット超えたらNG
        let width_limit = _mm256_cmpgt_epi16(x1, width_limit.load());
        let height_limit = _mm256_cmpgt_epi16(y1, height_limit.load());
        let over_limit = _mm256_or_si256(width_limit, height_limit);
        let over_limit = horizontal_or_u16(over_limit);

        if over_limit > 0 {
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

        // ピッタリくっついていないものがあったらNG
        let is_touching = horizontal_and_u16(is_touching);

        if is_touching == 0 {
            return None;
        }

        let heights = heights.load();
        let widths = widths.load();
        let new_width = _mm256_max_epu16(x1, widths);
        let new_height = _mm256_max_epu16(y1, heights);

        let op = Op::new(turn, rotate, Dir::Left, base);

        let avaliable_base_left_xor = available_base_left_xor | (1 << turn);
        let avaliable_base_up_xor = available_base_up_xor | (1 << turn);

        Some(Action::new(
            x0.into(),
            x1.into(),
            y0.into(),
            y1.into(),
            new_width.into(),
            new_height.into(),
            op,
            avaliable_base_left_xor,
            avaliable_base_up_xor,
            flip,
        ))
    }

    fn get_left_invalid_bases(&self, turn: usize) -> u128 {
        if turn == 0 {
            return 0;
        }

        let prev_turn = turn - 1;
        let placements_x1 = &self.placements_x1;
        let placements_y1 = &self.placements_y1;

        let new_x1 = self.placements_x1[prev_turn];
        let new_y0 = self.placements_y0[prev_turn];
        let new_y1 = self.placements_y1[prev_turn];
        let available_flag = self.avaliable_base_left;
        let width_limit = self.width_limit;
        let min_rect_size = self.min_rect_size;

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

    fn get_up_invalid_bases(&self, turn: usize) -> u128 {
        if turn == 0 {
            return 0;
        }

        // x, yを反転させて呼び出す
        let prev_turn = turn - 1;
        let placements_x1 = &self.placements_y1;
        let placements_y1 = &self.placements_x1;

        let new_x1 = self.placements_y1[prev_turn];
        let new_y0 = self.placements_x0[prev_turn];
        let new_y1 = self.placements_x1[prev_turn];
        let available_flag = self.avaliable_base_up;
        let width_limit = self.height_limit;
        let min_rect_size = self.min_rect_size;

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

                // invalidなものが1つでもあったらNG
                let invalid = _mm256_or_si256(invalid_base, invalid_width_limit);
                let invalid = _mm256_and_si256(pred_y, invalid);
                let is_invalid = horizontal_or_u16(invalid) > 0;
                invalid_flag |= (is_invalid as u128) << rect_i;
            }

            invalid_flag
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Action {
    placements_x0: AlignedU16,
    placements_x1: AlignedU16,
    placements_y0: AlignedU16,
    placements_y1: AlignedU16,
    placements_xor: AlignedU16, // x1 ^ y1 をハッシュとする
    new_widths: AlignedU16,
    new_heights: AlignedU16,
    op: Op,
    avaliable_base_left_xor: u128,
    avaliable_base_up_xor: u128,
}

impl Action {
    fn new(
        mut placements_x0: AlignedU16,
        mut placements_x1: AlignedU16,
        mut placements_y0: AlignedU16,
        mut placements_y1: AlignedU16,
        mut new_widths: AlignedU16,
        mut new_heights: AlignedU16,
        mut op: Op,
        mut avaliable_base_left_xor: u128,
        mut avaliable_base_up_xor: u128,
        flip: bool,
    ) -> Self {
        if flip {
            std::mem::swap(&mut placements_x0, &mut placements_y0);
            std::mem::swap(&mut placements_x1, &mut placements_y1);
            std::mem::swap(&mut new_widths, &mut new_heights);
            std::mem::swap(&mut avaliable_base_left_xor, &mut avaliable_base_up_xor);
            let dir = match op.dir() {
                Dir::Left => Dir::Up,
                Dir::Up => Dir::Left,
            };

            op = Op::new(op.rect_idx(), op.rotate(), dir, op.base());
        }

        let placements_xor =
            unsafe { _mm256_xor_si256(placements_x1.load(), placements_y1.load()).into() };

        Self {
            placements_x0,
            placements_x1,
            placements_y0,
            placements_y1,
            placements_xor,
            new_widths,
            new_heights,
            op,
            avaliable_base_left_xor,
            avaliable_base_up_xor,
        }
    }

    fn apply(&self, state: &mut State) {
        state.placements_x0[state.turn] = self.placements_x0;
        state.placements_x1[state.turn] = self.placements_x1;
        state.placements_y0[state.turn] = self.placements_y0;
        state.placements_y1[state.turn] = self.placements_y1;
        state.widths = self.new_widths;
        state.heights = self.new_heights;
        state.turn += 1;
        state.avaliable_base_left ^= self.avaliable_base_left_xor;
        state.avaliable_base_up ^= self.avaliable_base_up_xor;
    }

    fn bottom_left_value(&self) -> u64 {
        unsafe {
            let x0_sum = horizontal_add_u16(self.placements_x0.load()) as u64;
            let y0_sum = horizontal_add_u16(self.placements_y0.load()) as u64;
            (y0_sum << 32) | x0_sum
        }
    }

    #[target_feature(enable = "avx,avx2")]
    unsafe fn are_same_placement(&self, other: &Self) -> bool {
        // xorハッシュが一致していたら同じ配置と見なす
        let xor0 = self.placements_xor.load();
        let xor1 = other.placements_xor.load();

        let eq = _mm256_cmpeq_epi16(xor0, xor1);
        let mask = _mm256_movemask_epi8(eq);

        mask == -1
    }
}
