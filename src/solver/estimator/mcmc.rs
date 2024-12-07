use itertools::izip;

use super::Observation2d;
use crate::{
    problem::{
        Dir::{Left, Up},
        Input, Op,
    },
    solver::simd::{round_u16, AlignedF32, AlignedU16, SimdRectSet},
};
use core::arch::x86_64::*;

pub(crate) struct MCMCSampler {
    observations: Vec<Observation2d>,
    std_dev: f64,
}

pub(crate) fn test(input: &Input, observations: &[Observation2d], rects: SimdRectSet) {
    let env = Env::new(&observations, input.rect_cnt(), input.std_dev());
    let heights = rects.heights.iter().map(|&h| AlignedU16(h)).collect();
    let widths = rects.widths.iter().map(|&w| AlignedU16(w)).collect();
    let state = State::new(&env, widths, heights);
    eprintln!(
        "{:?} {:?}",
        state.log_likelihood[0].0, state.log_likelihood[1].0
    );
}

struct Env<'a> {
    observations: &'a [Observation2d],
    rect_cnt: usize,
    std_dev: f64,
}

impl<'a> Env<'a> {
    fn new(observations: &'a [Observation2d], rect_cnt: usize, std_dev: f64) -> Self {
        Self {
            observations,
            rect_cnt,
            std_dev,
        }
    }
}

#[derive(Debug, Clone)]
struct State {
    rect_w: Vec<AlignedU16>,
    rect_h: Vec<AlignedU16>,
    log_likelihood: [AlignedF32; 2],
}

impl State {
    fn new(env: &Env, rect_w: Vec<AlignedU16>, rect_h: Vec<AlignedU16>) -> Self {
        let log_likelihood = [AlignedF32::default(); 2];

        let mut s = Self {
            rect_w,
            rect_h,
            log_likelihood,
        };

        unsafe {
            s.log_likelihood = s.calc_log_likelihood(env);
        }

        s
    }

    #[target_feature(enable = "avx,avx2")]
    unsafe fn calc_log_likelihood(&self, env: &Env) -> [AlignedF32; 2] {
        let mut log_likelihood_low = _mm256_setzero_ps();
        let mut log_likelihood_high = _mm256_setzero_ps();
        let inv_std_dev = _mm256_set1_ps(1.0 / round_u16(env.std_dev.round() as u32) as f32);
        let mut x0_buf = [AlignedU16::default(); Input::MAX_RECT_CNT];
        let mut x1_buf = [AlignedU16::default(); Input::MAX_RECT_CNT];
        let mut y0_buf = [AlignedU16::default(); Input::MAX_RECT_CNT];
        let mut y1_buf = [AlignedU16::default(); Input::MAX_RECT_CNT];
        let mut x0_buf = &mut x0_buf[..env.rect_cnt];
        let mut x1_buf = &mut x1_buf[..env.rect_cnt];
        let mut y0_buf = &mut y0_buf[..env.rect_cnt];
        let mut y1_buf = &mut y1_buf[..env.rect_cnt];

        eprintln!("{}", 1.0 / round_u16(env.std_dev.round() as u32) as f32);

        for (i, observations) in env.observations.iter().enumerate() {
            let (width, height) = if observations.is_2d {
                self.pack_2d(
                    &observations.operations,
                    &self.rect_w,
                    &self.rect_h,
                    &mut x0_buf,
                    &mut x1_buf,
                    &mut y0_buf,
                    &mut y1_buf,
                )
            } else {
                let base = observations
                    .operations
                    .iter()
                    .flat_map(|op| op.base())
                    .next();

                self.pack_2d(
                    &observations.operations,
                    &self.rect_w,
                    &self.rect_h,
                    &mut x0_buf,
                    &mut x1_buf,
                    &mut y0_buf,
                    &mut y1_buf,
                )

                //self.pack_1d(
                //    &observations.operations,
                //    &self.rect_w,
                //    &self.rect_h,
                //    &mut x0_buf,
                //    &mut x1_buf,
                //    &mut y0_buf,
                //    &mut y1_buf,
                //    base,
                //)
            };

            // u16 x 16 -> (u32 x 8) x 2 -> (f32 x 8) x 2
            let width = width.load();
            let height = height.load();

            let width_low = _mm256_castsi256_si128(width);
            let width_high = _mm256_extracti128_si256(width, 1);
            let width_u32_low = _mm256_cvtepu16_epi32(width_low);
            let width_u32_high = _mm256_cvtepu16_epi32(width_high);
            let width_f32_low = _mm256_cvtepi32_ps(width_u32_low);
            let width_f32_high = _mm256_cvtepi32_ps(width_u32_high);

            let height_low = _mm256_castsi256_si128(height);
            let height_high = _mm256_extracti128_si256(height, 1);
            let height_u32_low = _mm256_cvtepu16_epi32(height_low);
            let height_u32_high = _mm256_cvtepu16_epi32(height_high);
            let height_f32_low = _mm256_cvtepi32_ps(height_u32_low);
            let height_f32_high = _mm256_cvtepi32_ps(height_u32_high);

            let observed_x = _mm256_set1_ps(round_u16(observations.len_x) as f32);
            let observed_y = _mm256_set1_ps(round_u16(observations.len_y) as f32);

            eprintln!("[Observed x] {:?} {:?}", width_f32_low, observed_x);
            eprintln!("[Observed y] {:?} {:?}", height_f32_low, observed_y);

            let x_diff_low = _mm256_sub_ps(observed_x, width_f32_low);
            let x_diff_high = _mm256_sub_ps(observed_x, width_f32_high);
            let y_diff_low = _mm256_sub_ps(observed_y, height_f32_low);
            let y_diff_high = _mm256_sub_ps(observed_y, height_f32_high);

            let x_diff_low = _mm256_mul_ps(x_diff_low, inv_std_dev);
            let x_diff_high = _mm256_mul_ps(x_diff_high, inv_std_dev);
            let y_diff_low = _mm256_mul_ps(y_diff_low, inv_std_dev);
            let y_diff_high = _mm256_mul_ps(y_diff_high, inv_std_dev);

            // logP' = (x - μ)^2 + logP
            log_likelihood_low = _mm256_fmadd_ps(x_diff_low, x_diff_low, log_likelihood_low);
            log_likelihood_high = _mm256_fmadd_ps(x_diff_high, x_diff_high, log_likelihood_high);
            log_likelihood_low = _mm256_fmadd_ps(y_diff_low, y_diff_low, log_likelihood_low);
            log_likelihood_high = _mm256_fmadd_ps(y_diff_high, y_diff_high, log_likelihood_high);

            eprintln!(
                "[log likelihood] {} {:?} {:?}",
                i, log_likelihood_low, log_likelihood_high
            );
        }

        // 対数尤度の-0.5は最後にかければよい
        let neg_inv_2 = _mm256_set1_ps(-0.5);
        let log_likelihood_low = _mm256_mul_ps(log_likelihood_low, neg_inv_2);
        let log_likelihood_high = _mm256_mul_ps(log_likelihood_high, neg_inv_2);
        eprintln!("{:?} {:?}", log_likelihood_low, log_likelihood_high);

        [log_likelihood_low.into(), log_likelihood_high.into()]
    }

    #[target_feature(enable = "avx,avx2")]
    unsafe fn pack_1d(
        &self,
        ops: &[Op],
        rect_w: &[AlignedU16],
        rect_h: &[AlignedU16],
        x0: &mut [AlignedU16],
        x1: &mut [AlignedU16],
        y0: &mut [AlignedU16],
        y1: &mut [AlignedU16],
        base: Option<usize>,
    ) -> (AlignedU16, AlignedU16) {
        const MAX_RECT_SIZE: u16 = round_u16(100000);
        let max_rect_size = _mm256_set1_epi16(MAX_RECT_SIZE as i16);
        let mut width = _mm256_setzero_si256();
        let mut height = _mm256_setzero_si256();

        x0.fill(AlignedU16::default());
        x1.fill(AlignedU16::default());
        y0.fill(AlignedU16::default());
        y1.fill(AlignedU16::default());

        // 座標がこれ以上になったら衝突判定をしなくても大丈夫という閾値
        // y座標はbase次第なので、baseが壁でない場合はINFを入れる
        let threshold_x = max_rect_size;
        let mut threshold_y = match base {
            Some(_) => _mm256_set1_epi16((u16::MAX >> 1) as i16),
            None => max_rect_size,
        };

        for (turn, &op) in ops.iter().enumerate() {
            // 全長方形がmax_rect_sizeを超えていれば安全なので、頑張って判定
            // x, y方向それぞれ閾値を超えていないか
            let x_ng = _mm256_cmpgt_epi16(threshold_x, width);
            let y_ng = _mm256_cmpgt_epi16(threshold_y, height);

            // DirectionがLeftなら全て0、Upなら全て1のマスク
            let y_mask = _mm256_set1_epi8(op.dir() as i8 - 1);

            // 判定したい方をを取り出して判定
            // 16並列全て条件を満たすのであればflagは0になる（不等号の向きに注意）
            let ng = _mm256_blendv_epi8(x_ng, y_ng, y_mask);
            let flag = _mm256_movemask_epi8(ng);

            if flag == 0 {
                // 座標がmax_rect_sizeを超えているので衝突判定不要
                // 高さ、幅どちらを取るかビット演算で求める
                // rotate_maskはrotateがtrueなら全て0、falseなら全て1
                let not_rotate_mask = _mm256_set1_epi8((op.rotate() as i8) - 1);
                let height_mask = _mm256_xor_si256(y_mask, not_rotate_mask);
                let index = op.rect_idx();
                let len =
                    _mm256_blendv_epi8(rect_h[index].load(), rect_w[index].load(), height_mask);

                // 高さ・幅を更新
                // 0を足すかlenを足すかをマスク演算で分岐
                let w_add = _mm256_andnot_si256(y_mask, len);
                let h_add = _mm256_and_si256(y_mask, len);
                width = _mm256_add_epi16(width, w_add);
                height = _mm256_add_epi16(height, h_add);
            } else {
                // 真面目に衝突判定をする
                // こちらが呼ばれる回数は少ないためちょっと処理をサボっても大丈夫
                let rotate = op.rotate();
                let rect_i = op.rect_idx();

                match op.dir() {
                    Left => {
                        let rect_h = rect_h[rect_i].load();
                        let rect_w = rect_w[rect_i].load();
                        let heights = &mut height;
                        let widths = &mut width;

                        self.pack_one_2d(
                            turn,
                            op.base(),
                            rotate,
                            rect_i,
                            rect_h,
                            rect_w,
                            x0,
                            x1,
                            y0,
                            y1,
                            heights,
                            widths,
                        );
                    }
                    Up => {
                        // x, y座標を反転させる
                        let (rect_h, rect_w) = (rect_w[rect_i].load(), rect_h[rect_i].load());
                        let (heights, widths) = (&mut width, &mut height);

                        self.pack_one_2d(
                            turn,
                            op.base(),
                            rotate,
                            rect_i,
                            rect_h,
                            rect_w,
                            y0,
                            y1,
                            x0,
                            x1,
                            heights,
                            widths,
                        );
                    }
                };

                // base == rect_index である場合、INFとしていたthreshold_yを更新
                if Some(op.rect_idx()) == base {
                    threshold_y = _mm256_add_epi16(y1[turn].load(), max_rect_size);
                }
            }
        }

        (width.into(), height.into())
    }

    #[target_feature(enable = "avx,avx2")]
    unsafe fn pack_2d(
        &self,
        ops: &[Op],
        rect_w: &[AlignedU16],
        rect_h: &[AlignedU16],
        x0: &mut [AlignedU16],
        x1: &mut [AlignedU16],
        y0: &mut [AlignedU16],
        y1: &mut [AlignedU16],
    ) -> (AlignedU16, AlignedU16) {
        // pack_1dとほぼ同じ
        // 毎回愚直に衝突判定をするためシンプル
        // 2次元的に詰め込む回数は少ないので計算量的になんとかなるはず
        let mut width = _mm256_setzero_si256();
        let mut height = _mm256_setzero_si256();

        x0.fill(AlignedU16::default());
        x1.fill(AlignedU16::default());
        y0.fill(AlignedU16::default());
        y1.fill(AlignedU16::default());

        for (turn, &op) in ops.iter().enumerate() {
            // 真面目に衝突判定をする
            let rotate = op.rotate();
            let rect_i = op.rect_idx();

            match op.dir() {
                Left => {
                    let rect_h = rect_h[rect_i].load();
                    let rect_w = rect_w[rect_i].load();
                    let heights = &mut height;
                    let widths = &mut width;

                    self.pack_one_2d(
                        turn,
                        op.base(),
                        rotate,
                        rect_i,
                        rect_h,
                        rect_w,
                        x0,
                        x1,
                        y0,
                        y1,
                        heights,
                        widths,
                    );
                }
                Up => {
                    // x, y座標を反転させる
                    let (rect_h, rect_w) = (rect_w[rect_i].load(), rect_h[rect_i].load());
                    let (heights, widths) = (&mut width, &mut height);

                    self.pack_one_2d(
                        turn,
                        op.base(),
                        rotate,
                        rect_i,
                        rect_h,
                        rect_w,
                        y0,
                        y1,
                        x0,
                        x1,
                        heights,
                        widths,
                    );
                }
            };
        }

        (width.into(), height.into())
    }

    #[target_feature(enable = "avx,avx2")]
    unsafe fn pack_one_2d(
        &self,
        turn: usize,
        base: Option<usize>,
        rotate: bool,
        rect_i: usize,
        mut rect_h: __m256i,
        mut rect_w: __m256i,
        placements_x0: &mut [AlignedU16],
        placements_x1: &mut [AlignedU16],
        placements_y0: &mut [AlignedU16],
        placements_y1: &mut [AlignedU16],
        heights: &mut __m256i,
        widths: &mut __m256i,
    ) {
        if rotate {
            std::mem::swap(&mut rect_w, &mut rect_h);
        }

        let y0 = match base {
            Some(index) => placements_y1[index].load(),
            None => _mm256_setzero_si256(),
        };
        let y1 = _mm256_add_epi16(y0, rect_h);

        // 長方形がどこに置かれるかを調べる
        let mut x0 = _mm256_setzero_si256();

        for (p_x1, p_y0, p_y1) in izip!(
            placements_x1[..rect_i].iter(),
            placements_y0[..rect_i].iter(),
            placements_y1[..rect_i].iter()
        ) {
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

        placements_x0[rect_i] = x0.into();
        placements_x1[rect_i] = x1.into();
        placements_y0[rect_i] = y0.into();
        placements_y1[rect_i] = y1.into();

        *widths = _mm256_max_epu16(*widths, x1);
        *heights = _mm256_max_epu16(*heights, y1);
    }
}
