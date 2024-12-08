use super::{Observation2d, RectStdDev};
use crate::{
    problem::{
        Dir::{Left, Up},
        Input, Judge, Op, Rect,
    },
    solver::simd::{
        expand_u16, round_i16, round_u16, AlignedF32, AlignedU16, SimdRectSet, AVX2_F32_W,
        AVX2_U16_W,
    },
};
use core::arch::x86_64::*;
use itertools::{izip, Itertools};
use rand::prelude::*;
use rand_distr::Normal;

pub(crate) struct MCMCSampler {
    env: Env,
    state: State,
}

impl MCMCSampler {
    pub(crate) fn new(
        input: &Input,
        observations: Vec<Observation2d>,
        rects: SimdRectSet,
        rect_std_dev: RectStdDev,
        init_duration: f64,
        rng: &mut impl Rng,
        judge: &impl Judge,
    ) -> Self {
        let env = Env::new(
            input,
            observations,
            input.rect_cnt(),
            input.std_dev(),
            rect_std_dev,
        );
        let heights = rects.heights.iter().map(|&h| AlignedU16(h)).collect();
        let widths = rects.widths.iter().map(|&w| AlignedU16(w)).collect();
        let state = State::new(&env, widths, heights);
        let mut states = mcmc(&env, state, init_duration, rng);
        unsafe {
            Self::dump_estimated(&states, judge.rects(), input.rect_cnt());
        }

        Self {
            env,
            state: states.pop().unwrap(),
        }
    }

    pub(crate) fn update(&mut self, observation: Observation2d) {
        self.env.observations.push(observation);
        self.state.log_likelihood = unsafe { self.state.calc_log_likelihood(&self.env) };
    }

    pub(crate) fn sample(&mut self, duration: f64, rng: &mut impl Rng) -> SimdRectSet {
        self.state = mcmc(&self.env, self.state.clone(), duration, rng)
            .pop()
            .unwrap();

        let mut heights = vec![];
        let mut widths = vec![];

        for (h, w) in izip!(&self.state.rect_h, &self.state.rect_w) {
            heights.push(h.0);
            widths.push(w.0);
        }

        SimdRectSet::new(heights, widths)
    }

    unsafe fn dump_estimated(states: &[State], actual_rects: Option<&[Rect]>, rect_cnt: usize) {
        let Some(rects) = actual_rects else { return };
        let mut sum_w = vec![0; rect_cnt];
        let mut sum_h = vec![0; rect_cnt];
        let count = states.len();

        for state in states {
            for (i, (w, h)) in izip!(&state.rect_w, &state.rect_h).enumerate() {
                for j in 0..AVX2_U16_W {
                    sum_w[i] += expand_u16(w.0[j]) as u64;
                    sum_h[i] += expand_u16(h.0[j]) as u64;
                }
            }
        }

        let mean_h = sum_h
            .iter()
            .map(|&s| s as f64 / (count * AVX2_U16_W as usize) as f64)
            .collect_vec();
        let mean_w = sum_w
            .iter()
            .map(|&s| s as f64 / (count * AVX2_U16_W as usize) as f64)
            .collect_vec();

        let mut sum_var_w = vec![0.0; rect_cnt];
        let mut sum_var_h = vec![0.0; rect_cnt];

        for state in states {
            for (i, (w, h)) in izip!(&state.rect_w, &state.rect_h).enumerate() {
                for j in 0..AVX2_U16_W {
                    sum_var_w[i] += (expand_u16(w.0[j]) as f64 - mean_w[i]).powi(2);
                    sum_var_h[i] += (expand_u16(h.0[j]) as f64 - mean_h[i]).powi(2);
                }
            }
        }

        let var_w = sum_var_w
            .iter()
            .map(|&s| s / (count * AVX2_U16_W as usize) as f64)
            .collect_vec();
        let var_h = sum_var_h
            .iter()
            .map(|&s| s / (count * AVX2_U16_W as usize) as f64)
            .collect_vec();

        let std_dev_w = var_w.iter().map(|&v| v.sqrt()).collect_vec();
        let std_dev_h = var_h.iter().map(|&v| v.sqrt()).collect_vec();

        eprintln!("[MCMC]");

        for i in 0..rect_cnt {
            eprint!(
                "{:>02} {:>6.0} ± {:>5.0} / {:>6.0} ± {:>5.0}",
                i, mean_h[i], std_dev_h[i], mean_w[i], std_dev_w[i]
            );

            let rect = rects[i];
            let sigma_h = (rect.height() as f64 - mean_h[i] as f64) / std_dev_h[i];
            let sigma_w = (rect.width() as f64 - mean_w[i] as f64) / std_dev_w[i];
            eprintln!(
                " (actual: {:>6.0} ({:+>5.2}σ) / {:>6.0} ({:+>5.2}σ))",
                rect.height(),
                sigma_h,
                rect.width(),
                sigma_w
            );
        }
    }
}

struct Env {
    observations: Vec<Observation2d>,
    init_measured_heights: Vec<AlignedF32>,
    init_measured_widths: Vec<AlignedF32>,
    rect_cnt: usize,
    std_dev: f64,
    rect_std_dev: RectStdDev,
}

impl Env {
    fn new(
        input: &Input,
        observations: Vec<Observation2d>,
        rect_cnt: usize,
        std_dev: f64,
        rect_std_dev: RectStdDev,
    ) -> Self {
        let init_measured_heights = input
            .rect_measures()
            .iter()
            .map(|m| {
                let h = round_u16(m.height()) as f32;
                AlignedF32([h; AVX2_F32_W])
            })
            .collect_vec();
        let init_measured_widths = input
            .rect_measures()
            .iter()
            .map(|m| {
                let w = round_u16(m.width()) as f32;
                AlignedF32([w; AVX2_F32_W])
            })
            .collect_vec();

        Self {
            observations,
            init_measured_heights,
            init_measured_widths,
            rect_cnt,
            std_dev,
            rect_std_dev,
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

        // 初期計測から
        for (rect_h, rect_w, measured_h, measured_w) in izip!(
            self.rect_h.iter(),
            self.rect_w.iter(),
            env.init_measured_heights.iter(),
            env.init_measured_widths.iter()
        ) {
            // u16 x 16 -> (u32 x 8) x 2 -> (f32 x 8) x 2
            let width = rect_w.load();
            let height = rect_h.load();

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

            let observed_x = measured_w.load();
            let observed_y = measured_h.load();
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
        }

        // 配置結果から
        let mut x0_buf = [AlignedU16::default(); Input::MAX_RECT_CNT];
        let mut x1_buf = [AlignedU16::default(); Input::MAX_RECT_CNT];
        let mut y0_buf = [AlignedU16::default(); Input::MAX_RECT_CNT];
        let mut y1_buf = [AlignedU16::default(); Input::MAX_RECT_CNT];
        let mut x0_buf = &mut x0_buf[..env.rect_cnt];
        let mut x1_buf = &mut x1_buf[..env.rect_cnt];
        let mut y0_buf = &mut y0_buf[..env.rect_cnt];
        let mut y1_buf = &mut y1_buf[..env.rect_cnt];

        for observation in env.observations.iter() {
            let (width, height) = if observation.is_2d {
                self.pack_2d(
                    &observation.operations,
                    &self.rect_w,
                    &self.rect_h,
                    &mut x0_buf,
                    &mut x1_buf,
                    &mut y0_buf,
                    &mut y1_buf,
                )
            } else {
                let base = observation
                    .operations
                    .iter()
                    .flat_map(|op| op.base())
                    .next();

                self.pack_1d(
                    &observation.operations,
                    &self.rect_w,
                    &self.rect_h,
                    &mut x0_buf,
                    &mut x1_buf,
                    &mut y0_buf,
                    &mut y1_buf,
                    base,
                )
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

            let observed_x = _mm256_set1_ps(round_u16(observation.len_x) as f32);
            let observed_y = _mm256_set1_ps(round_u16(observation.len_y) as f32);

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
        }

        // 対数尤度の-0.5は最後にかければよい
        let neg_inv_2 = _mm256_set1_ps(-0.5);
        let log_likelihood_low = _mm256_mul_ps(log_likelihood_low, neg_inv_2);
        let log_likelihood_high = _mm256_mul_ps(log_likelihood_high, neg_inv_2);

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
        // y座標はbase次第なので、baseが壁でない場合は0を入れる
        let threshold_x = max_rect_size;
        let mut threshold_y = match base {
            Some(_) => _mm256_setzero_si256(),
            None => max_rect_size,
        };

        for &op in ops.iter() {
            // 全長方形がmax_rect_size以上ならば安全なので、頑張って判定
            // x, y方向それぞれ閾値未満か（未満なら判定が必要）
            let x_ng = _mm256_cmpgt_epi16(threshold_x, width);
            let y_ng = _mm256_cmpgt_epi16(threshold_y, height);

            // DirectionがLeftなら全て0、Upなら全て1のマスク
            let y_mask = _mm256_set1_epi8(op.dir() as i8 - 1);

            // 判定したい方をを取り出して判定
            // 16並列全て条件を満たすのであればflagは0になる（不等号の向きに注意）
            let ng = _mm256_blendv_epi8(x_ng, y_ng, y_mask);
            let flag = _mm256_movemask_epi8(ng);
            let rect_i = op.rect_idx();

            if flag == 0 {
                // 座標がmax_rect_size以上なので衝突判定不要
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

                match op.dir() {
                    Left => {
                        let rect_h = rect_h[rect_i].load();
                        let rect_w = rect_w[rect_i].load();
                        let heights = &mut height;
                        let widths = &mut width;

                        self.pack_one_2d(
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

            // base == rect_index である場合、INFとしていたthreshold_yを更新
            // 併せて衝突判定用にx0, x1, y1も更新
            if Some(op.rect_idx()) == base {
                x0[rect_i] = _mm256_setzero_si256().into();
                x1[rect_i] = rect_w[rect_i];
                y0[rect_i] = _mm256_sub_epi16(height, rect_h[rect_i].load()).into();
                y1[rect_i] = height.into();
                threshold_y = _mm256_add_epi16(height, max_rect_size);
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

        for &op in ops.iter() {
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

struct Neighbor {
    delta: [i16; AVX2_U16_W],
    rect_i: usize,
    is_width: bool,
}

impl Neighbor {
    fn new(delta: [i16; AVX2_U16_W], rect_i: usize, is_width: bool) -> Self {
        Self {
            delta,
            rect_i,
            is_width,
        }
    }

    fn gen(env: &Env, state: &State, rng: &mut impl Rng) -> Self {
        let rect_i = rng.gen_range(0..env.rect_cnt);
        let is_width = rng.gen_bool(0.5);
        let std_dev = if is_width {
            env.rect_std_dev.widths[rect_i]
        } else {
            env.rect_std_dev.heights[rect_i]
        };
        let current = if is_width {
            state.rect_w[rect_i]
        } else {
            state.rect_h[rect_i]
        };

        let mut delta = [0; AVX2_U16_W];
        const LOWER_BOUND: u16 = round_u16(Input::MIN_RECT_SIZE);
        const UPPER_BOUND: u16 = round_u16(Input::MAX_RECT_SIZE);

        for i in 0..AVX2_U16_W {
            let dist = Normal::new(0.0, std_dev).unwrap();

            delta[i] = loop {
                let d = round_i16(dist.sample(rng).round() as i32);
                let new_x = current.0[i].wrapping_add_signed(d);

                if d != 0 && LOWER_BOUND <= new_x && new_x <= UPPER_BOUND {
                    break d;
                }
            }
        }

        Self::new(delta, rect_i, is_width)
    }
}

fn mcmc(env: &Env, mut state: State, duration: f64, rng: &mut impl Rng) -> Vec<State> {
    let since = std::time::Instant::now();
    let mut all_iter = 0;
    let mut accepted = 0;
    let mut rejected = 0;
    let mut all_states = vec![state.clone()];

    loop {
        if since.elapsed().as_secs_f64() >= duration {
            break;
        }

        // 変形
        let neighbor = Neighbor::gen(env, &state, rng);

        for i in 0..AVX2_U16_W {
            if neighbor.is_width {
                state.rect_w[neighbor.rect_i].0[i] =
                    state.rect_w[neighbor.rect_i].0[i].wrapping_add_signed(neighbor.delta[i]);
            } else {
                state.rect_h[neighbor.rect_i].0[i] =
                    state.rect_h[neighbor.rect_i].0[i].wrapping_add_signed(neighbor.delta[i]);
            }
        }

        // 対数尤度計算
        let new_log_likelihood = unsafe { state.calc_log_likelihood(env) };

        for i in 0..AVX2_U16_W {
            let (j, k) = (i / AVX2_F32_W, i % AVX2_F32_W);
            let prev_log_likelihood = state.log_likelihood[j].0[k];

            if rng.gen_range(0.0..1.0) < (-(prev_log_likelihood - new_log_likelihood[j].0[k])).exp()
            {
                accepted += 1;
                state.log_likelihood[j].0[k] = new_log_likelihood[j].0[k];
            } else {
                rejected += 1;
                if neighbor.is_width {
                    state.rect_w[neighbor.rect_i].0[i] =
                        state.rect_w[neighbor.rect_i].0[i].wrapping_add_signed(-neighbor.delta[i]);
                } else {
                    state.rect_h[neighbor.rect_i].0[i] =
                        state.rect_h[neighbor.rect_i].0[i].wrapping_add_signed(-neighbor.delta[i]);
                }
            }
        }

        all_iter += 1;
        all_states.push(state.clone());
    }

    eprintln!("mcmc_iter: {}", all_iter);
    eprintln!("mcmc accepted: {} / {}", accepted, accepted + rejected);

    all_states
}
