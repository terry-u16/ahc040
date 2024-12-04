use super::{Observation2d, Sampler, SimdRectSet, UpdatableSampler};
use crate::{
    problem::{
        Dir::{Left, Up},
        Input, Op,
    },
    solver::simd::{round_u16, SIMD_WIDTH},
};
use itertools::{izip, Itertools};
use ordered_float::OrderedFloat;
use rand::prelude::*;
use std::{arch::x86_64::*, cmp::Reverse};

#[derive(Debug, Clone)]
pub(super) struct MonteCarloSampler {
    candidates: Vec<Vec<RectU16>>,
    log_likelihoods: Vec<f64>,
    rect_cnt: usize,
    candidate_cnt: usize,
    std_dev: f64,
}

impl MonteCarloSampler {
    pub(super) fn new(
        input: &Input,
        sampler: &mut impl Sampler,
        rng: &mut impl Rng,
        candidate_cnt: usize,
    ) -> Self {
        assert!(
            candidate_cnt % SIMD_WIDTH == 0,
            "candidate_cnt must be a multiple of SIMD_WIDTH"
        );

        let mut candidates = vec![vec![RectU16::default(); input.rect_cnt()]; candidate_cnt];

        for group_i in 0..candidate_cnt / SIMD_WIDTH {
            let rects = sampler.sample(rng);

            for rect_i in 0..input.rect_cnt() {
                for simd_i in 0..SIMD_WIDTH {
                    let height = rects.heights[rect_i][simd_i];
                    let width = rects.widths[rect_i][simd_i];

                    let candidates_i = group_i * SIMD_WIDTH + simd_i;
                    candidates[candidates_i][rect_i] = RectU16::new(height, width);
                }
            }
        }

        let log_likelihoods = vec![0.0; candidate_cnt];
        let std_dev = round_u16(input.std_dev() as u32) as f64;

        Self {
            candidates,
            log_likelihoods,
            rect_cnt: input.rect_cnt(),
            candidate_cnt,
            std_dev,
        }
    }

    #[target_feature(enable = "avx,avx2")]
    unsafe fn update_unsafe(&mut self, observation: &Observation2d) {
        let zero = unsafe { _mm256_setzero_si256() };
        let mut rect_heights = vec![zero; self.rect_cnt];
        let mut rect_widths = vec![zero; self.rect_cnt];

        let mut placements_x0 = vec![];
        let mut placements_x1 = vec![];
        let mut placements_y0 = vec![];
        let mut placements_y1 = vec![];
        let mut pos = vec![!0; self.rect_cnt];

        let observed_x = round_u16(observation.len_x);
        let observed_y = round_u16(observation.len_y);

        for group_i in 0..self.candidate_cnt / SIMD_WIDTH {
            for rect_i in 0..self.rect_cnt {
                let mut heights = [0; SIMD_WIDTH];
                let mut widths = [0; SIMD_WIDTH];

                for simd_i in 0..SIMD_WIDTH {
                    let cand_i = group_i * SIMD_WIDTH + simd_i;
                    heights[simd_i] = self.candidates[cand_i][rect_i].height;
                    widths[simd_i] = self.candidates[cand_i][rect_i].width;
                }

                let heights = unsafe { _mm256_loadu_si256(heights.as_ptr() as *const __m256i) };
                let widths = unsafe { _mm256_loadu_si256(widths.as_ptr() as *const __m256i) };
                rect_heights[rect_i] = heights;
                rect_widths[rect_i] = widths;
            }

            let packed_rects = self.simulate_placement(
                &observation.operations,
                &rect_heights,
                &rect_widths,
                &mut placements_x0,
                &mut placements_x1,
                &mut placements_y0,
                &mut placements_y1,
                &mut pos,
            );

            for simd_i in 0..SIMD_WIDTH {
                let i = group_i * SIMD_WIDTH + simd_i;
                let rect = packed_rects[simd_i];
                let expected_len = [rect.height as f64, rect.width as f64];
                let observed_len = [observed_y as f64, observed_x as f64];

                for (&expected, &observed) in izip!(&expected_len, &observed_len) {
                    let x = (observed - expected) / self.std_dev;
                    self.log_likelihoods[i] += -0.5 * x * x;
                }
            }
        }
    }

    unsafe fn simulate_placement(
        &self,
        ops: &[Op],
        rect_heights: &[__m256i],
        rect_widths: &[__m256i],
        placements_x0: &mut Vec<__m256i>,
        placements_x1: &mut Vec<__m256i>,
        placements_y0: &mut Vec<__m256i>,
        placements_y1: &mut Vec<__m256i>,
        pos: &mut Vec<usize>,
    ) -> [RectU16; SIMD_WIDTH] {
        placements_x0.clear();
        placements_x1.clear();
        placements_y0.clear();
        placements_y1.clear();
        pos.fill(!0);

        let mut heights = _mm256_setzero_si256();
        let mut widths = _mm256_setzero_si256();

        for (op_i, op) in ops.iter().enumerate() {
            let rect_i = op.rect_idx();
            let rect_h = rect_heights[rect_i];
            let rect_w = rect_widths[rect_i];
            let rotate = op.rotate();
            let base = op.base().map(|i| pos[i]);
            pos[rect_i] = op_i;

            match op.dir() {
                Left => {
                    self.simulate_placement_once(
                        base,
                        rotate,
                        rect_h,
                        rect_w,
                        &mut heights,
                        &mut widths,
                        placements_x0,
                        placements_x1,
                        placements_y0,
                        placements_y1,
                    );
                }
                Up => {
                    // 水平・垂直をflipすることに注意
                    self.simulate_placement_once(
                        base,
                        rotate,
                        rect_w,
                        rect_h,
                        &mut widths,
                        &mut heights,
                        placements_y0,
                        placements_y1,
                        placements_x0,
                        placements_x1,
                    );
                }
            }
        }

        let mut rects = [RectU16::default(); SIMD_WIDTH];
        let mut heights_u16 = [0; SIMD_WIDTH];
        let mut widths_u16 = [0; SIMD_WIDTH];

        _mm256_storeu_si256(heights_u16.as_mut_ptr() as *mut __m256i, heights);
        _mm256_storeu_si256(widths_u16.as_mut_ptr() as *mut __m256i, widths);

        for simd_i in 0..SIMD_WIDTH {
            rects[simd_i] = RectU16::new(heights_u16[simd_i], widths_u16[simd_i]);
        }

        rects
    }

    /// 右からrectを置く候補を生成する
    /// 下から置く場合はx, yをflipして呼び出せばよい
    unsafe fn simulate_placement_once(
        &self,
        base: Option<usize>,
        rotate: bool,
        mut rect_h: __m256i,
        mut rect_w: __m256i,
        heights: &mut __m256i,
        widths: &mut __m256i,
        placements_x0: &mut Vec<__m256i>,
        placements_x1: &mut Vec<__m256i>,
        placements_y0: &mut Vec<__m256i>,
        placements_y1: &mut Vec<__m256i>,
    ) {
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

        for (p_x1, p_y0, p_y1) in izip!(
            placements_x1.iter(),
            placements_y0.iter(),
            placements_y1.iter()
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
            let max_y0 = _mm256_max_epu16(y0, *p_y0);
            let min_y1 = _mm256_min_epu16(y1, *p_y1);

            let gt = _mm256_cmpgt_epi16(min_y1, max_y0);
            let x = _mm256_and_si256(*p_x1, gt);
            x0 = _mm256_max_epu16(x0, x);
        }

        let x1 = _mm256_add_epi16(x0, rect_w);

        placements_x0.push(x0);
        placements_x1.push(x1);
        placements_y0.push(y0);
        placements_y1.push(y1);

        *heights = _mm256_max_epu16(y1, *heights);
        *widths = _mm256_max_epu16(x1, *widths);
    }
}

impl Sampler for MonteCarloSampler {
    fn sample(&mut self, _rng: &mut impl Rng) -> SimdRectSet {
        let mut indices = (0..self.candidate_cnt).collect_vec();
        indices.sort_unstable_by_key(|&i| Reverse(OrderedFloat(self.log_likelihoods[i])));

        let indices = &mut indices[..SIMD_WIDTH];
        indices.sort_unstable();

        let mut heights = vec![[0; SIMD_WIDTH]; self.rect_cnt];
        let mut widths = vec![[0; SIMD_WIDTH]; self.rect_cnt];

        for simd_i in 0..SIMD_WIDTH {
            for rect_i in 0..self.rect_cnt {
                heights[rect_i][simd_i] = self.candidates[indices[simd_i]][rect_i].height;
                widths[rect_i][simd_i] = self.candidates[indices[simd_i]][rect_i].width;
            }
        }

        // 2回同じサンプルを使うのは無駄なので削除する
        self.candidate_cnt -= SIMD_WIDTH;

        for &i in indices.iter().rev() {
            self.candidates.remove(i);
            self.log_likelihoods.remove(i);
        }

        SimdRectSet::new(heights, widths)
    }
}

impl UpdatableSampler for MonteCarloSampler {
    fn update(&mut self, observation: &super::Observation2d) {
        unsafe {
            self.update_unsafe(observation);
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RectU16 {
    height: u16,
    width: u16,
}

impl RectU16 {
    fn new(height: u16, width: u16) -> Self {
        Self { height, width }
    }
}
