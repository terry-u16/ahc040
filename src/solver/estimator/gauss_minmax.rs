mod annealing;

use std::collections::VecDeque;

use super::{RectStdDev, Sampler};
use crate::{
    problem::{Input, Op, Rect},
    solver::simd::{expand_u16, round_u16, SimdRectSet, AVX2_U16_W},
    util::ChangeMinMax,
};
use itertools::Itertools as _;
use nalgebra::{Cholesky, DMatrix, DVector, Matrix1};
use rand::prelude::*;
use rand_distr::StandardNormal;

#[derive(Debug, Clone)]
pub struct GaussEstimator {
    /// 平均ベクトル
    mean: DVector<f32>,

    /// 分散共分散行列
    variance: DMatrix<f32>,

    /// 計測誤差
    measure_variance: f32,

    /// 矩形の数
    rect_cnt: usize,

    compared_matrix: Vec<Vec<bool>>,

    sample_history: VecDeque<SimdRectSet>,
}

impl GaussEstimator {
    const HISTORY_QUEUE_SIZE: usize = 1000;

    pub fn new(input: &Input) -> Self {
        // 長方形ごとに縦横2次元
        let rect_cnt = input.rect_cnt();
        let len = rect_cnt * 2;

        // モンテカルロにより求めた事前分布
        const MEAN: f32 = 65000.0;
        const STDDEV: f32 = 21280.0;

        let mean = DVector::from_element(len, MEAN);
        let variance = DMatrix::from_diagonal_element(len, len, STDDEV * STDDEV);
        let compared_matrix = vec![vec![false; rect_cnt * 2]; rect_cnt * 2];
        let sample_history = VecDeque::new();

        Self {
            mean,
            variance,
            measure_variance: input.std_dev().powi(2) as f32,
            rect_cnt,
            compared_matrix,
            sample_history,
        }
    }

    pub fn update(&mut self, observation: Observation1d) {
        let y = Matrix1::new(observation.len as f32);
        let c = observation.edges;

        let c_t = c.transpose();
        let syx = &c_t * &self.variance;
        let sxy = syx.transpose();
        let syy = &c_t * &self.variance * &c + Matrix1::new(self.measure_variance);

        let inv_syy = syy.try_inverse().unwrap();

        let mean = &self.mean + &sxy * &inv_syy * (&y - &c_t * &self.mean);
        let variance = &self.variance - &sxy * &inv_syy * &syx;

        self.mean = mean;
        self.variance = variance;
    }

    pub fn get_sampler(&self) -> GaussSampler {
        GaussSampler::new(self)
    }

    pub fn dump_estimated(&self, actual_rects: Option<&[Rect]>) {
        let mean_heights = self.mean_height();
        let mean_widths = self.mean_width();
        let variance_heights = self.variance_height();
        let variance_widths = self.variance_width();

        for i in 0..self.rect_cnt {
            let std_dev_h = variance_heights[i].sqrt();
            let std_dev_w = variance_widths[i].sqrt();
            eprint!(
                "{:>02} {:>6.0} ± {:>5.0} / {:>6.0} ± {:>5.0}",
                i, mean_heights[i], std_dev_h, mean_widths[i], std_dev_w
            );

            if let Some(rects) = actual_rects {
                let rect = rects[i];
                let sigma_h = (rect.height() as f32 - mean_heights[i]) / std_dev_h;
                let sigma_w = (rect.width() as f32 - mean_widths[i]) / std_dev_w;
                eprintln!(
                    " (actual: {:>6.0} ({:+>5.2}σ) / {:>6.0} ({:+>5.2}σ))",
                    rect.height(),
                    sigma_h,
                    rect.width(),
                    sigma_w
                );
            } else {
                eprintln!();
            }
        }
    }

    pub fn enqueue_samples(&mut self, samples: impl Iterator<Item = SimdRectSet>) {
        for sample in samples {
            if self.sample_history.len() >= Self::HISTORY_QUEUE_SIZE {
                self.sample_history.pop_front();
            }

            self.sample_history.push_back(sample);
        }
    }

    pub fn get_next_placements(
        &mut self,
        input: &Input,
        duration: f64,
        rng: &mut impl Rng,
    ) -> (Vec<Op>, DVector<f32>, DVector<f32>) {
        let (edge0, edge1) = self.find_best_pair();
        self.compared_matrix[edge0.to_index(self.rect_cnt)][edge1.to_index(self.rect_cnt)] = true;
        self.compared_matrix[edge1.to_index(self.rect_cnt)][edge0.to_index(self.rect_cnt)] = true;

        annealing::solve(
            input,
            self,
            edge0,
            edge1,
            self.mean.clone(),
            self.variance.diagonal(),
            duration,
            rng,
        )
    }

    fn find_best_pair(&self) -> (RectEdge, RectEdge) {
        let (mean, variance) = self.calc_mean_and_variance();
        const MAX_DIST: usize = 10;

        let mut max_prob = 0.0;
        let mut best_pair = (RectEdge::new(0, false), RectEdge::new(0, false));

        for rect_i in 0..self.rect_cnt {
            let from = rect_i + 1;
            let to = (from + MAX_DIST).min(self.rect_cnt);
            for rect_j in from..to {
                for width_i in [false, true] {
                    for width_j in [false, true] {
                        let edge_i = RectEdge::new(rect_i, width_i);
                        let edge_j = RectEdge::new(rect_j, width_j);
                        let index_i = edge_i.to_index(self.rect_cnt);
                        let index_j = edge_j.to_index(self.rect_cnt);

                        if self.compared_matrix[index_i][index_j] {
                            continue;
                        }

                        let x = -(mean[index_i] - mean[index_j]).abs() as f64;
                        let var = (variance[index_i] + variance[index_j]) as f64;

                        // μ1 < μ2の正規分布においてx1 > x2となる確率
                        let prob = 0.5 * (1.0 + libm::erf(x / (2.0 * var).sqrt()));

                        if max_prob.change_max(prob) {
                            best_pair = (edge_i, edge_j);
                        }
                    }
                }
            }
        }

        eprintln!("Best pair: {:?}", best_pair);

        best_pair
    }

    fn calc_mean_and_variance(&self) -> (DVector<f32>, DVector<f32>) {
        if self.sample_history.is_empty() {
            let mean = self.mean.clone();
            let variance = self.variance.diagonal();
            (mean, variance)
        } else {
            let mut mean: DVector<f32> = DVector::zeros(self.mean.len());
            let mut variance: DVector<f32> = DVector::zeros(self.mean.len());
            let data_count = (self.sample_history.len() * AVX2_U16_W) as f32;

            for rect in self.sample_history.iter() {
                for rect_i in 0..self.rect_cnt {
                    for simd_i in 0..AVX2_U16_W {
                        let height = expand_u16(rect.heights[rect_i][simd_i]) as f32;
                        let width = expand_u16(rect.widths[rect_i][simd_i]) as f32;
                        mean[rect_i] += height;
                        mean[rect_i + self.rect_cnt] += width;
                        variance[rect_i] += height * height;
                        variance[rect_i + self.rect_cnt] += width * width;
                    }
                }
            }

            mean /= data_count;
            variance = variance / data_count - mean.component_mul(&mean);

            eprintln!("MEAN: {:?}", mean.iter().map(|v| v.round() as u32).collect_vec());

            (mean, variance)
        }
    }

    pub fn mean(&self) -> &[f32] {
        self.mean.as_slice()
    }

    pub fn mean_height(&self) -> &[f32] {
        &self.mean()[..self.rect_cnt]
    }

    pub fn mean_width(&self) -> &[f32] {
        &self.mean()[self.rect_cnt..]
    }

    pub fn variance_diag(&self) -> Vec<f32> {
        self.variance.diagonal().iter().copied().collect()
    }

    pub fn variance_height(&self) -> Vec<f32> {
        self.variance_diag()[..self.rect_cnt].to_vec()
    }

    pub fn variance_width(&self) -> Vec<f32> {
        self.variance_diag()[self.rect_cnt..].to_vec()
    }

    pub(crate) fn rect_std_dev(&self) -> RectStdDev {
        let std_dev_h = self
            .variance_height()
            .into_iter()
            .map(|v| (v as f64).sqrt())
            .collect();
        let std_dev_w = self
            .variance_width()
            .into_iter()
            .map(|v| (v as f64).sqrt())
            .collect();
        RectStdDev::new(std_dev_w, std_dev_h)
    }
}

#[derive(Debug, Clone, Copy)]
struct RectEdge {
    rect_i: usize,
    is_width: bool,
}

impl RectEdge {
    fn new(rect_i: usize, is_width: bool) -> Self {
        Self { rect_i, is_width }
    }

    fn to_index(&self, rect_cnt: usize) -> usize {
        self.rect_i + rect_cnt * self.is_width as usize
    }
}

#[derive(Debug, Clone)]
pub struct Observation1d {
    len: u32,
    edges: DVector<f32>,
}

impl Observation1d {
    pub fn new(len: u32, edges: DVector<f32>) -> Self {
        Self { len, edges }
    }

    pub fn single(input: &Input, len: u32, rect_i: usize, is_width: bool) -> Self {
        let mut edges = DVector::zeros(input.rect_cnt() * 2);
        edges[rect_i + if is_width { input.rect_cnt() } else { 0 }] = 1.0;
        Self::new(len, edges)
    }
}

pub struct GaussSampler<'a> {
    estimator: &'a GaussEstimator,
    cholesky: Cholesky<f32, nalgebra::Dyn>,
}

impl<'a> GaussSampler<'a> {
    fn new(estimator: &'a GaussEstimator) -> Self {
        let cholesky = estimator.variance.clone().cholesky().unwrap();

        Self {
            estimator,
            cholesky,
        }
    }
}

impl Sampler for GaussSampler<'_> {
    fn sample(&mut self, rng: &mut impl Rng) -> SimdRectSet {
        let mut heights = vec![[0; AVX2_U16_W]; self.estimator.rect_cnt];
        let mut widths = vec![[0; AVX2_U16_W]; self.estimator.rect_cnt];

        for simd_i in 0..AVX2_U16_W {
            let dist = StandardNormal;
            let values: Vec<f32> = (0..self.estimator.mean.len())
                .map(|_| dist.sample(rng))
                .collect_vec();
            let values = &self.cholesky.l() * DVector::from_vec(values) + &self.estimator.mean;

            for rect_i in 0..self.estimator.rect_cnt {
                let height = (values[rect_i].round() as u32)
                    .clamp(Input::MIN_RECT_SIZE, Input::MAX_RECT_SIZE);
                let width = (values[rect_i + self.estimator.rect_cnt] as u32)
                    .clamp(Input::MIN_RECT_SIZE, Input::MAX_RECT_SIZE);
                heights[rect_i][simd_i] = round_u16(height);
                widths[rect_i][simd_i] = round_u16(width);
            }
        }

        SimdRectSet::new(heights, widths)
    }
}
