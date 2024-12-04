mod annealing;

use super::Sampler;
use crate::{
    problem::{Input, Op, Rect},
    solver::simd::{round_u16, SimdRectSet, SIMD_WIDTH},
};
use itertools::Itertools as _;
use nalgebra::{Cholesky, DMatrix, DVector, Matrix1};
use rand::prelude::*;
use rand_distr::StandardNormal;

#[derive(Debug, Clone)]
pub struct GaussEstimator {
    /// 平均ベクトル
    mean: DVector<f64>,

    /// 分散共分散行列
    variance: DMatrix<f64>,

    /// 計測誤差
    measure_variance: f64,

    /// 矩形の数
    rect_cnt: usize,
}

impl GaussEstimator {
    pub fn new(input: &Input) -> Self {
        // 長方形ごとに縦横2次元
        let rect_cnt = input.rect_cnt();
        let len = rect_cnt * 2;

        // モンテカルロにより求めた事前分布
        const MEAN: f64 = 65000.0;
        const STDDEV: f64 = 21280.0;

        let mean = DVector::from_element(len, MEAN);
        let variance = DMatrix::from_diagonal_element(len, len, STDDEV * STDDEV);

        Self {
            mean,
            variance,
            measure_variance: input.std_dev().powi(2),
            rect_cnt,
        }
    }

    pub fn update(&mut self, observation: Observation1d) {
        let y = Matrix1::new(observation.len as f64);
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
                let sigma_h = (rect.height() as f64 - mean_heights[i]) / std_dev_h;
                let sigma_w = (rect.width() as f64 - mean_widths[i]) / std_dev_w;
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

    pub fn get_next_placements(
        &self,
        duration: f64,
        rng: &mut impl Rng,
    ) -> (Vec<Op>, DVector<f64>, DVector<f64>) {
        annealing::solve(self, duration, rng)
    }

    pub fn mean(&self) -> &[f64] {
        self.mean.as_slice()
    }

    pub fn mean_height(&self) -> &[f64] {
        &self.mean()[..self.rect_cnt]
    }

    pub fn mean_width(&self) -> &[f64] {
        &self.mean()[self.rect_cnt..]
    }

    pub fn variance_diag(&self) -> Vec<f64> {
        self.variance.diagonal().iter().copied().collect()
    }

    pub fn variance_height(&self) -> Vec<f64> {
        self.variance_diag()[..self.rect_cnt].to_vec()
    }

    pub fn variance_width(&self) -> Vec<f64> {
        self.variance_diag()[self.rect_cnt..].to_vec()
    }
}

#[derive(Debug, Clone)]
pub struct Observation1d {
    len: u32,
    edges: DVector<f64>,
}

impl Observation1d {
    pub fn new(len: u32, edges: DVector<f64>) -> Self {
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
    cholesky: Cholesky<f64, nalgebra::Dyn>,
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
        let mut heights = vec![[0; SIMD_WIDTH]; self.estimator.rect_cnt];
        let mut widths = vec![[0; SIMD_WIDTH]; self.estimator.rect_cnt];

        for simd_i in 0..SIMD_WIDTH {
            let dist = StandardNormal;
            let values: Vec<f64> = (0..self.estimator.mean.len())
                .map(|_| dist.sample(rng))
                .collect_vec();
            let values = &self.cholesky.l() * DVector::from_vec(values) + &self.estimator.mean;

            for rect_i in 0..self.estimator.rect_cnt {
                let height = (values[rect_i].round() as u32).clamp(20000, 100000);
                let width = (values[rect_i + self.estimator.rect_cnt] as u32).clamp(20000, 100000);
                heights[rect_i][simd_i] = round_u16(height);
                widths[rect_i][simd_i] = round_u16(width);
            }
        }

        SimdRectSet::new(heights, widths)
    }
}
