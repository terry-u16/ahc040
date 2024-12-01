use nalgebra::{DMatrix, DVector, Matrix1};

use crate::problem::Input;

#[derive(Debug, Clone)]
pub struct Estimator {
    /// 平均ベクトル
    mean: DVector<f64>,

    /// 分散共分散行列
    variance: DMatrix<f64>,

    /// 計測誤差
    measure_variance: f64,

    /// 矩形の数
    rect_cnt: usize,
}

impl Estimator {
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

    pub fn update(&mut self, observation: &Observation) {
        let y = Matrix1::new(observation.len as f64);
        let mut c = DVector::zeros(self.rect_cnt * 2);

        for &RectEdge(i, dir) in observation.edges.iter() {
            let i = match dir {
                RectDir::Vertical => i,
                RectDir::Horizontal => i + self.rect_cnt,
            };

            c[i] = 1.0;
        }

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
pub struct Observation {
    len: u32,
    edges: Vec<RectEdge>,
}

impl Observation {
    pub fn new(len: u32, edges: Vec<RectEdge>) -> Self {
        Self { len, edges }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RectEdge(pub usize, pub RectDir);

#[derive(Debug, Clone, Copy)]
pub enum RectDir {
    Vertical,
    Horizontal,
}
