use crate::{
    problem::{Dir, Input, Op, Rect},
    util::ChangeMinMax as _,
};
use itertools::Itertools;
use nalgebra::{Cholesky, DMatrix, DVector, Matrix1};
use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_pcg::Pcg64Mcg;

pub(super) fn get_placements_randomly(
    input: &Input,
    estimator: &Estimator,
    rng: &mut Pcg64Mcg,
) -> (Vec<Op>, Vec<RectEdge>, Vec<RectEdge>) {
    let dirs = loop {
        let dirs = (0..input.rect_cnt())
            .map(|_| {
                if rng.gen_bool(0.5) {
                    Dir::Up
                } else {
                    Dir::Left
                }
            })
            .collect_vec();

        // 5個あれば必ず右端・下端になる
        if dirs.iter().filter(|&&b| b == Dir::Up).count() >= 5
            && dirs.iter().filter(|&&b| b == Dir::Left).count() >= 5
        {
            break dirs;
        }
    };

    let rotates = (0..input.rect_cnt())
        .map(|_| rng.gen_bool(0.5))
        .collect_vec();

    let mut ops = vec![];

    for i in 0..input.rect_cnt() {
        ops.push(Op::new(i, rotates[i], dirs[i], None));
    }

    let sampler = estimator.get_sampler();

    let mut weights_v = vec![0; input.rect_cnt()];
    let mut weights_h = vec![0; input.rect_cnt()];

    // TODO: PARAMS
    const TRIAL_CNT: usize = 10;

    for _ in 0..TRIAL_CNT {
        let rects = sampler.sample(rng);

        let (mut x, mut y) = if rotates[0] {
            (rects[0].height(), rects[0].width())
        } else {
            (rects[0].width(), rects[0].height())
        };

        // 原点に近く衝突可能性のある矩形
        let mut near_origins: Vec<(usize, Placement)> = vec![];

        let mut parents_v = vec![None; input.rect_cnt()];
        let mut parents_h = vec![None; input.rect_cnt()];
        let mut last_v = 0;
        let mut last_h = 0;

        const MAX_RECT_SIZE: u32 = 100000;

        for i in 1..input.rect_cnt() {
            let (width, height) = if rotates[i] {
                (rects[i].height(), rects[i].width())
            } else {
                (rects[i].width(), rects[i].height())
            };

            match dirs[i] {
                Dir::Left => {
                    let mut x0 = x;
                    let mut parent = last_h;
                    let y0 = 0;
                    let y1 = height;

                    if x0 < MAX_RECT_SIZE {
                        for &(i, pl) in near_origins.iter() {
                            if y0.max(pl.y0) < y1.min(pl.y1) && x0.change_max(pl.x1) {
                                parent = i;
                            }
                        }
                    }

                    parents_h[i] = Some(parent);
                    last_h = i;
                    let pl = Placement::new(x0, x0 + width, y0, y1);
                    x = pl.x1;

                    if pl.x0 < MAX_RECT_SIZE || pl.y0 < MAX_RECT_SIZE {
                        near_origins.push((i, pl));
                    }
                }
                Dir::Up => {
                    let mut y0 = y;
                    let mut parent = last_v;
                    let x0 = 0;
                    let x1 = width;

                    if y0 < MAX_RECT_SIZE {
                        for &(i, pl) in near_origins.iter() {
                            if x0.max(pl.x0) < x1.min(pl.x1) && y0.change_max(pl.y1) {
                                parent = i;
                            }
                        }
                    }

                    parents_v[i] = Some(parent);
                    last_v = i;
                    let pl = Placement::new(x0, x1, y0, y0 + height);
                    y = pl.y1;

                    if pl.x0 < MAX_RECT_SIZE || pl.y0 < MAX_RECT_SIZE {
                        near_origins.push((i, pl));
                    }
                }
            }
        }

        // 寄与度を計算
        weights_h[last_h] += 1;

        while let Some(p) = parents_h[last_h] {
            weights_h[p] += 1;
            last_h = p;
        }

        weights_v[last_v] += 1;

        while let Some(p) = parents_v[last_v] {
            weights_v[p] += 1;
            last_v = p;
        }
    }

    let mut edges_h = vec![];
    let mut edges_v = vec![];

    for i in 0..input.rect_cnt() {
        if weights_h[i] > 0 {
            let weight = weights_h[i] as f64 / TRIAL_CNT as f64;
            let dir = if rotates[i] {
                RectDir::Vertical
            } else {
                RectDir::Horizontal
            };

            edges_h.push(RectEdge::new(i, dir, weight));
        }

        if weights_v[i] > 0 {
            let weight = weights_v[i] as f64 / TRIAL_CNT as f64;
            let dir = if rotates[i] {
                RectDir::Horizontal
            } else {
                RectDir::Vertical
            };

            edges_v.push(RectEdge::new(i, dir, weight));
        }
    }

    (ops, edges_h, edges_v)
}

#[derive(Debug, Clone, Copy)]
struct Placement {
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
}

impl Placement {
    fn new(x0: u32, x1: u32, y0: u32, y1: u32) -> Self {
        assert!(x0 < x1);
        assert!(y0 < y1);
        Self { x0, x1, y0, y1 }
    }
}

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

        for edge in observation.edges.iter() {
            let i = match edge.dir {
                RectDir::Vertical => edge.index,
                RectDir::Horizontal => edge.index + self.rect_cnt,
            };

            c[i] += edge.weight;
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

    pub fn get_sampler(&self) -> Sampler {
        Sampler::new(self)
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
pub struct RectEdge {
    index: usize,
    dir: RectDir,
    weight: f64,
}

impl RectEdge {
    pub fn new(index: usize, dir: RectDir, weight: f64) -> Self {
        Self { index, dir, weight }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RectDir {
    Vertical,
    Horizontal,
}

pub struct Sampler<'a> {
    estimator: &'a Estimator,
    cholesky: Cholesky<f64, nalgebra::Dyn>,
}

impl<'a> Sampler<'a> {
    fn new(estimator: &'a Estimator) -> Self {
        let cholesky = estimator.variance.clone().cholesky().unwrap();

        Self {
            estimator,
            cholesky,
        }
    }

    pub fn sample(&self, rng: &mut impl Rng) -> Vec<Rect> {
        let dist = StandardNormal;
        let values: Vec<f64> = (0..self.estimator.mean.len())
            .map(|_| dist.sample(rng))
            .collect_vec();
        let values = &self.cholesky.l() * DVector::from_vec(values) + &self.estimator.mean;

        let mut rects = Vec::with_capacity(self.estimator.rect_cnt);

        for i in 0..self.estimator.rect_cnt {
            let height = (values[i].round() as u32).clamp(20000, 100000);
            let width = (values[i + self.estimator.rect_cnt] as u32).clamp(20000, 100000);
            rects.push(Rect::new(height, width));
        }

        rects
    }
}
