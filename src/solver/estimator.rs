pub mod gauss;
pub mod gauss_minmax;
pub mod mcmc;

use super::simd::SimdRectSet;
use crate::problem::{Input, Op};
use gauss_minmax::GaussEstimator;
use nalgebra::DVector;
use rand::prelude::*;

pub(super) fn get_placements(
    input: &Input,
    estimator: &mut GaussEstimator,
    duration: f64,
    rng: &mut impl Rng,
) -> (Vec<Op>, DVector<f32>, DVector<f32>) {
    estimator.get_next_placements(input, duration, rng)
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

pub(super) trait Sampler {
    fn sample(&mut self, rng: &mut impl Rng) -> SimdRectSet;
}

#[derive(Debug, Clone)]
pub(crate) struct Observation2d {
    operations: Vec<Op>,
    len_x: u32,
    len_y: u32,
    is_2d: bool,
}

impl Observation2d {
    pub(super) fn new(operations: Vec<Op>, len_x: u32, len_y: u32, is_square: bool) -> Self {
        Self {
            operations,
            len_x,
            len_y,
            is_2d: is_square,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RectStdDev {
    widths: Vec<f64>,
    heights: Vec<f64>,
}

impl RectStdDev {
    fn new(widths: Vec<f64>, heights: Vec<f64>) -> Self {
        Self { widths, heights }
    }
}
