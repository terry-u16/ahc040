pub mod gauss;

use crate::problem::Op;
use gauss::GaussEstimator;
use nalgebra::DVector;
use rand::prelude::*;

pub(super) fn get_placements(
    estimator: &GaussEstimator,
    duration: f64,
    rng: &mut impl Rng,
) -> (Vec<Op>, DVector<f64>, DVector<f64>) {
    estimator.get_next_placements(duration, rng)
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
