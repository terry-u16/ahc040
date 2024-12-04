pub mod solver01;
mod estimator;
mod arranger;
mod simd;

use crate::problem::{Input, Judge};

pub trait Solver {
    fn solve(&self, input: &Input, judge: impl Judge);
}
