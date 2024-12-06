mod arranger;
mod estimator;
#[allow(dead_code)]
mod simd;
pub mod solver01;

use crate::problem::{Input, Judge};

pub trait Solver {
    fn solve(&self, input: &Input, judge: impl Judge);
}
