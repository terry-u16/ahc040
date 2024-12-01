pub mod solver01;
mod estimator;

use crate::problem::{Input, Judge};

pub trait Solver {
    fn solve(&self, input: &Input, judge: impl Judge);
}
