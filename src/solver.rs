pub mod stack;

use crate::problem::{Input, Judge};

pub trait Solver {
    fn solve(&self, input: &Input, judge: impl Judge);
}
