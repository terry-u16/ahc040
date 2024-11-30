use crate::problem::{Dir, Input, Judge, Op};

use super::Solver;

pub struct StackSolver;

impl Solver for StackSolver {
    fn solve(&self, input: &Input, mut judge: impl Judge) {
        for _ in 0..input.query_cnt() {
            let mut ops = vec![];

            for i in 0..input.rect_cnt() {
                ops.push(Op::new(i, false, Dir::Up, None));
            }

            _ = judge.query(&ops);
        }
    }
}
