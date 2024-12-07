pub(super) mod mcts;
pub(super) mod multi_beam_simd;

use super::{estimator::Sampler, simd::SimdRectSet};
use crate::problem::{Input, Op};
use rand::Rng;

pub(super) trait Arranger {
    fn arrange(
        &mut self,
        input: &Input,
        start_ops: &[Op],
        end_turn: usize,
        rects: SimdRectSet,
        rng: &mut impl Rng,
        duration_sec: f64,
    ) -> Vec<Op>;
}
