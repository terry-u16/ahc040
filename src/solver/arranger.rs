mod mcts;
mod multi_beam_simd;

use super::estimator::Sampler;
use crate::problem::{Input, Op};
use rand::Rng;

pub(super) trait Arranger {
    fn arrange(
        &mut self,
        input: &Input,
        sampler: &mut impl Sampler,
        rng: &mut impl Rng,
        duration_sec: f64,
    ) -> Vec<Op>;
}

pub(super) fn get_arranger() -> impl Arranger {
    //multi_beam_simd::MultiBeamArrangerSimd
    mcts::MCTSArranger
}
