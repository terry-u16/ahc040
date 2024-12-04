mod multi_beam_simd;

use super::estimator::Estimator;
use crate::problem::{Input, Op};
use rand::Rng;

pub(super) trait Arranger {
    fn arrange(&mut self, input: &Input) -> Vec<Op>;
}

pub(super) fn get_arranger<'a>(
    rng: &'a mut impl Rng,
    estimator: &'a Estimator,
    duration_sec: f64,
) -> impl Arranger + 'a {
    multi_beam_simd::MultiBeamArrangerSimd::new(&estimator, rng, duration_sec)
}
