mod multi_beam_simd;

use super::estimator::gauss::GaussEstimator;
use crate::problem::{Input, Op};
use rand::Rng;

pub(super) trait Arranger {
    fn arrange(&mut self, input: &Input) -> Vec<Op>;
}

pub(super) fn get_arranger<'a>(
    rng: &'a mut impl Rng,
    estimator: &'a GaussEstimator,
    duration_sec: f64,
) -> impl Arranger + 'a {
    multi_beam_simd::MultiBeamArrangerSimd::new(&estimator, rng, duration_sec)
}
