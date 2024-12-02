mod multi_beam;
mod single_beam;

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
    //single_beam::SingleBeamArranger::new(estimator, rng, duration_sec)
    multi_beam::MultiBeamArranger::new(&estimator, rng, duration_sec)
}
