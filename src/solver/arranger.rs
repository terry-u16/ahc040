use std::time::Duration;

use rand::Rng;

use crate::problem::{Input, Op};

use super::estimator::Sampler;

mod single_beam;

pub(super) trait Arranger {
    fn arrange(&mut self, input: &Input) -> Vec<Op>;
}

pub(super) fn get_arranger<'a>(
    rng: &'a mut impl Rng,
    sampler: &'a Sampler,
    duration: Duration,
) -> impl Arranger + 'a {
    single_beam::SingleBeamArranger::new(&sampler, rng, duration)
}
