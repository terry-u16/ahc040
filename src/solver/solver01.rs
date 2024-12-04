use super::{
    estimator::gauss::{GaussEstimator, Observation},
    Solver,
};
use crate::{
    problem::{Input, Judge},
    solver::{
        arranger::{self, Arranger},
        estimator,
    },
};
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

pub struct Solver01;

impl Solver for Solver01 {
    fn solve(&self, input: &Input, mut judge: impl Judge) {
        let mut rng = Pcg64Mcg::from_entropy();
        let mut estimator = GaussEstimator::new(input);

        for (i, &rect) in input.rect_measures().iter().enumerate() {
            estimator.update(Observation::single(input, rect.height(), i, false));
            estimator.update(Observation::single(input, rect.width(), i, true));
        }

        eprintln!("[Init]");
        estimator.dump_estimated(judge.rects());

        let arrange_count = (input.query_cnt() / 5).clamp(5, 10);
        let duration = 0.3 / (input.query_cnt() - arrange_count) as f64;

        for _ in 0..input.query_cnt() - arrange_count {
            let (ops, edges_v, edges_h) = estimator::get_placements(&estimator, duration, &mut rng);
            let measure = judge.query(&ops);
            let observation_x = Observation::new(measure.width(), edges_h);
            let observation_y = Observation::new(measure.height(), edges_v);

            estimator.update(observation_x);
            estimator.update(observation_y);
        }

        eprintln!("[Final]");
        estimator.dump_estimated(judge.rects());

        let each_duration = (2.85 - input.since().elapsed().as_secs_f64()) / arrange_count as f64;

        for _ in 0..arrange_count {
            let mut arranger = arranger::get_arranger(&mut rng, &estimator, each_duration);
            let ops = arranger.arrange(&input);

            _ = judge.query(&ops);
        }
    }
}
