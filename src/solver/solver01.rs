use std::time::Duration;

use super::{
    estimator::{Estimator, Observation, RectDir, RectEdge},
    Solver,
};
use crate::{
    problem::{Input, Judge},
    solver::{
        arranger::{self, Arranger},
        estimator,
    },
};
use rand_pcg::Pcg64Mcg;

pub struct Solver01;

impl Solver for Solver01 {
    fn solve(&self, input: &Input, mut judge: impl Judge) {
        let mut rng = Pcg64Mcg::new(42);
        let mut estimator = Estimator::new(input);

        for (i, &rect) in input.rect_measures().iter().enumerate() {
            estimator.update(&Observation::new(
                rect.height(),
                vec![RectEdge::new(i, RectDir::Vertical, 1.0)],
            ));
            estimator.update(&Observation::new(
                rect.width(),
                vec![RectEdge::new(i, RectDir::Horizontal, 1.0)],
            ));
        }

        eprintln!("[Init]");
        estimator.dump_estimated(judge.rects());

        const ARRANGE_COUNT: usize = 5;

        for _ in 0..input.query_cnt() - ARRANGE_COUNT {
            let (ops, edges_h, edges_v) =
                estimator::get_placements_randomly(input, &estimator, &mut rng);
            let measure = judge.query(&ops);
            let observation_x = Observation::new(measure.width(), edges_h);
            let observation_y = Observation::new(measure.height(), edges_v);

            estimator.update(&observation_x);
            estimator.update(&observation_y);
        }

        eprintln!("[Final]");
        estimator.dump_estimated(judge.rects());

        let sampler = estimator.get_sampler();

        for _ in 0..ARRANGE_COUNT {
            let mut arranger =
                arranger::get_arranger(&mut rng, &sampler, Duration::from_millis(500));
            let ops = arranger.arrange(&input);

            _ = judge.query(&ops);
        }
    }
}
