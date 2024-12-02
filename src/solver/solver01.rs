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
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

pub struct Solver01;

impl Solver for Solver01 {
    fn solve(&self, input: &Input, mut judge: impl Judge) {
        let mut rng = Pcg64Mcg::from_entropy();
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

        let arrange_count = (input.query_cnt() / 5).clamp(5, 15);

        for _ in 0..input.query_cnt() - arrange_count {
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

        let each_duration = (2.9 - input.since().elapsed().as_secs_f64()) / arrange_count as f64 * 10.0;

        for _ in 0..arrange_count {
            let mut arranger = arranger::get_arranger(&mut rng, &estimator, each_duration);
            let ops = arranger.arrange(&input);

            _ = judge.query(&ops);
        }
    }
}
