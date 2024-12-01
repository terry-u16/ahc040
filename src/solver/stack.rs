use rand::Rng;
use rand_pcg::Pcg64Mcg;

use crate::problem::{Dir, Input, Judge, Op};

use super::{
    estimator::{Estimator, Observation, RectDir, RectEdge},
    Solver,
};

pub struct StackSolver;

impl Solver for StackSolver {
    fn solve(&self, input: &Input, mut judge: impl Judge) {
        let mut rng = Pcg64Mcg::new(42);
        let mut estimator = Estimator::new(input);

        for (i, &rect) in input.rect_measures().iter().enumerate() {
            estimator.update(&Observation::new(
                rect.height(),
                vec![RectEdge(i, RectDir::Vertical)],
            ));
            estimator.update(&Observation::new(
                rect.width(),
                vec![RectEdge(i, RectDir::Horizontal)],
            ));
        }

        eprintln!("[Init]");
        dump_estimated(input, &estimator, &judge);

        for _ in 0..input.query_cnt() {
            let mut ops = vec![];
            let mut edges = vec![];

            for i in 0..input.rect_cnt() {
                if rng.gen_bool(0.5) {
                    continue;
                }

                let rotate = rng.gen_bool(0.5);
                ops.push(Op::new(i, rotate, Dir::Up, None));

                let rect_dir = if rotate {
                    RectDir::Horizontal
                } else {
                    RectDir::Vertical
                };
                edges.push(RectEdge(i, rect_dir));
            }

            let measure = judge.query(&ops);
            let observation_y = Observation::new(measure.height(), edges.clone());
            estimator.update(&observation_y);
        }

        eprintln!("[Final]");
        dump_estimated(input, &estimator, &judge);
    }
}

fn dump_estimated(input: &Input, estimator: &Estimator, judge: &impl Judge) {
    let mean_heights = estimator.mean_height();
    let mean_widths = estimator.mean_width();
    let variance_heights = estimator.variance_height();
    let variance_widths = estimator.variance_width();

    for i in 0..input.rect_cnt() {
        let std_dev_h = variance_heights[i].sqrt();
        let std_dev_w = variance_widths[i].sqrt();
        eprint!(
            "{:>02} {:>6.0} ± {:>5.0} / {:>6.0} ± {:>5.0}",
            i, mean_heights[i], std_dev_h, mean_widths[i], std_dev_w
        );

        if let Some(rects) = judge.rects() {
            let rect = rects[i];
            eprintln!(" (actual: {:>6.0} / {:>6.0})", rect.height(), rect.width());
        } else {
            eprintln!();
        }
    }
}
