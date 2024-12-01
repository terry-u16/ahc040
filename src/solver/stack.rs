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
        estimator.dump_estimated(judge.rects());

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
        estimator.dump_estimated(judge.rects());

        let sampler = estimator.get_sampler();
        let mut sampled = vec![];

        for _ in 0..10 {
            let s = sampler.sample(&mut rng);
            sampled.push(s.iter().last().copied().unwrap());
        }

        eprintln!("[Sampled]");
        for s in sampled {
            eprintln!("{:?}", s);
        }
    }
}
