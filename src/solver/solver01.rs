use super::{
    estimator::gauss::{GaussEstimator, Observation1d},
    Solver,
};
use crate::{
    problem::{Input, Judge},
    solver::{
        arranger::{self, Arranger},
        estimator::{self, Observation2d, UpdatableSampler},
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
            estimator.update(Observation1d::single(input, rect.height(), i, false));
            estimator.update(Observation1d::single(input, rect.width(), i, true));
        }

        eprintln!("[Init]");
        estimator.dump_estimated(judge.rects());

        let arrange_count = (input.query_cnt() / 5).clamp(5, 10);
        let duration = 0.3 / (input.query_cnt() - arrange_count) as f64;
        let mut observations = vec![];

        for _ in 0..input.query_cnt() - arrange_count {
            let (ops, edges_v, edges_h) = estimator::get_placements(&estimator, duration, &mut rng);
            let measure = judge.query(&ops);
            let observation_x = Observation1d::new(measure.width(), edges_h);
            let observation_y = Observation1d::new(measure.height(), edges_v);

            estimator.update(observation_x);
            estimator.update(observation_y);
            observations.push(Observation2d::new(ops, measure.width(), measure.height()));
        }

        eprintln!("[Final]");
        estimator.dump_estimated(judge.rects());
        let mut gauss_sampler = estimator.get_sampler();
        let mut use_monte_carlo = true;
        let mut monte_carlo_sampler =
            estimator::get_monte_carlo_sampler(input, &mut gauss_sampler, &mut rng, 1024);

        while let Some(observation) = observations.pop() {
            // 対数尤度の更新処理は重いので、時間がないときは多変量正規分布バージョンに切り替える
            if input.since().elapsed().as_millis() >= 1000 {
                use_monte_carlo = false;
                break;
            }

            monte_carlo_sampler.update(&observation);
        }

        let mut arranger = arranger::get_arranger();

        for i in 0..arrange_count {
            let remaining_arrange_count = arrange_count - i;
            let duration =
                (2.9 - input.since().elapsed().as_secs_f64()) / remaining_arrange_count as f64;

            let ops = if use_monte_carlo {
                arranger.arrange(&input, &mut monte_carlo_sampler, &mut rng, duration)
            } else {
                arranger.arrange(&input, &mut gauss_sampler, &mut rng, duration)
            };

            let measure = judge.query(&ops);

            if use_monte_carlo && i < arrange_count - 1 {
                let observation = Observation2d::new(ops, measure.width(), measure.height());
                monte_carlo_sampler.update(&observation);
            }
        }
    }
}
