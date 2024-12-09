use super::Solver;
use crate::{
    problem::{params::Params, Input, Judge},
    solver::{
        arranger::{mcts::MCTSArranger, multi_beam_simd::MultiBeamArrangerSimd},
        estimator::{
            self,
            gauss_minmax::{GaussEstimator, Observation1d},
            mcmc, Observation2d, Sampler as _,
        },
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

        let arrange_trial_count = Params::get().borrow().arrange_count;
        let duration = Params::get().borrow().query_annealing_duration_sec
            / (input.query_cnt() - arrange_trial_count) as f64;
        let mut observations = vec![];

        for _ in 0..input.query_cnt() - arrange_trial_count {
            let (ops, edges_v, edges_h) = estimator::get_placements(input, &mut estimator, duration, &mut rng);
            let measure = judge.query(&ops);
            let observation_x = Observation1d::new(measure.width(), edges_h);
            let observation_y = Observation1d::new(measure.height(), edges_v);

            estimator.update(observation_x);
            estimator.update(observation_y);
            observations.push(Observation2d::new(
                ops,
                measure.width(),
                measure.height(),
                false,
            ));
        }

        eprintln!("[Final]");
        estimator.dump_estimated(judge.rects());

        let mut gauss_sampler = estimator.get_sampler();

        let mut beam_arranger = MultiBeamArrangerSimd;
        let mut mcts_arranger = MCTSArranger;

        let gauss_rects = gauss_sampler.sample(&mut rng);
        let rect_std_dev = estimator.rect_std_dev();

        let mut mcmc_sampler = mcmc::MCMCSampler::new(
            input,
            observations.clone(),
            gauss_rects.clone(),
            rect_std_dev,
            Params::get().borrow().mcmc_init_duration_sec,
            &mut rng,
            &mut judge,
        );

        for i in 0..arrange_trial_count {
            let remaining_arrange_count = arrange_trial_count - i;
            let duration =
                (2.95 - input.since().elapsed().as_secs_f64()) / remaining_arrange_count as f64;

            let search_ratio = 1.0 - Params::get().borrow().mcmc_duration_ratio;
            let beam_duration =
                duration * search_ratio * Params::get().borrow().beam_mcts_duration_ratio;
            let mcts_duration =
                duration * search_ratio * (1.0 - Params::get().borrow().beam_mcts_duration_ratio);
            let mcmc_duration = duration * Params::get().borrow().mcmc_duration_ratio;
            let first_step_turn = input.rect_cnt() - Params::get().borrow().mcts_turn;

            let sampled_rects = mcmc_sampler.sample(mcmc_duration, &mut rng);

            let ops1 = beam_arranger.arrange(
                &input,
                first_step_turn,
                sampled_rects.clone(),
                beam_duration,
            );

            let ops2 = mcts_arranger.arrange(&input, &ops1, sampled_rects, &mut rng, mcts_duration);

            let mut ops = ops1;
            ops.extend_from_slice(&ops2);

            let measure = judge.query(&ops);

            if i < arrange_trial_count - 1 {
                let observation = Observation2d::new(ops, measure.width(), measure.height(), true);
                mcmc_sampler.update(observation);
            }
        }
    }
}
