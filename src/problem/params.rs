use std::{env, rc::Rc, str::FromStr};

thread_local! {
    static PARAMS: Rc<Params> = Rc::new(Params::new());
}

pub struct Params {
    pub arrange_count: usize,
    pub query_annealing_duration_sec: f64,
    pub mcmc_init_duration_sec: f64,
    pub beam_mcts_duration_ratio: f64,
    pub mcmc_duration_ratio: f64,
    pub mcts_turn: usize,
    pub mcts_expansion_threshold: usize,
    pub mcts_candidates_count: usize,
    pub parallel_score_mul: f32,
    pub width_buf: f64,
    pub ucb1_tuned_coef: f32,
}

impl Params {
    fn new() -> Self {
        let arrange_count = get_env("AHC_ARRANGE_COUNT", 10);
        let query_annealing_duration_sec = get_env("AHC_QUERY_ANNEALING_DURATION_SEC", 0.3);
        let mcmc_init_duration_sec = get_env("AHC_MCMC_INIT_DURATION_SEC", 0.1);
        let beam_mcts_duration_ratio = get_env("AHC_BEAM_MCTS_DURATION_RATIO", 0.5);
        let mcmc_duration_ratio = get_env("AHC_MCMC_DURATION_RATIO", 0.5);
        let mcts_turn = get_env("AHC_MCTS_TURN", 15);
        let mcts_expansion_threshold = get_env("AHC_MCTS_EXPANSION_THRESHOLD", 3);
        let mcts_candidates_count = get_env("AHC_MCTS_CANDIDATES_COUNT", 4);
        let parallel_score_mul = get_env("AHC_PARALLEL_SCORE_MUL", 0.9);
        let width_buf = get_env("AHC_WIDTH_BUF", 1.1);
        let ucb1_tuned_coef = get_env("AHC_UCB1_TUNED_COEF", 1.0);

        Self {
            arrange_count,
            query_annealing_duration_sec,
            mcmc_init_duration_sec,
            beam_mcts_duration_ratio,
            mcmc_duration_ratio,
            mcts_turn,
            mcts_expansion_threshold,
            mcts_candidates_count,
            parallel_score_mul,
            width_buf,
            ucb1_tuned_coef,
        }
    }

    pub fn get() -> Rc<Self> {
        PARAMS.with(|p| p.clone())
    }
}

fn get_env<T: FromStr>(name: &str, default: T) -> T {
    env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
