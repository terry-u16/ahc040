use nalgebra::{DMatrix, DVector, DVectorView};
use std::{cell::RefCell, env, fmt::Display, rc::Rc, str::FromStr};

thread_local! {
    static PARAMS: Rc<RefCell<Params>> = Rc::new(RefCell::new(Params::from_env()));
}

#[derive(Debug, Clone)]
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
    pub(super) fn new(n: usize, t: usize, sigma: f64) -> Self {
        let arrange_count = ParamSuggester::gen_arrange_count_pred()
            .suggest(n, t, sigma)
            .round() as usize;
        let query_annealing_duration_sec =
            ParamSuggester::gen_query_annealing_duration_sec().suggest(n, t, sigma);
        let mcmc_init_duration_sec =
            ParamSuggester::gen_mcmc_init_duration_sec().suggest(n, t, sigma);
        let beam_mcts_duration_ratio =
            ParamSuggester::gen_beam_mcts_duration_ratio().suggest(n, t, sigma);
        let mcmc_duration_ratio = ParamSuggester::gen_mcmc_duration_ratio().suggest(n, t, sigma);
        let mcts_turn = ParamSuggester::gen_mcts_turn().suggest(n, t, sigma).round() as usize;
        let mcts_expansion_threshold = ParamSuggester::gen_mcts_expansion_threshold()
            .suggest(n, t, sigma)
            .round() as usize;
        let mcts_candidates_count = ParamSuggester::gen_mcts_candidates_count()
            .suggest(n, t, sigma)
            .round() as usize;
        let parallel_score_mul =
            ParamSuggester::gen_parallel_score_mul().suggest(n, t, sigma) as f32;
        let width_buf = ParamSuggester::gen_width_buf().suggest(n, t, sigma);
        let ucb1_tuned_coef = ParamSuggester::gen_ucb1_tuned_coef().suggest(n, t, sigma) as f32;

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

    #[allow(dead_code)]
    pub(super) fn from_env() -> Self {
        let arrange_count = get_env("AHC_ARRANGE_COUNT", 10);
        let query_annealing_duration_sec = get_env("AHC_QUERY_ANNEALING_DURATION_SEC", 0.3);
        let mcmc_init_duration_sec = get_env("AHC_MCMC_INIT_DURATION_SEC", 0.1);
        let beam_mcts_duration_ratio = get_env("AHC_BEAM_MCTS_DURATION_RATIO", 0.5);
        let mcmc_duration_ratio = get_env("AHC_MCMC_DURATION_RATIO", 0.1);
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

    pub fn get() -> Rc<RefCell<Self>> {
        PARAMS.with(|p| p.clone())
    }
}

impl Display for Params {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "arrange_count: {}", self.arrange_count)?;
        writeln!(
            f,
            "query_annealing_duration_sec: {}",
            self.query_annealing_duration_sec
        )?;
        writeln!(f, "mcmc_init_duration_sec: {}", self.mcmc_init_duration_sec)?;
        writeln!(
            f,
            "beam_mcts_duration_ratio: {}",
            self.beam_mcts_duration_ratio
        )?;
        writeln!(f, "mcmc_duration_ratio: {}", self.mcmc_duration_ratio)?;
        writeln!(f, "mcts_turn: {}", self.mcts_turn)?;
        writeln!(
            f,
            "mcts_expansion_threshold: {}",
            self.mcts_expansion_threshold
        )?;
        writeln!(f, "mcts_candidates_count: {}", self.mcts_candidates_count)?;
        writeln!(f, "parallel_score_mul: {}", self.parallel_score_mul)?;
        writeln!(f, "width_buf: {}", self.width_buf)?;
        writeln!(f, "ucb1_tuned_coef: {}", self.ucb1_tuned_coef)
    }
}

const N: &[u8] = b"hDqogzqo4z+hDuqgDurgP5qZmZmZmck/6qAO6qAO6j9YfMVXfMXXPw==";
const T: &[u8] = b"2Efx9osP5L/q7/PoPs3vPyudqOl2ZeO/lG83Uro12T9vZP8B3OfQvw==";
const SIGMA: &[u8] = b"tA+fYVeAzD8dWmQ730/lP6z2dCYLI84/+hFd24xx7T/ZqBIWs+nlPw==";
const ARRANGE_COUNT: &[u8] = b"AAAAAAAAIEAAAAAAAAAkQAAAAAAAACZAAAAAAAAAJEAAAAAAAAAcQA==";
const QUERY_ANNEALING_DURATION_SEC: &[u8] =
    b"VpLUTGhZzj+Kt1pJfgfDPwFMm1m0Rq4/CjDJjdBA0T/Y0/SeXUixPw==";
const MCMC_INIT_DURATION_SEC: &[u8] = b"ZwGSG9musT8+lk2QJRe8P1r+l2fpeL8/xB1QdS+BtD/gw5mNkXK2Pw==";
const BEAM_MCTS_DURATION_RATIO: &[u8] = b"gFaiJw8x4z+Y3UGhiHzdPyMX33gUXeE/O+EbGfoD5j/jcvh7t1TgPw==";
const MCMC_DURATION_RATIO: &[u8] = b"EvBA31vKrj/FsgTng8/APxvT6TcPBMg/MBr2MknXuD9yZh8Eu+CtPw==";
const MCTS_TURN: &[u8] = b"AAAAAAAAM0AAAAAAAAAmQAAAAAAAACxAAAAAAAAAKkAAAAAAAAAwQA==";
const MCTS_EXPANSION_THRESHOLD: &[u8] = b"AAAAAAAAFEAAAAAAAADwPwAAAAAAABBAAAAAAAAA8D8AAAAAAAAAQA==";
const MCTS_CANDIDATES_COUNT: &[u8] = b"AAAAAAAACEAAAAAAAAAIQAAAAAAAABBAAAAAAAAACEAAAAAAAAAQQA==";
const PARALLEL_SCORE_MUL: &[u8] = b"bLoA5zId6z+FCyGhod/tP4LjVYDSB+o/bj1KWltz6D+oxD1/vvfsPw==";
const WIDTH_BUF: &[u8] = b"C9HmyMJu8T9O169jzYjxPxQpZTMBJvE/AmaGu7lt8T9SRrdenlHyPw==";
const UCB1_TUNED_COEF: &[u8] = b"QAMuHWjyzT8mxAidDqHZP/uEbDW2Rts/YNoxAWtK6T9tR46mczvcPw==";
const PARAM_ARRANGE_COUNT: &[u8] = b"9XfNlft8hD+dLVSRf8H+P4+/gGerw4s/KLQyqiL/zj8=";
const PARAM_QUERY_ANNEALING_DURATION_SEC: &[u8] = b"Slc03BR+hD9th4le65OEP+DWcg9hGbM/U4bHQ/OChD8=";
const PARAM_MCMC_INIT_DURATION_SEC: &[u8] = b"eR8o2NbLhD+RAvioXcOIP9Ib6MiuVuA/X+cHjNF+hD8=";
const PARAM_BEAM_MCTS_DURATION_RATIO: &[u8] = b"Za2ztxGFhD9GYldJmnyEP/65dQSF65A/spFU3nmNhD8=";
const PARAM_MCMC_DURATION_RATIO: &[u8] = b"NMQ33BiHhD/3SX0e46WEPwDorCiqNes/A+edbG+BhD8=";
const PARAM_MCTS_TURN: &[u8] = b"PScn/VX5DECDUrEs3iX0PyMG7oJ2upQ/0KAZpoEmCEA=";
const PARAM_MCTS_EXPANSION_THRESHOLD: &[u8] = b"0u4/VjB7DkD+oN52j3juP7VSkBbd9e8/yiry5fijhD8=";
const PARAM_MCTS_CANDIDATES_COUNT: &[u8] = b"T0waZNd+hD/NX22K2uG3P/KxomQ/LIY/vW7dwphSwj8=";
const PARAM_PARALLEL_SCORE_MUL: &[u8] = b"FHx9wduQhD/wrMiFluiGPya/3Bza4+0/NqFrlcKnhD8=";
const PARAM_WIDTH_BUF: &[u8] = b"UyOOcRWthz8zWHl3bcCEP8NKNMq57Oc/u1NxNUaVhD8=";
const PARAM_UCB1_TUNED_COEF: &[u8] = b"4d52/kOEhD8n1vX6dlWWPyhbYmV5PcY/aT6DukMJhz8=";

pub struct ParamSuggester {
    x_matrix: DMatrix<f64>,
    y_vector: DVector<f64>,
    hyper_param: DVector<f64>,
    y_inv_trans: fn(f64) -> f64,
    lower: f64,
    upper: f64,
}

impl ParamSuggester {
    fn new(
        hyper_param: DVector<f64>,
        x_matrix: DMatrix<f64>,
        y_vector: DVector<f64>,
        y_inv_trans: fn(f64) -> f64,
        lower: f64,
        upper: f64,
    ) -> Self {
        Self {
            hyper_param,
            x_matrix,
            y_vector,
            y_inv_trans,
            lower,
            upper,
        }
    }

    fn gen_x_matrix() -> DMatrix<f64> {
        let n = DVector::from_vec(decode_base64(N)).transpose();
        let t = DVector::from_vec(decode_base64(T)).transpose();
        let sigma = DVector::from_vec(decode_base64(SIGMA)).transpose();

        let x_matrix = DMatrix::from_rows(&[n, t, sigma]);

        x_matrix
    }

    pub fn gen_arrange_count_pred() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_ARRANGE_COUNT));
        let y_vector = DVector::from_vec(decode_base64(ARRANGE_COUNT));
        Self::new(
            hyper_param,
            Self::gen_x_matrix(),
            y_vector,
            |x| x,
            5.0,
            20.0,
        )
    }

    pub fn gen_query_annealing_duration_sec() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_QUERY_ANNEALING_DURATION_SEC));
        let y_vector = DVector::from_vec(decode_base64(QUERY_ANNEALING_DURATION_SEC));
        Self::new(
            hyper_param,
            Self::gen_x_matrix(),
            y_vector,
            |x| x,
            0.05,
            0.3,
        )
    }

    pub fn gen_mcmc_init_duration_sec() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_MCMC_INIT_DURATION_SEC));
        let y_vector = DVector::from_vec(decode_base64(MCMC_INIT_DURATION_SEC));
        Self::new(
            hyper_param,
            Self::gen_x_matrix(),
            y_vector,
            |x| x,
            0.05,
            0.15,
        )
    }

    pub fn gen_beam_mcts_duration_ratio() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_BEAM_MCTS_DURATION_RATIO));
        let y_vector = DVector::from_vec(decode_base64(BEAM_MCTS_DURATION_RATIO));
        Self::new(hyper_param, Self::gen_x_matrix(), y_vector, |x| x, 0.3, 0.7)
    }

    pub fn gen_mcmc_duration_ratio() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_MCMC_DURATION_RATIO));
        let y_vector = DVector::from_vec(decode_base64(MCMC_DURATION_RATIO));
        Self::new(
            hyper_param,
            Self::gen_x_matrix(),
            y_vector,
            |x| x,
            0.03,
            0.2,
        )
    }

    pub fn gen_mcts_turn() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_MCTS_TURN));
        let y_vector = DVector::from_vec(decode_base64(MCTS_TURN));
        Self::new(
            hyper_param,
            Self::gen_x_matrix(),
            y_vector,
            |x| x,
            0.8,
            20.0,
        )
    }

    pub fn gen_mcts_expansion_threshold() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_MCTS_EXPANSION_THRESHOLD));
        let y_vector = DVector::from_vec(decode_base64(MCTS_EXPANSION_THRESHOLD));
        Self::new(hyper_param, Self::gen_x_matrix(), y_vector, |x| x, 1.0, 5.0)
    }

    pub fn gen_mcts_candidates_count() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_MCTS_CANDIDATES_COUNT));
        let y_vector = DVector::from_vec(decode_base64(MCTS_CANDIDATES_COUNT));
        Self::new(hyper_param, Self::gen_x_matrix(), y_vector, |x| x, 2.0, 6.0)
    }

    pub fn gen_parallel_score_mul() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_PARALLEL_SCORE_MUL));
        let y_vector = DVector::from_vec(decode_base64(PARALLEL_SCORE_MUL));
        Self::new(hyper_param, Self::gen_x_matrix(), y_vector, |x| x, 0.7, 1.0)
    }

    pub fn gen_width_buf() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_WIDTH_BUF));
        let y_vector = DVector::from_vec(decode_base64(WIDTH_BUF));
        Self::new(
            hyper_param,
            Self::gen_x_matrix(),
            y_vector,
            |x| x,
            1.05,
            1.15,
        )
    }

    pub fn gen_ucb1_tuned_coef() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_UCB1_TUNED_COEF));
        let y_vector = DVector::from_vec(decode_base64(UCB1_TUNED_COEF));
        Self::new(
            hyper_param,
            Self::gen_x_matrix(),
            y_vector,
            |x| x,
            0.05,
            1.0,
        )
    }

    pub fn suggest(&self, n: usize, t: usize, sigma: f64) -> f64 {
        let t = (t as f64 / n as f64).ln();
        let n = (n - 30) as f64 / 70.0;
        let sigma = (sigma - 1000.0) / 9000.0;

        let len = self.x_matrix.shape().1;
        let y_mean = self.y_vector.mean();
        let y_mean = DVector::from_element(self.y_vector.len(), y_mean);
        let new_x = DMatrix::from_vec(3, 1, vec![n, t, sigma]);
        let noise = DMatrix::from_diagonal_element(len, len, self.hyper_param[3]);

        let k = self.calc_kernel_matrix(&self.x_matrix, &self.x_matrix) + noise;
        let kk = self.calc_kernel_matrix(&self.x_matrix, &new_x);

        let kernel_lu = k.lu();
        let new_y = kk.transpose() * kernel_lu.solve(&(&self.y_vector - &y_mean)).unwrap();

        (self.y_inv_trans)(new_y[(0, 0)] + y_mean[(0, 0)]).clamp(self.lower, self.upper)
    }

    fn calc_kernel_matrix(&self, x1: &DMatrix<f64>, x2: &DMatrix<f64>) -> DMatrix<f64> {
        let n = x1.shape().1;
        let m = x2.shape().1;
        let mut kernel = DMatrix::zeros(n, m);

        for i in 0..n {
            for j in 0..m {
                kernel[(i, j)] = self.gaussian_kernel(&x1.column(i), &x2.column(j));
            }
        }

        kernel
    }

    fn gaussian_kernel(&self, x1: &DVectorView<f64>, x2: &DVectorView<f64>) -> f64 {
        let t1 = self.hyper_param[0];
        let t2 = self.hyper_param[1];
        let t3 = self.hyper_param[2];

        let diff = x1 - x2;
        let norm_diff = diff.dot(&diff);
        let dot = x1.dot(&x2);
        t1 * dot + t2 * (-norm_diff / t3).exp()
    }
}

fn decode_base64(data: &[u8]) -> Vec<f64> {
    const BASE64_MAP: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut stream = vec![];

    let mut cursor = 0;

    while cursor + 4 <= data.len() {
        let mut buffer = 0u32;

        for i in 0..4 {
            let c = data[cursor + i];
            let shift = 6 * (3 - i);

            for (i, &d) in BASE64_MAP.iter().enumerate() {
                if c == d {
                    buffer |= (i as u32) << shift;
                }
            }
        }

        for i in 0..3 {
            let shift = 8 * (2 - i);
            let value = (buffer >> shift) as u8;
            stream.push(value);
        }

        cursor += 4;
    }

    let mut result = vec![];
    cursor = 0;

    while cursor + 8 <= stream.len() {
        let p = stream.as_ptr() as *const f64;
        let x = unsafe { *p.offset(cursor as isize / 8) };
        result.push(x);
        cursor += 8;
    }

    result
}

fn get_env<T: FromStr>(name: &str, default: T) -> T {
    env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
