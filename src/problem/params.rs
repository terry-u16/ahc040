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
    pub touching_threshold: usize,
    pub invalid_cnt_threshold: usize,
}

impl Params {
    pub(super) fn new(n: usize, t: usize, sigma: f64) -> Self {
        let arrange_count = ParamSuggester::gen_arrange_count_pred(t)
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
        let touching_threshold =
            ParamSuggester::gen_touching_threshold().suggest(n, t, sigma) as usize;
        let invalid_cnt_threshold =
            ParamSuggester::gen_invalid_cnt_threshold().suggest(n, t, sigma) as usize;

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
            touching_threshold,
            invalid_cnt_threshold,
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
        let touching_threshold = get_env("AHC_TOUCHING_THRESHOLD", 16);
        let invalid_cnt_threshold = get_env("AHC_INVALID_CNT_THRESHOLD", 10);

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
            touching_threshold,
            invalid_cnt_threshold,
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
        writeln!(f, "ucb1_tuned_coef: {}", self.ucb1_tuned_coef)?;
        writeln!(f, "touching_threshold: {}", self.touching_threshold)?;
        writeln!(f, "invalid_cnt_threshold: {}", self.invalid_cnt_threshold)
    }
}

// パラメータチューニングその1
const N1: &[u8] = b"Fl/xFV/xtT8d1EEd1EGtP8VXfMVXfOU/6qAO6qAO6j/btm3btm3bP8VXfMVXfOU/fMVXfMVX3D+amZmZmZm5P9u2bdu2bcs/JUmSJEmS5D9CHdRBHdThP6EO6qAO6uA/HdRBHdRB7T+w+Iqv+IrvP83MzMzMzOw/kiRJkiRJ0j+w+Iqv+IrvPzuogzqog+o/HdRBHdRBrT9YfMVXfMXXP+qgDuqgDuo/LL7iK77i6z+3bdu2bdvmP/EVX/EVX+E/mpmZmZmZ2T9YfMVXfMXnP5qZmZmZmek/kiRJkiRJwj9mZmZmZmbmPxZf8RVf8cU/hDqogzqo4z9YfMVXfMXXP5IkSZIkSdI/HdRBHdRBvT+hDuqgDurgP4uv+Iqv+Oo/6qAO6qAO6j/UQR3UQR3EP7dt27Zt2+Y/X/EVX/EVzz9QB3VQB3XgP5IkSZIkScI/+Yqv+Iqv6D+SJEmSJEniPzMzMzMzM9M/UAd1UAd1wD9QB3VQB3XgP1/xFV/xFe8/fMVXfMVX7D+SJEmSJEmyP7dt27Zt29Y/UAd1UAd1wD8AAAAAAADgP/EVX/EVX+E/MzMzMzMz4z+amZmZmZm5PzMzMzMzM9M/O6iDOqiD2j/btm3btm3rP1h8xVd8xdc/27Zt27Zt6z+amZmZmZnJP27btm3btu0/dVAHdVAH5T8P6qAO6qDuP1AHdVAHddA/mpmZmZmZuT9u27Zt27btP/EVX/EVX9E/mpmZmZmZyT+SJEmSJEniP4Q6qIM6qOM/X/EVX/EVzz++4iu+4ivuPw==";
const T1: &[u8] = b"k2DD4m+s3L/CU5lGkvrqP80YAXd+r8W/cRodhGTk6z8sPbRVMqrgP2iapt7KPNM/Yw21Xila3D8Em7cuz03lPwj8awCFXd6/G+rBNVJ58D+R2zQRYmnSP51eEMEtmPM/ktkJmojJ6z+SFkS7i6rhvzbGmVqqpcY/zG2Yr7Gu2L+uNzc+ZtzkP+TgEmghmLM/9RMjIAvG8D9vZP8B3OfQvygzznoMis+/RiLz124I7z94Ogl+RizQPwj+72zOj8E/TJi/7CPz2T/xNZuRWaDRP73c1BJlut4/zyYnGooU3j/J9fkmOavgP4CzUUeIhO4/2Efx9osP5L8RCCR1Evnjv/EWsxbOl9q/2zaZ1o5J2L//3DL2WHSnv6+GlbQTPNm/lG83Uro12T+XrRZk8v3Tv8rErlDUgus/CwOteuqT8T/ikQx8hZ/zPz4ZzFsdQ6q/ZuArt+Y09D9Zsb9wcYPhP7a5uMOIdsA/X8ebsdX+uD+sZw8zuA/Tv480E9nCiOW/mpxHhUSk8z9kAi00kYzjv0wbO1FUBe8/iyHFFQF/5D+LpH8DjgHpP5p2sFND7vI/ziYnGooU3r8v49gy7KzoP0tV3qR4798/ixIJnuI1kT9RFIH6rljwP/PaJj4MlOc/TPQM95R27T8bivukYcKmP2a/WDWPsPE/r1Rt+c9zsz9VxQTNEs7gP+PZijztk8q/Pv5lGQOF4j9cu1r9CFnWv9flxxbB9+0/K52o6XZl47/RQHBeX+jhv51IDnlLGOg/v8g5jxue4D/CXhzCi2HUPw==";
const SIGMA1: &[u8] = b"FBtVwmI25T/NhaQpFWfrP3BeTTwrGu0/buydgprl2z+ASXkPDG/oP5vlYxX6Ea0/bJsxLpFa5j/SlIqzD5/lP452K637Ftc/LR+r0I/o2j/fT42XbhLjP5KmVJx9+Ow/cWPvFNQs6z9TKs4+fIbVPzhCGb0ta+0/d3d3d3d36z/45tXEs6LBP5aPVehHdO0/q9CP6Npm6D/ZqBIWs+nlP+ZjFfoRXcs/XI/C9Shc1z9eJ1ft6UzeP+HBPFH/Ruw/VqEf0bXNqD9OYhBYObToP02pOPvwGbY/ltZ9i//Z0j+/WPKLJb/YPzn78Bl2Bcg/tA+fYVeAzD/n+6nx0k3SPxtVwmI2Pe8/xI29U1CzvD8LtmALtmDTP+MSqaUI0u0/+hFd24xx7T/vgeENB83nP1S9wF2UZOY/kceXAQWi2D92Bci95kKiPwu2YAu2YO8/wYN5ov6N2D/iM+wKkHvNPxyhjN6WteI/4C5KMudo4z967sgJsb7CP7lqT2eyMNI/lKv2dCYL6z+BuyjJnKPdP2So7DB1ue0/sEyDU73AvT/MpueOnBDjP4Y3HDR//dY/2vTckRNi6T+uR+F6FK7fP+mTPumTPtE/ZRqc6gXu7j9GtvP91HjRPz5WoR/RtdU/juM4juM43j+grSH8POTlP8CkvAeGN9Q/v+vkqj2d7T+W1n2L/9mKP+sr0+BUL+w/pHA9Ctej4D9y1Z7OZGHUP7Q1hJ+HPOo/rPZ0Jgsjzj9d24xxidTaPyfE+so0OMU/xPrKNDjV0z/xrGgk4JvnPw==";
const ARRANGE_COUNT: &[u8] = b"AAAAAAAAKEAAAAAAAAAiQAAAAAAAACBAAAAAAAAAKEAAAAAAAAAoQAAAAAAAABhAAAAAAAAAJkAAAAAAAAAmQAAAAAAAACRAAAAAAAAAJEAAAAAAAAAUQAAAAAAAACJAAAAAAAAAJEAAAAAAAAAkQAAAAAAAABRAAAAAAAAAIEAAAAAAAAAcQAAAAAAAABRAAAAAAAAAGEAAAAAAAAAcQAAAAAAAABhAAAAAAAAAIkAAAAAAAAAcQAAAAAAAABxAAAAAAAAAHEAAAAAAAAAYQAAAAAAAACBAAAAAAAAAGEAAAAAAAAAmQAAAAAAAABhAAAAAAAAAIEAAAAAAAAAmQAAAAAAAACRAAAAAAAAAGEAAAAAAAAAiQAAAAAAAABxAAAAAAAAAJEAAAAAAAAAiQAAAAAAAACZAAAAAAAAAJkAAAAAAAAAgQAAAAAAAABhAAAAAAAAAIkAAAAAAAAAgQAAAAAAAABxAAAAAAAAAGEAAAAAAAAAgQAAAAAAAABxAAAAAAAAAIkAAAAAAAAAmQAAAAAAAACxAAAAAAAAAIEAAAAAAAAAiQAAAAAAAACJAAAAAAAAAGEAAAAAAAAAYQAAAAAAAACRAAAAAAAAAHEAAAAAAAAAgQAAAAAAAACJAAAAAAAAAKEAAAAAAAAAUQAAAAAAAACBAAAAAAAAAGEAAAAAAAAAYQAAAAAAAACRAAAAAAAAAGEAAAAAAAAAcQAAAAAAAACJAAAAAAAAAJkAAAAAAAAAkQAAAAAAAACxAAAAAAAAAGEAAAAAAAAAoQA==";
const QUERY_ANNEALING_DURATION_SEC: &[u8] = b"95Ax5HV6uz+3nNK9X8nOP6+Jp+1vPsM/8Z35ADfkzz81Wvg+2CzTP8OVCxg25sQ/vHuCZ4ks0D/TCxE5KYvGPzMzMzMzM9M/7Bdt7yu1wD8qLNiB7PDHP1bxcEKBgMo/9FFvDmzZzj8+CvboZyTGP9e20LPpps4/91pqZ50SxD9Y1BNo1ZvRP4YswqmhjNA/Dg7xZMCpwT/Y0/SeXUixPzpZhzJs5sg/cgXTCFsGxj9F+ZmeXg3EP1pmguSRfsw/7uB5HK3pyz8FC+w0drjQPx0CRSC26sk/wuvkd5gazj//deaVL9PGP73g1UI43bc/VpLUTGhZzj/ZdzaXJzDJP9ioFYvi/sI/lhvwSFUUxj/0o2dmntjDPweMqrc+l6s/CjDJjdBA0T/aKEeigMPSP89BEJPe6s8/WahZoGRwsz9jals8llWtPx67V3qUTr4/zBrTzQPnzz9YjBMe6HvSP2IEKEEUt9E/1R0pElKG0j+ajEe+hyq7P1cVEKGEic0/SnsNoo7U0D8uSWeXbx+6P36tw90I0bs/NMWh/G5Wzj+OG0vxkSzOP52BphNFOcQ/CIZpPYUYwz9FcvPt6VXKP0A78trSXNE/GkAgin5v0T/QLFzG0u7DP7CF+l1XXsc/P2Z9DtHEtj/px5df4ce2Py80Kc9UX9A/ys5mmUZ00D+nk5QbmXrOPzMzMzMzM9M/Ila+UBDBzD+87emTYL7QP8qFPN6sFMg/AUybWbRGrj/KWocZz3zBP+I3brKyc7E/sU1IqXZ+zj+YgVDeSbjOPw==";
const MCMC_INIT_DURATION_SEC: &[u8] = b"ScJ8YM4psT/tiBZJzAO1P95kK0vp+Lg/hHAk6uejwj+GM/ppmRDDP/mO2FYjpa0/H3LT0KcXsT+T3LXX4neyP5qZmZmZmbk/1qQ4gi/Brj837NwNXL+2PyuN1iDBSb0/YC4zKvUFwj9sS6fzPE+4PwugocifsLY/Cc+W9IaptT/hs8Apu5uwPxEyelGDGbw/X2dHaC+QwD/gw5mNkXK2P9W2kz7xKcA/XsE4GHxdtD+6W3DJswaxP4WCsGULnbg/vSuxHGQCuD+M263c7P28PzHSjy4cYLg/v83e5/ZrvD8uYDZZpQ++P8sVK0jZzrs/ZwGSG9musT+SGcpMoFW6P1Td1zhgarQ/nZwCByRNrj+T4Tx3Zzq1P+nPDyITB68/xB1QdS+BtD+iQI3qDGWxP/UazHkynMI/z6FArLIQsj+P+24KSayzP8tWUETXJro/qOAjVBsBvT/AuPqgDGe2P4xovpqkT8E/nuQCBXmyuj9mHdtszXOwP9J7WP7nV7A/RIAUs08dsD+ySDg2wrG7P695WTnZm7M/BEWKB2d0wj+AnfkMLBW4P/EagZ8tA8A/VWO0Be3kuj9FnXssE8C4P+AU82HfdrM/PoSVHd1Ztz9QEdVNygXAP2IJjvsOAMM/cHN14eh+wT+x31ZMqZiyP1wWaoXD6cE/eKf9+IsRwj9JjbHRRnW0P5qZmZmZmbk/eY8/Axvluz8QXz8+hrC6Pwt9LNwHX7o/Wv6XZ+l4vz/9Nl/mLxCxP8tl37tvArU/zeqq3llTvD+sBfsJGLq/Pw==";
const BEAM_MCTS_DURATION_RATIO: &[u8] = b"TjrjHFMH5T8HtloAoLjfP9RGpI5Q5eE/4MAPWcRQ4T/09y/8zq3fPzdQvh3b3OU/FNdbk7TD5D8CXNliZTvfPwAAAAAAAOA/p+IipVA14D8CRALIcKvWPxjleqj4cOU/UtfUJyZ/4z+Fn3ReYezjP2QLJMJRBd4/G+f0G+q/4D/+x4Lg4nTiPzi37xmr5t0/jwwsEsDs3j/jcvh7t1TgP7YS6p/OaeI/EEqmjz1t4j+BMmepkDLfPz6MonZvNdg/yigehuFV2z882OKbT4HTP/NYFSGDIeI/aTvk5hh63z8CY4DVWoPjP1w4kZ/MUto/gFaiJw8x4z/alGc7xNLdP2kgLcBW9uM/5apDlfQF5D+n15cBCtvgP45mtkPz/N8/O+EbGfoD5j9r91SjjV3kPyqTKdnL+uU/RayeUyjQ5D9T44jEHdHeP+X2OX7ZYOM/yZrLoIw85D87/68a7wzgP3rCqFiPXuE//aW5xzgQ4T8RRUkxUEfaP5nF8tC6Sd8/Ud9Uu78y5j8ai3XrjL/lP18NBl9AXdw/5IiPyP+g2j9SctT8LQPkP7y8SgwDCOA/198WWm+M4j9U/OJ3BA/gP0Df3CsYxt4/0BTPKwzx3T+FWTtYe6fhP8rB1DlBm+A/Vo0fjl7M5T/ScuktHqTkPxKQK+0ecOA/zfD/f67/3D90Sxg/N5XfPwAAAAAAAOA/1gHKTz9n4T+YiGAZoAPhPynC1fa66uM/IxffeBRd4T8nnHYBPCHgPwtYEnViM+E/OlM3nhl82j9D7ywEKbPjPw==";
const MCMC_DURATION_RATIO: &[u8] = b"I4mDuInsxT9rK780xTO+P+NVThxkZME/lC6bTV4MwD/qnEJTLvCwP5xocJQSrbM/eLcdwlfQwj/goDqyBIXAP5qZmZmZmbk/kLqKgfEBsj8rEp/7LnWqPxfsq22AIq4/IMQ5aJLHxT/vVumQZmCxP+V6uf74Cro/1ktLT1hTuD+65OlMu3G9P7qeu1jy1sE/1IifptGZsD9yZh8Eu+CtP62inbvOfaA/8h3PaQpMuz8Dx0L2VAaxP3EoH3X8bsk/UTg5GqIbwj/cm/WnRkWtP2vDCv4EzbA/igTa2h0CqD+mKNrurcqzP58VAsOwZ7s/EvBA31vKrj/JN+07PUO9PxRx7RIa1cE/eIl1BIierz8QXdF1H0C0P0hA01NOeqw/MBr2MknXuD9SRVwv27q/P3ZnPLpeM78/lAq5gQqTrT8XokQ5XOW4P7RlJVaj1Lo//WV9oDNXtT/25v2xB2+zP6VnxA2Hzac/PpGYp3L4wj+E3SMES1mtP2tpEumCZ6g/Nrn/gjz6vj+uMPcBp6XAP93wieGTM8Q/dp5jYJ2Brz98hNtyJKHGPyM99CRugrM/lsGMapy9oj/V/zm2G+C1P6u355rbT74/HFj+3SvWxz9L6eOJbEusP+o9laUJ3qU/8LD3QHgstz+2pIiXDoLJP6QUMuBBX8Y/eK3zTZKCwj+CZCIWrPKxP5qZmZmZmbk/LQ+8ugfWuz9QsgIvYjvJPxd+SdD8Ybs/G9PpNw8EyD/Wu3MaoOnBP0y0BxWIjas/qa5vpEaxvz+oRB7w63G7Pw==";
const MCTS_TURN: &[u8] = b"AAAAAAAALEAAAAAAAAAyQAAAAAAAADNAAAAAAAAAM0AAAAAAAAAwQAAAAAAAACZAAAAAAAAAMkAAAAAAAAAsQAAAAAAAAC5AAAAAAAAAJkAAAAAAAAAxQAAAAAAAADJAAAAAAAAAKEAAAAAAAAA2QAAAAAAAADNAAAAAAAAAMEAAAAAAAAAqQAAAAAAAADNAAAAAAAAAMkAAAAAAAAAwQAAAAAAAADRAAAAAAAAAKEAAAAAAAAAzQAAAAAAAADNAAAAAAAAAMkAAAAAAAAA1QAAAAAAAAChAAAAAAAAALEAAAAAAAAAoQAAAAAAAACxAAAAAAAAAM0AAAAAAAAAwQAAAAAAAADdAAAAAAAAALEAAAAAAAAAyQAAAAAAAADRAAAAAAAAAKkAAAAAAAAAuQAAAAAAAAChAAAAAAAAALkAAAAAAAAAxQAAAAAAAADVAAAAAAAAAKEAAAAAAAAAzQAAAAAAAADBAAAAAAAAALEAAAAAAAAAyQAAAAAAAADVAAAAAAAAAJkAAAAAAAAAoQAAAAAAAADBAAAAAAAAALEAAAAAAAAAyQAAAAAAAADFAAAAAAAAANEAAAAAAAAAuQAAAAAAAADBAAAAAAAAAMUAAAAAAAAAoQAAAAAAAAC5AAAAAAAAAJkAAAAAAAAAsQAAAAAAAAChAAAAAAAAAMkAAAAAAAAAsQAAAAAAAAC5AAAAAAAAAM0AAAAAAAAAqQAAAAAAAADBAAAAAAAAALEAAAAAAAAAzQAAAAAAAACZAAAAAAAAALEAAAAAAAAAoQA==";
const MCTS_EXPANSION_THRESHOLD: &[u8] = b"AAAAAAAAEEAAAAAAAAAIQAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAAhAAAAAAAAAFEAAAAAAAAAIQAAAAAAAAAhAAAAAAAAAFEAAAAAAAAAAQAAAAAAAABBAAAAAAAAACEAAAAAAAAAAQAAAAAAAABRAAAAAAAAAFEAAAAAAAAAIQAAAAAAAABRAAAAAAAAAEEAAAAAAAAAAQAAAAAAAAABAAAAAAAAA8D8AAAAAAAAQQAAAAAAAAAhAAAAAAAAACEAAAAAAAAAQQAAAAAAAAPA/AAAAAAAACEAAAAAAAAAAQAAAAAAAABBAAAAAAAAAFEAAAAAAAAAIQAAAAAAAABRAAAAAAAAACEAAAAAAAAAIQAAAAAAAABRAAAAAAAAA8D8AAAAAAAAIQAAAAAAAAABAAAAAAAAACEAAAAAAAAAIQAAAAAAAABRAAAAAAAAACEAAAAAAAAAUQAAAAAAAABRAAAAAAAAAAEAAAAAAAAAAQAAAAAAAABBAAAAAAAAAFEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAAAEAAAAAAAAAIQAAAAAAAABRAAAAAAAAAEEAAAAAAAAAUQAAAAAAAAAhAAAAAAAAACEAAAAAAAADwPwAAAAAAAAhAAAAAAAAACEAAAAAAAAAIQAAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAAhAAAAAAAAAEEAAAAAAAAAAQAAAAAAAABBAAAAAAAAAEEAAAAAAAAAIQAAAAAAAABBAAAAAAAAACEAAAAAAAAAQQA==";
const MCTS_CANDIDATES_COUNT: &[u8] = b"AAAAAAAAEEAAAAAAAAAIQAAAAAAAABRAAAAAAAAACEAAAAAAAAAIQAAAAAAAABRAAAAAAAAACEAAAAAAAAAIQAAAAAAAABBAAAAAAAAAEEAAAAAAAAAIQAAAAAAAAAhAAAAAAAAAEEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAAFEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAAEEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAACEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAACEAAAAAAAAAIQAAAAAAAABRAAAAAAAAAEEAAAAAAAAAQQAAAAAAAABhAAAAAAAAACEAAAAAAAAAIQAAAAAAAABBAAAAAAAAAEEAAAAAAAAAIQAAAAAAAABRAAAAAAAAACEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAACEAAAAAAAAAQQAAAAAAAABBAAAAAAAAACEAAAAAAAAAIQAAAAAAAABBAAAAAAAAAEEAAAAAAAAAIQAAAAAAAAAhAAAAAAAAAFEAAAAAAAAAQQAAAAAAAABBAAAAAAAAAFEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAAFEAAAAAAAAAQQAAAAAAAABBAAAAAAAAAEEAAAAAAAAAQQAAAAAAAABBAAAAAAAAAEEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAAEEAAAAAAAAAQQAAAAAAAABBAAAAAAAAACEAAAAAAAAAIQAAAAAAAAAhAAAAAAAAAEEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAACEAAAAAAAAAQQA==";
const PARALLEL_SCORE_MUL: &[u8] = b"aWYdM2Rz6z/hbX1aJqnrP7xjGEhjxe4/SpZ7jx7k5z+HyR0E2oXqP4IoC9ZfYOg/8jGIaDGR5z+JlHqWA9vtP83MzMzMzOw/9nog2ahW5z8U4UArloXoP+aiPGMzeOc/LlKyoEOf6D9nv2cUl2vuP+NaaeRY4O4/RMH6kYMh7j8yTO7BmpXsP6HhRLP18e0/4gqjYlAJ6j+oxD1/vvfsPwCad650uOo/FnK7Q1d36z8Lgp3he23sPwW1EWMuNuo/CIxVHKQZ7z8UCXv9JNzvP4RMEl7MlO0/oB+ShcY+7T+6VVWccIjrP7o5YMsRnuw/bLoA5zId6z/ljU0xMLvsP9YAWxlf6ug/S2RZHyj57T+rlAmKz1DoP67T/2Q1ROk/bj1KWltz6D+u3UmemaHsP+VtlzIMmOw/q1YV3Dr27D/Qis13LTrvP36DLMrJ7Og/eFfr23RI7T/U6MjQDGLoP+3WJu+pdOw/ESH1+m467j9bKrv87k3uPzextb57Ze0/0H8DrhKt5z9qJi5KGtXpP9bji4ZY+ec/zUCA6KOy6z9iKWhu0QPuP6FRv3c/9OY/OcCdDD2f7z9+e4bOQ1PnP3EwNmNqCuk//pN6pWHh7D/ZHfrukHfoP3S/qDza+es/CWHev/Bt6D9eMXcgabjpP0vQ/6Lr++c/Mw7xzpjW6j+bNnDEbWXrP83MzMzMzOw/htsY65zh6T8gDuZfsMvsP4n0AbRuSe0/guNVgNIH6j9gSusR8bXtP7G0TkzAGOk/JTo7e1kJ7D/oWNtvElDtPw==";
const WIDTH_BUF: &[u8] = b"BX8871YY8j+SE/XQE1PxPyUSAP24KvI/BBm8L8u68D9/dis9jMPxP2Ik4wZsGPE/xH//jKK18T9hL1iHRkfxP5qZmZmZmfE/y+ETjk+X8T8oXZ1V3a3xPw5qGGx1mvE/qmAxM2KK8D/ATT7OiZzxP/axm/HFMPE/upGsEzWH8T+jjL46/y/xP2ATupOYEPI/ijoaqifn8D9SRrdenlHyPzumifPV0vE//HAIME1R8T9zz1ZqlTrxPzc9zETn/fE/Z+yxwafQ8D89ciSGk3DxP+D+kYxb8vA/qy/CXHuI8T9QF+u8vznxPx+KB+1p1vA/C9HmyMJu8T/lZxf4urLxP6qzoO6oZfI/Oh3pt5WG8T98zbaVWxjxP1eXvAJayvE/AmaGu7lt8T+ho14l//nxPwuBV2N+pvA/KzDWxILF8D/azsKcTXjwP3qHrU2OxPE/wd0iyjZj8D+TKRg9bJ7wP039uG9pHfE/K1cOG1ps8T+SmO4YNvXxP9E90EX1vPE/vEW5rZLE8D9/VqFTNUrxP72px1tpCfE/EHL3AibN8D+GU8OlXXfxP7QkbLWH9PA/p/Fzgy9j8j/fSTjTQrbxPzs55eSs1/E/ZIifaYP18T/b/x4FE87wP5PfbHKpmfE/jSkDbckb8T9KfIk9rdDxPz2hRvtNEvE/pQZfCCdj8T8n0NyO8jrxP5qZmZmZmfE//1QmoTXC8T9xFkpSHqDxP0crAfT2ufE/FCllMwEm8T9RU8qaS6vxP0aimCW98fA/sutf19ES8T8+4Mkj2xfyPw==";
const UCB1_TUNED_COEF: &[u8] = b"0NEtiC9D5z81OaQZ8QzQP0BA0nhkytM/byT+hef7rj/I2kVpeA3uP4Umj2YWYrg/ytNOgz7NxD/dSfBxsBHjPwAAAAAAAOA/d/SSv3mI3j9eaxY7wWjhP+DuJSLyM8U/gHplJlBxxj+e+bkf+RLRP+5VKGh96Nc/D/LtE6f55D+/lbjI5LvAP1YEdP/yJds/ZedwecNsyD9tR46mczvcP7/xQSEnWL8/xf9trNJZ5D8+bo0a9nS8Pwtc50qaJNE/bg3tTf7Wwz8+Ypu6CZjNP6WvzcwxAMQ/V2T3y0Da2z/sikbdKUK/P5Fg1Evat9A/QAMuHWjyzT/ihMmM0JrcPzWnnRAKpeQ/vdcEhfTB2j+3T0tTq4HEP2uHpupitN8/YNoxAWtK6T/aLUgArejhP6DjEU+N8eo/4/1Vy9OUxD9+WqU3uI21P3vqGFDB690/zoOLZdcR4D9EmV+yjRC/P6vbK1Noido/alNCubcn3z+Ur0nJpvTDP9bWCrpIlMk/q2iS46/Dsz8j9GQPnnTiPxhzIXA118o/LsRdJEmeyj+AuXks7fyxP5gO/rvdocM/GmmPe1Qb5T8Kvaimuw/bP31dLYtwXsQ/YZ63SjHJzT8n007MyS2/PxqKVNzlb9s/mRyuXSJu6D+Gr6ZBKY7gPxX8aootb9s/3vsvUMy/zj/vYZJOz1KvPwAAAAAAAOA/zPxBxhiPzz99ljGjKYDlPxkatL5RM+A/+4RsNbZG2z8LgJ2feT7dPzNopt1JmdY/4QJz/6Rt4T9e8TYbuUTDPw==";
const PARAM_ARRANGE_COUNT: &[u8] = b"jRBJnsXimj9wgcTetawKQIcxcuL8y7A/NCUOC8Bw+j8=";
const PARAM_QUERY_ANNEALING_DURATION_SEC: &[u8] = b"7A62czmYhD/GjKRcUWyHPz2Fb6tOYOY/VSGPVoWIhD8=";
const PARAM_MCMC_INIT_DURATION_SEC: &[u8] = b"8BWA5jGPjz+moQ71mJOEP2dbHJFP8u8/+GYKJceEhD8=";
const PARAM_BEAM_MCTS_DURATION_RATIO: &[u8] = b"b3ogiF+OhD9RRq4VLFyKP9sih4iUQ+0/L+rx20SGhD8=";
const PARAM_MCMC_DURATION_RATIO: &[u8] = b"R6R94sHBiT/DaGF8jlaOP1gBiEh1pe8/xIaGuSqGhD8=";
const PARAM_MCTS_TURN: &[u8] = b"e6ZVr0qEnz9RfXVpEPMcQPV2DK0dk+o/rHilJnNUFUA=";
const PARAM_MCTS_EXPANSION_THRESHOLD: &[u8] = b"3oDEaG7ThD9+42LRvRf2PyZ/9efGWYw/29vFm0ncpz8=";
const PARAM_MCTS_CANDIDATES_COUNT: &[u8] = b"i4Q0dFZzhz/Hz4A2vy/QP65UrBlRXcA/NU+sS//I0D8=";
const PARAM_PARALLEL_SCORE_MUL: &[u8] = b"WrcOSSP4hD/mENpZ046EP9NDoYc4Wu8/8dfS3NuQhD8=";
const PARAM_WIDTH_BUF: &[u8] = b"QCLB65GMhj/N8DHQy/mLP5d6NK2Hw+w/VqFp6Bp+hD8=";
const PARAM_UCB1_TUNED_COEF: &[u8] = b"+GmSQpB9jD/hPYjWRKmEP23WXKJNVdI/4gh3743KoD8=";

// パラメータチューニングその2
const N2: &[u8] = b"mpmZmZmZ2T+SJEmSJEnCPyy+4iu+4us/X/EVX/EVzz8d1EEd1EGtP5IkSZIkSdI/HdRBHdRBjT9f8RVf8RXPPzMzMzMzM9M/1EEd1EEd1D8=";
const T2: &[u8] = b"uLm4w4h2wL+Md2WPDvnxPwoz8vcP+/A/WicKmZpV4z+oYhHAMAqvv4xfcwPXA+w/6ncJLSNN3T+veJu1ES7cP6I/9/f54OI/xsf+QCsG0z8=";
const SIGMA2: &[u8] = b"FK5H4XoU1j/Gkl8s+cWSPz1R/0Yoo+8/PVH/Riijtz/ZqBIWs+ntP+Psw2fYFcA/u2/xP1tD2D9xY+8U1CznPxMWs+m5I9c/u9z+IENl1z8=";
const TOUCHING_THRESHOLD: &[u8] = b"AAAAAAAALEAAAAAAAAAwQAAAAAAAACpAAAAAAAAAKkAAAAAAAAAuQAAAAAAAACxAAAAAAAAALkAAAAAAAAAwQAAAAAAAADBAAAAAAAAALEA=";
const INVALID_CNT_THRESHOLD: &[u8] = b"AAAAAAAA8D8AAAAAAAAIQAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA8D8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
const PARAM_TOUCHING_THRESHOLD: &[u8] = b"74kcIv2WhD9JRwZppGXwP+mNMIRHJ4Y/tawKTt0AzD8=";
const PARAM_INVALID_CNT_THRESHOLD: &[u8] = b"uArHZjCJhD9D2VENXISEP4HNJiNj5Z8//7FMyc/78D8=";

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

    fn gen_x1_matrix() -> DMatrix<f64> {
        let n = DVector::from_vec(decode_base64(N1)).transpose();
        let t = DVector::from_vec(decode_base64(T1)).transpose();
        let sigma = DVector::from_vec(decode_base64(SIGMA1)).transpose();

        let x_matrix = DMatrix::from_rows(&[n, t, sigma]);

        x_matrix
    }

    fn gen_x2_matrix() -> DMatrix<f64> {
        let n = DVector::from_vec(decode_base64(N2)).transpose();
        let t = DVector::from_vec(decode_base64(T2)).transpose();
        let sigma = DVector::from_vec(decode_base64(SIGMA2)).transpose();

        let x_matrix = DMatrix::from_rows(&[n, t, sigma]);

        x_matrix
    }

    pub fn gen_arrange_count_pred(t: usize) -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_ARRANGE_COUNT));
        let y_vector = DVector::from_vec(decode_base64(ARRANGE_COUNT));
        Self::new(
            hyper_param,
            Self::gen_x1_matrix(),
            y_vector,
            |x| x,
            5.0,
            (t - 1) as f64,
        )
    }

    pub fn gen_query_annealing_duration_sec() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_QUERY_ANNEALING_DURATION_SEC));
        let y_vector = DVector::from_vec(decode_base64(QUERY_ANNEALING_DURATION_SEC));
        Self::new(
            hyper_param,
            Self::gen_x1_matrix(),
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
            Self::gen_x1_matrix(),
            y_vector,
            |x| x,
            0.05,
            0.15,
        )
    }

    pub fn gen_beam_mcts_duration_ratio() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_BEAM_MCTS_DURATION_RATIO));
        let y_vector = DVector::from_vec(decode_base64(BEAM_MCTS_DURATION_RATIO));
        Self::new(
            hyper_param,
            Self::gen_x1_matrix(),
            y_vector,
            |x| x,
            0.3,
            0.7,
        )
    }

    pub fn gen_mcmc_duration_ratio() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_MCMC_DURATION_RATIO));
        let y_vector = DVector::from_vec(decode_base64(MCMC_DURATION_RATIO));
        Self::new(
            hyper_param,
            Self::gen_x1_matrix(),
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
            Self::gen_x1_matrix(),
            y_vector,
            |x| x,
            0.8,
            20.0,
        )
    }

    pub fn gen_mcts_expansion_threshold() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_MCTS_EXPANSION_THRESHOLD));
        let y_vector = DVector::from_vec(decode_base64(MCTS_EXPANSION_THRESHOLD));
        Self::new(
            hyper_param,
            Self::gen_x1_matrix(),
            y_vector,
            |x| x,
            1.0,
            5.0,
        )
    }

    pub fn gen_mcts_candidates_count() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_MCTS_CANDIDATES_COUNT));
        let y_vector = DVector::from_vec(decode_base64(MCTS_CANDIDATES_COUNT));
        Self::new(
            hyper_param,
            Self::gen_x1_matrix(),
            y_vector,
            |x| x,
            2.0,
            6.0,
        )
    }

    pub fn gen_parallel_score_mul() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_PARALLEL_SCORE_MUL));
        let y_vector = DVector::from_vec(decode_base64(PARALLEL_SCORE_MUL));
        Self::new(
            hyper_param,
            Self::gen_x1_matrix(),
            y_vector,
            |x| x,
            0.7,
            1.0,
        )
    }

    pub fn gen_width_buf() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_WIDTH_BUF));
        let y_vector = DVector::from_vec(decode_base64(WIDTH_BUF));
        Self::new(
            hyper_param,
            Self::gen_x1_matrix(),
            y_vector,
            |x| x,
            1.02,
            1.15,
        )
    }

    pub fn gen_ucb1_tuned_coef() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_UCB1_TUNED_COEF));
        let y_vector = DVector::from_vec(decode_base64(UCB1_TUNED_COEF));
        Self::new(
            hyper_param,
            Self::gen_x1_matrix(),
            y_vector,
            |x| x,
            0.05,
            1.0,
        )
    }

    fn gen_touching_threshold() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_TOUCHING_THRESHOLD));
        let y_vector = DVector::from_vec(decode_base64(TOUCHING_THRESHOLD));
        Self::new(
            hyper_param,
            Self::gen_x2_matrix(),
            y_vector,
            |x| x,
            12.0,
            16.0,
        )
    }

    fn gen_invalid_cnt_threshold() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_INVALID_CNT_THRESHOLD));
        let y_vector = DVector::from_vec(decode_base64(INVALID_CNT_THRESHOLD));
        Self::new(
            hyper_param,
            Self::gen_x2_matrix(),
            y_vector,
            |x| x,
            0.0,
            3.0,
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
