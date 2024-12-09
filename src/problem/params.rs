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

const N: &[u8] = b"Fl/xFV/xtT8d1EEd1EGtP8VXfMVXfOU/6qAO6qAO6j/btm3btm3bP8VXfMVXfOU/fMVXfMVX3D+amZmZmZm5P9u2bdu2bcs/JUmSJEmS5D9CHdRBHdThP6EO6qAO6uA/HdRBHdRB7T+w+Iqv+IrvP83MzMzMzOw/kiRJkiRJ0j+w+Iqv+IrvPzuogzqog+o/HdRBHdRBrT9YfMVXfMXXP+qgDuqgDuo/LL7iK77i6z+3bdu2bdvmP/EVX/EVX+E/mpmZmZmZ6T+SJEmSJEnCP2ZmZmZmZuY/Fl/xFV/xxT+EOqiDOqjjP1h8xVd8xdc/kiRJkiRJ0j+hDuqgDurgP4uv+Iqv+Oo/6qAO6qAO6j/UQR3UQR3EP1/xFV/xFc8/UAd1UAd14D+SJEmSJEnCP/mKr/iKr+g/MzMzMzMz0z9QB3VQB3XAP1AHdVAHdeA/X/EVX/EV7z+SJEmSJEmyP7dt27Zt29Y/UAd1UAd1wD8AAAAAAADgP/EVX/EVX+E/MzMzMzMz4z+amZmZmZm5PzMzMzMzM9M/O6iDOqiD2j/btm3btm3rP1h8xVd8xdc/27Zt27Zt6z+amZmZmZnJP27btm3btu0/dVAHdVAH5T8P6qAO6qDuP1AHdVAHddA/btu2bdu27T+amZmZmZnJP5IkSZIkSeI/hDqogzqo4z9f8RVf8RXPP77iK77iK+4/";
const T: &[u8] = b"k2DD4m+s3L/CU5lGkvrqP80YAXd+r8W/cRodhGTk6z8sPbRVMqrgP2iapt7KPNM/Yw21Xila3D8Em7cuz03lPwj8awCFXd6/G+rBNVJ58D+R2zQRYmnSP51eEMEtmPM/ktkJmojJ6z+SFkS7i6rhvzbGmVqqpcY/zG2Yr7Gu2L+uNzc+ZtzkP+TgEmghmLM/9RMjIAvG8D9vZP8B3OfQvygzznoMis+/RiLz124I7z94Ogl+RizQPwj+72zOj8E/vdzUEmW63j/PJicaihTeP8n1+SY5q+A/gLNRR4iE7j/YR/H2iw/kvxEIJHUS+eO/8RazFs6X2r//3DL2WHSnv6+GlbQTPNm/lG83Uro12T+XrRZk8v3TvwsDrXrqk/E/4pEMfIWf8z8+GcxbHUOqv2bgK7fmNPQ/trm4w4h2wD9fx5ux1f64P6xnDzO4D9O/jzQT2cKI5b9kAi00kYzjv0wbO1FUBe8/iyHFFQF/5D+LpH8DjgHpP5p2sFND7vI/ziYnGooU3r8v49gy7KzoP0tV3qR4798/ixIJnuI1kT9RFIH6rljwP/PaJj4MlOc/TPQM95R27T8bivukYcKmP2a/WDWPsPE/r1Rt+c9zsz9VxQTNEs7gP+PZijztk8q/XLta/QhZ1r8rnajpdmXjv9FAcF5f6OG/nUgOeUsY6D+/yDmPG57gP8JeHMKLYdQ/";
const SIGMA: &[u8] = b"FBtVwmI25T/NhaQpFWfrP3BeTTwrGu0/buydgprl2z+ASXkPDG/oP5vlYxX6Ea0/bJsxLpFa5j/SlIqzD5/lP452K637Ftc/LR+r0I/o2j/fT42XbhLjP5KmVJx9+Ow/cWPvFNQs6z9TKs4+fIbVPzhCGb0ta+0/d3d3d3d36z/45tXEs6LBP5aPVehHdO0/q9CP6Npm6D/ZqBIWs+nlP+ZjFfoRXcs/XI/C9Shc1z9eJ1ft6UzeP+HBPFH/Ruw/Tak4+/AZtj+W1n2L/9nSP79Y8oslv9g/OfvwGXYFyD+0D59hV4DMP+f7qfHSTdI/G1XCYjY97z8LtmALtmDTP+MSqaUI0u0/+hFd24xx7T/vgeENB83nP5HHlwEFotg/dgXIveZCoj8LtmALtmDvP8GDeaL+jdg/HKGM3pa14j/gLkoy52jjP3ruyAmxvsI/uWpPZ7Iw0j+BuyjJnKPdP2So7DB1ue0/sEyDU73AvT/MpueOnBDjP4Y3HDR//dY/2vTckRNi6T+uR+F6FK7fP+mTPumTPtE/ZRqc6gXu7j9GtvP91HjRPz5WoR/RtdU/juM4juM43j+grSH8POTlP8CkvAeGN9Q/v+vkqj2d7T+W1n2L/9mKP+sr0+BUL+w/ctWezmRh1D+s9nQmCyPOP13bjHGJ1No/J8T6yjQ4xT/E+so0ONXTP/GsaCTgm+c/";
const ARRANGE_COUNT: &[u8] = b"AAAAAAAAKEAAAAAAAAAiQAAAAAAAACBAAAAAAAAAKEAAAAAAAAAoQAAAAAAAABhAAAAAAAAAJkAAAAAAAAAmQAAAAAAAACRAAAAAAAAAJEAAAAAAAAAUQAAAAAAAACJAAAAAAAAAJEAAAAAAAAAkQAAAAAAAABRAAAAAAAAAIEAAAAAAAAAcQAAAAAAAABRAAAAAAAAAGEAAAAAAAAAcQAAAAAAAABhAAAAAAAAAIkAAAAAAAAAcQAAAAAAAABxAAAAAAAAAIEAAAAAAAAAYQAAAAAAAACZAAAAAAAAAGEAAAAAAAAAgQAAAAAAAACZAAAAAAAAAJEAAAAAAAAAiQAAAAAAAABxAAAAAAAAAJEAAAAAAAAAiQAAAAAAAACZAAAAAAAAAIEAAAAAAAAAYQAAAAAAAACJAAAAAAAAAHEAAAAAAAAAYQAAAAAAAACBAAAAAAAAAHEAAAAAAAAAmQAAAAAAAACxAAAAAAAAAIEAAAAAAAAAiQAAAAAAAACJAAAAAAAAAGEAAAAAAAAAYQAAAAAAAACRAAAAAAAAAHEAAAAAAAAAgQAAAAAAAACJAAAAAAAAAKEAAAAAAAAAUQAAAAAAAACBAAAAAAAAAGEAAAAAAAAAYQAAAAAAAACRAAAAAAAAAHEAAAAAAAAAmQAAAAAAAACRAAAAAAAAALEAAAAAAAAAYQAAAAAAAAChA";
const QUERY_ANNEALING_DURATION_SEC: &[u8] = b"95Ax5HV6uz+3nNK9X8nOP6+Jp+1vPsM/8Z35ADfkzz81Wvg+2CzTP8OVCxg25sQ/vHuCZ4ks0D/TCxE5KYvGPzMzMzMzM9M/7Bdt7yu1wD8qLNiB7PDHP1bxcEKBgMo/9FFvDmzZzj8+CvboZyTGP9e20LPpps4/91pqZ50SxD9Y1BNo1ZvRP4YswqmhjNA/Dg7xZMCpwT/Y0/SeXUixPzpZhzJs5sg/cgXTCFsGxj9F+ZmeXg3EP1pmguSRfsw/HQJFILbqyT/C6+R3mBrOP/915pUv08Y/veDVQjjdtz9WktRMaFnOP9l3NpcnMMk/2KgVi+L+wj/0o2dmntjDPweMqrc+l6s/CjDJjdBA0T/aKEeigMPSP1moWaBkcLM/Y2pbPJZVrT8eu1d6lE6+P8wa080D588/YgQoQRS30T/VHSkSUobSP5qMR76HKrs/VxUQoYSJzT8uSWeXbx+6P36tw90I0bs/NMWh/G5Wzj+OG0vxkSzOP52BphNFOcQ/CIZpPYUYwz9FcvPt6VXKP0A78trSXNE/GkAgin5v0T/QLFzG0u7DP7CF+l1XXsc/P2Z9DtHEtj/px5df4ce2Py80Kc9UX9A/ys5mmUZ00D+nk5QbmXrOPzMzMzMzM9M/vO3pk2C+0D8BTJtZtEauP8pahxnPfME/4jdusrJzsT+xTUipdn7OP5iBUN5JuM4/";
const MCMC_INIT_DURATION_SEC: &[u8] = b"ScJ8YM4psT/tiBZJzAO1P95kK0vp+Lg/hHAk6uejwj+GM/ppmRDDP/mO2FYjpa0/H3LT0KcXsT+T3LXX4neyP5qZmZmZmbk/1qQ4gi/Brj837NwNXL+2PyuN1iDBSb0/YC4zKvUFwj9sS6fzPE+4PwugocifsLY/Cc+W9IaptT/hs8Apu5uwPxEyelGDGbw/X2dHaC+QwD/gw5mNkXK2P9W2kz7xKcA/XsE4GHxdtD+6W3DJswaxP4WCsGULnbg/MdKPLhxguD+/zd7n9mu8Py5gNlmlD74/yxUrSNnOuz9nAZIb2a6xP5IZykygVbo/VN3XOGBqtD+T4Tx3Zzq1P+nPDyITB68/xB1QdS+BtD+iQI3qDGWxP8+hQKyyELI/j/tuCkmssz/LVlBE1ya6P6jgI1QbAb0/jGi+mqRPwT+e5AIFebK6P2Yd22zNc7A/0ntY/udXsD+ySDg2wrG7P695WTnZm7M/BEWKB2d0wj+AnfkMLBW4P/EagZ8tA8A/VWO0Be3kuj9FnXssE8C4P+AU82HfdrM/PoSVHd1Ztz9QEdVNygXAP2IJjvsOAMM/cHN14eh+wT+x31ZMqZiyP1wWaoXD6cE/eKf9+IsRwj9JjbHRRnW0P5qZmZmZmbk/EF8/Poawuj9a/pdn6Xi/P/02X+YvELE/y2Xfu28CtT/N6qreWVO8P6wF+wkYur8/";
const BEAM_MCTS_DURATION_RATIO: &[u8] = b"TjrjHFMH5T8HtloAoLjfP9RGpI5Q5eE/4MAPWcRQ4T/09y/8zq3fPzdQvh3b3OU/FNdbk7TD5D8CXNliZTvfPwAAAAAAAOA/p+IipVA14D8CRALIcKvWPxjleqj4cOU/UtfUJyZ/4z+Fn3ReYezjP2QLJMJRBd4/G+f0G+q/4D/+x4Lg4nTiPzi37xmr5t0/jwwsEsDs3j/jcvh7t1TgP7YS6p/OaeI/EEqmjz1t4j+BMmepkDLfPz6MonZvNdg/81gVIYMh4j9pO+TmGHrfPwJjgNVag+M/XDiRn8xS2j+AVqInDzHjP9qUZzvE0t0/aSAtwFb24z+n15cBCtvgP45mtkPz/N8/O+EbGfoD5j9r91SjjV3kP0WsnlMo0OQ/U+OIxB3R3j/l9jl+2WDjP8may6CMPOQ/esKoWI9e4T/9pbnHOBDhPxFFSTFQR9o/mcXy0LpJ3z8ai3XrjL/lP18NBl9AXdw/5IiPyP+g2j9SctT8LQPkP7y8SgwDCOA/198WWm+M4j9U/OJ3BA/gP0Df3CsYxt4/0BTPKwzx3T+FWTtYe6fhP8rB1DlBm+A/Vo0fjl7M5T/ScuktHqTkPxKQK+0ecOA/zfD/f67/3D90Sxg/N5XfPwAAAAAAAOA/mIhgGaAD4T8jF994FF3hPyecdgE8IeA/C1gSdWIz4T86UzeeGXzaP0PvLAQps+M/";
const MCMC_DURATION_RATIO: &[u8] = b"I4mDuInsxT9rK780xTO+P+NVThxkZME/lC6bTV4MwD/qnEJTLvCwP5xocJQSrbM/eLcdwlfQwj/goDqyBIXAP5qZmZmZmbk/kLqKgfEBsj8rEp/7LnWqPxfsq22AIq4/IMQ5aJLHxT/vVumQZmCxP+V6uf74Cro/1ktLT1hTuD+65OlMu3G9P7qeu1jy1sE/1IifptGZsD9yZh8Eu+CtP62inbvOfaA/8h3PaQpMuz8Dx0L2VAaxP3EoH3X8bsk/a8MK/gTNsD+KBNraHQKoP6Yo2u6tyrM/nxUCw7Bnuz8S8EDfW8quP8k37Ts9Q70/FHHtEhrVwT8QXdF1H0C0P0hA01NOeqw/MBr2MknXuD9SRVwv27q/P5QKuYEKk60/F6JEOVzluD+0ZSVWo9S6P/1lfaAzV7U/pWfEDYfNpz8+kZincvjCP4TdIwRLWa0/a2kS6YJnqD+uMPcBp6XAP93wieGTM8Q/dp5jYJ2Brz98hNtyJKHGPyM99CRugrM/lsGMapy9oj/V/zm2G+C1P6u355rbT74/HFj+3SvWxz9L6eOJbEusP+o9laUJ3qU/8LD3QHgstz+2pIiXDoLJP6QUMuBBX8Y/eK3zTZKCwj+CZCIWrPKxP5qZmZmZmbk/ULICL2I7yT8b0+k3DwTIP9a7cxqg6cE/TLQHFYiNqz+prm+kRrG/P6hEHvDrcbs/";
const MCTS_TURN: &[u8] = b"AAAAAAAALEAAAAAAAAAyQAAAAAAAADNAAAAAAAAAM0AAAAAAAAAwQAAAAAAAACZAAAAAAAAAMkAAAAAAAAAsQAAAAAAAAC5AAAAAAAAAJkAAAAAAAAAxQAAAAAAAADJAAAAAAAAAKEAAAAAAAAA2QAAAAAAAADNAAAAAAAAAMEAAAAAAAAAqQAAAAAAAADNAAAAAAAAAMkAAAAAAAAAwQAAAAAAAADRAAAAAAAAAKEAAAAAAAAAzQAAAAAAAADNAAAAAAAAAKEAAAAAAAAAsQAAAAAAAAChAAAAAAAAALEAAAAAAAAAzQAAAAAAAADBAAAAAAAAAN0AAAAAAAAAyQAAAAAAAADRAAAAAAAAAKkAAAAAAAAAuQAAAAAAAAC5AAAAAAAAAMUAAAAAAAAA1QAAAAAAAAChAAAAAAAAAMEAAAAAAAAAsQAAAAAAAADJAAAAAAAAANUAAAAAAAAAoQAAAAAAAADBAAAAAAAAALEAAAAAAAAAyQAAAAAAAADFAAAAAAAAANEAAAAAAAAAuQAAAAAAAADBAAAAAAAAAMUAAAAAAAAAoQAAAAAAAAC5AAAAAAAAAJkAAAAAAAAAsQAAAAAAAAChAAAAAAAAAMkAAAAAAAAAsQAAAAAAAAC5AAAAAAAAAKkAAAAAAAAAsQAAAAAAAADNAAAAAAAAAJkAAAAAAAAAsQAAAAAAAAChA";
const MCTS_EXPANSION_THRESHOLD: &[u8] = b"AAAAAAAAEEAAAAAAAAAIQAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAAhAAAAAAAAAFEAAAAAAAAAIQAAAAAAAAAhAAAAAAAAAFEAAAAAAAAAAQAAAAAAAABBAAAAAAAAACEAAAAAAAAAAQAAAAAAAABRAAAAAAAAAFEAAAAAAAAAIQAAAAAAAABRAAAAAAAAAEEAAAAAAAAAAQAAAAAAAAABAAAAAAAAA8D8AAAAAAAAQQAAAAAAAAAhAAAAAAAAA8D8AAAAAAAAIQAAAAAAAAABAAAAAAAAAEEAAAAAAAAAUQAAAAAAAAAhAAAAAAAAAFEAAAAAAAAAIQAAAAAAAABRAAAAAAAAA8D8AAAAAAAAIQAAAAAAAAAhAAAAAAAAACEAAAAAAAAAUQAAAAAAAAAhAAAAAAAAAFEAAAAAAAAAAQAAAAAAAAABAAAAAAAAAEEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAAAEAAAAAAAAAIQAAAAAAAABRAAAAAAAAAEEAAAAAAAAAUQAAAAAAAAAhAAAAAAAAACEAAAAAAAADwPwAAAAAAAAhAAAAAAAAACEAAAAAAAAAIQAAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAAhAAAAAAAAAAEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAAEEAAAAAAAAAIQAAAAAAAABBA";
const MCTS_CANDIDATES_COUNT: &[u8] = b"AAAAAAAAEEAAAAAAAAAIQAAAAAAAABRAAAAAAAAACEAAAAAAAAAIQAAAAAAAABRAAAAAAAAACEAAAAAAAAAIQAAAAAAAABBAAAAAAAAAEEAAAAAAAAAIQAAAAAAAAAhAAAAAAAAAEEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAAFEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAAEEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAACEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAAFEAAAAAAAAAQQAAAAAAAABBAAAAAAAAAGEAAAAAAAAAIQAAAAAAAAAhAAAAAAAAAEEAAAAAAAAAIQAAAAAAAABRAAAAAAAAACEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAAEEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAAEEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAACEAAAAAAAAAQQAAAAAAAABBAAAAAAAAAFEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAAFEAAAAAAAAAQQAAAAAAAABBAAAAAAAAAEEAAAAAAAAAQQAAAAAAAABBAAAAAAAAAEEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAAEEAAAAAAAAAQQAAAAAAAABBAAAAAAAAACEAAAAAAAAAQQAAAAAAAABBAAAAAAAAACEAAAAAAAAAIQAAAAAAAABBA";
const PARALLEL_SCORE_MUL: &[u8] = b"aWYdM2Rz6z/hbX1aJqnrP7xjGEhjxe4/SpZ7jx7k5z+HyR0E2oXqP4IoC9ZfYOg/8jGIaDGR5z+JlHqWA9vtP83MzMzMzOw/9nog2ahW5z8U4UArloXoP+aiPGMzeOc/LlKyoEOf6D9nv2cUl2vuP+NaaeRY4O4/RMH6kYMh7j8yTO7BmpXsP6HhRLP18e0/4gqjYlAJ6j+oxD1/vvfsPwCad650uOo/FnK7Q1d36z8Lgp3he23sPwW1EWMuNuo/hEwSXsyU7T+gH5KFxj7tP7pVVZxwiOs/ujlgyxGe7D9sugDnMh3rP+WNTTEwu+w/1gBbGV/q6D+rlAmKz1DoP67T/2Q1ROk/bj1KWltz6D+u3UmemaHsP6tWFdw69uw/0IrNdy067z9+gyzKyezoP3hX69t0SO0/7dYm76l07D8RIfX6bjruP1squ/zuTe4/N7G1vntl7T9qJi5KGtXpP9bji4ZY+ec/zUCA6KOy6z9iKWhu0QPuP6FRv3c/9OY/OcCdDD2f7z9+e4bOQ1PnP3EwNmNqCuk//pN6pWHh7D/ZHfrukHfoP3S/qDza+es/CWHev/Bt6D9eMXcgabjpP0vQ/6Lr++c/Mw7xzpjW6j+bNnDEbWXrP83MzMzMzOw/IA7mX7DL7D+C41WA0gfqP2BK6xHxte0/sbROTMAY6T8lOjt7WQnsP+hY228SUO0/";
const WIDTH_BUF: &[u8] = b"BX8871YY8j+SE/XQE1PxPyUSAP24KvI/BBm8L8u68D9/dis9jMPxP2Ik4wZsGPE/xH//jKK18T9hL1iHRkfxP5qZmZmZmfE/y+ETjk+X8T8oXZ1V3a3xPw5qGGx1mvE/qmAxM2KK8D/ATT7OiZzxP/axm/HFMPE/upGsEzWH8T+jjL46/y/xP2ATupOYEPI/ijoaqifn8D9SRrdenlHyPzumifPV0vE//HAIME1R8T9zz1ZqlTrxPzc9zETn/fE/4P6RjFvy8D+rL8Jce4jxP1AX67y/OfE/H4oH7WnW8D8L0ebIwm7xP+VnF/i6svE/qrOg7qhl8j98zbaVWxjxP1eXvAJayvE/AmaGu7lt8T+ho14l//nxPysw1sSCxfA/2s7CnE148D96h61NjsTxP8HdIso2Y/A/Tf24b2kd8T8rVw4bWmzxP5KY7hg29fE/0T3QRfW88T9/VqFTNUrxP72px1tpCfE/EHL3AibN8D+GU8OlXXfxP7QkbLWH9PA/p/Fzgy9j8j/fSTjTQrbxPzs55eSs1/E/ZIifaYP18T/b/x4FE87wP5PfbHKpmfE/jSkDbckb8T9KfIk9rdDxPz2hRvtNEvE/pQZfCCdj8T8n0NyO8jrxP5qZmZmZmfE/cRZKUh6g8T8UKWUzASbxP1FTyppLq/E/RqKYJb3x8D+y61/X0RLxPz7gySPbF/I/";
const UCB1_TUNED_COEF: &[u8] = b"0NEtiC9D5z81OaQZ8QzQP0BA0nhkytM/byT+hef7rj/I2kVpeA3uP4Umj2YWYrg/ytNOgz7NxD/dSfBxsBHjPwAAAAAAAOA/d/SSv3mI3j9eaxY7wWjhP+DuJSLyM8U/gHplJlBxxj+e+bkf+RLRP+5VKGh96Nc/D/LtE6f55D+/lbjI5LvAP1YEdP/yJds/ZedwecNsyD9tR46mczvcP7/xQSEnWL8/xf9trNJZ5D8+bo0a9nS8Pwtc50qaJNE/pa/NzDEAxD9XZPfLQNrbP+yKRt0pQr8/kWDUS9q30D9AAy4daPLNP+KEyYzQmtw/NaedEAql5D+3T0tTq4HEP2uHpupitN8/YNoxAWtK6T/aLUgArejhP+P9VcvTlMQ/flqlN7iNtT976hhQwevdP86Di2XXEeA/q9srU2iJ2j9qU0K5tyffP5SvScmm9MM/1tYKukiUyT8j9GQPnnTiPxhzIXA118o/LsRdJEmeyj+AuXks7fyxP5gO/rvdocM/GmmPe1Qb5T8Kvaimuw/bP31dLYtwXsQ/YZ63SjHJzT8n007MyS2/PxqKVNzlb9s/mRyuXSJu6D+Gr6ZBKY7gPxX8aootb9s/3vsvUMy/zj/vYZJOz1KvPwAAAAAAAOA/fZYxoymA5T/7hGw1tkbbPwuAnZ95Pt0/M2im3UmZ1j/hAnP/pG3hP17xNhu5RMM/";
const PARAM_ARRANGE_COUNT: &[u8] = b"LpwKIAKjhD+f0RCuF/QRQMqyY+sKn60/rMABrwU16z8=";
const PARAM_QUERY_ANNEALING_DURATION_SEC: &[u8] = b"EIk7nth/hD9KzY996IOHPzRJWXjP2+8/GXRbsm+NhD8=";
const PARAM_MCMC_INIT_DURATION_SEC: &[u8] = b"LY4sNsuIhD+3ace94XCHPzTM4PuJ8+8/TvQSPcuAhD8=";
const PARAM_BEAM_MCTS_DURATION_RATIO: &[u8] = b"p4lL8Qu/hD+JyxoFMIeIP2FBaw2r4u8/H2UusYZ+hD8=";
const PARAM_MCMC_DURATION_RATIO: &[u8] = b"Q1SB63uZhD9TavdISoaEP+qCzWCm2+8/PVXsfeSKhD8=";
const PARAM_MCTS_TURN: &[u8] = b"pYTWe8z6wD9TpCTmlZIdQGG7ahyy7u8/uFS+5GO2E0A=";
const PARAM_MCTS_EXPANSION_THRESHOLD: &[u8] = b"JVt68k37jj9IGWwT4Qz3P8LOf4XQT44/d2AramC2kD8=";
const PARAM_MCTS_CANDIDATES_COUNT: &[u8] = b"X1rf8ByFhD85v+8wX2TUP/lybQGTccM/mZcxqFcYzz8=";
const PARAM_PARALLEL_SCORE_MUL: &[u8] = b"uPbK/C8Rkz+JhFN8ZUuHP2TYYTOxXu0/3jPQLoCRhD8=";
const PARAM_WIDTH_BUF: &[u8] = b"BOqkVux0iz9tj8pLF9OEP+PqZpBn4u8/SwXW/pSEhD8=";
const PARAM_UCB1_TUNED_COEF: &[u8] = b"nGF18PGMhD/f2ZYqwXyEPy/oapsl08o/CyhfUa1FoD8=";

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

    pub fn gen_arrange_count_pred(t: usize) -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_ARRANGE_COUNT));
        let y_vector = DVector::from_vec(decode_base64(ARRANGE_COUNT));
        Self::new(
            hyper_param,
            Self::gen_x_matrix(),
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
            1.02,
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
