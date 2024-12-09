use super::{GaussEstimator, RectEdge};
use crate::{
    problem::{Dir, Input, Op, Rect},
    sa::{self},
};
use itertools::Itertools;
use nalgebra::DVector;
use rand::prelude::*;
use std::u32;

pub(super) fn solve(
    input: &Input,
    estimator: &GaussEstimator,
    edge0: RectEdge,
    edge1: RectEdge,
    mean: DVector<f32>,
    variance: DVector<f32>,
    duration: f64,
    rng: &mut impl Rng,
) -> (Vec<Op>, DVector<f32>, DVector<f32>) {
    let env = Env::new(input, estimator.clone(), edge0, edge1, mean, variance);
    let state = State::new(&env);
    let annealer = sa::Annealer::new(1e4, 1e2, rng.gen(), 1);
    let (state, _stats) = annealer.run(&env, state, &NeighGen, duration);

    //eprintln!("{}", stats);

    state.to_op_and_edges(&env)
}

struct Env {
    estimator: GaussEstimator,
    rects: Vec<Rect>,
    edge0: RectEdge,
    edge1: RectEdge,
    edge01_width: f32,
    edge01_height: f32,
    mean: DVector<f32>,
    variance: DVector<f32>,
}

impl Env {
    fn new(
        input: &Input,
        estimator: GaussEstimator,
        edge0: RectEdge,
        edge1: RectEdge,
        mean: DVector<f32>,
        variance: DVector<f32>,
    ) -> Self {
        let rects = (0..estimator.rect_cnt)
            .map(|i| {
                let height = estimator.mean_height()[i].round() as u32;
                let width = estimator.mean_width()[i].round() as u32;
                let height = height.clamp(Input::MIN_RECT_SIZE, Input::MAX_RECT_SIZE);
                let width = width.clamp(Input::MIN_RECT_SIZE, Input::MAX_RECT_SIZE);
                Rect::new(height, width)
            })
            .collect_vec();

        // 高さ方向の干渉回避制約（小さい方に干渉しないように）
        // 干渉回避のため3σのマージンを取る
        let std_dev_h01 = (variance[edge0.to_index(estimator.rect_cnt)]
            .max(variance[edge1.to_index(estimator.rect_cnt)]))
        .sqrt();
        let edge01_height = (mean[edge0.to_index(estimator.rect_cnt)]
            .min(mean[edge1.to_index(estimator.rect_cnt)]))
            - 3.0 * std_dev_h01;

        // 幅方向の干渉回避制約
        let inv_edge0 = RectEdge::new(edge0.rect_i, !edge0.is_width);
        let inv_edge1 = RectEdge::new(edge1.rect_i, !edge1.is_width);
        let std_dev_w01 = (variance[inv_edge0.to_index(estimator.rect_cnt)]
            + variance[inv_edge1.to_index(estimator.rect_cnt)])
        .sqrt();
        let edge01_width = (mean[edge1.to_index(estimator.rect_cnt)]
            + mean[edge0.to_index(estimator.rect_cnt)])
            - 3.0 * std_dev_w01;

        Self {
            estimator,
            rects,
            edge0,
            edge1,
            edge01_width,
            edge01_height,
            mean,
            variance,
        }
    }

    fn mean_height(&self) -> &[f32] {
        &self.mean.as_slice()[..self.estimator.rect_cnt]
    }

    fn mean_width(&self) -> &[f32] {
        &self.mean.as_slice()[self.estimator.rect_cnt..]
    }

    fn variance_height(&self) -> &[f32] {
        &self.variance.as_slice()[..self.estimator.rect_cnt]
    }

    fn variance_width(&self) -> &[f32] {
        &self.variance.as_slice()[self.estimator.rect_cnt..]
    }
}

#[derive(Debug, Clone)]
struct State {
    placements: Vec<Option<(Dir, bool)>>,
    calculator_v: ScoreCalculator,
    calculator_h: ScoreCalculator,
}

impl State {
    fn new(env: &Env) -> Self {
        let mut placements = vec![None; env.estimator.rect_cnt];

        // #####@@@@@
        // ##0##@@1@@
        // #####            XXX
        //             <--- XXX
        //                  XXX
        // 0と1のどちらの高さの方が大きいかを調べたい
        // 長方形の幅を調べる場合は回転が必要
        placements[env.edge0.rect_i] = Some((Dir::Up, env.edge0.is_width));
        placements[env.edge1.rect_i] = Some((Dir::Left, env.edge1.is_width));

        let (vec_v, vec_h) = Self::get_box_vec(&env, &placements);
        let calculator_v = ScoreCalculator::new(&env.estimator, vec_v);
        let calculator_h = ScoreCalculator::new(&env.estimator, vec_h);

        Self {
            placements,
            calculator_v,
            calculator_h,
        }
    }

    fn get_box_vec(env: &Env, directions: &[Option<(Dir, bool)>]) -> (DVector<f32>, DVector<f32>) {
        // 本来ちゃんとシミュレーションすべきだが、単純な線形和として簡略化する
        let mut vec_v = DVector::zeros(env.estimator.rect_cnt * 2);
        let mut vec_h = DVector::zeros(env.estimator.rect_cnt * 2);

        for (rect_i, pair) in directions.iter().enumerate() {
            let &Some((dir, rotate)) = pair else {
                continue;
            };

            if rect_i == env.edge0.rect_i {
                let (v_index, h_index) = if rotate {
                    (rect_i + env.estimator.rect_cnt, rect_i)
                } else {
                    (rect_i, rect_i + env.estimator.rect_cnt)
                };

                vec_v[v_index] = 1.0;
                vec_h[h_index] = 1.0;
            } else {
                match dir {
                    Dir::Up => {
                        let h_index = if rotate {
                            rect_i
                        } else {
                            rect_i + env.estimator.rect_cnt
                        };
                        vec_v[h_index] = 1.0;
                    }
                    Dir::Left => {
                        let v_index = if rotate {
                            rect_i + env.estimator.rect_cnt
                        } else {
                            rect_i
                        };
                        vec_h[v_index] = 1.0;
                    }
                }
            }
        }

        (vec_v, vec_h)
    }

    fn to_op_and_edges(&self, env: &Env) -> (Vec<Op>, DVector<f32>, DVector<f32>) {
        let mut ops = vec![];
        let mut prev = None;
        let mut base = None;

        for (i, &placement) in self.placements.iter().enumerate() {
            let Some((dir, rotate)) = placement else {
                continue;
            };

            if i == env.edge0.rect_i {
                base = prev;
            }

            prev = Some(i);

            let current_base = match dir {
                Dir::Up => None,
                Dir::Left => base,
            };

            let op = Op::new(i, rotate, dir, current_base);
            ops.push(op);

            if i == env.edge1.rect_i {
                // 左の奴に合わせる
                base = Some(env.edge0.rect_i);
            }
        }

        (
            ops,
            self.calculator_v.v.clone(),
            self.calculator_h.v.clone(),
        )
    }
}

impl sa::State for State {
    type Env = Env;
    type Score = ScoreF32;

    fn score(&self, _env: &Self::Env) -> Self::Score {
        ScoreF32(self.calculator_v.score() + self.calculator_h.score())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct ScoreF32(f32);

impl sa::Score for ScoreF32 {
    fn raw_score(&self) -> f64 {
        self.0 as f64
    }
}

#[derive(Debug, Clone)]
struct ScoreCalculator {
    v: DVector<f32>,
    sigma_c_sum: DVector<f32>,
    tr_numerator: f32,
    tr_denominator: f32,
}

impl ScoreCalculator {
    fn new(estimator: &GaussEstimator, v: DVector<f32>) -> Self {
        let tr_numerator = (&estimator.variance * &v).norm_squared();
        let tr_denominator =
            (v.transpose() * &estimator.variance * &v)[0] + estimator.measure_variance;
        let sigma_c_sum = &estimator.variance * &v;

        Self {
            v,
            sigma_c_sum,
            tr_numerator,
            tr_denominator,
        }
    }

    fn update(&mut self, estimator: &GaussEstimator, new_v: &mut DVector<f32>) {
        let delta = &*new_v - &self.v;

        for (i, &dv) in delta.iter().enumerate() {
            if dv == 0.0 {
                continue;
            }

            self.tr_numerator += 2.0 * dv * (estimator.variance.column(i).dot(&self.sigma_c_sum));
            self.tr_numerator += dv * dv * estimator.variance.column(i).norm_squared();

            self.tr_denominator += 2.0 * dv * (estimator.variance.column(i).dot(&self.v));
            self.tr_denominator += dv * dv * estimator.variance[(i, i)];

            self.sigma_c_sum += dv * &estimator.variance.column(i);

            // swapする
            let temp = new_v[i];
            new_v[i] = self.v[i];
            self.v[i] = temp;
        }
    }

    fn score(&self) -> f32 {
        self.tr_numerator / self.tr_denominator
    }
}

struct NeighGen;

impl sa::NeighborGenerator for NeighGen {
    type Env = Env;
    type State = State;

    fn generate(
        &self,
        env: &Self::Env,
        state: &Self::State,
        rng: &mut impl Rng,
    ) -> Box<dyn sa::Neighbor<Env = Self::Env, State = Self::State>> {
        ChangeDirNeigh::gen(env, state, rng)
    }
}

#[derive(Debug, Clone)]
struct ChangeDirNeigh {
    index: usize,
    placement: Option<(Dir, bool)>,
    old_placement: Option<(Dir, bool)>,
    vec_v: Option<DVector<f32>>,
    vec_h: Option<DVector<f32>>,
}

impl ChangeDirNeigh {
    fn gen(
        env: &Env,
        state: &State,
        rng: &mut impl Rng,
    ) -> Box<dyn sa::Neighbor<Env = Env, State = State>> {
        'generate: loop {
            let index = rng.gen_range(0..env.estimator.rect_cnt);

            // edge0とedge1は固定
            if index == env.edge0.rect_i || index == env.edge1.rect_i {
                continue;
            }

            let placement = if rng.gen_bool(0.1) && state.placements[index].is_some() {
                // これによって干渉が生じないかチェック
                if env.edge0.rect_i < index
                    && state.placements[index].map(|(dir, _)| dir) == Some(Dir::Up)
                {
                    // 消してもpivotの下の箱に変化がない場合はOK
                    let mut has_buffer = false;

                    for i in env.edge0.rect_i + 1..index {
                        if state.placements[i].map(|(dir, _)| dir) == Some(Dir::Up) {
                            has_buffer = true;
                            break;
                        }
                    }

                    if !has_buffer {
                        // 新たにpivotの下に来る箱
                        let mut next = None;

                        for i in index + 1..env.rects.len() {
                            if let Some((dir, rotate)) = state.placements[i] {
                                if dir == Dir::Up {
                                    next = Some((i, rotate));
                                    break;
                                }
                            }
                        }

                        // 箱の幅が閾値より大きい場合はNG
                        if let Some((i, rotate)) = next {
                            let width = if rotate {
                                env.mean_height()[i]
                            } else {
                                env.mean_width()[i]
                            };

                            if width >= env.edge01_width {
                                continue 'generate;
                            }
                        }
                    }
                }

                None
            } else {
                let dir = if rng.gen_bool(0.5) {
                    Dir::Up
                } else {
                    Dir::Left
                };

                // pivotより先に置くものは下から上のみ
                if index < env.edge0.rect_i && dir == Dir::Left {
                    continue;
                }

                let rotate = rng.gen_bool(0.5);

                if state.placements[index] == Some((dir, rotate)) {
                    continue;
                }

                // 干渉が生じないかチェック
                // edge0とedge1の間に入る場合は高さ制限あり
                if env.edge0.rect_i < index && index < env.edge1.rect_i && dir == Dir::Left {
                    let height = if rotate {
                        env.mean_width()[index]
                    } else {
                        env.mean_height()[index]
                    };

                    if height >= env.edge01_height {
                        continue;
                    }
                }

                // pivotのすぐ下に来る場合は幅制限あり
                let mut has_buffer = false;

                for i in env.edge0.rect_i + 1..index {
                    if state.placements[i].map(|(dir, _)| dir) == Some(Dir::Up) {
                        has_buffer = true;
                        break;
                    }
                }

                if !has_buffer {
                    let width = if rotate {
                        env.mean_height()[index]
                    } else {
                        env.mean_width()[index]
                    };

                    if width >= env.edge01_width {
                        continue;
                    }
                }

                Some((dir, rotate))
            };

            let mut placements = state.placements.clone();
            placements[index] = placement;

            return Box::new(Self {
                index,
                placement,
                old_placement: state.placements[index],
                vec_v: None,
                vec_h: None,
            });
        }
    }
}

impl sa::Neighbor for ChangeDirNeigh {
    type Env = Env;
    type State = State;

    fn preprocess(&mut self, env: &Self::Env, state: &mut Self::State) {
        state.placements[self.index] = self.placement;

        let (mut vec_v, mut vec_h) = State::get_box_vec(&env, &state.placements);
        state.calculator_v.update(&env.estimator, &mut vec_v);
        state.calculator_h.update(&env.estimator, &mut vec_h);

        self.vec_v = Some(vec_v);
        self.vec_h = Some(vec_h);
    }

    fn postprocess(self: Box<Self>, _env: &Self::Env, _state: &mut Self::State) {
        // do nothing
    }

    fn rollback(self: Box<Self>, env: &Self::Env, state: &mut Self::State) {
        state.placements[self.index] = self.old_placement;

        state
            .calculator_v
            .update(&env.estimator, &mut self.vec_v.unwrap());
        state
            .calculator_h
            .update(&env.estimator, &mut self.vec_h.unwrap());
    }
}
