use super::GaussEstimator;
use crate::{
    problem::{Dir, Op, Rect},
    sa::{self},
    solver::estimator::Placement,
    util::ChangeMinMax,
};
use itertools::Itertools;
use nalgebra::DVector;
use rand::prelude::*;
use smallvec::SmallVec;
use std::u32;

pub(super) fn solve(
    estimator: &GaussEstimator,
    duration: f64,
    rng: &mut impl Rng,
) -> (Vec<Op>, DVector<f64>, DVector<f64>) {
    let env = Env::new(estimator.clone());
    let state = State::init(&env, rng);
    let annealer = sa::Annealer::new(1e4, 1e2, rng.gen(), 1);
    let (state, stats) = annealer.run(&env, state, &NeighGen, duration);

    eprintln!("{}", stats);

    state.to_op_and_edges()
}

struct Env {
    estimator: GaussEstimator,
    rects: Vec<Rect>,
}

impl Env {
    fn new(estimator: GaussEstimator) -> Self {
        let rects = (0..estimator.rect_cnt)
            .map(|i| {
                let height = estimator.mean_height()[i].round() as u32;
                let width = estimator.mean_width()[i].round() as u32;
                let height = height.clamp(20000, 100000);
                let width = width.clamp(20000, 100000);
                Rect::new(height, width)
            })
            .collect_vec();

        Self { estimator, rects }
    }
}

#[derive(Debug, Clone)]
struct State {
    placements: Vec<Option<(Dir, bool)>>,
    pivot: usize,
    calculator_v: ScoreCalculator,
    calculator_h: ScoreCalculator,
}

impl State {
    fn new(env: &Env, placements: Vec<Option<(Dir, bool)>>, pivot: usize) -> Self {
        let (vec_v, vec_h) = Self::get_box_vec(&env, &placements, pivot);
        let calculator_v = ScoreCalculator::new(&env.estimator, vec_v);
        let calculator_h = ScoreCalculator::new(&env.estimator, vec_h);

        Self {
            placements,
            pivot,
            calculator_v,
            calculator_h,
        }
    }

    fn init(env: &Env, rng: &mut impl Rng) -> Self {
        let mut placements = (0..env.estimator.rect_cnt)
            .map(|_| {
                let rotate = rng.gen_bool(0.5);
                Some(if rng.gen_bool(0.5) {
                    (Dir::Left, rotate)
                } else {
                    (Dir::Up, rotate)
                })
            })
            .collect_vec();
        placements[0] = Some((Dir::Up, false));
        let pivot = 0;

        Self::new(env, placements, pivot)
    }

    fn get_box_vec(
        env: &Env,
        directions: &[Option<(Dir, bool)>],
        pivot: usize,
    ) -> (DVector<f64>, DVector<f64>) {
        let mut parents_v = vec![None; env.estimator.rect_cnt];
        let mut parents_h = vec![None; env.estimator.rect_cnt];
        let mut placements_v = SmallVec::<[(Placement, usize, usize); 5]>::new();
        let mut placements_h = SmallVec::<[(Placement, usize, usize); 5]>::new();
        let mut vec_v = DVector::zeros(env.estimator.rect_cnt * 2);
        let mut vec_h = DVector::zeros(env.estimator.rect_cnt * 2);

        let mut x = 0;
        let mut y = 0;
        let mut pivot_y = u32::MAX / 2;

        // v, h方向に置いている箱のうち、壁にピッタリくっついているものの番号
        let mut last_h = None;
        let mut last_v = None;

        let mut max_x = 0;
        let mut max_y = 0;
        let mut max_x_i = !0;
        let mut max_y_i = !0;

        const MAX_BOX_SIZE: u32 = 100000;

        for (i, dir) in directions.iter().enumerate() {
            let &Some((dir, rotate)) = dir else {
                continue;
            };

            let rect = &env.rects[i];

            match dir {
                Dir::Up => {
                    let (w, h, vec_i) = if rotate {
                        (rect.height(), rect.width(), i + env.estimator.rect_cnt)
                    } else {
                        (rect.width(), rect.height(), i)
                    };

                    let x0 = 0;
                    let x1 = x0 + w;

                    if y >= MAX_BOX_SIZE + pivot_y {
                        // 絶対に他の箱と干渉しない

                        // pivotの場合は水平方向のrootとなる
                        if pivot == i {
                            pivot_y = y;
                            last_h = Some(i);
                            x = x1;
                        }

                        y += h;
                        parents_v[i] = last_v;
                        last_v = Some(i);

                        if max_x.change_max(x1) {
                            max_x_i = i;
                        }

                        if max_y.change_max(y) {
                            max_y_i = i;
                        }

                        continue;
                    }

                    let mut y0 = y;
                    let mut parent = last_v;

                    if let Some((y, p)) = placements_v
                        .iter()
                        .rev()
                        .filter_map(|(p, i, _)| {
                            if p.x0 < w {
                                Some((p.y1, Some(*i)))
                            } else {
                                None
                            }
                        })
                        .max()
                    {
                        if y0.change_max(y) {
                            parent = p;
                        }
                    }

                    // pivotの場合は水平方向のrootとなる
                    if pivot == i {
                        // vec_vにカウントされるかどうかは未確定
                        pivot_y = y0;
                        last_h = Some(i);
                        x = x1;
                    }

                    let y1 = y0 + h;
                    y = y1;
                    last_v = Some(i);
                    parents_v[i] = parent;

                    if max_x.change_max(x1) {
                        max_x_i = i;
                    }

                    if max_y.change_max(y) {
                        max_y_i = i;
                    }

                    if pivot_y < y0 && y0 < MAX_BOX_SIZE + pivot_y {
                        let pl = Placement::new(x0, x1, y0, y1);
                        placements_h.push((pl, i, vec_i));
                    }
                }
                Dir::Left => {
                    let (w, h, vec_i) = if rotate {
                        (rect.height(), rect.width(), i)
                    } else {
                        (rect.width(), rect.height(), i + env.estimator.rect_cnt)
                    };

                    assert!(pivot_y != !0);

                    let y0 = pivot_y;
                    let y1 = y0 + h;

                    if x >= MAX_BOX_SIZE {
                        // 絶対に他の箱と干渉しない
                        x += w;
                        parents_h[i] = last_h;
                        last_h = Some(i);

                        if max_x.change_max(x) {
                            max_x_i = i;
                        }

                        if max_y.change_max(y1) {
                            max_y_i = i;
                        }

                        continue;
                    }

                    let mut x0 = x;
                    let mut parent = last_h;

                    if let Some((x, p)) = placements_h
                        .iter()
                        .rev()
                        .filter_map(|(p, i, _)| {
                            if p.y0 < h + pivot_y {
                                Some((p.x1, Some(*i)))
                            } else {
                                None
                            }
                        })
                        .max()
                    {
                        if x0.change_max(x) {
                            parent = p;
                        }
                    }

                    let x1 = x0 + w;
                    x = x1;
                    last_h = Some(i);
                    parents_h[i] = parent;
                    parents_v[i] = parents_v[pivot];

                    if max_x.change_max(x) {
                        max_x_i = i;
                    }

                    if max_y.change_max(y1) {
                        max_y_i = i;
                    }

                    if 0 < x0 && x0 < MAX_BOX_SIZE {
                        let pl = Placement::new(x0, x1, y0, y1);
                        placements_v.push((pl, i, vec_i));
                    }
                }
            }
        }

        let mut v = Some(max_y_i);

        while let Some(i) = v {
            let vec_i = if directions[i].unwrap().1 {
                i + env.estimator.rect_cnt
            } else {
                i
            };
            vec_v[vec_i] = 1.0;
            v = parents_v[i];
        }

        let mut v = Some(max_x_i);

        while let Some(i) = v {
            let vec_i = if directions[i].unwrap().1 {
                i
            } else {
                i + env.estimator.rect_cnt
            };
            vec_h[vec_i] = 1.0;
            v = parents_h[i];
        }

        (vec_v, vec_h)
    }

    fn to_op_and_edges(&self) -> (Vec<Op>, DVector<f64>, DVector<f64>) {
        let mut ops = vec![];
        let mut prev = None;
        let mut base = None;

        for (i, &placement) in self.placements.iter().enumerate() {
            let Some((dir, rotate)) = placement else {
                continue;
            };

            if i == self.pivot {
                base = prev;
            }

            prev = Some(i);

            let base = match dir {
                Dir::Up => None,
                Dir::Left => base,
            };

            let op = Op::new(i, rotate, dir, base);
            ops.push(op);
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
    type Score = ScoreF64;

    fn score(&self, _env: &Self::Env) -> Self::Score {
        ScoreF64(self.calculator_v.score() + self.calculator_h.score())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct ScoreF64(f64);

impl sa::Score for ScoreF64 {
    fn raw_score(&self) -> f64 {
        self.0
    }
}

#[derive(Debug, Clone)]
struct ScoreCalculator {
    v: DVector<f64>,
    sigma_c_sum: DVector<f64>,
    tr_numerator: f64,
    tr_denominator: f64,
}

impl ScoreCalculator {
    fn new(estimator: &GaussEstimator, v: DVector<f64>) -> Self {
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

    fn update(&mut self, estimator: &GaussEstimator, new_v: &mut DVector<f64>) {
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

    fn score(&self) -> f64 {
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
        loop {
            if rng.gen_bool(0.9) {
                return ChangeDirNeigh::gen(env, state, rng);
            } else {
                if let Some(neigh) = ChangePivotNeigh::gen(env, state, rng) {
                    return neigh;
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct ChangeDirNeigh {
    index: usize,
    placement: Option<(Dir, bool)>,
    old_placement: Option<(Dir, bool)>,
    vec_v: Option<DVector<f64>>,
    vec_h: Option<DVector<f64>>,
}

impl ChangeDirNeigh {
    fn gen(
        env: &Env,
        state: &State,
        rng: &mut impl Rng,
    ) -> Box<dyn sa::Neighbor<Env = Env, State = State>> {
        loop {
            let index = rng.gen_range(0..env.estimator.rect_cnt);
            let placement = if rng.gen_bool(0.1) && index != state.pivot {
                None
            } else {
                let dir = if rng.gen_bool(0.5) {
                    Dir::Up
                } else {
                    Dir::Left
                };

                if index <= state.pivot && dir == Dir::Left {
                    continue;
                }

                let rotate = rng.gen_bool(0.5);

                Some((dir, rotate))
            };

            if state.placements[index] != placement {
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
}

impl sa::Neighbor for ChangeDirNeigh {
    type Env = Env;
    type State = State;

    fn preprocess(&mut self, env: &Self::Env, state: &mut Self::State) {
        state.placements[self.index] = self.placement;

        let (mut vec_v, mut vec_h) = State::get_box_vec(&env, &state.placements, state.pivot);
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

#[derive(Debug, Clone)]
struct ChangePivotNeigh {
    old_pivot: usize,
    new_pivot: usize,
    vec_v: Option<DVector<f64>>,
    vec_h: Option<DVector<f64>>,
}

impl ChangePivotNeigh {
    fn gen(
        env: &Env,
        state: &State,
        rng: &mut impl Rng,
    ) -> Option<Box<dyn sa::Neighbor<Env = Env, State = State>>> {
        let new_pivot = if rng.gen_bool(0.5) {
            (0..state.pivot)
                .rev()
                .filter(|&i| state.placements[i].is_some())
                .next()
        } else {
            (state.pivot + 1..env.estimator.rect_cnt)
                .filter(|&i| state.placements[i].is_some())
                .next()
        };

        let Some(new_pivot) = new_pivot else {
            return None;
        };

        if state.placements[new_pivot].unwrap().0 == Dir::Left {
            return None;
        }

        let old_pivot = state.pivot;

        Some(Box::new(Self {
            old_pivot,
            new_pivot,
            vec_v: None,
            vec_h: None,
        }))
    }
}

impl sa::Neighbor for ChangePivotNeigh {
    type Env = Env;
    type State = State;

    fn preprocess(&mut self, env: &Self::Env, state: &mut Self::State) {
        state.pivot = self.new_pivot;

        let (mut vec_v, mut vec_h) = State::get_box_vec(&env, &state.placements, state.pivot);
        state.calculator_v.update(&env.estimator, &mut vec_v);
        state.calculator_h.update(&env.estimator, &mut vec_h);

        self.vec_v = Some(vec_v);
        self.vec_h = Some(vec_h);
    }

    fn postprocess(self: Box<Self>, _env: &Self::Env, _state: &mut Self::State) {
        // do nothing
    }

    fn rollback(self: Box<Self>, env: &Self::Env, state: &mut Self::State) {
        state.pivot = self.old_pivot;

        state
            .calculator_v
            .update(&env.estimator, &mut self.vec_v.unwrap());
        state
            .calculator_h
            .update(&env.estimator, &mut self.vec_h.unwrap());
    }
}
