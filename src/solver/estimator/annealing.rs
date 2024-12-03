use std::u32;

use super::{Estimator, RectDir, RectEdge};
use crate::{
    problem::{Dir, Input, Op, Rect},
    sa::{self, State as _},
    solver::estimator::{self, Placement},
    util::ChangeMinMax,
};
use itertools::Itertools;
use nalgebra::DVector;
use rand::prelude::*;
use smallvec::SmallVec;

pub(super) fn solve(
    estimator: &Estimator,
    rng: &mut impl Rng,
) -> (Vec<Op>, DVector<f64>, DVector<f64>) {
    let env = Env::new(estimator.clone());
    let state = State::init(&env, rng);
    let anneler = sa::Annealer::new(1e5, 1e2, 42, 1);
    let (state, stats) = anneler.run(&env, state, &NeighGen, 0.01);

    eprintln!("{}", stats);

    state.to_op_and_edges()
}

struct Env {
    estimator: Estimator,
    rects: Vec<Rect>,
}

impl Env {
    fn new(estimator: Estimator) -> Self {
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

                    let x0 = x;
                    let x1 = x + w;

                    if y >= MAX_BOX_SIZE + pivot_y {
                        // 絶対に他の箱と干渉しない

                        // pivotの場合は水平方向のrootとなる
                        if pivot == i {
                            // vec_vにカウントされるかどうかは未確定
                            pivot_y = y;
                            last_h = Some(i);
                            x = x1;
                        }

                        y += h;
                        vec_v[vec_i] = 1.0;
                        parents_v[i] = last_v;
                        last_v = Some(i);
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

                    let y0 = y + pivot_y;
                    let y1 = y + h;

                    if x >= MAX_BOX_SIZE {
                        // 絶対に他の箱と干渉しない
                        x += w;
                        vec_h[vec_i] = 1.0;
                        parents_h[i] = last_h;
                        last_h = Some(i);
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

                    if 0 < x0 && x0 < MAX_BOX_SIZE {
                        let pl = Placement::new(x0, x1, y0, y1);
                        placements_v.push((pl, i, vec_i));
                    }
                }
            }
        }

        while let Some(i) = last_v {
            let vec_i = if directions[i].unwrap().1 {
                i + env.estimator.rect_cnt
            } else {
                i
            };
            vec_v[vec_i] = 1.0;
            last_v = parents_v[i];
        }

        while let Some(i) = last_h {
            let vec_i = if directions[i].unwrap().1 {
                i
            } else {
                i + env.estimator.rect_cnt
            };
            vec_h[vec_i] = 1.0;
            last_h = parents_h[i];
        }

        (vec_v, vec_h)
    }

    fn to_op_and_edges(&self) -> (Vec<Op>, DVector<f64>, DVector<f64>) {
        let mut ops = vec![];

        for (i, &placement) in self.placements.iter().enumerate() {
            let Some((dir, rotate)) = placement else {
                continue;
            };

            let base = match dir {
                Dir::Up => None,
                Dir::Left => {
                    if self.pivot == 0 {
                        None
                    } else {
                        Some(self.pivot - 1)
                    }
                }
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
    //sigma_c_sum: DVector<f64>,
    tr_numerator: f64,
    tr_denominator: f64,
}

impl ScoreCalculator {
    fn new(estimator: &Estimator, v: DVector<f64>) -> Self {
        let tr_numerator = (&estimator.variance * &v).norm_squared();
        let tr_denominator =
            (v.transpose() * &estimator.variance * &v)[0] + estimator.measure_variance;
        //let sigma_c_sum = &estimator.variance * &v;

        Self {
            v,
            //sigma_c_sum,
            tr_numerator,
            tr_denominator,
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
        ChangeDirNeigh::gen(env, state, rng)
    }
}

#[derive(Debug, Clone)]
struct ChangeDirNeigh {
    index: usize,
    placement: Option<(Dir, bool)>,
    placements: Vec<Option<(Dir, bool)>>,
    calculator_v: ScoreCalculator,
    calculator_h: ScoreCalculator,
}

impl ChangeDirNeigh {
    fn gen(
        env: &Env,
        state: &State,
        rng: &mut impl Rng,
    ) -> Box<dyn sa::Neighbor<Env = Env, State = State>> {
        loop {
            let index = rng.gen_range(0..env.estimator.rect_cnt);
            let placement = if rng.gen_bool(0.1) && index != 0 {
                None
            } else {
                let dir = if rng.gen_bool(0.5) {
                    Dir::Up
                } else {
                    Dir::Left
                };

                if index == 0 && dir == Dir::Left {
                    continue;
                }

                let rotate = rng.gen_bool(0.5);

                Some((dir, rotate))
            };

            if state.placements[index] != placement {
                let mut placements = state.placements.clone();
                placements[index] = placement;
                let (vec_v, vec_h) = State::get_box_vec(&env, &placements, state.pivot);
                let calculator_v = ScoreCalculator::new(&env.estimator, vec_v);
                let calculator_h = ScoreCalculator::new(&env.estimator, vec_h);

                return Box::new(Self {
                    index,
                    placement,
                    placements,
                    calculator_v,
                    calculator_h,
                });
            }
        }
    }
}

impl sa::Neighbor for ChangeDirNeigh {
    type Env = Env;
    type State = State;

    fn preprocess(&mut self, _env: &Self::Env, state: &mut Self::State) {
        std::mem::swap(&mut self.placements, &mut state.placements);
        std::mem::swap(&mut self.calculator_v, &mut state.calculator_v);
        std::mem::swap(&mut self.calculator_h, &mut state.calculator_h);
    }

    fn postprocess(self: Box<Self>, _env: &Self::Env, _state: &mut Self::State) {
        // do nothing
    }

    fn rollback(mut self: Box<Self>, _env: &Self::Env, state: &mut Self::State) {
        std::mem::swap(&mut self.placements, &mut state.placements);
        std::mem::swap(&mut self.calculator_v, &mut state.calculator_v);
        std::mem::swap(&mut self.calculator_h, &mut state.calculator_h);
    }
}
