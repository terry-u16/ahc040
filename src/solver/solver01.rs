use itertools::Itertools;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

use crate::{
    problem::{Dir, Input, Judge, Op},
    util::ChangeMinMax,
};

use super::{
    estimator::{Estimator, Observation, RectDir, RectEdge},
    Solver,
};

pub struct Solver01;

impl Solver for Solver01 {
    fn solve(&self, input: &Input, mut judge: impl Judge) {
        let mut rng = Pcg64Mcg::new(42);
        let mut estimator = Estimator::new(input);

        for (i, &rect) in input.rect_measures().iter().enumerate() {
            estimator.update(&Observation::new(
                rect.height(),
                vec![RectEdge::new(i, RectDir::Vertical, 1.0)],
            ));
            estimator.update(&Observation::new(
                rect.width(),
                vec![RectEdge::new(i, RectDir::Horizontal, 1.0)],
            ));
        }

        eprintln!("[Init]");
        estimator.dump_estimated(judge.rects());

        for _ in 0..input.query_cnt() {
            let (ops, edges_h, edges_v) = get_placements_randomly(input, &estimator, &mut rng);
            let measure = judge.query(&ops);
            let observation_x = Observation::new(measure.width(), edges_h);
            let observation_y = Observation::new(measure.height(), edges_v);

            estimator.update(&observation_x);
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

fn get_placements_randomly(
    input: &Input,
    estimator: &Estimator,
    rng: &mut Pcg64Mcg,
) -> (Vec<Op>, Vec<RectEdge>, Vec<RectEdge>) {
    let dirs = loop {
        let dirs = (0..input.rect_cnt())
            .map(|_| {
                if rng.gen_bool(0.5) {
                    Dir::Up
                } else {
                    Dir::Left
                }
            })
            .collect_vec();

        // 5個あれば必ず右端・下端になる
        if dirs.iter().filter(|&&b| b == Dir::Up).count() >= 5
            && dirs.iter().filter(|&&b| b == Dir::Left).count() >= 5
        {
            break dirs;
        }
    };

    let rotates = (0..input.rect_cnt())
        .map(|_| rng.gen_bool(0.5))
        .collect_vec();

    let mut ops = vec![];

    for i in 0..input.rect_cnt() {
        ops.push(Op::new(i, rotates[i], dirs[i], None));
    }

    let sampler = estimator.get_sampler();

    let mut weights_v = vec![0; input.rect_cnt()];
    let mut weights_h = vec![0; input.rect_cnt()];
    const TRIAL_CNT: usize = 10;

    for _ in 0..TRIAL_CNT {
        let rects = sampler.sample(rng);

        let (mut x, mut y) = if rotates[0] {
            (rects[0].height(), rects[0].width())
        } else {
            (rects[0].width(), rects[0].height())
        };

        // 原点に近く衝突可能性のある矩形
        let mut near_origins: Vec<(usize, Placement)> = vec![];

        let mut parents_v = vec![None; input.rect_cnt()];
        let mut parents_h = vec![None; input.rect_cnt()];
        let mut last_v = 0;
        let mut last_h = 0;

        const MAX_RECT_SIZE: u32 = 100000;

        for i in 1..input.rect_cnt() {
            let (width, height) = if rotates[i] {
                (rects[i].height(), rects[i].width())
            } else {
                (rects[i].width(), rects[i].height())
            };

            match dirs[i] {
                Dir::Left => {
                    let mut x0 = x;
                    let mut parent = last_h;
                    let y0 = 0;
                    let y1 = height;

                    if x0 < MAX_RECT_SIZE {
                        for &(i, pl) in near_origins.iter() {
                            if y0.max(pl.y0) < y1.min(pl.y1) && x0.change_max(pl.x1) {
                                parent = i;
                            }
                        }
                    }

                    parents_h[i] = Some(parent);
                    last_h = i;
                    let pl = Placement::new(x0, x0 + width, y0, y1);
                    x = pl.x1;

                    if pl.x0 < MAX_RECT_SIZE || pl.y0 < MAX_RECT_SIZE {
                        near_origins.push((i, pl));
                    }
                }
                Dir::Up => {
                    let mut y0 = y;
                    let mut parent = last_v;
                    let x0 = 0;
                    let x1 = width;

                    if y0 < MAX_RECT_SIZE {
                        for &(i, pl) in near_origins.iter() {
                            if x0.max(pl.x0) < x1.min(pl.x1) && y0.change_max(pl.y1) {
                                parent = i;
                            }
                        }
                    }

                    parents_v[i] = Some(parent);
                    last_v = i;
                    let pl = Placement::new(x0, x1, y0, y0 + height);
                    y = pl.y1;

                    if pl.x0 < MAX_RECT_SIZE || pl.y0 < MAX_RECT_SIZE {
                        near_origins.push((i, pl));
                    }
                }
            }
        }

        // 寄与度を計算
        weights_h[last_h] += 1;

        while let Some(p) = parents_h[last_h] {
            weights_h[p] += 1;
            last_h = p;
        }

        weights_v[last_v] += 1;

        while let Some(p) = parents_v[last_v] {
            weights_v[p] += 1;
            last_v = p;
        }
    }

    let mut edges_h = vec![];
    let mut edges_v = vec![];

    for i in 0..input.rect_cnt() {
        if weights_h[i] > 0 {
            let weight = weights_h[i] as f64 / TRIAL_CNT as f64;
            let dir = if rotates[i] {
                RectDir::Vertical
            } else {
                RectDir::Horizontal
            };

            edges_h.push(RectEdge::new(i, dir, weight));
        }

        if weights_v[i] > 0 {
            let weight = weights_v[i] as f64 / TRIAL_CNT as f64;
            let dir = if rotates[i] {
                RectDir::Horizontal
            } else {
                RectDir::Vertical
            };

            edges_v.push(RectEdge::new(i, dir, weight));
        }
    }

    (ops, edges_h, edges_v)
}

#[derive(Debug, Clone, Copy)]
struct Placement {
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
}

impl Placement {
    fn new(x0: u32, x1: u32, y0: u32, y1: u32) -> Self {
        assert!(x0 < x1);
        assert!(y0 < y1);
        Self { x0, x1, y0, y1 }
    }
}
