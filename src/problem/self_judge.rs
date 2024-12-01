use super::{Input, Judge, Rect};
use crate::{
    problem::{Dir, Measure},
    util::ChangeMinMax,
};
use proconio::input_interactive;

pub struct SelfJudge {
    input: Input,
    turn: usize,
    score: u32,
    rects: Vec<Rect>,
    errors: Vec<(i32, i32)>,
}

#[allow(dead_code)]
impl SelfJudge {
    pub fn read(input: &Input) -> Self {
        input_interactive! {
            rects: [(u32, u32); input.rect_cnt()],
            errors: [(i32, i32); input.query_cnt()],
        }

        let rects = rects.into_iter().map(|(w, h)| Rect::new(h, w)).collect();

        Self::new(input.clone(), rects, errors)
    }

    fn new(input: Input, rects: Vec<Rect>, errors: Vec<(i32, i32)>) -> Self {
        let score = rects.iter().map(|r| r.width + r.height).sum();

        Self {
            input,
            turn: 0,
            score,
            errors,
            rects,
        }
    }

    pub fn score(&self) -> u32 {
        self.score
    }
}

impl Judge for SelfJudge {
    fn query(&mut self, query: &[super::Op]) -> super::Measure {
        assert!(self.turn < self.input.query_cnt());
        println!("{}", query.len());

        for op in query {
            println!("{}", op);
        }

        let mut width = 0;
        let mut height = 0;
        let mut prev = -1;
        let mut pos = vec![P0; self.input.rect_cnt()];

        for (t, &op) in query.iter().enumerate() {
            assert!(
                prev.change_max(op.rect_idx as i32),
                "rect_idx must be in ascending order.",
            );

            if let Some(base) = op.base {
                assert!(pos[base].t >= 0, "Rectangle {} is not placed.", base);
            }

            let Rect {
                width: w,
                height: h,
            } = self.rects[op.rect_idx as usize];
            let mut w = w as i32;
            let mut h = h as i32;

            if op.rotate {
                std::mem::swap(&mut w, &mut h);
            }

            match op.dir {
                Dir::Up => {
                    let x1 = match op.base {
                        Some(base) => pos[base].x2,
                        None => 0,
                    };
                    let x2 = x1 + w;
                    let y1 = pos
                        .iter()
                        .filter_map(|p| {
                            if p.t >= 0 && x1.max(p.x1) < x2.min(p.x2) {
                                Some(p.y2)
                            } else {
                                None
                            }
                        })
                        .max()
                        .unwrap_or(0);
                    let y2 = y1 + h;
                    pos[op.rect_idx] = Pos::new(x1, x2, y1, y2, op.rotate, t as i32);
                }
                Dir::Left => {
                    let y1 = match op.base {
                        Some(base) => pos[base].y2,
                        None => 0,
                    };
                    let y2 = y1 + h;
                    let x1 = pos
                        .iter()
                        .filter_map(|p| {
                            if p.t >= 0 && y1.max(p.y1) < y2.min(p.y2) {
                                Some(p.x2)
                            } else {
                                None
                            }
                        })
                        .max()
                        .unwrap_or(0);
                    let x2 = x1 + w;
                    pos[op.rect_idx] = Pos::new(x1, x2, y1, y2, op.rotate, t as i32);
                }
            }

            width.change_max(pos[op.rect_idx].x2);
            height.change_max(pos[op.rect_idx].y2);
        }

        let measured_w = (width + self.errors[self.turn].0).clamp(1, 1_000_000_000);
        let measured_h = (height + self.errors[self.turn].1).clamp(1, 1_000_000_000);
        let mut score = (width + height) as u32;

        for i in 0..self.input.rect_cnt() {
            if pos[i].t < 0 {
                score += self.rects[i].width + self.rects[i].height;
            }
        }

        self.score.change_min(score);
        self.turn += 1;

        if self.turn == self.input.query_cnt() {
            eprintln!("Score = {}", self.score);
        }

        Measure::new(measured_h as u32, measured_w as u32)
    }

    fn query_cnt(&self) -> usize {
        self.turn
    }

    fn rects(&self) -> Option<&[Rect]> {
        Some(&self.rects)
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct Pos {
    x1: i32,
    x2: i32,
    y1: i32,
    y2: i32,
    r: bool,
    t: i32,
}

impl Pos {
    const fn new(x1: i32, x2: i32, y1: i32, y2: i32, r: bool, t: i32) -> Self {
        Self {
            x1,
            x2,
            y1,
            y2,
            r,
            t,
        }
    }
}

const P0: Pos = Pos::new(-1, -1, -1, -1, false, -1);
