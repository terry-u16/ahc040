mod atcoder_judge;
mod self_judge;
pub mod params;

use proconio::input_interactive;
use std::{fmt::Display, time::Instant};

#[derive(Debug, Clone)]
pub struct Input {
    rect_cnt: usize,
    query_cnt: usize,
    std_dev: f64,
    rect_measures: Vec<Measure>,
    since: Instant,
}

impl Input {
    pub const MAX_RECT_CNT: usize = 100;
    pub const MAX_RECT_SIZE: u32 = 100000;
    pub const MIN_RECT_SIZE: u32 = 10000;

    pub fn read() -> Self {
        input_interactive! {
            rect_cnt: usize,
            query_cnt: usize,
            std_dev: f64,
        }

        let since = Instant::now();
        let mut rectangles = Vec::with_capacity(rect_cnt);

        for _ in 0..rect_cnt {
            input_interactive! {
                width: u32,
                height: u32,
            }

            rectangles.push(Measure::new(height, width));
        }

        Self {
            rect_cnt,
            query_cnt,
            std_dev,
            rect_measures: rectangles,
            since,
        }
    }

    pub fn rect_cnt(&self) -> usize {
        self.rect_cnt
    }

    pub fn query_cnt(&self) -> usize {
        self.query_cnt
    }

    pub fn std_dev(&self) -> f64 {
        self.std_dev
    }

    pub fn rect_measures(&self) -> &[Measure] {
        &self.rect_measures
    }

    pub fn since(&self) -> Instant {
        self.since
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rect {
    height: u32,
    width: u32,
}

impl Rect {
    pub fn new(height: u32, width: u32) -> Self {
        Self { height, width }
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn width(&self) -> u32 {
        self.width
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Measure {
    height: u32,
    width: u32,
}

impl Measure {
    fn new(height: u32, width: u32) -> Self {
        Self { height, width }
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn width(&self) -> u32 {
        self.width
    }
}

impl Display for Measure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} x {}", self.height, self.width)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Dir {
    #[default]
    Up = 0x00,
    Left = 0x01,
}

impl Display for Dir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dir::Up => write!(f, "U"),
            Dir::Left => write!(f, "L"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Op {
    rect_idx: usize,
    rotate: bool,
    dir: Dir,
    base: Option<usize>,
}

impl Op {
    pub fn new(rect_idx: usize, rotate: bool, dir: Dir, base: Option<usize>) -> Self {
        Self {
            rect_idx,
            rotate,
            dir,
            base,
        }
    }

    pub fn rect_idx(&self) -> usize {
        self.rect_idx
    }

    pub fn rotate(&self) -> bool {
        self.rotate
    }

    pub fn dir(&self) -> Dir {
        self.dir
    }

    pub fn base(&self) -> Option<usize> {
        self.base
    }
}

impl Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} {}",
            self.rect_idx,
            if self.rotate { 1 } else { 0 },
            self.dir
        )?;

        match self.base {
            Some(base) => write!(f, " {}", base),
            None => write!(f, " -1"),
        }
    }
}

pub trait Judge {
    fn query(&mut self, query: &[Op]) -> Measure;
    fn query_cnt(&self) -> usize;
    fn rects(&self) -> Option<&[Rect]>;
}

#[cfg(feature = "local")]
pub fn gen_judge(input: &Input) -> impl Judge {
    self_judge::SelfJudge::read(input)
}

#[cfg(not(feature = "local"))]
pub fn gen_judge(_input: &Input) -> impl Judge {
    atcoder_judge::AtCoderJudge::new()
}
