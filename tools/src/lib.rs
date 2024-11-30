#![allow(non_snake_case, unused_macros)]

use proconio::input;
use rand::prelude::*;
use std::io::{prelude::*, BufReader};
use std::ops::RangeBounds;
use std::process::ChildStdout;
use svg::node::element::{Group, Rectangle, Style, Title};

pub trait SetMinMax {
    fn setmin(&mut self, v: Self) -> bool;
    fn setmax(&mut self, v: Self) -> bool;
}
impl<T> SetMinMax for T
where
    T: PartialOrd,
{
    fn setmin(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }
    fn setmax(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

#[macro_export]
macro_rules! mat {
	($($e:expr),*) => { Vec::from(vec![$($e),*]) };
	($($e:expr,)*) => { Vec::from(vec![$($e),*]) };
	($e:expr; $d:expr) => { Vec::from(vec![$e; $d]) };
	($e:expr; $d:expr $(; $ds:expr)+) => { Vec::from(vec![mat![$e $(; $ds)*]; $d]) };
}

#[derive(Clone, Debug)]
pub struct Input {
    pub N: usize,
    pub T: usize,
    pub sigma: i32,
    pub wh2: Vec<(i32, i32)>,
    pub wh: Vec<(i32, i32)>,
    pub es: Vec<(i32, i32)>,
}

impl std::fmt::Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} {} {}", self.N, self.T, self.sigma)?;
        for i in 0..self.N {
            writeln!(f, "{} {}", self.wh2[i].0, self.wh2[i].1)?;
        }
        for i in 0..self.N {
            writeln!(f, "{} {}", self.wh[i].0, self.wh[i].1)?;
        }
        for i in 0..self.T {
            writeln!(f, "{} {}", self.es[i].0, self.es[i].1)?;
        }
        Ok(())
    }
}

pub fn parse_input(f: &str) -> Input {
    let f = proconio::source::once::OnceSource::from(f);
    input! {
        from f,
        N: usize, T: usize, sigma: i32,
        wh2: [(i32, i32); N],
        wh: [(i32, i32); N],
        es: [(i32, i32); T],
    }
    Input {
        N,
        T,
        sigma,
        wh2,
        wh,
        es,
    }
}

pub fn read<T: Copy + PartialOrd + std::fmt::Display + std::str::FromStr, R: RangeBounds<T>>(
    token: Option<&str>,
    range: R,
) -> Result<T, String> {
    if let Some(v) = token {
        if let Ok(v) = v.parse::<T>() {
            if !range.contains(&v) {
                Err(format!("Out of range: {}", v))
            } else {
                Ok(v)
            }
        } else {
            Err(format!("Parse error: {}", v))
        }
    } else {
        Err("Unexpected EOF".to_owned())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Cmd {
    p: usize,
    r: bool,
    d: char,
    b: i32,
}

pub struct Output {
    pub out: Vec<Vec<Cmd>>,
    pub comments: Vec<String>,
}

fn next_line<'a>(lines: &'a mut std::str::Lines, comment: &mut String) -> Option<&'a str> {
    while let Some(line) = lines.next() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') {
            let line = line.trim_start_matches('#').trim();
            *comment += line;
            comment.push('\n');
        } else {
            return Some(line);
        }
    }
    None
}

pub fn parse_output(input: &Input, f: &str) -> Result<Output, String> {
    let mut out = vec![];
    let mut comments = vec![];
    let mut lines = f.lines();
    for _ in 0..input.T {
        let mut comment = String::new();
        let line = next_line(&mut lines, &mut comment);
        if line.is_none() {
            break;
        }
        let n = read(line, 0..=input.N)?;
        let mut cmd = vec![];
        for _ in 0..n {
            let line = next_line(&mut lines, &mut comment).ok_or("Unexpected EOF")?;
            let mut tokens = line.split_whitespace();
            let p = read(tokens.next(), 0..input.N)?;
            let r = read(tokens.next(), 0..=1)? == 1;
            let d = read(tokens.next(), 'A'..='Z')?;
            if d != 'U' && d != 'L' {
                return Err(format!("Unknown direction: {}", d));
            }
            let b = read(tokens.next(), -1..input.N as i32)?;
            cmd.push(Cmd { p, r, d, b });
        }
        out.push(cmd);
        comments.push(comment);
    }
    if next_line(&mut lines, &mut String::new()).is_some() {
        return Err("Too many output".to_owned());
    }
    Ok(Output { out, comments })
}

pub fn gen(seed: u64, fix_N: Option<usize>, fix_T: Option<usize>, fix_sigma: Option<i32>) -> Input {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed ^ 210);
    let mut N = rng.gen_range(30i32..=100) as usize;
    if let Some(v) = fix_N {
        N = v;
    }
    let mut T = (N as f64 * f64::powf(2.0, rng.gen_range(-1.0..2.0))).round() as usize;
    if let Some(v) = fix_T {
        T = v;
    }
    let ub = 100000;
    let mut sigma = rng.gen_range(ub / 100..=ub / 10);
    if let Some(v) = fix_sigma {
        sigma = v;
    }
    let lb = rng.gen_range(ub / 10..=ub / 2);
    let wh = (0..N)
        .map(|_| (rng.gen_range(lb..=ub), rng.gen_range(lb..=ub)))
        .collect::<Vec<_>>();
    let wh2 = wh
        .iter()
        .map(|&(w, h)| {
            let dw: f64 = rng.sample(rand_distr::StandardNormal);
            let dh: f64 = rng.sample(rand_distr::StandardNormal);
            let w2 = (w as f64 + sigma as f64 * dw).round().max(1.0).min(1e9) as i32;
            let h2 = (h as f64 + sigma as f64 * dh).round().max(1.0).min(1e9) as i32;
            (w2, h2)
        })
        .collect::<Vec<_>>();
    let es = (0..T)
        .map(|_| {
            let dw: f64 = rng.sample(rand_distr::StandardNormal);
            let dh: f64 = rng.sample(rand_distr::StandardNormal);
            (
                (sigma as f64 * dw).round() as i32,
                (sigma as f64 * dh).round() as i32,
            )
        })
        .collect();
    Input {
        N,
        T,
        sigma,
        wh2,
        wh,
        es,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Pos {
    pub x1: i32,
    pub x2: i32,
    pub y1: i32,
    pub y2: i32,
    pub r: bool,
    pub t: i32,
}

pub const P0: Pos = Pos {
    x1: -1,
    x2: -1,
    y1: -1,
    y2: -1,
    r: false,
    t: -1,
};

#[derive(Clone)]
pub struct State {
    pub turn: usize,
    /// (x1, x2, y1, y2, rot, t)
    pub pos: Vec<Pos>,
    pub W: i32,
    pub H: i32,
    pub W2: i32,
    pub H2: i32,
    pub score_t: i32,
    pub score: i32,
    pub comment: String,
}

impl State {
    pub fn new(input: &Input) -> Self {
        let score = input.wh.iter().map(|&(w, h)| w + h).sum::<i32>();
        Self {
            turn: 0,
            pos: vec![P0; input.N],
            W: 0,
            H: 0,
            W2: 0,
            H2: 0,
            score_t: score,
            score,
            comment: String::new(),
        }
    }
    pub fn query(&mut self, input: &Input, cmd: &[Cmd]) -> Result<(), String> {
        self.pos.fill(P0);
        self.W = 0;
        self.H = 0;
        let mut prev = -1;
        for (t, c) in cmd.iter().enumerate() {
            if !prev.setmax(c.p as i32) {
                return Err(format!("p must be in ascending order."));
            }
            if self.pos[c.p].t >= 0 {
                return Err(format!("Rectangle {} is already used", c.p));
            } else if c.b >= 0 && self.pos[c.b as usize].t < 0 {
                return Err(format!("Rectangle {} is not used", c.b));
            }
            let (mut w, mut h) = input.wh[c.p];
            if c.r {
                std::mem::swap(&mut w, &mut h);
            }
            if c.d == 'U' {
                let x1 = if c.b < 0 {
                    0
                } else {
                    self.pos[c.b as usize].x2
                };
                let x2 = x1 + w;
                let y1 = self
                    .pos
                    .iter()
                    .filter_map(|q| {
                        if q.t >= 0 && x1.max(q.x1) < x2.min(q.x2) {
                            Some(q.y2)
                        } else {
                            None
                        }
                    })
                    .max()
                    .unwrap_or(0);
                let y2 = y1 + h;
                self.pos[c.p] = Pos {
                    x1,
                    x2,
                    y1,
                    y2,
                    r: c.r,
                    t: t as i32,
                };
            } else {
                let y1 = if c.b < 0 {
                    0
                } else {
                    self.pos[c.b as usize].y2
                };
                let y2 = y1 + h;
                let x1 = self
                    .pos
                    .iter()
                    .filter_map(|q| {
                        if q.t >= 0 && y1.max(q.y1) < y2.min(q.y2) {
                            Some(q.x2)
                        } else {
                            None
                        }
                    })
                    .max()
                    .unwrap_or(0);
                let x2 = x1 + w;
                self.pos[c.p] = Pos {
                    x1,
                    x2,
                    y1,
                    y2,
                    r: c.r,
                    t: t as i32,
                };
            }
            self.W.setmax(self.pos[c.p].x2);
            self.H.setmax(self.pos[c.p].y2);
        }
        self.W2 = (self.W + input.es[self.turn].0).max(1).min(1000000000);
        self.H2 = (self.H + input.es[self.turn].1).max(1).min(1000000000);
        self.score_t = self.W + self.H;
        for i in 0..input.N {
            if self.pos[i].t < 0 {
                self.score_t += input.wh[i].0 + input.wh[i].1;
            }
        }
        self.score.setmin(self.score_t);
        self.turn += 1;
        Ok(())
    }
}

pub fn compute_score(input: &Input, out: &Output) -> (i64, String) {
    let (mut score, mut err, _) = compute_score_details(input, &out.out);
    if err.len() > 0 {
        score = 0;
    } else if out.out.len() < input.T {
        err = "Unexpected EOF".to_owned();
        score = 0;
    }
    (score, err)
}

pub fn compute_score_details(input: &Input, out: &[Vec<Cmd>]) -> (i64, String, State) {
    let mut state = State::new(input);
    for cmd in out {
        if let Err(err) = state.query(input, cmd) {
            return (state.score as i64, err, state);
        }
    }
    (state.score as i64, String::new(), state)
}

/// 0 <= val <= 1
pub fn color(mut val: f64) -> String {
    val.setmin(1.0);
    val.setmax(0.0);
    let (r, g, b) = if val < 0.5 {
        let x = val * 2.0;
        (
            30. * (1.0 - x) + 144. * x,
            144. * (1.0 - x) + 255. * x,
            255. * (1.0 - x) + 30. * x,
        )
    } else {
        let x = val * 2.0 - 1.0;
        (
            144. * (1.0 - x) + 255. * x,
            255. * (1.0 - x) + 30. * x,
            30. * (1.0 - x) + 70. * x,
        )
    };
    format!(
        "#{:02x}{:02x}{:02x}",
        r.round() as i32,
        g.round() as i32,
        b.round() as i32
    )
}

pub fn rect(x: i32, y: i32, w: i32, h: i32, fill: &str) -> Rectangle {
    Rectangle::new()
        .set("x", x)
        .set("y", y)
        .set("width", w)
        .set("height", h)
        .set("fill", fill)
}

pub fn group(title: String) -> Group {
    Group::new().add(Title::new(title))
}

pub struct VisResult {
    pub score: i64,
    pub err: String,
    pub svg: String,
    pub state: State,
}

pub fn vis_default(input: &Input, out: &Output) -> VisResult {
    let mut ret = vis(input, &out.out);
    if ret.err.len() > 0 {
        ret.score = 0;
    } else if out.out.len() < input.T {
        ret.err = "Unexpected EOF".to_owned();
        ret.score = 0;
    }
    ret
}

pub fn vis(input: &Input, out: &[Vec<Cmd>]) -> VisResult {
    let W = 600;
    let (score, err, state) = compute_score_details(input, &out);
    let max = state.W.max(state.H);
    let mut doc = svg::Document::new()
        .set("id", "vis")
        .set("viewBox", (-1000, -1000, max + 2000, max + 2000))
        .set("width", W)
        .set("height", W)
        .set("style", "background-color:white");
    doc = doc.add(Style::new(format!(
        "text {{text-anchor: middle;dominant-baseline: central;}}"
    )));
    doc = doc.add(
        rect(0, 0, state.W, state.H, "white")
            .set("stroke", "black")
            .set("stroke-width", 1000),
    );
    for i in 0..input.N {
        if state.pos[i].t < 0 {
            continue;
        }
        let (mut w, mut h) = input.wh[i];
        let (mut w2, mut h2) = input.wh2[i];
        let p = state.pos[i];
        if p.r {
            std::mem::swap(&mut w, &mut h);
            std::mem::swap(&mut w2, &mut h2);
        }
        doc = doc.add(
            group(format!(
                "{}\n[{}, {}] × [{}, {}]\n(w, h) = ({}, {})\n(w', h') = ({}, {})",
                i, p.x1, p.x2, p.y1, p.y2, w, h, w2, h2
            ))
            .add(
                rect(p.x1, p.y1, p.x2 - p.x1, p.y2 - p.y1, "gray")
                    .set("stroke-width", 1000)
                    .set("stroke", "black"),
            ),
        );
    }
    VisResult {
        score,
        err,
        svg: doc.to_string(),
        state,
    }
}

pub fn evaluate(input: &Input, out: &Output) -> Vec<VisResult> {
    let mut state = State::new(input);
    let mut ret = vec![VisResult {
        score: state.score as i64,
        err: String::new(),
        svg: String::new(),
        state: state.clone(),
    }];
    for (cmd, comment) in out.out.iter().zip(out.comments.iter()) {
        state.comment = comment.clone();
        if let Err(err) = state.query(input, cmd) {
            ret.push(VisResult {
                score: state.score as i64,
                err,
                svg: String::new(),
                state: state.clone(),
            });
            break;
        } else {
            ret.push(VisResult {
                score: state.score as i64,
                err: String::new(),
                svg: String::new(),
                state: state.clone(),
            });
        }
    }
    if out.out.len() < input.T {
        ret.last_mut().unwrap().err = "Unexpected EOF".to_owned();
    }
    ret
}

fn read_line(stdout: &mut BufReader<ChildStdout>, local: bool) -> Result<String, String> {
    loop {
        let mut out = String::new();
        match stdout.read_line(&mut out) {
            Ok(0) | Err(_) => {
                return Err(format!("Your program has terminated unexpectedly"));
            }
            _ => (),
        }
        if local {
            print!("{}", out);
        }
        let v = out.trim();
        if v.len() == 0 || v.starts_with("#") {
            continue;
        }
        return Ok(v.to_owned());
    }
}

pub fn exec(p: &mut std::process::Child, local: bool) -> Result<i64, String> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    let input = parse_input(&input);
    let mut stdin = std::io::BufWriter::new(p.stdin.take().unwrap());
    let mut stdout = std::io::BufReader::new(p.stdout.take().unwrap());
    let _ = writeln!(stdin, "{} {} {}", input.N, input.T, input.sigma);
    for i in 0..input.N {
        let _ = writeln!(stdin, "{} {}", input.wh2[i].0, input.wh2[i].1);
    }
    let _ = stdin.flush();
    let mut state = State::new(&input);
    for _ in 0..input.T {
        let n = read(Some(&read_line(&mut stdout, local)?), 0..=input.N)?;
        let mut cmd = vec![];
        for _ in 0..n {
            let line = read_line(&mut stdout, local)?;
            let mut tokens = line.split_whitespace();
            let p = read(tokens.next(), 0..input.N)?;
            let r = read(tokens.next(), 0..=1)? == 1;
            let d = read(tokens.next(), 'A'..='Z')?;
            if d != 'U' && d != 'L' {
                return Err(format!("Unknown direction: {}", d));
            }
            let b = read(tokens.next(), -1..input.N as i32)?;
            cmd.push(Cmd { p, r, d, b });
        }
        state.query(&input, &cmd)?;
        let _ = writeln!(stdin, "{} {}", state.W2, state.H2);
        let _ = stdin.flush();
    }
    if read_line(&mut stdout, local).is_ok() {
        return Err("Too many output".to_owned());
    }
    Ok(state.score as i64)
}
