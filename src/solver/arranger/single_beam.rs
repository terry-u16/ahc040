use super::Arranger;
use crate::{
    beam::{self, BayesianBeamWidthSuggester},
    problem::{Dir, Input, Op, Rect},
    solver::estimator::Sampler,
};
use rand::Rng;

pub(super) struct SingleBeamArranger<'a, R: Rng> {
    sampler: &'a Sampler<'a>,
    rng: &'a mut R,
    duration_sec: f64,
}

impl<'a, R: Rng> SingleBeamArranger<'a, R> {
    pub(super) fn new(sampler: &'a Sampler<'a>, rng: &'a mut R, duration_sec: f64) -> Self {
        Self {
            sampler,
            rng,
            duration_sec,
        }
    }
}

impl<R: Rng> Arranger for SingleBeamArranger<'_, R> {
    fn arrange(&mut self, input: &Input) -> Vec<Op> {
        let since = std::time::Instant::now();
        let rects = self.sampler.sample(self.rng);
        let large_state = LargeState::new(input.clone(), rects, self.rng);
        let small_state = SmallState::default();
        let act_gen = ActGen;

        let remaining_time = self.duration_sec - since.elapsed().as_secs_f64();
        let mut beam = beam::BeamSearch::new(large_state, small_state, act_gen);
        let standard_beam_width = 100_000_000 / (input.rect_cnt() as usize).pow(3);
        let beam_width_suggester = BayesianBeamWidthSuggester::new(
            input.rect_cnt(),
            5,
            remaining_time,
            standard_beam_width,
            1,
            10000,
            1,
        );
        let deduplicator = beam::HashSingleDeduplicator::new();
        let (ops, score) = beam.run(input.rect_cnt(), beam_width_suggester, deduplicator);

        eprintln!("score: {}", score);
        ops
    }
}

#[derive(Debug, Clone)]
struct LargeState {
    rects: Vec<Rect>,
    width: u32,
    height: u32,
    placements: Vec<Placement>,
    hash: u64,
    hash_base_x: Vec<u64>,
    hash_base_y: Vec<u64>,
    hash_base_rot: Vec<u64>,
    turn: usize,
}

impl LargeState {
    fn new(input: Input, rects: Vec<Rect>, rng: &mut impl rand::Rng) -> Self {
        let hash_base_x = (0..input.rect_cnt()).map(|_| rng.gen()).collect();
        let hash_base_y = (0..input.rect_cnt()).map(|_| rng.gen()).collect();
        let hash_base_rot = (0..input.rect_cnt()).map(|_| rng.gen()).collect();

        Self {
            rects,
            width: 0,
            height: 0,
            placements: vec![],
            hash: 0,
            hash_base_x,
            hash_base_y,
            hash_base_rot,
            turn: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SmallState {
    placement: Placement,
    old_width: u32,
    old_height: u32,
    new_width: u32,
    new_height: u32,
    hash: u64,
    hash_xor: u64,
    op: Op,
}

impl beam::SmallState for SmallState {
    type Score = i32;
    type Hash = u64;
    type LargeState = LargeState;
    type Action = Op;

    fn raw_score(&self) -> Self::Score {
        (self.new_height + self.new_width) as i32
    }

    fn beam_score(&self) -> Self::Score {
        -self.raw_score()
    }

    fn hash(&self) -> Self::Hash {
        self.hash
    }

    fn apply(&self, state: &mut Self::LargeState) {
        state.placements.push(self.placement);
        state.width = self.new_width;
        state.height = self.new_height;
        state.hash ^= self.hash_xor;
        state.turn += 1;
    }

    fn rollback(&self, state: &mut Self::LargeState) {
        state.placements.pop();
        state.width = self.old_width;
        state.height = self.old_height;
        state.hash ^= self.hash_xor;
        state.turn -= 1;
    }

    fn action(&self) -> Self::Action {
        self.op
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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

struct ActGen;

impl ActGen {
    fn gen_left_cand(
        &self,
        large_state: &LargeState,
        base: Option<usize>,
        rotate: bool,
    ) -> Option<SmallState> {
        let rect = &large_state.rects[large_state.turn];
        let (width, height) = if rotate {
            (rect.height(), rect.width())
        } else {
            (rect.width(), rect.height())
        };

        let turn = large_state.turn;
        let y0 = match base {
            Some(index) => large_state.placements[index].y1,
            None => 0,
        };
        let y1 = y0 + height;

        let x0 = large_state
            .placements
            .iter()
            .filter_map(|p| {
                if y0.max(p.y0) < y1.min(p.y1) {
                    Some(p.x1)
                } else {
                    None
                }
            })
            .max()
            .unwrap_or(0);
        let x1 = x0 + width;

        // 側面がピッタリくっついているかチェック
        let is_touching = match base {
            Some(base) => {
                let p = large_state.placements[base];
                x0.max(p.x0) < x1.min(p.x1)
            }
            None => true,
        };

        if !is_touching {
            return None;
        }

        let placement = Placement::new(x0, x1, y0, y1);
        let hash_x = large_state.hash_base_x[turn].wrapping_mul(x0 as u64);
        let hash_y = large_state.hash_base_y[turn].wrapping_mul(y0 as u64);
        let hash_rot = large_state.hash_base_rot[turn].wrapping_mul(rotate as u64);
        let hash_xor = hash_x ^ hash_y ^ hash_rot;

        let new_width = x1.max(large_state.width);
        let new_height = y1.max(large_state.height);

        let hash = large_state.hash ^ hash_xor;

        let op = Op::new(turn, rotate, Dir::Left, base);

        Some(SmallState {
            placement,
            old_width: large_state.width,
            old_height: large_state.height,
            new_width,
            new_height,
            hash,
            hash_xor,
            op,
        })
    }

    fn gen_up_cand(
        &self,
        large_state: &LargeState,
        base: Option<usize>,
        rotate: bool,
    ) -> Option<SmallState> {
        let rect = &large_state.rects[large_state.turn];
        let (width, height) = if rotate {
            (rect.height(), rect.width())
        } else {
            (rect.width(), rect.height())
        };

        let turn = large_state.turn;
        let x0 = match base {
            Some(index) => large_state.placements[index].x1,
            None => 0,
        };
        let x1 = x0 + width;

        let y0 = large_state
            .placements
            .iter()
            .filter_map(|p| {
                if x0.max(p.x0) < x1.min(p.x1) {
                    Some(p.y1)
                } else {
                    None
                }
            })
            .max()
            .unwrap_or(0);
        let y1 = y0 + height;

        // 側面がピッタリくっついているかチェック
        let is_touching = match base {
            Some(base) => {
                let p = large_state.placements[base];
                y0.max(p.y0) < y1.min(p.y1)
            }
            None => true,
        };

        if !is_touching {
            return None;
        }

        let placement = Placement::new(x0, x1, y0, y1);
        let hash_x = large_state.hash_base_x[turn].wrapping_mul(x0 as u64);
        let hash_y = large_state.hash_base_y[turn].wrapping_mul(y0 as u64);
        let hash_rot = large_state.hash_base_rot[turn].wrapping_mul(rotate as u64);
        let hash_xor = hash_x ^ hash_y ^ hash_rot;

        let new_width = x1.max(large_state.width);
        let new_height = y1.max(large_state.height);

        let hash = large_state.hash ^ hash_xor;

        let op = Op::new(turn, rotate, Dir::Up, base);

        Some(SmallState {
            placement,
            old_width: large_state.width,
            old_height: large_state.height,
            new_width,
            new_height,
            hash,
            hash_xor,
            op,
        })
    }
}

impl beam::ActGen<SmallState> for ActGen {
    fn generate(
        &self,
        _small_state: &SmallState,
        large_state: &<SmallState as beam::SmallState>::LargeState,
        next_states: &mut Vec<SmallState>,
    ) {
        // Left
        next_states.extend(self.gen_left_cand(&large_state, None, false));
        next_states.extend(self.gen_left_cand(&large_state, None, true));

        for i in 0..large_state.turn {
            next_states.extend(self.gen_left_cand(&large_state, Some(i), false));
            next_states.extend(self.gen_left_cand(&large_state, Some(i), true));
        }

        // Up
        next_states.extend(self.gen_up_cand(&large_state, None, false));
        next_states.extend(self.gen_up_cand(&large_state, None, true));

        for i in 0..large_state.turn {
            next_states.extend(self.gen_up_cand(&large_state, Some(i), false));
            next_states.extend(self.gen_up_cand(&large_state, Some(i), true));
        }
    }
}
