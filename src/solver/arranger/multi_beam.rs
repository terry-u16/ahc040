use super::Arranger;
use crate::{
    beam::{self, BayesianBeamWidthSuggester},
    problem::{Dir, Input, Op, Rect},
    solver::estimator::Estimator,
};
use rand::Rng;

pub(super) struct MultiBeamArranger<'a, R: Rng> {
    estimator: &'a Estimator,
    rng: &'a mut R,
    duration_sec: f64,
}

impl<'a, R: Rng> MultiBeamArranger<'a, R> {
    pub(super) fn new(estimator: &'a Estimator, rng: &'a mut R, duration_sec: f64) -> Self {
        Self {
            estimator,
            rng,
            duration_sec,
        }
    }
}

impl<R: Rng> Arranger for MultiBeamArranger<'_, R> {
    fn arrange(&mut self, input: &Input) -> Vec<Op> {
        let since = std::time::Instant::now();
        let sampler = self.estimator.get_sampler();
        let parallel_cnt = 16;

        let mut rects = vec![vec![]; input.rect_cnt()];

        for _ in 0..parallel_cnt {
            let rects_i = sampler.sample(self.rng);

            for j in 0..input.rect_cnt() {
                rects[j].push(rects_i[j]);
            }
        }

        let large_state = LargeState::new(input.clone(), rects, parallel_cnt, self.rng);
        let small_state = SmallState::default();
        let act_gen = ActGen;

        let remaining_time = self.duration_sec - since.elapsed().as_secs_f64();
        let mut beam = beam::BeamSearch::new(large_state, small_state, act_gen);
        let standard_beam_width = 10_000_000 / (input.rect_cnt() as usize).pow(3);
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
        let (ops, _) = beam.run(input.rect_cnt(), beam_width_suggester, deduplicator);

        ops
    }
}

#[derive(Debug, Clone)]
struct LargeState {
    rects: Vec<Vec<Rect>>,
    widths: Vec<u32>,
    heights: Vec<u32>,
    placements: Vec<Vec<Placement>>,
    hash: u64,
    hash_base_x: Vec<Vec<u64>>,
    hash_base_y: Vec<Vec<u64>>,
    hash_base_rot: Vec<u64>,
    turn: usize,
    parallel_cnt: usize,
}

impl LargeState {
    fn new(
        input: Input,
        rects: Vec<Vec<Rect>>,
        parallel_cnt: usize,
        rng: &mut impl rand::Rng,
    ) -> Self {
        let heights = vec![0; parallel_cnt];
        let widths = vec![0; parallel_cnt];
        let hash_base_x = (0..input.rect_cnt())
            .map(|_| (0..parallel_cnt).map(|_| rng.gen()).collect())
            .collect();
        let hash_base_y = (0..input.rect_cnt())
            .map(|_| (0..parallel_cnt).map(|_| rng.gen()).collect())
            .collect();
        let hash_base_rot = (0..input.rect_cnt()).map(|_| rng.gen()).collect();

        Self {
            rects,
            widths,
            heights,
            placements: vec![],
            hash: 0,
            hash_base_x,
            hash_base_y,
            hash_base_rot,
            turn: 0,
            parallel_cnt,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct SmallState {
    placements: Vec<Placement>,
    old_widths: Vec<u32>,
    old_heights: Vec<u32>,
    new_widths: Vec<u32>,
    new_heights: Vec<u32>,
    hash: u64,
    hash_xor: u64,
    op: Op,
    score: i32,
}

impl beam::SmallState for SmallState {
    type Score = i32;
    type Hash = u64;
    type LargeState = LargeState;
    type Action = Op;

    fn raw_score(&self) -> Self::Score {
        self.score
    }

    fn hash(&self) -> Self::Hash {
        self.hash
    }

    fn apply(&self, state: &mut Self::LargeState) {
        state.placements.push(self.placements.clone());
        state.widths.copy_from_slice(&self.new_widths);
        state.heights.copy_from_slice(&self.new_heights);
        state.hash ^= self.hash_xor;
        state.turn += 1;
    }

    fn rollback(&self, state: &mut Self::LargeState) {
        state.placements.pop();
        state.widths.copy_from_slice(&self.old_widths);
        state.heights.copy_from_slice(&self.old_heights);
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
    rotate: bool,
}

impl Placement {
    fn new(x0: u32, x1: u32, y0: u32, y1: u32, rotate: bool) -> Self {
        assert!(x0 < x1);
        assert!(y0 < y1);
        Self {
            x0,
            x1,
            y0,
            y1,
            rotate,
        }
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
        let turn = large_state.turn;

        let mut new_placements = vec![];
        let mut old_heights = vec![];
        let mut new_heights = vec![];
        let mut old_widths = vec![];
        let mut new_widths = vec![];
        let mut hash_xor = 0;
        let mut score = 0;

        let rects = &large_state.rects[large_state.turn];

        let mut touching_cnt = 0;

        for parallel_i in 0..large_state.parallel_cnt {
            let rect = &rects[parallel_i];
            let placements = &large_state.placements;

            let (width, height) = if rotate {
                (rect.height(), rect.width())
            } else {
                (rect.width(), rect.height())
            };

            let y0 = match base {
                Some(index) => placements[index][parallel_i].y1,
                None => 0,
            };
            let y1 = y0 + height;

            let x0 = placements
                .iter()
                .filter_map(|p| {
                    let p = &p[parallel_i];

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
                    let p = placements[base][parallel_i];
                    x0.max(p.x0) < x1.min(p.x1)
                }
                None => true,
            };

            touching_cnt += is_touching as u32;

            let placement = Placement::new(x0, x1, y0, y1, rotate);
            new_placements.push(placement);

            let hash_x = large_state.hash_base_x[turn][parallel_i].wrapping_mul(x0 as u64);
            let hash_y = large_state.hash_base_y[turn][parallel_i].wrapping_mul(y0 as u64);
            hash_xor ^= hash_x ^ hash_y;

            let new_width = x1.max(large_state.widths[parallel_i]);
            let new_height = y1.max(large_state.heights[parallel_i]);

            old_heights.push(large_state.heights[parallel_i]);
            old_widths.push(large_state.widths[parallel_i]);
            new_heights.push(new_height);
            new_widths.push(new_width);

            score -= new_height as i32;
            score -= new_width as i32;
        }

        if touching_cnt == 0 {
            return None;
        }

        hash_xor ^= if rotate {
            large_state.hash_base_rot[turn]
        } else {
            0
        };
        
        let hash = large_state.hash ^ hash_xor;

        let op = Op::new(turn, rotate, Dir::Left, base);

        Some(SmallState {
            placements: new_placements,
            old_widths,
            old_heights,
            new_widths,
            new_heights,
            hash,
            hash_xor,
            op,
            score,
        })
    }

    fn gen_up_cand(
        &self,
        large_state: &LargeState,
        base: Option<usize>,
        rotate: bool,
    ) -> Option<SmallState> {
        let turn = large_state.turn;

        let mut new_placements = vec![];
        let mut old_heights = vec![];
        let mut new_heights = vec![];
        let mut old_widths = vec![];
        let mut new_widths = vec![];
        let mut hash_xor = 0;
        let mut score = 0;

        let rects = &large_state.rects[large_state.turn];

        let mut touching_cnt = 0;

        for parallel_i in 0..large_state.parallel_cnt {
            let rect = &rects[parallel_i];
            let placements = &large_state.placements;

            let (width, height) = if rotate {
                (rect.height(), rect.width())
            } else {
                (rect.width(), rect.height())
            };

            let x0 = match base {
                Some(index) => placements[index][parallel_i].x1,
                None => 0,
            };
            let x1 = x0 + width;

            let y0 = placements
                .iter()
                .filter_map(|p| {
                    let p = &p[parallel_i];

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
                    let p = placements[base][parallel_i];
                    y0.max(p.y0) < y1.min(p.y1)
                }
                None => true,
            };

            touching_cnt += is_touching as u32;

            let placement = Placement::new(x0, x1, y0, y1, rotate);

            new_placements.push(placement);

            let hash_x = large_state.hash_base_x[turn][parallel_i].wrapping_mul(x0 as u64);
            let hash_y = large_state.hash_base_y[turn][parallel_i].wrapping_mul(y0 as u64);
            hash_xor ^= hash_x ^ hash_y;

            let new_width = x1.max(large_state.widths[parallel_i]);
            let new_height = y1.max(large_state.heights[parallel_i]);

            old_heights.push(large_state.heights[parallel_i]);
            old_widths.push(large_state.widths[parallel_i]);
            new_heights.push(new_height);
            new_widths.push(new_width);

            score -= new_height as i32;
            score -= new_width as i32;
        }

        if touching_cnt == 0 {
            return None;
        }

        hash_xor ^= if rotate {
            large_state.hash_base_rot[turn]
        } else {
            0
        };

        let hash = large_state.hash ^ hash_xor;

        let op = Op::new(turn, rotate, Dir::Up, base);

        Some(SmallState {
            placements: new_placements,
            old_widths,
            old_heights,
            new_widths,
            new_heights,
            hash,
            hash_xor,
            op,
            score,
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
        let rotates = if large_state.turn == 0 {
            vec![false]
        } else {
            vec![false, true]
        };

        for rotate in rotates {
            // Left
            next_states.extend(self.gen_left_cand(&large_state, None, rotate));

            for i in 0..large_state.turn {
                next_states.extend(self.gen_left_cand(&large_state, Some(i), rotate));
            }

            // Up
            next_states.extend(self.gen_up_cand(&large_state, None, rotate));

            for i in 0..large_state.turn {
                next_states.extend(self.gen_up_cand(&large_state, Some(i), rotate));
            }
        }
    }
}
