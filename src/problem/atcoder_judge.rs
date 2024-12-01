use proconio::input_interactive;

use super::{Judge, Measure, Op};

#[derive(Debug, Clone)]
pub struct AtCoderJudge {
    query_cnt: usize,
}

#[allow(dead_code)]
impl AtCoderJudge {
    pub fn new() -> Self {
        Self { query_cnt: 0 }
    }
}

impl Judge for AtCoderJudge {
    fn query(&mut self, ops: &[Op]) -> Measure {
        self.query_cnt += 1;

        println!("{}", ops.len());

        for op in ops {
            println!("{}", op);
        }

        input_interactive! {
            width: u32,
            height: u32,
        }

        Measure::new(height, width)
    }

    fn query_cnt(&self) -> usize {
        self.query_cnt
    }

    fn rects(&self) -> Option<&[super::Rect]> {
        None
    }
}
