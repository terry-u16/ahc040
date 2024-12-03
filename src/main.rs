use problem::Input;
use solver::Solver;

#[allow(dead_code)]
mod sa;
#[allow(dead_code)]
mod beam;
mod problem;
pub mod solver;
mod util;

fn main() {
    let input = Input::read();
    let judge = problem::gen_judge(&input);
    let solver = solver::solver01::Solver01;
    solver.solve(&input, judge);
}
