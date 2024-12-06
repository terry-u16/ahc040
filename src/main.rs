use problem::Input;
use solver::Solver;

#[allow(dead_code)]
mod beam;
mod problem;
#[allow(dead_code)]
mod sa;
pub mod solver;
#[allow(dead_code)]
mod util;

fn main() {
    let input = Input::read();
    let judge = problem::gen_judge(&input);
    let solver = solver::solver01::Solver01;
    solver.solve(&input, judge);
    eprintln!("Elapsed: {:?}", input.since().elapsed());
}
