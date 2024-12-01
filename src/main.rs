use problem::Input;
use solver::Solver;

mod problem;
pub mod solver;
mod util;

fn main() {
    let input = Input::read();
    let judge = problem::gen_judge(&input);
    let solver = solver::stack::StackSolver;
    solver.solve(&input, judge);
}
