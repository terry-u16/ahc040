use problem::Input;
use solver::Solver;

mod problem;
pub mod solver;

fn main() {
    let input = Input::read();
    let judge = problem::ActualJudge::new();
    let solver = solver::stack::StackSolver;
    solver.solve(&input, judge);
}
