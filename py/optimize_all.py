import datetime
import json
import os
import random
import subprocess

import optimize
import optuna

OPT_PATH = "data/opt"

for iteration in range(1, 1000):
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"ahc040-{start_time}-group-{iteration:000}"

    n = random.randint(30, 100)
    t_pow = random.random() * 3 - 1
    t = int(round(n * pow(2, t_pow)))
    sigma = random.randint(1000, 10000)

    cmd = [
        "cargo",
        "run",
        "--release",
        "--bin",
        "gen",
        "./seeds.txt",
        "--dir",
        "../pahcer/in",
        "--N",
        f"{n}",
        "--T",
        f"{t}",
        "--sigma",
        f"{sigma}",
    ]
    print(cmd)

    subprocess.run(cmd, cwd="./tools").check_returncode()

    if os.path.exists("./pahcer/best_scores.json"):
        os.remove("./pahcer/best_scores.json")

    # ベストスコアを更新しておく
    for _ in range(3):
        cmd = [
            "pahcer",
            "run",
            "--setting-file",
            "pahcer_config_optuna.toml",
        ]
        subprocess.run(cmd).check_returncode()

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        pruner=optuna.pruners.WilcoxonPruner(n_startup_steps=88),
        sampler=optuna.samplers.TPESampler(),
    )

    study.enqueue_trial(
        {
            "arrange_count": 10,
            "query_annealing_duration_sec": 0.3,
            "mcmc_init_duration_sec": 0.1,
            "beam_mcts_duration_ratio": 0.5,
            "mcmc_duration_ratio": 0.1,
            "mcts_turn": 15,
            "mcts_expansion_threshold": 3,
            "mcts_candidates_count": 4,
            "parallel_score_mul": 0.9,
            "width_buf": 1.1,
            "ucb1_tuned_coef": 1.0,
        }
    )

    optimize.run_optimization(study)

    print(f"best params = {study.best_params}")
    print(f"best score  = {study.best_value}")

    dictionary = {
        "study_name": study_name,
        "n": n,
        "t": t,
        "sigma": sigma,
        "params": study.best_trial.params,
    }

    filename = (
        "optimized_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    with open(f"{OPT_PATH}/{filename}", "w") as f:
        json.dump(dictionary, f, indent=2)
