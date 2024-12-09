import json
import math
import os

import numpy as np
import optuna
import pack

PARAM_NAMES = [
    "touching_threshold",
    "invalid_cnt_threshold",
]


def gaussian_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
) -> float:
    return t1 * np.dot(x1, x2) + t2 * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / t3)


def calc_kernel_matrix(
    x1: np.ndarray,
    x2: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
) -> np.ndarray:
    n = x1.shape[0]
    m = x2.shape[0]
    k = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            k[i, j] = gaussian_kernel(x1[i, :], x2[j, :], t1, t2, t3)

    return k


def predict_y(
    x: np.ndarray,
    y: np.ndarray,
    xx: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    t4: float,
) -> float:
    y_mean = np.mean(y)
    k = calc_kernel_matrix(x, x, t1, t2, t3) + t4 * np.eye(x.shape[0])
    kk = calc_kernel_matrix(x, xx, t1, t2, t3)
    yy = kk.transpose() @ np.linalg.solve(k, y - y_mean)
    return yy + y_mean


def calc_log_likelihood(
    x: np.ndarray,
    y: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    t4: float,
) -> float:
    y_mean = np.mean(y)
    y = y - y_mean
    k = calc_kernel_matrix(x, x, t1, t2, t3) + t4 * np.eye(x.shape[0])
    yy = y.transpose() @ np.linalg.solve(k, y)
    return -np.log(np.linalg.det(k)) - yy


class Objective:
    x: np.ndarray
    y: np.ndarray

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    def __call__(self, trial: optuna.trial.Trial) -> float:
        t1 = trial.suggest_float("t1", 0.01, 10.0, log=True)
        t2 = trial.suggest_float("t2", 0.01, 10.0, log=True)
        t3 = trial.suggest_float("t3", 0.01, 1.0, log=True)
        t4 = trial.suggest_float("t4", 0.01, 10.0, log=True)
        return calc_log_likelihood(self.x, self.y, t1, t2, t3, t4)


def load_data():
    x_list = []
    p_list = [[] for _ in range(11)]

    OPT_RESULT_DIR = "data/opt2"

    files = os.listdir(OPT_RESULT_DIR)

    for file in files:
        if not file.endswith(".json"):
            continue

        with open(f"{OPT_RESULT_DIR}/{file}", "r") as f:
            data = json.load(f)
            x = []
            x.append((data["n"] - 30) / 70)
            x.append(math.log(data["t"] / data["n"]))
            x.append((data["sigma"] - 1000) / 9000)
            x_list.append(x)

            for i, name in enumerate(PARAM_NAMES):
                p_list[i].append(data["params"][name])

    x_matrix = np.array(x_list, dtype=np.float64)
    p_arrays = [np.array(p, dtype=np.float64) for p in p_list]

    return x_matrix, p_arrays


def predict_one(
    x_matrix: np.ndarray, data_array: np.ndarray, new_x: np.ndarray, n_trials: int = 500
) -> tuple[float, float, float, float, float]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
    )
    objective = Objective(x_matrix, data_array)
    study.optimize(objective, n_trials=n_trials)

    print("param", study.best_trial.params)

    t1 = study.best_trial.params["t1"]
    t2 = study.best_trial.params["t2"]
    t3 = study.best_trial.params["t3"]
    t4 = study.best_trial.params["t4"]

    optuna.logging.set_verbosity(optuna.logging.INFO)

    pred = predict_y(x_matrix, data_array, new_x, t1, t2, t3, t4)

    return pred, t1, t2, t3, t4


def predict(
    n: int, t: int, sigma: float, n_trials: int = 500
) -> tuple[float, float, float, float]:
    (x_matrix, p_arrays) = load_data()

    new_x = np.array(
        [[(n - 30) / 70, math.log(t / n), (sigma - 1000) / 9000]],
        dtype=np.float64,
    )

    preds = []

    for p_array in p_arrays:
        pred, _, _, _, _ = predict_one(x_matrix, p_array, new_x, n_trials)
        preds.append(pred)

    return [p[0] for p in preds]


if __name__ == "__main__":
    (x_matrix, p_arrays) = load_data()

    n = 30
    t = 60
    sigma = 5574

    print(f"n={n}, t={t}, sigma={sigma}")

    new_x = np.array(
        [[(n - 30) / 70, math.log(t / n), (sigma - 1000) / 9000]],
        dtype=np.float64,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    params = []

    for p_array, name in zip(p_arrays, PARAM_NAMES):
        print(f"=== {name} ===")
        pred, t1, t2, t3, t4 = predict_one(x_matrix, p_array, new_x)
        params.append([t1, t2, t3, t4])
        print(f"pred_{name}", pred)

    PARAM_PATH = "data/params2.txt"

    with open(PARAM_PATH, "w") as f:
        n_vec = x_matrix[:, 0]
        t_vec = x_matrix[:, 1]
        sigma_vec = x_matrix[:, 2]

        f.write(f'const N2: &[u8] = b"{pack.pack_vec(n_vec)}";\n')
        f.write(f'const T2: &[u8] = b"{pack.pack_vec(t_vec)}";\n')
        f.write(f'const SIGMA2: &[u8] = b"{pack.pack_vec(sigma_vec)}";\n')

        for param_name, p_array in zip(PARAM_NAMES, p_arrays):
            name = f"{param_name.upper()}"
            f.write(f'const {name}: &[u8] = b"{pack.pack_vec(p_array)}";\n')

        for param_name, param in zip(PARAM_NAMES, params):
            name = f"{param_name.upper()}"
            f.write(
                f'const PARAM_{name}: &[u8] = b"{pack.pack_vec(np.array(param, dtype=np.float64))}";\n'
            )
