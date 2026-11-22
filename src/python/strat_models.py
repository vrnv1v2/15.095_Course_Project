"""Random forest, XGBoost, Gaussian-kernel logistic regression, and Nadaraya–Watson kernel
regression on the stratified split.

Follows MLProjectPart1_Orig.ipynb's protocol so these rows can sit next to its CART / OCT /
logistic rows:
  * development set = biodeg_train_strat + biodeg_val_strat (844), test = biodeg_test_strat
  * the notebook's stratified 5 folds and k = 3, 5, 10 / all descriptor lists
    (from outputs/strat, written by src/julia/strat_setup.jl — run that first)
  * every grid setting is scored by mean CV AUC over the 5 folds; the best is refit on all
    844 chemicals
  * thresholds from the out-of-fold scores: balanced-accuracy threshold and the NRB-first
    threshold (catch >= 90% of NRB), both frozen before test
  * the test set is scored only with --run-test

Usage (from the repo root):
    python3 src/python/strat_models.py [--models rf xgb klr nw] [--run-test]

Outputs per model in outputs/strat/<model>/: cv_results.csv (notebook columns), cv_grid.csv
(every setting's CV AUC), oof_predictions.csv, and with --run-test: test_results.csv and
predictions_test_<set>.csv.
"""

import argparse
import itertools

import numpy as np
import pandas as pd

from common import ensure_openmp_for_xgboost
from strat_common import (SEED, SETS, STRAT_OUT, RESULT_COLS, TEST_COLS, cv_generic, load_strat,
                          result_row, test_row)


def expand(grid):
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


# ---------------------------------------------------------------- random forest

RF_GRID = expand({
    "max_features": ["sqrt", 0.33, 0.5],
    "min_samples_leaf": [1, 3, 5, 10],
    "class_weight": [None, "balanced_subsample"],
})


def fit_rf(X, y, prm):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=500, **prm, random_state=SEED, n_jobs=-1).fit(
        X.to_numpy(), y)


def predict_rf(m, X):
    return m.predict_proba(X.to_numpy())[:, 1]


# ---------------------------------------------------------------- XGBoost

XGB_GRID = expand({
    "n_estimators": [200, 500],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.1],
    "colsample_bytree": [0.6, 1.0],
    "min_child_weight": [1, 5],
})


def fit_xgb(X, y, prm):
    from xgboost import XGBClassifier
    return XGBClassifier(**prm, subsample=0.8, objective="binary:logistic", eval_metric="logloss",
                         tree_method="hist", random_state=SEED, n_jobs=4).fit(X.to_numpy(), y)


def predict_xgb(m, X):
    return m.predict_proba(X.to_numpy())[:, 1]


# ---------------------------------------------------------------- Gaussian-kernel logistic regression
#
# Descriptors are standardized with the training part's means / std devs (as the notebook does
# for logistic regression). With K_ij = exp(-γ ‖z_i − z_j‖²), γ = g / (number of descriptors),
# the model is  P(NRB | x) = σ(Σ_i α_i K(x, x_i) + b)  and we minimize
#
#     Σ_i [log(1 + e^{η_i}) − y_i η_i] + (λ/2) αᵀ K α,      η = K α + b
#
# by Newton / IRLS: each step solves the weighted kernel-ridge system
#     (W K + λ I) α + W 1 b = W η + (y − p),     1ᵀ W K α + 1ᵀ W 1 b = 1ᵀ (W η + y − p)
# with backtracking so the objective never increases.

KLR_GRID = expand({
    "g": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0],   # γ = g / n_descriptors
    "lam": [0.001, 0.01, 0.1, 1.0, 10.0],
})


def _rbf(A, B, gamma):
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def _klr_objective(K, alpha, b, y, lam):
    eta = K @ alpha + b
    return np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * lam * alpha @ K @ alpha


def fit_klr(X, y, prm, max_iter=100, tol=1e-8):
    Xm = X.to_numpy(float)
    mu = Xm.mean(0)
    sd = np.maximum(Xm.std(0, ddof=1), 1e-12)
    Z = (Xm - mu) / sd
    gamma = prm["g"] / Z.shape[1]
    lam = prm["lam"]
    K = _rbf(Z, Z, gamma)
    n = len(y)
    yf = y.astype(float)

    alpha, b = np.zeros(n), 0.0
    obj = _klr_objective(K, alpha, b, yf, lam)
    for _ in range(max_iter):
        eta = K @ alpha + b
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-10, None)
        rhs = w * eta + (yf - p)
        A = np.empty((n + 1, n + 1))
        A[:n, :n] = w[:, None] * K + lam * np.eye(n)
        A[:n, n] = w
        A[n, :n] = w @ K
        A[n, n] = w.sum()
        sol = np.linalg.solve(A, np.append(rhs, rhs.sum()))
        d_alpha, d_b = sol[:n] - alpha, sol[n] - b
        step = 1.0
        while True:
            new_obj = _klr_objective(K, alpha + step * d_alpha, b + step * d_b, yf, lam)
            if new_obj <= obj + 1e-12 or step < 1e-6:
                break
            step /= 2
        alpha, b = alpha + step * d_alpha, b + step * d_b
        converged = obj - new_obj < tol * max(1.0, abs(obj))
        obj = new_obj
        if converged:
            break
    return dict(alpha=alpha, b=b, Z=Z, mu=mu, sd=sd, gamma=gamma)


def predict_klr(m, X):
    Zn = (X.to_numpy(float) - m["mu"]) / m["sd"]
    eta = _rbf(Zn, m["Z"], m["gamma"]) @ m["alpha"] + m["b"]
    return 1.0 / (1.0 + np.exp(-eta))


# ---------------------------------------------------------------- Nadaraya–Watson kernel regression
#
# The NRB score of x is the kernel-weighted average of the training labels,
#
#     m(x) = Σ_i K(x, x_i) y_i / Σ_i K(x, x_i),     K(x, x') = exp(-γ ‖z − z'‖²),
#
# on descriptors standardized with the training part's means / std devs, with γ = g / (number of
# descriptors). Nothing is fitted; the only setting is the bandwidth g. Weights are computed in
# log space (softmax over −γ‖z − z_i‖²) so narrow bandwidths don't underflow to 0/0.

NW_GRID = expand({
    "g": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],   # γ = g / n_descriptors
})


def fit_nw(X, y, prm):
    Xm = X.to_numpy(float)
    mu = Xm.mean(0)
    sd = np.maximum(Xm.std(0, ddof=1), 1e-12)
    return dict(Z=(Xm - mu) / sd, y=y.astype(float), mu=mu, sd=sd, gamma=prm["g"] / Xm.shape[1])


def predict_nw(m, X):
    Zn = (X.to_numpy(float) - m["mu"]) / m["sd"]
    sq = (Zn ** 2).sum(1)[:, None] + (m["Z"] ** 2).sum(1)[None, :] - 2 * Zn @ m["Z"].T
    logw = -m["gamma"] * np.maximum(sq, 0.0)
    logw -= logw.max(1, keepdims=True)
    w = np.exp(logw)
    return (w @ m["y"]) / w.sum(1)


# ---------------------------------------------------------------- driver

MODELS = {
    "rf": ("Random forest", "random_forest", RF_GRID, fit_rf, predict_rf),
    "xgb": ("XGBoost", "xgboost", XGB_GRID, fit_xgb, predict_xgb),
    "klr": ("Kernel logistic", "kernel_logistic", KLR_GRID, fit_klr, predict_klr),
    "nw": ("Nadaraya-Watson", "nadaraya_watson", NW_GRID, fit_nw, predict_nw),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--run-test", action="store_true",
                    help="score the frozen final models on the 211 test chemicals")
    args = ap.parse_args()
    if "xgb" in args.models:
        ensure_openmp_for_xgboost()

    X_dev, y_dev, X_te, y_te, fold, feature_sets = load_strat()
    print(f"RUN_TEST={args.run_test}\n")

    for key in args.models:
        name, folder, grid, fit_fn, predict_fn = MODELS[key]
        out = STRAT_OUT / folder
        out.mkdir(parents=True, exist_ok=True)
        rows, tests, grids = [], [], []
        oof_tbl = pd.DataFrame({"dev_row": np.arange(1, len(y_dev) + 1),
                                "class": np.where(y_dev == 1, "NRB", "RB"), "fold": fold})
        print(f"=== {name}: {len(grid)} settings × {len(SETS)} descriptor sets × 5 folds")
        for s in SETS:
            fs = feature_sets[s]
            r = cv_generic(X_dev, y_dev, fold, fs, grid, fit_fn, predict_fn)
            row = result_row(s, name, r, y_dev, n_used=len(fs))
            rows.append(row)
            grids.append(r["grid"].assign(set=s))
            oof_tbl[f"score_nrb_{s}"] = r["oof"]
            print(f"{s:<5} {name:<16} CV AUC {r['cv_auc']:.3f} ± {r['cv_sd']:.3f} | "
                  f"{r['settings']} | {r['time']:.1f} s")
            if args.run_test:
                p = r["predict"](X_te)
                tests.append(test_row(row, p, y_te))
                pd.DataFrame({"row_id": np.arange(len(p)), "score_nrb": p,
                              "predicted_label": (p >= row["threshold"]).astype(int)}).to_csv(
                    out / f"predictions_test_{s}.csv", index=False)

        res = pd.DataFrame(rows)[RESULT_COLS]
        res.to_csv(out / "cv_results.csv", index=False)
        pd.concat(grids).sort_values(["set", "cv_auc"], ascending=[True, False]).to_csv(
            out / "cv_grid.csv", index=False)
        oof_tbl.to_csv(out / "oof_predictions.csv", index=False)
        print(res[["set", "cv_auc", "cv_auc_sd", "cv_bal_acc", "threshold", "thr_prec",
                   "prec_rb_cleared"]].round(3).to_string(index=False))
        if args.run_test:
            t = pd.DataFrame(tests)[TEST_COLS]
            t.to_csv(out / "test_results.csv", index=False)
            print("\nTest (frozen thresholds):\n" + t.round(3).to_string(index=False))
        print(f"Wrote outputs/strat/{folder}/\n")

    if not args.run_test:
        print("Test evaluation is switched off (pass --run-test).")


if __name__ == "__main__":
    main()
