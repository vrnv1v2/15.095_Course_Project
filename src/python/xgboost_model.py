"""XGBoost baseline for QSAR biodegradation (NRB = 1, RB = 0).

1. Tune hyperparameters with non-stratified K-fold CV on the training split.
2. Refit the best setting on the full training split and report validation metrics.
3. Evaluate that finalized model once on the test split.

Usage (from the repo root):
    python3 src/python/xgboost_model.py [--folds 5] [--seed 42] [--metric balanced_accuracy]

(Named xgboost_model.py, not xgboost.py, so it doesn't shadow the xgboost package.)
"""

from common import ensure_openmp_for_xgboost

ensure_openmp_for_xgboost()

import argparse  # noqa: E402
import itertools  # noqa: E402

import pandas as pd  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from common import (OUTPUTS_DIR, ROOT, evaluate_and_export, load_all, metrics,  # noqa: E402
                    summarize_folds, write_json)

OUT_DIR = OUTPUTS_DIR / "xgboost"

PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [2, 3, 4, 6],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.5, 0.8],
    "min_child_weight": [1, 5],
    # filled in at runtime: 1 (unweighted) and n_RB / n_NRB (balanced; NRB is the majority)
    "scale_pos_weight": None,
}


def make_model(params, seed):
    return XGBClassifier(**params, objective="binary:logistic", eval_metric="logloss",
                         tree_method="hist", random_state=seed, n_jobs=-1)


def cross_validate(X, y, params, kf, seed):
    folds = []
    for tr_idx, te_idx in kf.split(X):
        model = make_model(params, seed).fit(X[tr_idx], y[tr_idx])
        folds.append(metrics(y[te_idx], model.predict_proba(X[te_idx])[:, 1]))
    return summarize_folds(folds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metric", default="balanced_accuracy",
                    choices=["accuracy", "balanced_accuracy", "auc", "nrb_recall"],
                    help="CV metric used to pick hyperparameters")
    args = ap.parse_args()

    splits = load_all()
    X_tr, y_tr = splits["train"]

    grid = dict(PARAM_GRID)
    grid["scale_pos_weight"] = [1.0, round(float((y_tr == 0).sum() / (y_tr == 1).sum()), 4)]

    # Non-stratified K-fold (shuffled so folds aren't tied to file order)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    keys, values = zip(*grid.items())
    rows = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        rows.append({**params, **cross_validate(X_tr, y_tr, params, kf, args.seed)})
    cv = pd.DataFrame(rows).sort_values(f"cv_{args.metric}_mean", ascending=False)

    best = cv.iloc[0][list(keys)].to_dict()
    for k in ["n_estimators", "max_depth", "min_child_weight"]:
        best[k] = int(best[k])
    for k in ["learning_rate", "subsample", "colsample_bytree", "scale_pos_weight"]:
        best[k] = float(best[k])

    print(f"\n{args.folds}-fold CV on train ({len(cv)} settings), top 5 by {args.metric}:")
    show = list(keys) + [f"cv_{args.metric}_mean", f"cv_{args.metric}_std"]
    print(cv[show].head().to_string(index=False))
    print(f"\nBest params: {best}")

    model = make_model(best, args.seed).fit(X_tr, y_tr)
    results = {"best_params": best, "cv_folds": args.folds, "cv_metric": args.metric,
               "cv_best": cv.iloc[0][[c for c in cv.columns if c.startswith("cv_")]].to_dict()}
    results.update(evaluate_and_export("XGBoost", lambda X: model.predict_proba(X)[:, 1],
                                       splits, OUT_DIR))

    importances = pd.Series(model.feature_importances_,
                            index=[f"x{i + 1}" for i in range(X_tr.shape[1])])
    print("\nTop 10 feature importances (gain):")
    print(importances.sort_values(ascending=False).head(10).round(4).to_string())

    cv.to_csv(OUT_DIR / "cv_results.csv", index=False)
    importances.sort_values(ascending=False).to_csv(OUT_DIR / "feature_importances.csv",
                                                    header=["importance"])
    write_json(OUT_DIR / "metrics.json", results)
    print(f"\nWrote outputs to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
