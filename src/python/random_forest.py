"""Random forest baseline for QSAR biodegradation (NRB = 1, RB = 0).

1. Tune hyperparameters with non-stratified K-fold CV on the training split.
2. Refit the best setting on the full training split and report validation metrics.
3. Evaluate that finalized model once on the test split.

Usage (from the repo root):
    python3 src/python/random_forest.py [--folds 5] [--seed 42] [--metric balanced_accuracy]
"""

import argparse
import itertools

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from common import (OUTPUTS_DIR, ROOT, evaluate_and_export, load_all, metrics,
                    summarize_folds, write_json)

OUT_DIR = OUTPUTS_DIR / "random_forest"

PARAM_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [None, 10, 20],
    "max_features": ["sqrt", 0.33, 0.5],
    "min_samples_leaf": [1, 2, 5],
    "class_weight": [None, "balanced"],
}


def make_model(params, seed):
    return RandomForestClassifier(**params, random_state=seed, n_jobs=-1)


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

    # Non-stratified K-fold (shuffled so folds aren't tied to file order)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    keys, values = zip(*PARAM_GRID.items())
    rows = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        rows.append({**params, **cross_validate(X_tr, y_tr, params, kf, args.seed)})
    cv = pd.DataFrame(rows).sort_values(f"cv_{args.metric}_mean", ascending=False)

    best = cv.iloc[0][list(keys)].to_dict()
    best = {k: (None if pd.isna(v) else v) for k, v in best.items()}
    best["n_estimators"] = int(best["n_estimators"])
    best["min_samples_leaf"] = int(best["min_samples_leaf"])
    if best["max_depth"] is not None:
        best["max_depth"] = int(best["max_depth"])

    print(f"\n{args.folds}-fold CV on train ({len(cv)} settings), top 5 by {args.metric}:")
    show = list(keys) + [f"cv_{args.metric}_mean", f"cv_{args.metric}_std"]
    print(cv[show].head().to_string(index=False))
    print(f"\nBest params: {best}")

    model = make_model(best, args.seed).fit(X_tr, y_tr)
    results = {"best_params": best, "cv_folds": args.folds, "cv_metric": args.metric,
               "cv_best": cv.iloc[0][[c for c in cv.columns if c.startswith("cv_")]].to_dict()}
    results.update(evaluate_and_export("random forest", lambda X: model.predict_proba(X)[:, 1],
                                       splits, OUT_DIR))

    importances = pd.Series(model.feature_importances_,
                            index=[f"x{i + 1}" for i in range(X_tr.shape[1])])
    print("\nTop 10 feature importances:")
    print(importances.sort_values(ascending=False).head(10).round(4).to_string())

    cv.to_csv(OUT_DIR / "cv_results.csv", index=False)
    importances.sort_values(ascending=False).to_csv(OUT_DIR / "feature_importances.csv",
                                                    header=["importance"])
    write_json(OUT_DIR / "metrics.json", results)
    print(f"\nWrote outputs to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
