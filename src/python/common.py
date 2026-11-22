"""Shared data loading, metrics, and output helpers for the Python baselines (NRB = 1, RB = 0)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

LABEL_MAP = {"NRB": 1, "RB": 0}  # NRB is the positive class
METRIC_KEYS = ["accuracy", "nrb_recall", "rb_specificity", "balanced_accuracy", "auc"]


def ensure_openmp_for_xgboost():
    """XGBoost on macOS needs libomp.dylib. If it isn't installed (e.g. no Homebrew),
    re-exec this process with the loader pointed at the copy bundled in scikit-learn's wheel."""
    import os
    import sys
    try:
        import xgboost  # noqa: F401
        return
    except Exception:
        if os.environ.get("_XGB_OMP_RETRY"):
            raise
    import sklearn
    dylibs = Path(sklearn.__file__).parent / ".dylibs"
    env = dict(os.environ, _XGB_OMP_RETRY="1")
    env["DYLD_FALLBACK_LIBRARY_PATH"] = os.pathsep.join(
        p for p in [str(dylibs), env.get("DYLD_FALLBACK_LIBRARY_PATH")] if p)
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


def load_split(name: str):
    """Split CSVs are ';'-separated, no header: 41 descriptors then the RB/NRB label."""
    df = pd.read_csv(DATA_DIR / f"biodeg_{name}.csv", sep=";", header=None)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].str.strip().map(LABEL_MAP)
    if y.isna().any():
        raise ValueError(f"Unexpected labels in {name}: {set(df.iloc[:, -1]) - set(LABEL_MAP)}")
    return X, y.to_numpy(dtype=int)


def load_all(verbose=True):
    splits = {name: load_split(name) for name in ["train", "val", "test"]}
    if verbose:
        print(", ".join(f"{n} {X.shape}" for n, (X, _) in splits.items())
              + "; NRB share " + " ".join(f"{n}={y.mean():.3f}" for n, (_, y) in splits.items()))
    return splits


def metrics(y_true, score, threshold=0.5):
    y_pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "nrb_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "rb_specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, score),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def summarize_folds(fold_metrics):
    return {
        **{f"cv_{k}_mean": float(np.mean([f[k] for f in fold_metrics])) for k in METRIC_KEYS},
        **{f"cv_{k}_std": float(np.std([f[k] for f in fold_metrics])) for k in METRIC_KEYS},
    }


def print_metrics(title, m):
    cm = m["confusion_matrix"]
    print(f"\n{title}")
    for k in METRIC_KEYS:
        print(f"  {k:<18} {m[k]:.4f}")
    print(f"  confusion (rows=true RB/NRB, cols=pred RB/NRB): "
          f"[[{cm['tn']}, {cm['fp']}], [{cm['fn']}, {cm['tp']}]]")


def evaluate_and_export(model_name, predict_score, splits, out_dir, threshold=0.5):
    """Score val and test, print metrics, and write predictions_{split}.csv files.

    row_id = 0-based row position within the split CSV (until data/splits.csv exists).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ["val", "test"]:
        X, y = splits[name]
        score = predict_score(X)
        results[name] = metrics(y, score, threshold)
        print_metrics(f"{name.upper()} ({model_name} fit on train)", results[name])
        pd.DataFrame({
            "row_id": np.arange(len(score)),
            "score_nrb": score,
            "predicted_label": (score >= threshold).astype(int),
        }).to_csv(out_dir / f"predictions_{name}.csv", index=False)
    return results


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=float))
