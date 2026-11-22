"""Shared helpers for the stratified-split pipeline (Python side).

Mirrors MLProjectPart1_Orig.ipynb so results line up with the notebook's CART / OCT /
logistic rows: same development set (train_strat + val_strat), same folds and descriptor
lists (written by src/julia/strat_setup.jl), same metric / threshold definitions, and the
same results-table columns.
"""

import time

import numpy as np
import pandas as pd

from common import DATA_DIR, OUTPUTS_DIR

STRAT_OUT = OUTPUTS_DIR / "strat"

SEED = 1
N_FOLDS = 5
SETS = ["k3", "k5", "k10", "all"]
NRB_RECALL_TARGET = 0.90

DESCRIPTORS = [
    "SpMax_L", "J_Dz(e)", "nHM", "F01[N-N]", "F04[C-N]", "NssssC", "nCb-",
    "C%", "nCp", "nO", "F03[C-N]", "SdssC", "HyWi_B(m)", "LOC", "SM6_L",
    "F03[C-O]", "Me", "Mi", "nN-N", "nArNO2", "nCRX3", "SpPosA_B(p)", "nCIR",
    "B01[C-Br]", "B03[C-Cl]", "N-073", "SpMax_A", "Psi_i_1d", "B04[C-Br]",
    "SdO", "TI2_L", "nCrt", "C-026", "F02[C-N]", "nHDon", "SpMax_B(m)",
    "Psi_i_A", "nN", "SM6_B(m)", "nArCOOR", "nX",
]

RESULT_COLS = ["set", "model", "settings", "leaves", "descriptors_used", "threshold",
               "cv_auc", "cv_auc_sd", "cv_bal_acc", "cv_recall", "cv_specificity",
               "cv_nrb_missed", "cv_rb_cleared", "thr_prec", "prec_recall",
               "prec_specificity", "prec_nrb_missed", "prec_rb_cleared", "time_s"]

TEST_COLS = ["set", "model", "test_auc", "test_bal_acc", "test_recall", "test_specificity",
             "nrb_missed", "rb_cleared", "prec_recall", "prec_specificity", "prec_nrb_missed",
             "cv_auc", "cv_bal_acc"]


# ---------------------------------------------------------------- data

def read_part(name):
    """The *_strat.xls files are ';'-separated text (not Excel), no header: 41 descriptors + class."""
    d = pd.read_csv(DATA_DIR / f"biodeg_{name}_strat.xls", sep=";", header=None)
    assert d.shape[1] == 42, f"{name} has {d.shape[1]} columns"
    d.columns = DESCRIPTORS + ["class"]
    d["class"] = d["class"].astype(str).str.strip()
    assert set(d["class"]) <= {"NRB", "RB"}
    return d


def load_strat(verbose=True):
    """Returns X_dev, y_dev, X_te, y_te (DataFrames of descriptors; y: 1 = NRB), fold, feature_sets."""
    dev = pd.concat([read_part("train"), read_part("val")], ignore_index=True)
    test = read_part("test")
    y_dev = (dev["class"] == "NRB").to_numpy(int)
    y_te = (test["class"] == "NRB").to_numpy(int)

    folds = pd.read_csv(STRAT_OUT / "cv_folds.csv")
    if len(folds) != len(dev) or (folds["class"].to_numpy() != dev["class"].to_numpy()).any():
        raise RuntimeError("outputs/strat/cv_folds.csv doesn't match the development set; "
                           "rerun src/julia/strat_setup.jl")
    fold = folds.sort_values("dev_row")["fold"].to_numpy(int)

    fs_tbl = pd.read_csv(STRAT_OUT / "feature_sets.csv")
    feature_sets = {s: fs_tbl.loc[fs_tbl["set"] == s].sort_values("order")["descriptor"].tolist()
                    for s in SETS}

    if verbose:
        for name, y in [("development", y_dev), ("test", y_te)]:
            print(f"{name:<12}{len(y)} chemicals | NRB {y.sum()} ({100 * y.mean():.1f}%) | "
                  f"RB {len(y) - y.sum()}")
        for s in SETS[:-1]:
            print(f"{s:<5} {', '.join(feature_sets[s])}")
    return dev[DESCRIPTORS], y_dev, test[DESCRIPTORS], y_te, fold, feature_sets


# ---------------------------------------------------------------- metrics (same as the notebook)

def metrics(p, y, t=0.5):
    pred = p >= t
    rec = np.sum(pred & (y == 1)) / np.sum(y == 1)
    spec = np.sum(~pred & (y == 0)) / np.sum(y == 0)
    return dict(acc=np.mean(pred == (y == 1)), recall=rec, specificity=spec,
                bal_acc=(rec + spec) / 2,
                nrb_missed=int(np.sum(~pred & (y == 1))), rb_cleared=int(np.sum(~pred & (y == 0))))


def best_threshold(p, y):
    cands = np.unique(np.append(p, 0.5))
    return cands[int(np.argmax([metrics(p, y, t)["bal_acc"] for t in cands]))]


def recall_threshold(p, y, target):
    """Highest threshold that still catches at least `target` of NRB."""
    for t in np.unique(p)[::-1]:
        if metrics(p, y, t)["recall"] >= target:
            return t
    return p.min()


def auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    return ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (
        len(pos) * len(neg))


# ---------------------------------------------------------------- CV driver (notebook's cv_generic)

def cv_generic(X_dev, y_dev, fold, fs, grid, fit_fn, predict_fn):
    """For each setting: fit on 4 folds, score the 5th, average fold AUCs. Pick the best mean
    AUC, refit on all development rows. Returns a dict like the notebook's NamedTuple, plus the
    whole grid's CV table."""
    t0 = time.time()
    res = []
    for prm in grid:
        oof, aucs = np.zeros(len(y_dev)), []
        for f in range(1, N_FOLDS + 1):
            tr, va = fold != f, fold == f
            m = fit_fn(X_dev.loc[tr, fs], y_dev[tr], prm)
            oof[va] = predict_fn(m, X_dev.loc[va, fs])
            aucs.append(auc(oof[va], y_dev[va]))
        res.append(dict(prm=prm, cv_auc=float(np.mean(aucs)), cv_sd=float(np.std(aucs, ddof=1)), oof=oof))
    ch = res[int(np.argmax([r["cv_auc"] for r in res]))]
    final = fit_fn(X_dev[fs], y_dev, ch["prm"])
    grid_tbl = pd.DataFrame([{**r["prm"], "cv_auc": r["cv_auc"], "cv_auc_sd": r["cv_sd"]} for r in res])
    return dict(predict=lambda X: predict_fn(final, X[fs]), model=final,
                settings=", ".join(f"{k}={v}" for k, v in ch["prm"].items()),
                cv_auc=ch["cv_auc"], cv_sd=ch["cv_sd"], oof=ch["oof"],
                grid=grid_tbl, time=time.time() - t0)


def result_row(s, model, r, y_dev, n_used, leaves=None):
    oof = r["oof"]
    thr = best_threshold(oof, y_dev)
    mt = metrics(oof, y_dev, thr)
    tp = recall_threshold(oof, y_dev, NRB_RECALL_TARGET)
    mp = metrics(oof, y_dev, tp)
    return dict(set=s, model=model, settings=r["settings"], leaves=leaves, descriptors_used=n_used,
                threshold=thr, cv_auc=r["cv_auc"], cv_auc_sd=r["cv_sd"], cv_bal_acc=mt["bal_acc"],
                cv_recall=mt["recall"], cv_specificity=mt["specificity"],
                cv_nrb_missed=mt["nrb_missed"], cv_rb_cleared=mt["rb_cleared"],
                thr_prec=tp, prec_recall=mp["recall"], prec_specificity=mp["specificity"],
                prec_nrb_missed=mp["nrb_missed"], prec_rb_cleared=mp["rb_cleared"],
                time_s=r["time"])


def test_row(row, p, y):
    mt = metrics(p, y, row["threshold"])
    mp = metrics(p, y, row["thr_prec"])
    return dict(set=row["set"], model=row["model"], test_auc=auc(p, y), test_bal_acc=mt["bal_acc"],
                test_recall=mt["recall"], test_specificity=mt["specificity"],
                nrb_missed=mt["nrb_missed"], rb_cleared=mt["rb_cleared"],
                prec_recall=mp["recall"], prec_specificity=mp["specificity"],
                prec_nrb_missed=mp["nrb_missed"], cv_auc=row["cv_auc"], cv_bal_acc=row["cv_bal_acc"])
