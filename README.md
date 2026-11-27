# How Much Does Interpretability Cost? Biodegradability Screening

How much accuracy does a readable model give up compared with a flexible one, and when? We predict ready biodegradability (UCI QSAR Biodegradation: 1,055 chemicals, 41 descriptors; NRB = not readily biodegradable = positive class) with descriptor budgets of k = 3, 5, 10 and all 41, plus two refinements:

1. **Choosing descriptors with a classification objective:** sparse logistic regression instead of best-subset least squares (Ch. 6 §6.2–6.3).
2. **Interpretable clustering and errors per descriptor group:** k-means on percentile-scaled descriptors, explained by a shallow tree (Ch. 20 §20.2).

## Models analysed (7)

| Readable | Flexible |
|---|---|
| OCT, OCT-H, CART, Logistic (ridge) | Random forest, XGBoost, Kernel regression (Nadaraya–Watson) |

Notebook 03 also fits an additive kernel and kernel logistic regression; their output files stay in `outputs/` but are **not used** in the analysis (`MODELS_USED` in `04_results.ipynb`).

## Run order

| Step | File | Produces |
|---|---|---|
| 0 | `DataPrep_ML.ipynb` (Python) | `data/biodeg_{train,val,test}_strat.xls`: stratified 60/20/20 split |
| 1 | `01_folds_and_descriptors.ipynb` | `outputs/folds.csv`, `feature_sets.csv`, `selection_stability.csv`, `descriptor_association.csv`; figures `01_*` |
| 2 | `02_trees_and_logistic.ipynb` | `cv_`, `oof_`, `test_` files for CART, OCT, OCT-H, Logistic; `depth_*.csv`; trees in `figures/trees/` |
| 3 | `03_kernels_and_ensembles.ipynb` | `cv_`, `oof_`, `test_` files for Nadaraya–Watson, Random forest, XGBoost (and the two unused kernel models) |
| 4 | `04_results.ipynb` | every results table and figure below; fits no model except the small tree that explains the groups |

- Notebooks 01–03 use the Julia (IAI) kernel and take a long time to run; steps 2 and 3 are independent. Notebook 04 only reads `outputs/` and reruns in a few minutes.
- Every notebook starts with `include("common.jl")` (shared settings, data loading, metrics, CV loop, output format, plot style). **Change shared settings only there.**
- Switches for notebook 04 are in its first cell: `MODELS_USED`, `MAIN_SETS`, `AUDIT_SETS`, `RUN_TEST`, `SHOW_ROC`.

**Requirements:** Julia 1.11 with Interpretable AI, packages `CSV`, `DataFrames`, `Plots`. Python with `numpy` and `pandas` for step 0.

## Design

- **Development set:** training + validation rows, 844 chemicals (559 NRB / 285 RB), `dev_row` 1–844. **Test set:** 211 chemicals (140 NRB / 71 RB).
- **Cross-validation:** 5 stratified folds; duplicate descriptor rows are kept in one fold (`outputs/folds.csv`).
- **Feature sets:** `reg_k3, log_k3, reg_k5, log_k5, reg_k10, log_k10, all`. `reg` = best-subset least squares (baseline), `log` = sparse logistic regression (Refinement 1). Both are chosen inside each training fold (folds 1–5) and once on all 844 chemicals for the final model (fold 0).
- **Tuning:** by mean 5-fold AUC; trees use the simplest setting within 0.01 AUC of the best.
- **Thresholds** (from out-of-fold scores, same rule for every model): *balanced* maximizes balanced accuracy; *NRB-first* is the highest threshold that still catches ≥ 90% of NRB. **RB cleared** = RB chemicals correctly let through, i.e. unnecessary lab tests avoided.
- **Test set:** scored by the final models in notebooks 02–03, evaluated only once in Part 6 of `04_results.ipynb` with the parameters frozen in Part 5.

## Results produced by `04_results.ipynb`

**Tables (`outputs/`)**

| File | Contents |
|---|---|
| `all_results.csv` | every model × feature set: CV AUC (+ s.d.), both thresholds, balanced accuracy, NRB caught, RB cleared, leaves, descriptors |
| `main_results.csv` | baseline: CV AUC and balanced accuracy, models × the 4 budgets |
| `refinement1.csv` | regression vs. logistic selector at k = 3, 5, 10: AUC, balanced accuracy, RB cleared, leaves, and differences |
| `refinement2_groups.csv` | descriptor groups: size, share NRB, median of key descriptors |
| `refinement2_audit.csv` | errors per group (`log_k3`, `all`): balanced accuracy, % NRB missed, % RB cleared |
| `final_parameters.csv` | frozen settings, final descriptor list and both thresholds per model × set |
| `test_results.csv` | test AUC, balanced accuracy, NRB caught, RB cleared at the frozen thresholds |
| `test_refinement2_audit.csv` | errors per group on the test set |

**Figures (`figures/`)**

| File | Shows |
|---|---|
| `04_A_auc_by_budget.png` | CV AUC of all 7 models at k = 3, 5, 10, all 41 |
| `04_B_rb_cleared_nrb_first.png` | RB cleared (unnecessary tests avoided) when each model catches ≥ 90% of NRB |
| `04_C_depth_tradeoff.png` | tree AUC vs. maximum depth, with the best flexible model as a dotted line |
| `04_D_refinement1_selectors.png` | readable models with the regression (hollow) vs. logistic (filled) selector |
| `04_E_group_profiles.png` | heatmap of what characterizes each descriptor group |
| `04_F_errors_by_group.png` | balanced accuracy of every model within each group |
| `04_G_test_vs_cv.png` | test AUC vs. CV AUC for every model and set |
| `04_H_roc.png` | optional ROC curves (only when `SHOW_ROC = true`) |
| `trees/04_group_tree.html` | the shallow tree that explains the groups |

`01_*` figures come from notebook 01; trees from notebook 02 are saved as interactive HTML in `figures/trees/`.

## Output format (for adding a model from any language)

A model fitted elsewhere appears in `04_results.ipynb` once its name is added to `MODELS_READ` or `MODELS_FLEX` (and to `STYLE` in `common.jl`) and it writes these files to `outputs/`, where `<model>` is the name in lower case with spaces and symbols replaced by `_` (e.g. `random_forest`):

- **`cv_<model>.csv`**, one row per set: `set, model, settings, leaves, descriptors_used, max_per_split, cv_auc, cv_auc_sd, time_s` (leave `leaves`/`max_per_split` empty for non-trees).
- **`oof_<model>.csv`**: `dev_row` (1–844), `fold` (must equal `outputs/folds.csv`), `set`, `model`, `score` = out-of-fold P(NRB).
- **`test_<model>.csv`**: `test_row` (1–211), `set`, `model`, `score` = P(NRB) from the final model fitted on all 844 chemicals with the fold-0 descriptor list.
- Optional for trees: `depth_<model>.csv` with `max_depth, cv_auc, cv_sd, cv_bal_acc, mean_leaves, chosen, set, model`.

Rules: use `outputs/folds.csv` and `outputs/feature_sets.csv` (fold *f* model uses that fold's list; the final model uses fold 0); choose settings by mean 5-fold AUC; standardize on the training part only.
