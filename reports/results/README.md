# Results snapshot

Out-of-fold cross-validation results read by `reports/results_notebook.ipynb`, so the notebook runs
without re-fitting any model. `outputs/` is gitignored; these are committed copies.

| File | Source |
|---|---|
| `cv_results_all_models.csv` | `outputs/strat/cv_results_all_models.csv` (RF, XGBoost, kernel logistic, Nadaraya–Watson, additive kernel, OCT-H) |
| `oof_<model>.csv` | `outputs/strat/<model>/oof_predictions.csv` |
| `oct_h_depth_curve.csv` | `outputs/strat/oct_h/cv_depth_curve.csv` |
| `additive_kernel_selected.csv` | `outputs/strat/additive_kernel/selected_features.csv` |
| `cv_folds.csv`, `feature_sets.csv` | `outputs/strat/` (written by `src/julia/strat_setup.jl`) |
| `notebook_models_cv.csv` | CART / OCT / ridge logistic numbers copied by hand from `MLProjectPart1_Orig.ipynb` (Parts 7–8 outputs) |

## Refreshing after re-running the models

```sh
julia +1.11.6/IAI/3.2.2 src/julia/strat_setup.jl
python3 src/python/strat_models.py
julia +1.11.6/IAI/3.2.2 -t auto src/julia/strat_oct_h.jl
julia +1.11.6/IAI/3.2.2 -t auto src/julia/strat_additive_kernel.jl

python3 -c "import pandas as pd; ms=['random_forest','xgboost','kernel_logistic','nadaraya_watson','additive_kernel','oct_h']; pd.concat([pd.read_csv(f'outputs/strat/{m}/cv_results.csv') for m in ms]).to_csv('outputs/strat/cv_results_all_models.csv', index=False)"

R=reports/results
cp outputs/strat/cv_results_all_models.csv outputs/strat/cv_folds.csv outputs/strat/feature_sets.csv $R/
for m in random_forest xgboost kernel_logistic nadaraya_watson additive_kernel oct_h; do
  cp outputs/strat/$m/oof_predictions.csv $R/oof_$m.csv
done
cp outputs/strat/oct_h/cv_depth_curve.csv $R/oct_h_depth_curve.csv
cp outputs/strat/additive_kernel/selected_features.csv $R/additive_kernel_selected.csv
```

Then re-run the notebook top to bottom.
