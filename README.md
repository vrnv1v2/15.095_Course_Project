# ML Biodegradability Pilot

MIT 15.095 project on interpretable biodegradability screening with Optimal Classification Trees.

Can a shallow optimal classification tree perform comparably to flexible models and help prioritize chemicals for follow-up testing under a fixed budget?

## Data and split

Use the [UCI QSAR Biodegradation dataset](https://doi.org/10.24432/C5H60M). See [data/README.md](data/README.md) for where to put the data and the existing split.

## Modeling and evaluation

Compare depth-2 and depth-3 OCT with OCT-H, CART, logistic regression, random forest, XGBoost, and an always-NRB baseline. Set NRB = 1 and RB = 0. Fit on training data, choose settings on validation, and evaluate the finalized models on test.

Export predictions as `row_id,score_nrb,predicted_label` so every model can use a shared evaluator. Report accuracy, NRB recall, RB specificity, balanced accuracy, and confusion matrices. For screening, compare NRB discoveries and precision at 10%, 20%, and 30% budgets against repeated random selection. Use seeded random tie-breaking for equal scores.

## Code

- `src/julia/` for IAI OCT and OCT-H.
- `src/python/` for Python baselines and shared evaluation if you use Python.
- `outputs/` for local experiment outputs, created when needed.
