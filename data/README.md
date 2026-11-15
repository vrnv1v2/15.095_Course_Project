# Data setup

Download the source data from the [UCI QSAR Biodegradation dataset page](https://doi.org/10.24432/C5H60M) and put it in `data/raw/`. Record the original filename used.

Export the existing split as `data/splits.csv` with columns `row_id,split`, using `train`, `validation`, and `test` for split names. Record exactly how row IDs map to the source data before sharing the split. Use IDs only for alignment.

Verify 1,055 rows and 41 descriptors, split counts of 633 / 210 / 212, and that identical descriptor rows are assigned to the same split.
