import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "biodeg.csv"
SEED = 42

# 1. Detect delimiter from the first line
with open(path, newline="") as f:
    sep = csv.Sniffer().sniff(f.readline(), delimiters=",;\t|").delimiter

# 2. Detect header: a column that is numeric in the body but not in row 0 means row 0 is a header
raw = pd.read_csv(path, sep=sep, header=None, dtype=str)
body = raw.iloc[1:]
numeric_cols = [
    c for c in raw.columns
    if (pd.to_numeric(body[c], errors="coerce").notna() | body[c].isna()).all()
]
has_header = any(pd.isna(pd.to_numeric(raw.loc[0, c], errors="coerce")) for c in numeric_cols)
df = pd.read_csv(path, sep=sep, header=0 if has_header else None)

# 3. Give identical rows the same group id so duplicates stay together
group = df.groupby(list(df.columns), sort=False, dropna=False).ngroup()

# 4. Shuffle the groups, then assign them to splits by cumulative row count (60/20/20)
rng = np.random.default_rng(SEED)
ids = rng.permutation(group.unique())
sizes = group.value_counts()[ids]
frac = (sizes.cumsum() - sizes / 2) / len(df)
bucket = pd.cut(frac, [0, 0.6, 0.8, 1.0], labels=["train", "val", "test"])
row_split = group.map(bucket)

# 5. Write one CSV per split (same delimiter, header only if the input had one)
stem = Path(path).stem
for name in ["train", "val", "test"]:
    part = df[row_split == name].sample(frac=1, random_state=SEED)
    part.to_csv(f"{stem}_{name}.csv", sep=sep, index=False, header=has_header)
    print(f"{stem}_{name}.csv: {len(part)} rows")