"""Figure 1 of the combined report: cross-validated AUC by descriptor budget.

Values are mean 5-fold CV AUC on the 844 development chemicals: CART / OCT / ridge logistic from
MLProjectPart1_Orig.ipynb (Part 7), the rest from outputs/strat/cv_results_all_models.csv.
"Best flexible model" is the highest of random forest, XGBoost and kernel logistic regression
at each budget (all on the fixed descriptor lists).

Usage (from the repo root):  python3 reports/figures/make_auc_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).with_name("auc_by_budget.pdf")

BUDGETS = ["3", "5", "10", "all 41"]
REFERENCE = ("Best flexible model", [0.855, 0.917, 0.932, 0.943])
SERIES = [  # (label, values, color, marker) — categorical slots 1-4, fixed order
    ("Ridge logistic", [0.830, 0.881, 0.910, 0.931], "#2a78d6", "o"),
    ("Additive kernel (own descriptors)", [0.878, 0.904, 0.919, 0.936], "#eb6834", "s"),
    ("OCT-H", [0.806, 0.866, 0.866, 0.875], "#1baf7a", "D"),
    ("OCT", [0.792, 0.842, 0.830, 0.834], "#eda100", "^"),
]

TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
})

fig, ax = plt.subplots(figsize=(6.0, 3.3))
x = range(len(BUDGETS))

ax.plot(x, REFERENCE[1], color="#8a8984", lw=1.5, ls=(0, (4, 3)), marker="o", ms=4,
        mfc="white", mec="#8a8984", mew=1.2, label=REFERENCE[0], zorder=2)
for label, ys, color, marker in SERIES:
    ax.plot(x, ys, color=color, lw=2, marker=marker, ms=6, mec="white", mew=1.2,
            label=label, zorder=3)

ax.set_xticks(list(x), BUDGETS)
ax.set_xlabel("Descriptors allowed")
ax.set_ylabel("Mean 5-fold CV AUC")
ax.set_ylim(0.78, 0.955)
ax.set_xlim(-0.25, len(BUDGETS) - 0.75)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.tick_params(length=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=8,
          handlelength=2.6, columnspacing=1.6)

fig.tight_layout()
fig.savefig(OUT)
print(f"wrote {OUT}")
