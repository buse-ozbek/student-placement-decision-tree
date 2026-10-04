"""Exploratory data analysis (EDA) for the placement project.

This module produces a small, focused set of report-ready figures and a
summary-statistics table. It deliberately stays lean: a handful of plots that
each support a specific claim in the report, rather than a large gallery.

Figures are written as PNGs at ``dpi=150`` under ``results/figures/eda/`` and
the numeric summary table as ``results/eda_summary.csv``.

Outputs:

1. ``target_balance.png``       -- Placed vs Not Placed class balance (the
   78/22 imbalance the report discusses).
2. ``numeric_distributions.png``-- histograms of the key numeric predictors,
   useful for the skewness / Box-Cox discussion in the guideline.
3. ``placement_rate_by_category.png`` -- placement rate across the categorical
   predictors (Major, University_Year, English_Proficiency, ...).
4. ``numeric_correlation.png``  -- correlation heatmap of the numeric
   predictors, supporting the PCA discussion.
5. ``eda_summary.csv``          -- describe() table of the numeric predictors.

All plots reuse :func:`load_data` and the column constants from
``data_preprocessing`` so the EDA sees exactly the modelling view of the data
(same target encoding, same leakage columns excluded).

Run standalone::

    python -m src.eda
    python src/eda.py

or as part of the pipeline via ``python -m src.run_all --eda``.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dual-entry guard so `python src/eda.py` works as well as `-m src.eda`.
# ---------------------------------------------------------------------------
if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib  # noqa: E402

# Non-interactive backend: saving figures never needs a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (after backend select)
import numpy as np  # noqa: E402

from src.data_preprocessing import (  # noqa: E402
    LEAKAGE_COLS,
    NOMINAL_COLS,
    NUMERIC_COLS,
    ORDINAL_SPEC,
    TARGET,
    load_data,
)

# Project root resolved from this file so paths work from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
EDA_FIG_DIR = RESULTS_DIR / "figures" / "eda"

# Standard DPI for every saved figure (matches src/plots.py).
DPI = 150

# A muted green palette consistent across the EDA figures.
C_PLACED = "#4F6B52"
C_NOTPLACED = "#B45A4B"
C_BAR = "#4F6B52"
C_ACCENT = "#26352A"


def _ensure_parent(path):
    """Make sure a figure's parent directory exists; return a Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 1. Target balance
# ---------------------------------------------------------------------------

def plot_target_balance(df, path):
    """Bar chart of the Placed vs Not Placed class counts with percentages."""
    path = _ensure_parent(path)
    counts = df[TARGET].value_counts()
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = [C_PLACED if lbl == "Placed" else C_NOTPLACED for lbl in counts.index]
    bars = ax.bar(counts.index.astype(str), counts.values, color=colors, width=0.6)
    for bar, value in zip(bars, counts.values):
        pct = 100 * value / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{value:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=C_ACCENT,
        )
    ax.set_title("Target balance: Placement_Status", fontweight="bold")
    ax.set_ylabel("Number of students")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. Numeric distributions
# ---------------------------------------------------------------------------

def plot_numeric_distributions(df, path, cols=None):
    """Grid of histograms for the numeric predictors.

    Each subplot is annotated with the attribute's skewness, which motivates the
    report's note about Box-Cox / power transforms for skewed inputs.
    """
    path = _ensure_parent(path)
    if cols is None:
        cols = [c for c in NUMERIC_COLS if c in df.columns]

    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        series = df[col].dropna()
        ax.hist(series, bins=30, color=C_BAR, edgecolor="white", linewidth=0.4)
        skew = series.skew()
        ax.set_title(f"{col}\n(skew={skew:.2f})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide any unused axes in the final row.
    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Numeric predictor distributions", fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Placement rate by category
# ---------------------------------------------------------------------------

def plot_placement_rate_by_category(df, path, cols=None):
    """For each categorical predictor, show the placement rate per level.

    A horizontal reference line marks the overall placement rate so it is easy
    to see which categories place above or below average.
    """
    path = _ensure_parent(path)
    if cols is None:
        # Nominal predictors plus the two ordinal ones.
        cols = [c for c in NOMINAL_COLS if c in df.columns]
        cols += [c for c in ORDINAL_SPEC if c in df.columns]

    placed = df[TARGET].eq("Placed").astype(int)
    overall = placed.mean()

    n = len(cols)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        rate = placed.groupby(df[col]).mean().sort_values(ascending=False)
        # Keep ordinal columns in their natural order where known.
        if col in ORDINAL_SPEC:
            order = [c for c in ORDINAL_SPEC[col] if c in rate.index]
            rate = rate.reindex(order)
        ax.bar(rate.index.astype(str), rate.values, color=C_BAR, width=0.65)
        ax.axhline(overall, color=C_NOTPLACED, linestyle="--", linewidth=1,
                   label=f"overall {overall:.2f}")
        ax.set_title(f"Placement rate by {col}", fontsize=10, fontweight="bold")
        ax.set_ylabel("P(Placed)")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 4. Numeric correlation heatmap
# ---------------------------------------------------------------------------

def plot_numeric_correlation(df, path, cols=None):
    """Correlation heatmap of the numeric predictors (supports the PCA case)."""
    path = _ensure_parent(path)
    if cols is None:
        cols = [c for c in NUMERIC_COLS if c in df.columns]

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(0.6 * len(cols) + 2, 0.6 * len(cols) + 2))
    im = ax.imshow(corr.values, cmap="BrBG", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    # Annotate each cell with the rounded correlation.
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=5,
                    color="black" if abs(corr.values[i, j]) < 0.6 else "white")
    ax.set_title("Numeric predictor correlations", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 5. Summary statistics table
# ---------------------------------------------------------------------------

def write_summary_table(df, path, cols=None):
    """Write describe() statistics for the numeric predictors to CSV."""
    path = _ensure_parent(path)
    if cols is None:
        cols = [c for c in NUMERIC_COLS if c in df.columns]
    summary = df[cols].describe().T
    summary["skew"] = df[cols].skew()
    summary.to_csv(path, index=True)
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_eda(df=None, fig_dir=EDA_FIG_DIR, results_dir=RESULTS_DIR):
    """Generate every EDA figure and the summary table.

    Loads the data if ``df`` is None. The leakage columns are excluded from the
    plots (we never explore attributes we won't model on), but the target column
    is kept so placement rates can be computed.

    Returns
    -------
    list[pathlib.Path]
        Paths of all artifacts written.
    """
    if df is None:
        df = load_data()

    # Drop leakage columns up front so EDA matches the modelling view, but keep
    # the target itself (needed for class balance and placement rates).
    leak = [c for c in LEAKAGE_COLS if c in df.columns and c != TARGET]
    df = df.drop(columns=leak)

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    outputs.append(plot_target_balance(df, fig_dir / "target_balance.png"))
    outputs.append(
        plot_numeric_distributions(df, fig_dir / "numeric_distributions.png")
    )
    outputs.append(
        plot_placement_rate_by_category(
            df, fig_dir / "placement_rate_by_category.png"
        )
    )
    outputs.append(
        plot_numeric_correlation(df, fig_dir / "numeric_correlation.png")
    )
    outputs.append(write_summary_table(df, results_dir / "eda_summary.csv"))
    return outputs


def main():
    """Standalone entry point: run the EDA and report what was written."""
    print("=== EDA: generating figures and summary table ===")
    outputs = run_eda()
    for path in outputs:
        print(f"  wrote {path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
