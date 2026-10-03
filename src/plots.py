"""Figure generation for the placement project.

Every figure is written as a PNG at ``dpi=150`` under ``results/figures/``. The
functions take already-computed results (held-out evaluation dicts, the model
comparison table, fitted pipelines) and turn them into the plots the report
uses:

* :func:`plot_roc_overlay`          -- ROC curves of every model on one axis.
* :func:`plot_pr_overlay`           -- precision/recall curves overlaid.
* :func:`plot_cv_metric_bars`       -- CV mean +/- std bars for roc_auc and f1.
* :func:`plot_confusion_matrices`   -- a grid of per-model confusion matrices.
* :func:`plot_feature_importance`   -- impurity/gain importances for one of the
  tree-ensemble models, mapped back to readable feature names.

Matplotlib runs headless (the ``Agg`` backend) so this works on a server or in
a plain script with no display.
"""

from pathlib import Path

import matplotlib

# Use a non-interactive backend so saving figures never needs a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (import after backend select)
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

# Project root resolved from this file so paths work from any cwd.
# This file is <root>/src/plots.py, so parents[1] is <root>.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Standard DPI for every saved figure.
DPI = 150

# Only these four models expose ``feature_importances_`` that we plot.
IMPORTANCE_MODELS = ("RandomForest", "XGBoost", "LightGBM", "CatBoost")


def _ensure_parent(path):
    """Make sure the figure's parent directory exists; return a Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# ROC / PR overlays
# ---------------------------------------------------------------------------

def plot_roc_overlay(holdout_results, path):
    """Overlay the ROC curve of every model on a single axis.

    Parameters
    ----------
    holdout_results : dict[str, dict]
        Maps model name -> the dict returned by
        :func:`src.evaluate.evaluate_holdout` (needs ``y_test`` and
        ``y_proba``).
    path : str or Path
        Where to save the PNG.
    """
    path = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in holdout_results.items():
        RocCurveDisplay.from_predictions(
            res["y_test"],
            res["y_proba"],
            name=name,
            ax=ax,
        )
    # Chance line for reference.
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_title("ROC curves (held-out test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_pr_overlay(holdout_results, path):
    """Overlay the precision/recall curve of every model on one axis."""
    path = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in holdout_results.items():
        PrecisionRecallDisplay.from_predictions(
            res["y_test"],
            res["y_proba"],
            name=name,
            ax=ax,
        )
    ax.set_title("Precision-Recall curves (held-out test set)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CV metric bars
# ---------------------------------------------------------------------------

def plot_cv_metric_bars(comparison_df, path):
    """Bar chart of CV mean +/- std for roc_auc and f1 across models.

    Parameters
    ----------
    comparison_df : pandas.DataFrame
        Indexed by model name with ``<metric>_mean`` / ``<metric>_std`` columns
        (as produced by :func:`src.evaluate.cross_validate_model`).
    path : str or Path
    """
    path = _ensure_parent(path)
    metrics = ["roc_auc", "f1"]
    models = list(comparison_df.index)
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 3, 6))
    for i, metric in enumerate(metrics):
        means = comparison_df[f"{metric}_mean"].to_numpy()
        stds = comparison_df[f"{metric}_std"].to_numpy()
        ax.bar(
            x + (i - 0.5) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=metric,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Cross-validated scores (mean +/- std)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Confusion matrices grid
# ---------------------------------------------------------------------------

def plot_confusion_matrices(holdout_results, path):
    """Draw every model's held-out confusion matrix in one grid figure.

    Parameters
    ----------
    holdout_results : dict[str, dict]
        Maps model name -> evaluate_holdout dict (needs ``confusion_matrix``).
    path : str or Path
    """
    path = _ensure_parent(path)
    names = list(holdout_results.keys())
    n = len(names)

    # Lay the panels out in a roughly square grid.
    ncols = min(3, n) if n else 1
    nrows = int(np.ceil(n / ncols)) if n else 1

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.6 * nrows), squeeze=False
    )
    # Row = true class, column = predicted class; label 0 = Not Placed.
    tick_labels = ["Not Placed (0)", "Placed (1)"]

    for idx, name in enumerate(names):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        cm = np.asarray(holdout_results[name]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(tick_labels, rotation=20, ha="right")
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        # Annotate each cell with its count.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # Pick a readable text color against the cell shade.
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(
                    j, i, str(cm[i, j]), ha="center", va="center", color=color
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused panels in the grid.
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.suptitle("Confusion matrices (held-out test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(name, fitted_pipeline, path, top_n=20):
    """Plot the top feature importances for one tree-ensemble model.

    Reads ``feature_importances_`` off the ``'clf'`` step and maps the values to
    the transformed feature names via the ``'prep'`` ColumnTransformer's
    ``get_feature_names_out``. Only RandomForest / XGBoost / LightGBM / CatBoost
    expose these importances; other models are skipped with no file written.

    Parameters
    ----------
    name : str
        Model name (used only in the title / skip check).
    fitted_pipeline : sklearn Pipeline
        A pipeline already fitted (steps ``'prep'`` and ``'clf'``).
    path : str or Path
    top_n : int
        How many of the most important features to show.

    Returns
    -------
    bool
        True if a figure was written, False if the model was skipped.
    """
    if name not in IMPORTANCE_MODELS:
        return False

    clf = fitted_pipeline.named_steps["clf"]
    prep = fitted_pipeline.named_steps["prep"]

    importances = getattr(clf, "feature_importances_", None)
    if importances is None:
        return False

    feature_names = prep.get_feature_names_out()
    importances = np.asarray(importances)

    # Rank and keep the top_n features by importance.
    order = np.argsort(importances)[::-1][:top_n]
    top_names = np.asarray(feature_names)[order]
    top_values = importances[order]

    path = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(top_names) + 2))
    # Highest importance at the top of the chart.
    y_pos = np.arange(len(top_names))[::-1]
    ax.barh(y_pos, top_values, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.set_xlabel("importance")
    ax.set_title(f"{name}: top {len(top_names)} feature importances")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return True
