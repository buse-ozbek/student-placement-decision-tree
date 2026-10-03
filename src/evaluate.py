"""Model evaluation for the placement project.

Three pieces live here:

* :func:`cross_validate_model` -- repeated stratified cross-validation on the
  training data, aggregated to a mean/std summary row per metric.
* :func:`evaluate_holdout`     -- fit on the full training set, score on the
  untouched test set, and return everything the plots need (metrics, confusion
  matrix, predicted probabilities, true labels).
* :func:`weighting_demo`       -- a small experiment showing how class weighting
  changes recall on the minority class, using XGBoost with and without
  ``scale_pos_weight``.

The target is encoded ``Placed=1`` / ``Not Placed=0`` (see
:mod:`src.data_preprocessing`), so **label 0 = "Not Placed" is the minority
class** (~22% of rows). All imbalance-aware metrics below focus on that class.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from src.models import compute_scale_pos_weight, make_xgboost

# Project root resolved from this file so paths work from any cwd.
# This file is <root>/src/evaluate.py, so parents[1] is <root>.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANDOM_STATE = 42

# Label 0 ("Not Placed") is the minority class we care about catching.
MINORITY_LABEL = 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Metric name -> sklearn scorer (string names reuse sklearn's built-ins; the
# probability-based ones need explicit scorers). These are the columns that end
# up in every CV summary row.
SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(name, estimator, X, y, quick=False, results_dir=None):
    """Repeated stratified cross-validation for one fitted-estimator recipe.

    Runs :func:`sklearn.model_selection.cross_validate` with every metric in
    :data:`SCORING` over a :class:`RepeatedStratifiedKFold`
    (5 splits x 3 repeats for a full run, 3 splits x 1 repeat in quick mode).
    The per-fold scores are aggregated to a mean and std per metric.

    The full per-fold table is saved to ``results/cv_<name>.csv`` and a one-row
    summary (``<metric>_mean`` / ``<metric>_std`` columns) is returned so the
    orchestrator can stack the models into a comparison table.

    Parameters
    ----------
    name : str
        Model name, used for the output filename and the summary row index.
    estimator : sklearn estimator / Pipeline
        The (tuned) model to evaluate. It is cloned per fold by sklearn.
    X, y : training features / target.
    quick : bool
        Use the smaller CV scheme for the smoke test.
    results_dir : str or Path, optional
        Directory for ``cv_<name>.csv``. Defaults to ``PROJECT_ROOT/results``.

    Returns
    -------
    pandas.DataFrame
        A single-row summary indexed by ``name``.
    """
    if quick:
        n_splits, n_repeats = 3, 1
    else:
        n_splits, n_repeats = 5, 3

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )

    cv_results = cross_validate(
        estimator,
        X,
        y,
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )

    # Keep the raw per-fold test scores (one column per metric) for the CSV.
    per_fold = pd.DataFrame(
        {metric: cv_results[f"test_{metric}"] for metric in SCORING}
    )
    per_fold.index.name = "fold"

    # Resolve the output directory and persist the per-fold table.
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(results_dir / f"cv_{name}.csv", index=True)

    # Aggregate to a one-row mean/std summary indexed by the model name.
    summary = {}
    for metric in SCORING:
        summary[f"{metric}_mean"] = per_fold[metric].mean()
        summary[f"{metric}_std"] = per_fold[metric].std()
    summary_df = pd.DataFrame([summary], index=[name])
    summary_df.index.name = "model"
    return summary_df


# ---------------------------------------------------------------------------
# Held-out test evaluation
# ---------------------------------------------------------------------------

def evaluate_holdout(name, estimator, X_train, y_train, X_test, y_test):
    """Fit on the training set and score once on the held-out test set.

    Returns a dict carrying both the scalar metrics and the arrays the plotting
    code needs (confusion matrix, predicted probabilities, true labels). The
    probability column used for ROC / PR is the probability of the positive
    class (label 1 = "Placed").

    Returns
    -------
    dict
        Keys: ``name``, metric names (accuracy, balanced_accuracy, precision,
        recall, f1, roc_auc, average_precision), ``confusion_matrix`` (2x2
        numpy array), ``y_proba`` (positive-class probabilities), ``y_test``.
    """
    estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_test)
    # Probability of the positive class (column for label 1).
    y_proba = estimator.predict_proba(X_test)[:, 1]

    y_test_arr = np.asarray(y_test)

    result = {
        "name": name,
        "accuracy": accuracy_score(y_test_arr, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test_arr, y_pred),
        "precision": precision_score(y_test_arr, y_pred),
        "recall": recall_score(y_test_arr, y_pred),
        "f1": f1_score(y_test_arr, y_pred),
        "roc_auc": roc_auc_score(y_test_arr, y_proba),
        "average_precision": average_precision_score(y_test_arr, y_proba),
        "confusion_matrix": confusion_matrix(y_test_arr, y_pred),
        "y_proba": y_proba,
        "y_test": y_test_arr,
    }
    return result


# ---------------------------------------------------------------------------
# Class-weighting demonstration
# ---------------------------------------------------------------------------

def weighting_demo(
    X_train,
    y_train,
    X_test,
    y_test,
    preprocessor,
    quick=False,
    results_dir=None,
):
    """Show how class weighting changes minority-class recall.

    Trains XGBoost twice on identical data and features:

    * **weighted**   -- ``scale_pos_weight = n_negative / n_positive`` so the
      minority class is up-weighted, and
    * **unweighted** -- ``scale_pos_weight = 1`` (classes treated evenly).

    For each we measure recall on the minority class (label 0 = "Not Placed"),
    i.e. the share of truly-not-placed students the model actually catches, and
    report the difference. Results are written to ``results/weighting_demo.csv``
    with columns ``model, weighted, minority_recall`` plus a ``delta`` row-pair
    captured in a final ``delta`` column.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : the split data.
    preprocessor : ColumnTransformer
        Shared preprocessor injected into both XGBoost pipelines.
    quick : bool
        Use fewer trees for a fast smoke run.
    results_dir : str or Path, optional
        Directory for ``weighting_demo.csv``. Defaults to
        ``PROJECT_ROOT/results``.

    Returns
    -------
    pandas.DataFrame
        The table written to disk.
    """
    # Fewer trees keeps the quick smoke run fast; this demo is about the effect
    # of weighting, not peak accuracy, so the exact count does not matter much.
    n_estimators = 100 if quick else 300

    # Up-weight the minority class: ratio of negatives to positives.
    pos_weight = compute_scale_pos_weight(y_train)

    def _minority_recall(scale_pos_weight):
        """Fit XGBoost with the given weight and return recall on label 0."""
        model = make_xgboost(
            preprocessor,
            scale_pos_weight=scale_pos_weight,
            n_estimators=n_estimators,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # pos_label=0 -> recall for the "Not Placed" minority class.
        return recall_score(
            np.asarray(y_test), y_pred, pos_label=MINORITY_LABEL
        )

    weighted_recall = _minority_recall(pos_weight)
    unweighted_recall = _minority_recall(1)

    # Difference = how much weighting improved (or hurt) minority recall. The
    # same delta is attached to both rows so a single-column read is easy.
    delta = weighted_recall - unweighted_recall

    table = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "weighted": True,
                "minority_recall": weighted_recall,
                "delta": delta,
            },
            {
                "model": "XGBoost",
                "weighted": False,
                "minority_recall": unweighted_recall,
                "delta": delta,
            },
        ]
    )

    if results_dir is None:
        results_dir = PROJECT_ROOT / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "weighting_demo.csv", index=False)

    return table
