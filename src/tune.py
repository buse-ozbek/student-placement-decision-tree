"""Per-model hyperparameter tuning for the placement project.

Each model is tuned with cross-validated search over a small, student-readable
space. Parameter keys are prefixed ``clf__`` because the estimator lives under
the ``'clf'`` step of the shared Pipeline (see :mod:`src.models`).

Two sizes of search space are provided:

* ``SEARCH_SPACES``        -- the full spaces used for a real run.
* ``QUICK_SEARCH_SPACES``  -- smaller spaces for the ``--quick`` smoke test.

CART is small enough to search exhaustively with GridSearchCV; the ensembles use
RandomizedSearchCV so the run stays affordable. Best parameters for each model
are written to ``results/best_params/<name>.json`` for later features to reuse.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)

from src.models import build_model

# Project root resolved from this file so paths work from any cwd.
# This file is <root>/src/tune.py, so parents[1] is <root>.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Search spaces (keys prefixed 'clf__' to target the estimator in the Pipeline)
# ---------------------------------------------------------------------------

SEARCH_SPACES = {
    "CART": {
        "clf__max_depth": [3, 5, 8, 12, None],
        "clf__min_samples_leaf": [1, 5, 10, 20, 50],
        "clf__min_samples_split": [2, 10, 20, 50],
        "clf__ccp_alpha": [0.0, 0.0001, 0.001, 0.01],
    },
    "RandomForest": {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [5, 10, 20, None],
        "clf__max_features": ["sqrt", "log2", 0.5],
        "clf__min_samples_leaf": [1, 2, 5, 10],
    },
    "XGBoost": {
        "clf__n_estimators": [200, 400, 600, 800],
        "clf__max_depth": [3, 5, 7, 9],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__reg_lambda": [0.0, 1.0, 5.0, 10.0],
        "clf__reg_alpha": [0.0, 0.1, 1.0],
        "clf__min_child_weight": [1, 3, 5, 10],
    },
    "LightGBM": {
        "clf__n_estimators": [200, 400, 600, 800],
        "clf__num_leaves": [15, 31, 63, 127],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__reg_lambda": [0.0, 1.0, 5.0, 10.0],
        "clf__reg_alpha": [0.0, 0.1, 1.0],
        "clf__min_child_samples": [10, 20, 50, 100],
    },
    "CatBoost": {
        "clf__iterations": [200, 400, 600, 800],
        "clf__depth": [4, 6, 8, 10],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
    },
}

# Smaller spaces for the fast smoke run.
QUICK_SEARCH_SPACES = {
    "CART": {
        "clf__max_depth": [3, 5, None],
        "clf__min_samples_leaf": [1, 10],
    },
    "RandomForest": {
        "clf__n_estimators": [200],
        "clf__max_depth": [5, None],
        "clf__max_features": ["sqrt"],
    },
    "XGBoost": {
        "clf__n_estimators": [200],
        "clf__max_depth": [3, 5],
        "clf__learning_rate": [0.05, 0.1],
    },
    "LightGBM": {
        "clf__n_estimators": [200],
        "clf__num_leaves": [15, 31],
        "clf__learning_rate": [0.05, 0.1],
    },
    "CatBoost": {
        "clf__iterations": [200],
        "clf__depth": [4, 6],
        "clf__learning_rate": [0.05, 0.1],
    },
}


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------

def _jsonify(value):
    """Convert numpy scalar types to plain Python so json.dump works."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# ---------------------------------------------------------------------------
# Tuning routine
# ---------------------------------------------------------------------------

def tune_model(
    name,
    X,
    y,
    preprocessor,
    quick=False,
    random_state=RANDOM_STATE,
    results_dir=None,
):
    """Tune one model and persist its best parameters.

    Parameters
    ----------
    name : str
        Model key from :data:`src.models.MODEL_FACTORIES`.
    X, y : features / target to tune on.
    preprocessor : ColumnTransformer
        The shared preprocessor injected into the model Pipeline.
    quick : bool
        Use the smaller search space, fewer CV folds, and fewer random
        iterations for a fast smoke test.
    random_state : int
        Seed for the CV splitter and the randomized search.
    results_dir : str or Path, optional
        Where to write ``<name>.json``. Defaults to
        ``PROJECT_ROOT/results/best_params``.

    Returns
    -------
    tuple(estimator, dict)
        The refit best estimator and the best parameter dict.
    """
    # Build the base pipeline (shared preprocessor + classifier).
    base = build_model(name, preprocessor)

    # Pick the search space size.
    spaces = QUICK_SEARCH_SPACES if quick else SEARCH_SPACES
    param_space = spaces[name]

    # Cross-validation: fewer folds in quick mode.
    n_splits = 3 if quick else 5
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    # CatBoost's own progress is already silenced in its factory; keep the
    # search quiet too so logs stay readable.
    if name == "CART":
        # Small enough to search exhaustively.
        search = GridSearchCV(
            base,
            param_grid=param_space,
            scoring="roc_auc",
            cv=cv,
            refit=True,
            n_jobs=-1,
        )
    else:
        n_iter = 5 if quick else 25
        search = RandomizedSearchCV(
            base,
            param_distributions=param_space,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            refit=True,
            n_jobs=-1,
            random_state=random_state,
        )

    search.fit(X, y)

    best_estimator = search.best_estimator_
    best_params = search.best_params_

    # Resolve the output directory and write the best params as JSON.
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results" / "best_params"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    serializable = {k: _jsonify(v) for k, v in best_params.items()}
    out_path = results_dir / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    return best_estimator, best_params
