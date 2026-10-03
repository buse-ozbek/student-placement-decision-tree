"""Model factories for the placement project.

Every model is wrapped in an sklearn ``Pipeline`` whose first step is the single
shared preprocessor from :mod:`src.data_preprocessing` and whose second step is
the classifier itself::

    Pipeline([('prep', preprocessor), ('clf', estimator)])

Putting the same preprocessor in front of every estimator keeps the five models
directly comparable: they all see the exact same transformed features. The
factory functions take the preprocessor as an argument so the tuning and
evaluation code can build (or rebuild) the preprocessor once and inject it.

The five models:

* CART  -- a single DecisionTreeClassifier, the interpretable starting point.
* RandomForest, XGBoost -- known baselines we have used before.
* LightGBM, CatBoost    -- the new / focus boosters for this course.

Class imbalance (~78% Placed) is handled per model with each library's own
weighting option so the minority "Not Placed" class is not ignored.
"""

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data_preprocessing import build_preprocessor

# Single random seed used everywhere so results are reproducible.
RANDOM_STATE = 42


def compute_scale_pos_weight(y):
    """Return the XGBoost ``scale_pos_weight`` value for the target ``y``.

    XGBoost does not take ``class_weight='balanced'``; instead it accepts a
    single ratio of negative to positive examples. The convention is
    ``count(negative) / count(positive)``, which up-weights the minority class.

    Parameters
    ----------
    y : array-like of 0/1 labels.

    Returns
    -------
    float
        ``n_negative / n_positive``.
    """
    y = list(y)
    n_pos = sum(1 for v in y if v == 1)
    n_neg = sum(1 for v in y if v == 0)
    # Guard against divide-by-zero on degenerate subsamples.
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_cart(preprocessor, class_weight="balanced", **kw):
    """Build a CART (single decision tree) pipeline."""
    clf = DecisionTreeClassifier(
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        **kw,
    )
    return Pipeline([("prep", preprocessor), ("clf", clf)])


def make_random_forest(preprocessor, class_weight="balanced", n_jobs=-1, **kw):
    """Build a RandomForest pipeline (known baseline)."""
    clf = RandomForestClassifier(
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
        **kw,
    )
    return Pipeline([("prep", preprocessor), ("clf", clf)])


def make_xgboost(preprocessor, scale_pos_weight=None, n_jobs=-1, **kw):
    """Build an XGBoost pipeline (known baseline).

    Pass ``scale_pos_weight`` (see :func:`compute_scale_pos_weight`) to handle
    class imbalance; leaving it ``None`` lets XGBoost treat the classes evenly.
    """
    clf = XGBClassifier(
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
        **kw,
    )
    return Pipeline([("prep", preprocessor), ("clf", clf)])


def make_lightgbm(preprocessor, class_weight="balanced", n_jobs=-1, **kw):
    """Build a LightGBM pipeline (new / focus model)."""
    clf = LGBMClassifier(
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
        verbose=-1,
        **kw,
    )
    return Pipeline([("prep", preprocessor), ("clf", clf)])


def make_catboost(preprocessor, auto_class_weights="Balanced", **kw):
    """Build a CatBoost pipeline (new / focus model).

    NOTE: CatBoost runs on the *shared* preprocessor output (one-hot / ordinal /
    scaled columns) rather than its own native categorical handling. We do this
    on purpose so every model sees identical features and the comparison stays
    fair. ``verbose=0`` and ``allow_writing_files=False`` keep it quiet and stop
    it from writing a ``catboost_info/`` directory into the working folder.
    """
    clf = CatBoostClassifier(
        auto_class_weights=auto_class_weights,
        random_state=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,
        **kw,
    )
    return Pipeline([("prep", preprocessor), ("clf", clf)])


# ---------------------------------------------------------------------------
# Registry and convenience builder
# ---------------------------------------------------------------------------

# Name -> factory. This dict drives the tuning and evaluation loops so adding a
# model is a one-line change here.
MODEL_FACTORIES = {
    "CART": make_cart,
    "RandomForest": make_random_forest,
    "XGBoost": make_xgboost,
    "LightGBM": make_lightgbm,
    "CatBoost": make_catboost,
}


def build_model(name, preprocessor=None, **kw):
    """Build a model Pipeline by name.

    Parameters
    ----------
    name : str
        One of the keys in :data:`MODEL_FACTORIES`.
    preprocessor : ColumnTransformer, optional
        The shared preprocessor to put in front of the estimator. A fresh one is
        built via :func:`build_preprocessor` when omitted.
    **kw
        Extra keyword arguments forwarded to the chosen factory / estimator.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if name not in MODEL_FACTORIES:
        raise KeyError(
            f"Unknown model '{name}'. Choose one of: {list(MODEL_FACTORIES)}"
        )
    if preprocessor is None:
        preprocessor = build_preprocessor()
    return MODEL_FACTORIES[name](preprocessor, **kw)
