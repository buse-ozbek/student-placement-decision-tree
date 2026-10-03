"""Data loading, feature typing, and preprocessing for the placement project.

This module centralizes every decision about the data so the model code stays
simple:

* which columns leak the label and must be dropped,
* which columns are nominal / ordinal / numeric,
* how the target is encoded,
* the single shared ``ColumnTransformer`` every model reuses, and
* the stratified train/test split.

Keeping all of this in one place means each model Pipeline just plugs the same
preprocessor in front of its estimator.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# ---------------------------------------------------------------------------
# Paths and column constants
# ---------------------------------------------------------------------------

# Resolve the project root from this file so scripts work from any directory.
# This file lives at <root>/src/data_preprocessing.py, so parents[1] is <root>.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The label we predict (binary: Placed vs Not Placed).
TARGET = "Placement_Status"

# Default location of the dataset CSV.
DATA_PATH = PROJECT_ROOT / "student_career_success_dataset.csv"

# Columns that leak the answer. Each one takes a fixed sentinel value exactly
# when a student is Not Placed, so keeping them would let a model "cheat".
# Student_ID is just a per-row identifier with no signal.
LEAKAGE_COLS = [
    "Student_ID",
    "Company_Tier",
    "Career_Field",
    "Placement_Mode",
    "Starting_Salary_USD",
]

# Three score columns are kept by default (their values overlap across classes,
# so they are not trivial leakers). Flip DROP_SCORE_COLS to True to run a
# sensitivity check that excludes them, in case they are derived from the label.
SCORE_COLS = ["Employability_Score", "Interview_Score", "Resume_Score"]
DROP_SCORE_COLS = False

# Nominal (unordered) categorical features -> one-hot encoded.
NOMINAL_COLS = [
    "Gender",
    "University_Year",
    "Major",
    "GitHub_Profile",
    "Leadership_Experience",
    "LinkedIn_Profile",
]

# Ordinal (naturally ordered) categoricals -> ordinal encoded in this order.
ORDINAL_SPEC = {
    "Academic_Performance": ["Poor", "Average", "Good", "Excellent"],
    "English_Proficiency": ["Basic", "Intermediate", "Advanced"],
}

# Numeric features -> standard-scaled. Note the three score columns are listed
# here; get_feature_lists() removes them when drop_scores is True.
NUMERIC_COLS = [
    "Age",
    "Attendance_Percentage",
    "Study_Hours_Per_Week",
    "CGPA",
    "Programming_Skill",
    "Projects_Completed",
    "Certifications",
    "Hackathons",
    "Internships",
    "Resume_Score",
    "Communication_Skills",
    "Teamwork",
    "Problem_Solving",
    "Interview_Score",
    "Employability_Score",
]


# ---------------------------------------------------------------------------
# Loading and feature typing
# ---------------------------------------------------------------------------

def load_data(path=None):
    """Read the dataset CSV into a DataFrame.

    Parameters
    ----------
    path : str or Path, optional
        CSV location. Defaults to :data:`DATA_PATH`.

    Returns
    -------
    pandas.DataFrame
    """
    if path is None:
        path = DATA_PATH
    return pd.read_csv(path)


def get_feature_lists(drop_scores=DROP_SCORE_COLS):
    """Return the (nominal, ordinal_cols, numeric) column lists.

    When ``drop_scores`` is True the three score columns are removed from the
    numeric list; the nominal and ordinal lists are unchanged.

    Returns
    -------
    tuple(list, list, list)
        ``(nominal_cols, ordinal_cols, numeric_cols)``.
    """
    nominal = list(NOMINAL_COLS)
    ordinal_cols = list(ORDINAL_SPEC.keys())
    if drop_scores:
        numeric = [c for c in NUMERIC_COLS if c not in SCORE_COLS]
    else:
        numeric = list(NUMERIC_COLS)
    return nominal, ordinal_cols, numeric


# ---------------------------------------------------------------------------
# Shared preprocessor
# ---------------------------------------------------------------------------

def build_preprocessor(drop_scores=DROP_SCORE_COLS):
    """Build the shared ColumnTransformer used by every model Pipeline.

    * One-hot encode the nominal categoricals (unknown categories ignored).
    * Ordinal encode the two ordered categoricals using their explicit order
      (unknown categories map to -1 instead of raising).
    * Standard-scale the numeric columns.

    Anything not listed is dropped (``remainder='drop'``). Column order is kept
    deterministic so transformed output is reproducible.
    """
    nominal, ordinal_cols, numeric = get_feature_lists(drop_scores=drop_scores)

    # Category order per ordinal column, kept in the fixed ordinal_cols order.
    ordinal_categories = [ORDINAL_SPEC[col] for col in ordinal_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "nominal",
                OneHotEncoder(handle_unknown="ignore"),
                nominal,
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_cols,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def make_train_test_split(
    df=None,
    drop_scores=DROP_SCORE_COLS,
    test_size=0.2,
    random_state=42,
):
    """Build a stratified train/test split of features and encoded target.

    Steps:

    1. Load the data if ``df`` is None.
    2. Encode the target: ``Placed -> 1``, ``Not Placed -> 0``.
    3. Drop the leakage columns and the target from the features.
    4. Optionally drop the three score columns.
    5. Stratified split on the target so both classes keep their proportions.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)`` as pandas objects.
    """
    # Imported here to keep the module import light for callers that only need
    # the constants or the preprocessor.
    from sklearn.model_selection import train_test_split

    if df is None:
        df = load_data()

    # Encode the target to 1 / 0.
    y = df[TARGET].map({"Placed": 1, "Not Placed": 0})

    # Features: drop the leakage columns and the target itself.
    drop_cols = LEAKAGE_COLS + [TARGET]
    X = df.drop(columns=drop_cols)

    # Optional sensitivity check: drop the score columns too.
    if drop_scores:
        X = X.drop(columns=[c for c in SCORE_COLS if c in X.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
