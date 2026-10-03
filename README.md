# Student Placement Prediction — Decision Trees & Boosting

A course project that predicts whether a student gets placed (`Placement_Status`)
from the `student_career_success_dataset.csv` dataset. This is a **binary
classification** task: the target is encoded as `Placed = 1` and `Not Placed = 0`.

The dataset is imbalanced on purpose (~78% of students are placed), which is why
the project also looks at class weighting rather than just raw accuracy.

## Models

The project compares five tree-based models:

| Model | Role |
| --- | --- |
| **CART** (scikit-learn `DecisionTreeClassifier`) | Interpretable starter — a single, readable tree. |
| **RandomForest** | Known baseline (used in a previous course). |
| **XGBoost** | Known baseline (used in a previous course). |
| **LightGBM** | New / focus model — something not covered before. |
| **CatBoost** | New / focus model — something not covered before. |

RandomForest and XGBoost are included as baselines; LightGBM and CatBoost are the
newer gradient-boosting libraries the course wants us to learn, and CART gives an
interpretable reference point.

## Dropped leakage columns

Five columns are dropped before training because each one is a dead giveaway for
the label — every value is a fixed sentinel exactly when a student is *Not Placed*:

- `Student_ID` — a per-row identifier, no predictive signal.
- `Company_Tier` — equals `"No Company"` iff the student is Not Placed.
- `Career_Field` — equals `"Not Placed"` iff the student is Not Placed.
- `Placement_Mode` — equals `"Not Applicable"` iff the student is Not Placed.
- `Starting_Salary_USD` — equals `0` iff the student is Not Placed.

Keeping any of these would let the model "cheat" by reading the answer straight
from a feature, so they are removed in `make_train_test_split`.

### A note on the three score columns

`Employability_Score`, `Interview_Score`, and `Resume_Score` are **kept by
default**. Their values overlap between placed and not-placed students, so they
are not trivial leakers. If you suspect they are computed from the outcome, set
`DROP_SCORE_COLS = True` (in `src/data_preprocessing.py`) to run the pipeline
without them as a sensitivity check.

## Encoding

A single shared scikit-learn `ColumnTransformer` preprocesses every model's input:

- **Ordinal encoding** for the two naturally ordered categoricals, with an
  explicit order: `Academic_Performance` (Poor < Average < Good < Excellent) and
  `English_Proficiency` (Basic < Intermediate < Advanced).
- **One-hot encoding** (`handle_unknown='ignore'`) for the other nominal
  categoricals: `Gender`, `University_Year`, `Major`, `GitHub_Profile`,
  `Leadership_Experience`, `LinkedIn_Profile`.
- **StandardScaler** for all numeric columns.

## Setup

Use the project virtual environment's Python for everything:

```bash
# macOS one-time: OpenMP runtime for XGBoost / LightGBM
brew install libomp

# dependencies (already installed in .venv)
.venv/bin/python -m pip install -r requirements.txt
```

Always invoke the project Python as `.venv/bin/python` (not bare `python`).

## How to run

```bash
# full run
.venv/bin/python -m src.run_all

# fast smoke run (smaller search, quicker)
.venv/bin/python -m src.run_all --quick
```

Both `python -m src.run_all` and `python src/run_all.py` work.

## Results artifacts

A full run writes to the gitignored `results/` directory:

- `results/model_comparison.csv` — metrics for every model side by side.
- `results/best_params/<Model>.json` — tuned hyperparameters per model.
- `results/cv_<Model>.csv` — cross-validation scores per model.
- `results/weighting_demo.csv` — class-weighting comparison.
- `results/figures/*.png` — ROC/PR overlays, CV metric bars, confusion
  matrices, and per-model feature-importance plots.
