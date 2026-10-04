"""End-to-end orchestrator for the placement project.

This single script ties the whole pipeline together:

1. Load the data and make a stratified train/test split.
2. Build the shared preprocessor.
3. For each of the five models: tune hyperparameters, run repeated-CV on the
   training set, then evaluate once on the held-out test set.
4. Collect the CV summaries into ``results/model_comparison.csv``.
5. Run the class-weighting demo (XGBoost with / without ``scale_pos_weight``).
6. Generate every figure under ``results/figures/``.
7. Print a concise summary table to the console.

Run it either way::

    python -m src.run_all --quick
    python src/run_all.py --quick

``--quick`` shrinks the search spaces and CV scheme for a fast smoke test;
``--drop-scores`` reruns the sensitivity variant that excludes the three score
columns. All paths are resolved from this file, so the cwd does not matter.
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dual-entry guard: make `python src/run_all.py` work as well as `-m src.run_all`
# ---------------------------------------------------------------------------
# When run as a plain script (`python src/run_all.py`) there is no package
# context, so the `src.` imports below would fail. Put the project root on
# sys.path in that case so `import src.*` resolves either way.
if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data_preprocessing import (  # noqa: E402
    build_preprocessor,
    load_data,
    make_train_test_split,
)
from src.evaluate import (  # noqa: E402
    cross_validate_model,
    evaluate_holdout,
    weighting_demo,
)
from src.models import MODEL_FACTORIES  # noqa: E402
from src.plots import (  # noqa: E402
    IMPORTANCE_MODELS,
    plot_confusion_matrices,
    plot_cv_metric_bars,
    plot_feature_importance,
    plot_pr_overlay,
    plot_roc_overlay,
)
from src.tune import tune_model  # noqa: E402

# Project root resolved from this file so every output path is stable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Metrics shown in the printed summary and worth highlighting in the comparison.
KEY_METRICS = ["roc_auc", "average_precision", "f1", "recall", "balanced_accuracy"]


def parse_args(argv=None):
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(
        description="Run the full placement-prediction pipeline."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke run: smaller search spaces and CV scheme.",
    )
    parser.add_argument(
        "--drop-scores",
        action="store_true",
        help="Sensitivity run that drops the three score columns.",
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Generate the EDA figures and summary table before modelling.",
    )
    parser.add_argument(
        "--eda-only",
        action="store_true",
        help="Only generate the EDA outputs, then exit (skip modelling).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the whole pipeline and write every artifact under results/."""
    args = parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    mode = "QUICK smoke run" if args.quick else "FULL run"
    print(f"=== Placement pipeline: {mode} ===")
    if args.drop_scores:
        print("(score columns dropped for sensitivity analysis)")

    # 0. EDA (optional) -----------------------------------------------------
    # Run when --eda or --eda-only is passed. --eda-only exits right after.
    if args.eda or args.eda_only:
        print("\n--- Exploratory data analysis ---")
        from src.eda import run_eda

        eda_outputs = run_eda()
        for path in eda_outputs:
            print(f"  wrote {path}")
        if args.eda_only:
            print("\nDone (EDA only).")
            return 0

    # 1. Data + split -------------------------------------------------------
    df = load_data()
    X_train, X_test, y_train, y_test = make_train_test_split(
        df, drop_scores=args.drop_scores
    )
    print(
        f"Loaded {len(df)} rows; train={len(X_train)}, test={len(X_test)}, "
        f"train prevalence(Placed)={y_train.mean():.3f}"
    )

    # 2. Shared preprocessor ------------------------------------------------
    # Each model gets its OWN preprocessor instance (same recipe) so fitting one
    # pipeline never mutates another's transformer.
    def fresh_preprocessor():
        return build_preprocessor(drop_scores=args.drop_scores)

    # 3. Per-model: tune -> CV -> held-out --------------------------------
    comparison_rows = []
    holdout_results = {}
    fitted_pipelines = {}

    for name in MODEL_FACTORIES:
        print(f"\n--- {name} ---")

        # Tune hyperparameters (writes results/best_params/<name>.json).
        print("  tuning...")
        best_estimator, best_params = tune_model(
            name, X_train, y_train, fresh_preprocessor(), quick=args.quick
        )
        print(f"  best params: {best_params}")

        # Repeated-CV on the training set (writes results/cv_<name>.csv).
        print("  cross-validating...")
        summary_row = cross_validate_model(
            name,
            best_estimator,
            X_train,
            y_train,
            quick=args.quick,
            results_dir=RESULTS_DIR,
        )
        comparison_rows.append(summary_row)

        # Held-out test evaluation (refits on the full training set).
        print("  evaluating on held-out test set...")
        holdout = evaluate_holdout(
            name, best_estimator, X_train, y_train, X_test, y_test
        )
        holdout_results[name] = holdout
        # Keep the now-fitted pipeline for feature-importance plots.
        fitted_pipelines[name] = best_estimator
        print(
            f"  test roc_auc={holdout['roc_auc']:.3f} "
            f"f1={holdout['f1']:.3f} recall={holdout['recall']:.3f}"
        )

    # 4. Comparison table ---------------------------------------------------
    comparison_df = pd.concat(comparison_rows)
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=True)
    print(f"\nWrote {comparison_path}")

    # 5. Class-weighting demo ----------------------------------------------
    print("\n--- Class-weighting demo (XGBoost) ---")
    demo_df = weighting_demo(
        X_train,
        y_train,
        X_test,
        y_test,
        fresh_preprocessor(),
        quick=args.quick,
        results_dir=RESULTS_DIR,
    )
    print(demo_df.to_string(index=False))

    # 6. Figures ------------------------------------------------------------
    print("\n--- Generating figures ---")
    plot_roc_overlay(holdout_results, FIGURES_DIR / "roc_overlay.png")
    plot_pr_overlay(holdout_results, FIGURES_DIR / "pr_overlay.png")
    plot_cv_metric_bars(comparison_df, FIGURES_DIR / "cv_metric_bars.png")
    plot_confusion_matrices(
        holdout_results, FIGURES_DIR / "confusion_matrices.png"
    )
    for name in IMPORTANCE_MODELS:
        if name in fitted_pipelines:
            plot_feature_importance(
                name,
                fitted_pipelines[name],
                FIGURES_DIR / f"feature_importance_{name}.png",
            )
    print(f"Figures written under {FIGURES_DIR}")

    # 7. Console summary ----------------------------------------------------
    print("\n=== CV summary (mean) ===")
    mean_cols = [f"{m}_mean" for m in KEY_METRICS]
    summary_view = comparison_df[mean_cols].copy()
    summary_view.columns = KEY_METRICS
    print(summary_view.round(3).to_string())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
