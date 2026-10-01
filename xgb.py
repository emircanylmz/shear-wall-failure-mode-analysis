# -*- coding: utf-8 -*-
"""Leakage-safe XGBoost benchmark for the shear-wall database."""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "shear_wall_matplotlib")
)

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, __version__ as xgboost_version

from main import (
    CLASS_LABELS,
    FAILURE_NAMES,
    GROUP_COL,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    build_evaluation_groups,
    choose_balanced_group_splits,
    classification_metrics,
    configure_reproducibility,
    file_sha256,
    group_bootstrap_intervals,
    load_and_validate_data,
    package_versions,
    save_json,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Author-grouped XGBoost benchmark.")
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "processed" / "Shear_Wall_Database_clean.xlsx",
        help="Input .xlsx file containing a Database sheet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "xgboost_analysis",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def make_pipeline(seed: int, estimators: int) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "section",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
    model = XGBClassifier(
        n_estimators=estimators,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=len(CLASS_LABELS),
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_metrics: pd.DataFrame,
    importance: pd.DataFrame,
    output_path: Path,
) -> None:
    class_names = [FAILURE_NAMES[label] for label in CLASS_LABELS]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("XGBoost — Author-Grouped Cross-Validation", fontsize=14, fontweight="bold")

    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
        ax=axes[0], cmap="Blues", colorbar=False
    )
    axes[0].set_title("Out-of-fold confusion matrix")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(
        fold_metrics["fold"].astype(str),
        fold_metrics["f1_macro"],
        color="#4472C4",
        alpha=0.85,
    )
    mean_f1 = fold_metrics["f1_macro"].mean()
    axes[1].axhline(
        mean_f1,
        color="#C00000",
        linestyle="--",
        label=f"Mean={mean_f1:.3f}",
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Fold-level performance")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    ordered = importance.sort_values("mean_permutation_importance")
    axes[2].barh(
        ordered["feature"],
        ordered["mean_permutation_importance"],
        xerr=ordered["std_across_folds"],
        color="#70AD47",
        alpha=0.85,
    )
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Macro-F1 decrease after permutation")
    axes[2].set_title("Held-out permutation importance")
    axes[2].grid(axis="x", alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_xgboost(args: argparse.Namespace) -> dict[str, Any]:
    configure_reproducibility(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data.resolve()

    frame = load_and_validate_data(data_path)
    groups = build_evaluation_groups(frame)
    X = frame[MODEL_FEATURES]
    y = frame[TARGET_COL].to_numpy(dtype=int)
    y_zero_based = y - 1

    estimators = 80 if args.quick else 300
    permutation_repeats = 3 if args.quick else 20
    bootstrap_repeats = 100 if args.quick else 1_000
    pipeline_template = make_pipeline(args.seed, estimators)
    splits, split_seed = choose_balanced_group_splits(
        X, y, groups, n_splits=5, seed=args.seed
    )

    out_of_fold_pred = np.zeros(len(frame), dtype=int)
    out_of_fold_prob = np.zeros((len(frame), len(CLASS_LABELS)), dtype=float)
    fold_rows: list[dict[str, Any]] = []
    importance_rows: list[np.ndarray] = []

    for fold, (train_index, test_index) in enumerate(splits, start=1):
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        if train_groups & test_groups:
            raise AssertionError("Evaluation groups overlap between train and test.")

        pipeline = clone(pipeline_template)
        weights = compute_sample_weight("balanced", y_zero_based[train_index])
        pipeline.fit(
            X.iloc[train_index],
            y_zero_based[train_index],
            model__sample_weight=weights,
        )
        predicted_zero = pipeline.predict(X.iloc[test_index]).astype(int)
        probabilities = pipeline.predict_proba(X.iloc[test_index])
        predictions = predicted_zero + 1
        out_of_fold_pred[test_index] = predictions
        out_of_fold_prob[test_index] = probabilities

        metrics = classification_metrics(y[test_index], predictions, probabilities)
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "test_rows": len(test_index),
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                "test_class_counts": np.bincount(
                    y[test_index] - 1, minlength=4
                ).tolist(),
                **metrics,
            }
        )

        importance = permutation_importance(
            pipeline,
            X.iloc[test_index],
            y_zero_based[test_index],
            scoring="f1_macro",
            n_repeats=permutation_repeats,
            random_state=args.seed + fold,
            n_jobs=1,
        )
        importance_rows.append(importance.importances_mean)

    overall = classification_metrics(y, out_of_fold_pred, out_of_fold_prob)
    intervals = group_bootstrap_intervals(
        y,
        out_of_fold_pred,
        groups,
        repeats=bootstrap_repeats,
        seed=args.seed,
    )
    report = classification_report(
        y,
        out_of_fold_pred,
        labels=CLASS_LABELS,
        target_names=[FAILURE_NAMES[label] for label in CLASS_LABELS],
        output_dict=True,
        zero_division=0,
    )

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(output_dir / "xgb_grouped_fold_metrics.csv", index=False)
    predictions = frame[["RowID", "SourceID", GROUP_COL, "Specimen", TARGET_COL]].copy()
    predictions["EvaluationGroup"] = groups
    predictions["PredictedFailureMode"] = out_of_fold_pred
    predictions["Correct"] = (y == out_of_fold_pred).astype(int)
    for position, label in enumerate(CLASS_LABELS):
        predictions[f"Probability_{FAILURE_NAMES[label]}"] = out_of_fold_prob[:, position]
    predictions.to_csv(output_dir / "xgb_grouped_oof_predictions.csv", index=False)

    importance_matrix = np.vstack(importance_rows)
    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "mean_permutation_importance": importance_matrix.mean(axis=0),
            "std_across_folds": importance_matrix.std(axis=0),
        }
    ).sort_values("mean_permutation_importance", ascending=False)
    importance.to_csv(output_dir / "xgb_grouped_permutation_importance.csv", index=False)
    plot_results(
        y,
        out_of_fold_pred,
        fold_frame,
        importance,
        output_dir / "xgb_grouped_results.png",
    )

    final_pipeline = clone(pipeline_template)
    final_weights = compute_sample_weight("balanced", y_zero_based)
    final_pipeline.fit(X, y_zero_based, model__sample_weight=final_weights)
    joblib.dump(final_pipeline, output_dir / "xgb_pipeline_zero_based.joblib")

    metadata = {
        "data_file": data_path.name,
        "data_sha256": file_sha256(data_path),
        "evaluation": "Author and repeated-feature-vector grouped cross-validation",
        "split_seed_selected_from_label_balance_only": split_seed,
        "n_splits": 5,
        "n_groups": int(len(np.unique(groups))),
        "features": MODEL_FEATURES,
        "classes": FAILURE_NAMES,
        "class_imbalance": "Fold-local balanced sample weights; no synthetic categorical interpolation",
        "saved_pipeline_prediction_codes": {
            "0": "Flexural / FailureMode 1",
            "1": "Shear / FailureMode 2",
            "2": "Flex-Shear / FailureMode 3",
            "3": "Sliding / FailureMode 4",
        },
        "fold_metrics": fold_rows,
        "overall_oof_metrics": overall,
        "group_bootstrap_95_intervals": intervals,
        "classification_report": report,
        "versions": {**package_versions(), "xgboost": xgboost_version},
    }
    save_json(output_dir / "xgb_metadata.json", metadata)

    print("\nXGBoost analysis complete")
    print(f"Rows: {len(frame)}, evaluation groups: {len(np.unique(groups))}")
    print(f"Grouped OOF macro F1: {overall['f1_macro']:.3f}")
    print(f"Outputs: {output_dir}")
    return metadata


def main() -> None:
    run_xgboost(parse_args())


if __name__ == "__main__":
    main()
