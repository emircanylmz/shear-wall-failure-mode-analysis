# -*- coding: utf-8 -*-
"""Shear-wall failure-mode analysis with leakage-safe evaluation.

The script has three independent responsibilities:
1. Data-quality anomaly review on the complete research database.
2. Author-grouped, fold-local Random Forest evaluation.
3. A Fourier-activation MLP with disjoint train/validation/test author groups.

Anomaly flags are descriptive quality-review signals. They are not silently
removed from the classifier data and are not presented as ground truth.
"""

from __future__ import annotations

import os
import tempfile

# Set numerical-library limits before NumPy, sklearn, or PyTorch imports.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "shear_wall_matplotlib")
)

import argparse
import copy
import hashlib
import json
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn import __version__ as imblearn_version
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
NUMERIC_FEATURES = [
    "M/Vlw",
    "lw/tw",
    "ρvwFy,vw/fc",
    "ρhwFy,vw/fc",
    "ρvcFy,vc/fc",
    "ρhcFy,hc/fc",
    "P/fcAg",
    "Ab/Ag",
]
CATEGORICAL_FEATURES = ["Section"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "FailureMode"
GROUP_COL = "Author"
CLASS_LABELS = [1, 2, 3, 4]
FAILURE_NAMES = {
    1: "Flexural",
    2: "Shear",
    3: "Flex-Shear",
    4: "Sliding",
}


@dataclass(frozen=True)
class RunConfig:
    seed: int = SEED
    cv_splits: int = 5
    rf_estimators: int = 300
    permutation_repeats: int = 20
    bootstrap_repeats: int = 1_000
    mlp_epochs: int = 250
    mlp_patience: int = 30
    mlp_hidden: int = 64
    mlp_dropout: float = 0.25
    anomaly_contamination: float = 0.08
    lof_neighbors: int = 20


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Leakage-safe shear-wall classification and anomaly review."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "processed" / "Shear_Wall_Database_clean.xlsx",
        help="Input .xlsx file containing a Database sheet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "shear_wall_analysis",
        help="Directory for models, metrics, predictions, and figures.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller settings for a fast smoke run.",
    )
    parser.add_argument(
        "--skip-mlp",
        action="store_true",
        help="Run data checks, anomalies, and Random Forest only.",
    )
    return parser.parse_args()


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        # Some older backends do not support all deterministic kernels.
        pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def load_and_validate_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    frame = pd.read_excel(path, sheet_name="Database")
    if "RowID" not in frame.columns:
        frame.insert(0, "RowID", np.arange(1, len(frame) + 1, dtype=int))
    if "SourceID" not in frame.columns:
        legacy_id = "Unnamed: 0"
        if legacy_id in frame.columns:
            frame.insert(1, "SourceID", frame[legacy_id])
        else:
            frame.insert(1, "SourceID", frame["RowID"])

    required = ["RowID", "SourceID", GROUP_COL, "Specimen", TARGET_COL, *MODEL_FEATURES]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    frame = frame.loc[:, required].copy()
    if frame.empty:
        raise ValueError("The Database sheet contains no records.")
    if not frame["RowID"].is_unique:
        raise ValueError("RowID must be unique in the cleaned analysis frame.")
    if frame[required].isna().any().any():
        missing = frame[required].isna().sum()
        raise ValueError(f"Missing required values: {missing[missing.gt(0)].to_dict()}")

    for column in ["RowID", "SourceID", TARGET_COL, *NUMERIC_FEATURES]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    numeric_values = frame[NUMERIC_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Model features contain non-finite values.")
    if (frame[NUMERIC_FEATURES] < 0).any().any():
        raise ValueError("Ratio features must not be negative.")
    if (frame[["M/Vlw", "lw/tw"]] <= 0).any().any():
        raise ValueError("M/Vlw and lw/tw must be strictly positive.")
    if not set(frame[TARGET_COL].astype(int)).issubset(CLASS_LABELS):
        raise ValueError(f"FailureMode must contain only {CLASS_LABELS}.")
    if not set(frame["Section"].astype(str)).issubset({"R", "B", "F"}):
        raise ValueError("Section must contain only R, B, or F.")
    if frame.duplicated([GROUP_COL, "Specimen"]).any():
        duplicated = frame.loc[
            frame.duplicated([GROUP_COL, "Specimen"], keep=False),
            [GROUP_COL, "Specimen"],
        ]
        raise ValueError(
            "Author + Specimen must be unique. Duplicates: "
            f"{duplicated.head(10).to_dict(orient='records')}"
        )

    frame[TARGET_COL] = frame[TARGET_COL].astype(int)
    frame["Section"] = frame["Section"].astype(str)
    frame[GROUP_COL] = frame[GROUP_COL].astype(str)
    frame["Specimen"] = frame["Specimen"].astype(str)
    return frame


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def build_evaluation_groups(frame: pd.DataFrame) -> np.ndarray:
    """Keep each author and each repeated feature vector in one group."""
    union_find = UnionFind(len(frame))
    for _, indices in frame.groupby(GROUP_COL, sort=False).indices.items():
        indices = list(indices)
        for index in indices[1:]:
            union_find.union(indices[0], index)

    for _, indices in frame.groupby(MODEL_FEATURES, sort=False, dropna=False).indices.items():
        indices = list(indices)
        for index in indices[1:]:
            union_find.union(indices[0], index)

    roots = [union_find.find(index) for index in range(len(frame))]
    root_to_group = {root: f"G{position:03d}" for position, root in enumerate(sorted(set(roots)), 1)}
    return np.asarray([root_to_group[root] for root in roots])


def split_balance_score(splits: Iterable[tuple[np.ndarray, np.ndarray]], y: np.ndarray) -> float:
    concrete_splits = list(splits)
    global_distribution = np.bincount(
        y - min(CLASS_LABELS), minlength=len(CLASS_LABELS)
    ) / len(y)
    expected_fraction = 1.0 / len(concrete_splits)
    score = 0.0
    for _, test_index in concrete_splits:
        test_counts = np.bincount(
            y[test_index] - min(CLASS_LABELS), minlength=len(CLASS_LABELS)
        )
        if np.any(test_counts == 0):
            return float("inf")
        test_distribution = test_counts / len(test_index)
        score += abs(len(test_index) / len(y) - expected_fraction)
        score += float(np.abs(test_distribution - global_distribution).mean())
    return score


def choose_balanced_group_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    best_splits: list[tuple[np.ndarray, np.ndarray]] | None = None
    best_seed = seed
    best_score = float("inf")
    for candidate_seed in range(seed, seed + 100):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=candidate_seed,
        )
        candidate = list(splitter.split(X, y, groups))
        score = split_balance_score(candidate, y)
        if score < best_score:
            best_score = score
            best_splits = candidate
            best_seed = candidate_seed
    if best_splits is None:
        raise RuntimeError("Could not construct group-disjoint folds containing every class.")
    return best_splits, best_seed


def choose_group_holdout(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    splits, selected_seed = choose_balanced_group_splits(X, y, groups, n_splits, seed)
    global_distribution = np.bincount(
        y - min(CLASS_LABELS), minlength=len(CLASS_LABELS)
    ) / len(y)

    def fold_score(split: tuple[np.ndarray, np.ndarray]) -> float:
        _, test_index = split
        counts = np.bincount(
            y[test_index] - min(CLASS_LABELS), minlength=len(CLASS_LABELS)
        )
        distribution = counts / len(test_index)
        return (
            abs(len(test_index) / len(y) - 1.0 / n_splits)
            + float(np.abs(distribution - global_distribution).mean())
        )

    train_index, test_index = min(splits, key=fold_score)
    return train_index, test_index, selected_seed


def make_rf_pipeline(config: RunConfig) -> ImbPipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "section",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return ImbPipeline(
        steps=[
            ("preprocess", preprocessing),
            (
                "smote",
                SMOTENC(
                    categorical_features=[len(NUMERIC_FEATURES)],
                    random_state=config.seed,
                    k_neighbors=5,
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=config.rf_estimators,
                    min_samples_split=4,
                    random_state=config.seed,
                    n_jobs=1,
                ),
            ),
        ]
    )


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0
        ),
    }
    if probabilities is not None and set(np.unique(y_true)) == set(CLASS_LABELS):
        metrics["roc_auc_ovr_macro"] = roc_auc_score(
            y_true,
            probabilities,
            labels=CLASS_LABELS,
            multi_class="ovr",
            average="macro",
        )
    return {key: float(value) for key, value in metrics.items()}


def group_bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    samples: dict[str, list[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "f1_macro": [],
    }
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    for _ in range(repeats):
        selected_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_rows = np.concatenate([group_rows[group] for group in selected_groups])
        sampled_true = y_true[sampled_rows]
        sampled_pred = y_pred[sampled_rows]
        samples["accuracy"].append(accuracy_score(sampled_true, sampled_pred))
        samples["balanced_accuracy"].append(
            balanced_accuracy_score(sampled_true, sampled_pred)
        )
        samples["f1_macro"].append(
            f1_score(
                sampled_true,
                sampled_pred,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        )
    return {
        metric: {
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
        }
        for metric, values in samples.items()
    }


def evaluate_random_forest(
    frame: pd.DataFrame,
    groups: np.ndarray,
    config: RunConfig,
    output_dir: Path,
) -> tuple[ImbPipeline, dict[str, Any]]:
    X = frame[MODEL_FEATURES]
    y = frame[TARGET_COL].to_numpy(dtype=int)
    splits, split_seed = choose_balanced_group_splits(
        X, y, groups, config.cv_splits, config.seed
    )

    out_of_fold_pred = np.zeros(len(frame), dtype=int)
    out_of_fold_prob = np.zeros((len(frame), len(CLASS_LABELS)), dtype=float)
    fold_metrics: list[dict[str, Any]] = []
    importances: list[np.ndarray] = []
    base_pipeline = make_rf_pipeline(config)

    for fold, (train_index, test_index) in enumerate(splits, start=1):
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        if train_groups & test_groups:
            raise AssertionError("Evaluation groups overlap between train and test.")

        pipeline = clone(base_pipeline)
        pipeline.fit(X.iloc[train_index], y[train_index])
        predictions = pipeline.predict(X.iloc[test_index]).astype(int)
        probabilities = pipeline.predict_proba(X.iloc[test_index])
        out_of_fold_pred[test_index] = predictions
        out_of_fold_prob[test_index] = probabilities

        metrics = classification_metrics(y[test_index], predictions, probabilities)
        fold_metrics.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "test_rows": len(test_index),
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                "test_class_counts": np.bincount(
                    y[test_index] - 1, minlength=len(CLASS_LABELS)
                ).tolist(),
                **metrics,
            }
        )

        importance = permutation_importance(
            pipeline,
            X.iloc[test_index],
            y[test_index],
            scoring="f1_macro",
            n_repeats=config.permutation_repeats,
            random_state=config.seed + fold,
            n_jobs=1,
        )
        importances.append(importance.importances_mean)

    overall = classification_metrics(y, out_of_fold_pred, out_of_fold_prob)
    intervals = group_bootstrap_intervals(
        y,
        out_of_fold_pred,
        groups,
        repeats=config.bootstrap_repeats,
        seed=config.seed,
    )
    report = classification_report(
        y,
        out_of_fold_pred,
        labels=CLASS_LABELS,
        target_names=[FAILURE_NAMES[label] for label in CLASS_LABELS],
        output_dict=True,
        zero_division=0,
    )

    fold_frame = pd.DataFrame(fold_metrics)
    fold_frame.to_csv(output_dir / "rf_grouped_fold_metrics.csv", index=False)

    prediction_frame = frame[["RowID", "SourceID", GROUP_COL, "Specimen", TARGET_COL]].copy()
    prediction_frame["EvaluationGroup"] = groups
    prediction_frame["PredictedFailureMode"] = out_of_fold_pred
    prediction_frame["Correct"] = (
        prediction_frame[TARGET_COL] == prediction_frame["PredictedFailureMode"]
    ).astype(int)
    for position, label in enumerate(CLASS_LABELS):
        prediction_frame[f"Probability_{FAILURE_NAMES[label]}"] = out_of_fold_prob[:, position]
    prediction_frame.to_csv(output_dir / "rf_grouped_oof_predictions.csv", index=False)

    importance_matrix = np.vstack(importances)
    importance_frame = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "mean_permutation_importance": importance_matrix.mean(axis=0),
            "std_across_folds": importance_matrix.std(axis=0),
        }
    ).sort_values("mean_permutation_importance", ascending=False)
    importance_frame.to_csv(output_dir / "rf_grouped_permutation_importance.csv", index=False)

    plot_random_forest_results(
        y,
        out_of_fold_pred,
        fold_frame,
        importance_frame,
        report,
        output_dir / "01_grouped_rf_results.png",
    )

    final_pipeline = clone(base_pipeline)
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, output_dir / "rf_pipeline.joblib")

    metadata = {
        "evaluation": "Author and repeated-feature-vector grouped cross-validation",
        "split_seed_selected_from_label_balance_only": split_seed,
        "n_splits": config.cv_splits,
        "n_groups": int(len(np.unique(groups))),
        "features": MODEL_FEATURES,
        "classes": FAILURE_NAMES,
        "fold_metrics": fold_metrics,
        "overall_oof_metrics": overall,
        "group_bootstrap_95_intervals": intervals,
        "classification_report": report,
    }
    save_json(output_dir / "rf_metadata.json", metadata)
    return final_pipeline, metadata


def plot_random_forest_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_frame: pd.DataFrame,
    importance_frame: pd.DataFrame,
    report: dict[str, Any],
    output_path: Path,
) -> None:
    class_names = [FAILURE_NAMES[label] for label in CLASS_LABELS]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Random Forest — Author-Grouped Cross-Validation", fontsize=14, fontweight="bold")

    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
        ax=axes[0, 0], cmap="Blues", colorbar=False
    )
    axes[0, 0].set_title("Out-of-fold confusion matrix")
    axes[0, 0].tick_params(axis="x", rotation=25)

    axes[0, 1].bar(
        fold_frame["fold"].astype(str),
        fold_frame["f1_macro"],
        color="#4472C4",
        alpha=0.85,
    )
    axes[0, 1].axhline(
        fold_frame["f1_macro"].mean(),
        color="#C00000",
        linestyle="--",
        label=f"Mean={fold_frame['f1_macro'].mean():.3f}",
    )
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xlabel("Fold")
    axes[0, 1].set_ylabel("Macro F1")
    axes[0, 1].set_title("Fold-level performance")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.25)

    ordered = importance_frame.sort_values("mean_permutation_importance")
    axes[1, 0].barh(
        ordered["feature"],
        ordered["mean_permutation_importance"],
        xerr=ordered["std_across_folds"],
        color="#70AD47",
        alpha=0.85,
    )
    axes[1, 0].axvline(0, color="black", linewidth=0.8)
    axes[1, 0].set_xlabel("Macro-F1 decrease after permutation")
    axes[1, 0].set_title("Held-out permutation importance")
    axes[1, 0].grid(axis="x", alpha=0.25)

    f1_values = [report[name]["f1-score"] for name in class_names]
    supports = [int(report[name]["support"]) for name in class_names]
    bars = axes[1, 1].bar(class_names, f1_values, color="#5B9BD5", alpha=0.85)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("Out-of-fold F1")
    axes[1, 1].set_title("Class F1 with actual support")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].grid(axis="y", alpha=0.25)
    for bar, support in zip(bars, supports):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={support}",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def run_anomaly_quality_review(
    frame: pd.DataFrame,
    config: RunConfig,
    output_dir: Path,
) -> dict[str, Any]:
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "section",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
    transformed = preprocessing.fit_transform(frame[MODEL_FEATURES])

    isolation = IsolationForest(
        n_estimators=300,
        contamination=config.anomaly_contamination,
        random_state=config.seed,
        n_jobs=1,
    )
    iso_flag = isolation.fit_predict(transformed) == -1
    iso_score = isolation.decision_function(transformed)

    lof = LocalOutlierFactor(
        n_neighbors=config.lof_neighbors,
        contamination=config.anomaly_contamination,
        n_jobs=1,
    )
    lof_flag = lof.fit_predict(transformed) == -1
    lof_score = lof.negative_outlier_factor_
    consensus_flag = iso_flag & lof_flag

    anomaly_frame = frame[["RowID", "SourceID", GROUP_COL, "Specimen", TARGET_COL]].copy()
    anomaly_frame["IsolationForestFlag"] = iso_flag.astype(int)
    anomaly_frame["IsolationForestDecisionScore"] = iso_score
    anomaly_frame["LOFFlag"] = lof_flag.astype(int)
    anomaly_frame["LOFNegativeOutlierFactor"] = lof_score
    anomaly_frame["ConsensusQualityReviewFlag"] = consensus_flag.astype(int)
    anomaly_frame.to_csv(output_dir / "anomaly_quality_review.csv", index=False)

    baseline = set(np.flatnonzero(consensus_flag))
    sensitivity_rows: list[dict[str, Any]] = []
    for contamination in (0.05, 0.08, 0.10):
        candidate_iso = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=config.seed,
            n_jobs=1,
        ).fit_predict(transformed) == -1
        candidate_lof = LocalOutlierFactor(
            n_neighbors=config.lof_neighbors,
            contamination=contamination,
            n_jobs=1,
        ).fit_predict(transformed) == -1
        candidate = set(np.flatnonzero(candidate_iso & candidate_lof))
        sensitivity_rows.append(
            {
                "parameter": "contamination",
                "value": contamination,
                "consensus_count": len(candidate),
                "jaccard_to_baseline": jaccard(baseline, candidate),
            }
        )
    for neighbors in (10, 20, 30, 40, 50):
        candidate_lof = LocalOutlierFactor(
            n_neighbors=neighbors,
            contamination=config.anomaly_contamination,
            n_jobs=1,
        ).fit_predict(transformed) == -1
        candidate = set(np.flatnonzero(iso_flag & candidate_lof))
        sensitivity_rows.append(
            {
                "parameter": "lof_neighbors",
                "value": neighbors,
                "consensus_count": len(candidate),
                "jaccard_to_baseline": jaccard(baseline, candidate),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(output_dir / "anomaly_sensitivity.csv", index=False)

    pca = PCA(n_components=2)
    projected = pca.fit_transform(transformed)
    plot_anomaly_review(
        iso_score,
        iso_flag,
        lof_score,
        lof_flag,
        lof.offset_,
        projected,
        consensus_flag,
        pca.explained_variance_ratio_,
        output_dir / "02_anomaly_quality_review.png",
    )
    plot_anomaly_sensitivity(
        sensitivity,
        output_dir / "03_anomaly_sensitivity.png",
    )

    metadata = {
        "interpretation": (
            "Descriptive data-quality review only; flags are not verified ground-truth errors "
            "and are not removed from classification."
        ),
        "preprocessing": "Standardized numeric fields plus one-hot Section",
        "contamination": config.anomaly_contamination,
        "lof_neighbors": config.lof_neighbors,
        "isolation_forest_flags": int(iso_flag.sum()),
        "lof_flags": int(lof_flag.sum()),
        "consensus_flags": int(consensus_flag.sum()),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "sensitivity": sensitivity_rows,
    }
    save_json(output_dir / "anomaly_metadata.json", metadata)
    joblib.dump(preprocessing, output_dir / "anomaly_preprocessor.joblib")
    joblib.dump(isolation, output_dir / "isolation_forest.joblib")
    return metadata


def plot_anomaly_review(
    iso_score: np.ndarray,
    iso_flag: np.ndarray,
    lof_score: np.ndarray,
    lof_flag: np.ndarray,
    lof_threshold: float,
    projected: np.ndarray,
    consensus_flag: np.ndarray,
    explained_variance: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Anomaly Signals — Descriptive Data-Quality Review", fontsize=14, fontweight="bold")

    axes[0].hist(iso_score[~iso_flag], bins=28, alpha=0.75, label="Not flagged", color="#5B9BD5")
    axes[0].hist(iso_score[iso_flag], bins=14, alpha=0.85, label="Flagged", color="#C00000")
    axes[0].axvline(0, color="black", linestyle="--", label="Decision threshold = 0")
    axes[0].set_title("Isolation Forest decision function")
    axes[0].set_xlabel("Decision score")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].hist(lof_score[~lof_flag], bins=28, alpha=0.75, label="Not flagged", color="#70AD47")
    axes[1].hist(lof_score[lof_flag], bins=14, alpha=0.85, label="Flagged", color="#ED7D31")
    axes[1].axvline(
        lof_threshold,
        color="black",
        linestyle="--",
        label=f"Threshold={lof_threshold:.3f}",
    )
    axes[1].set_title("Local Outlier Factor")
    axes[1].set_xlabel("Negative outlier factor")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].scatter(
        projected[~consensus_flag, 0],
        projected[~consensus_flag, 1],
        s=22,
        alpha=0.6,
        color="#A5A5A5",
        label="Not consensus-flagged",
    )
    axes[2].scatter(
        projected[consensus_flag, 0],
        projected[consensus_flag, 1],
        s=75,
        marker="X",
        color="#C00000",
        label=f"Consensus flags (n={consensus_flag.sum()})",
    )
    axes[2].set_xlabel(f"PC1 ({explained_variance[0] * 100:.1f}%)")
    axes[2].set_ylabel(f"PC2 ({explained_variance[1] * 100:.1f}%)")
    axes[2].set_title("PCA projection for visualization only")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_anomaly_sensitivity(sensitivity: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Anomaly Consensus Sensitivity", fontsize=14, fontweight="bold")
    for axis, parameter, title in [
        (axes[0], "contamination", "Contamination sensitivity"),
        (axes[1], "lof_neighbors", "LOF neighbor sensitivity"),
    ]:
        data = sensitivity.loc[sensitivity["parameter"].eq(parameter)]
        labels = [str(value) for value in data["value"]]
        bars = axis.bar(labels, data["jaccard_to_baseline"], color="#5B9BD5", alpha=0.85)
        axis.set_ylim(0, 1.05)
        axis.set_xlabel(parameter)
        axis.set_ylabel("Jaccard similarity to baseline")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        for bar, count in zip(bars, data["consensus_count"]):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"n={int(count)}",
                ha="center",
                fontsize=8,
            )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


class FourierActivation(nn.Module):
    """Learned Fourier residual activation; this is not a KAN layer."""

    def __init__(self, frequencies: int = 6) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(frequencies) * 0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expanded = self.frequencies.view(-1, 1, 1) * inputs.unsqueeze(0)
        return inputs + 0.1 * expanded.sin().sum(0) + 0.1 * expanded.cos().sum(0)


class ShearWallMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        class_count: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            FourierActivation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            FourierActivation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, class_count),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def make_mlp_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "section",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def train_mlp(
    frame: pd.DataFrame,
    groups: np.ndarray,
    config: RunConfig,
    output_dir: Path,
) -> dict[str, Any]:
    X = frame[MODEL_FEATURES]
    y = frame[TARGET_COL].to_numpy(dtype=int)

    development_index, test_index, test_split_seed = choose_group_holdout(
        X, y, groups, n_splits=5, seed=config.seed + 1_000
    )
    development_X = X.iloc[development_index].reset_index(drop=True)
    development_y = y[development_index]
    development_groups = groups[development_index]
    train_local, validation_local, validation_split_seed = choose_group_holdout(
        development_X,
        development_y,
        development_groups,
        n_splits=5,
        seed=config.seed + 2_000,
    )
    train_index = development_index[train_local]
    validation_index = development_index[validation_local]

    split_group_sets = {
        "train": set(groups[train_index]),
        "validation": set(groups[validation_index]),
        "test": set(groups[test_index]),
    }
    if (
        split_group_sets["train"] & split_group_sets["validation"]
        or split_group_sets["train"] & split_group_sets["test"]
        or split_group_sets["validation"] & split_group_sets["test"]
    ):
        raise AssertionError("MLP train, validation, and test groups must be disjoint.")

    preprocessor = make_mlp_preprocessor()
    X_train = preprocessor.fit_transform(X.iloc[train_index]).astype(np.float32)
    X_validation = preprocessor.transform(X.iloc[validation_index]).astype(np.float32)
    X_test = preprocessor.transform(X.iloc[test_index]).astype(np.float32)

    y_zero_based = y - 1
    y_train = y_zero_based[train_index]
    y_validation = y_zero_based[validation_index]
    y_test = y_zero_based[test_index]

    class_counts = np.bincount(y_train, minlength=len(CLASS_LABELS))
    class_weights = len(y_train) / (len(CLASS_LABELS) * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32)
    )

    model = ShearWallMLP(
        input_size=X_train.shape[1],
        hidden_size=config.mlp_hidden,
        class_count=len(CLASS_LABELS),
        dropout=config.mlp_dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.mlp_epochs
    )

    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=32,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_tensor = torch.tensor(X_validation, dtype=torch.float32)

    best_state: dict[str, torch.Tensor] | None = None
    best_validation_f1 = -np.inf
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.mlp_epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for features, targets in loader:
            optimizer.zero_grad()
            noisy_features = features + torch.randn_like(features) * 0.02
            loss = criterion(model(noisy_features), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        scheduler.step()

        model.eval()
        with torch.no_grad():
            validation_pred = model(validation_tensor).argmax(dim=1).numpy()
        validation_f1 = f1_score(
            y_validation,
            validation_pred,
            labels=list(range(len(CLASS_LABELS))),
            average="macro",
            zero_division=0,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "validation_f1_macro": float(validation_f1),
            }
        )

        if validation_f1 > best_validation_f1 + 1e-8:
            best_validation_f1 = float(validation_f1)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.mlp_patience:
                break

    if best_state is None:
        raise RuntimeError("MLP training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.tensor(X_test, dtype=torch.float32))
        test_probabilities = F.softmax(test_logits, dim=1).numpy()
        test_predictions_zero = test_logits.argmax(dim=1).numpy()
    test_predictions = test_predictions_zero + 1
    test_metrics = classification_metrics(y[test_index], test_predictions, test_probabilities)
    test_report = classification_report(
        y[test_index],
        test_predictions,
        labels=CLASS_LABELS,
        target_names=[FAILURE_NAMES[label] for label in CLASS_LABELS],
        output_dict=True,
        zero_division=0,
    )

    torch.save(best_state, output_dir / "best_mlp_weights.pth")
    joblib.dump(preprocessor, output_dir / "mlp_preprocessor.joblib")
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "mlp_training_history.csv", index=False)

    test_frame = frame.loc[
        test_index, ["RowID", "SourceID", GROUP_COL, "Specimen", TARGET_COL]
    ].copy()
    test_frame["EvaluationGroup"] = groups[test_index]
    test_frame["PredictedFailureMode"] = test_predictions
    for position, label in enumerate(CLASS_LABELS):
        test_frame[f"Probability_{FAILURE_NAMES[label]}"] = test_probabilities[:, position]
    test_frame.to_csv(output_dir / "mlp_independent_test_predictions.csv", index=False)

    plot_mlp_results(
        history_frame,
        y[test_index],
        test_predictions,
        test_report,
        best_epoch,
        output_dir / "04_mlp_holdout_results.png",
    )

    metadata = {
        "evaluation": "Disjoint author/feature-group train-validation-test split",
        "test_split_seed_selected_from_label_balance_only": test_split_seed,
        "validation_split_seed_selected_from_label_balance_only": validation_split_seed,
        "train_rows": len(train_index),
        "validation_rows": len(validation_index),
        "test_rows": len(test_index),
        "train_groups": len(split_group_sets["train"]),
        "validation_groups": len(split_group_sets["validation"]),
        "test_groups": len(split_group_sets["test"]),
        "train_class_counts": np.bincount(y[train_index] - 1, minlength=4).tolist(),
        "validation_class_counts": np.bincount(
            y[validation_index] - 1, minlength=4
        ).tolist(),
        "test_class_counts": np.bincount(y[test_index] - 1, minlength=4).tolist(),
        "input_size": X_train.shape[1],
        "hidden_size": config.mlp_hidden,
        "dropout": config.mlp_dropout,
        "best_epoch": best_epoch,
        "best_validation_f1_macro": best_validation_f1,
        "independent_test_metrics": test_metrics,
        "independent_test_classification_report": test_report,
        "features": MODEL_FEATURES,
        "classes": FAILURE_NAMES,
        "checkpoint_loading": "Use torch.load(..., weights_only=True)",
    }
    save_json(output_dir / "mlp_metadata.json", metadata)
    return metadata


def plot_mlp_results(
    history: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    report: dict[str, Any],
    best_epoch: int,
    output_path: Path,
) -> None:
    class_names = [FAILURE_NAMES[label] for label in CLASS_LABELS]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Fourier MLP — Independent Group Holdout", fontsize=14, fontweight="bold")

    axes[0].plot(history["epoch"], history["train_loss"], color="#C00000")
    axes[0].axvline(best_epoch, color="black", linestyle="--", label=f"Best epoch={best_epoch}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[0].set_title("Training history")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        history["epoch"], history["validation_f1_macro"], color="#70AD47"
    )
    axes[1].axvline(best_epoch, color="black", linestyle="--")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation macro F1")
    axes[1].set_title("Validation used for early stopping")
    axes[1].grid(alpha=0.25)

    matrix = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
        ax=axes[2], cmap="Greens", colorbar=False
    )
    supports = [int(report[name]["support"]) for name in class_names]
    axes[2].set_title("Independent test confusion matrix\n" + ", ".join(
        f"{name}: n={support}" for name, support in zip(class_names, supports)
    ))
    axes[2].tick_params(axis="x", rotation=25)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "imbalanced_learn": imblearn_version,
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
    }


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    config = RunConfig(seed=args.seed)
    if args.quick:
        config = RunConfig(
            seed=args.seed,
            rf_estimators=80,
            permutation_repeats=3,
            bootstrap_repeats=100,
            mlp_epochs=50,
            mlp_patience=8,
        )

    configure_reproducibility(config.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data.resolve()
    frame = load_and_validate_data(data_path)
    groups = build_evaluation_groups(frame)

    rf_pipeline, rf_metadata = evaluate_random_forest(
        frame, groups, config, output_dir
    )
    anomaly_metadata = run_anomaly_quality_review(
        frame, config, output_dir
    )
    mlp_metadata = None
    if not args.skip_mlp:
        mlp_metadata = train_mlp(frame, groups, config, output_dir)

    run_metadata = {
        "data_file": data_path.name,
        "data_sha256": file_sha256(data_path),
        "records": len(frame),
        "authors": int(frame[GROUP_COL].nunique()),
        "evaluation_groups": int(len(np.unique(groups))),
        "class_counts": frame[TARGET_COL].value_counts().sort_index().to_dict(),
        "source_id_repeated_rows": int(
            frame["SourceID"].duplicated(keep=False).sum()
        ),
        "repeated_feature_vector_rows": int(
            frame.duplicated(MODEL_FEATURES, keep=False).sum()
        ),
        "config": asdict(config),
        "versions": package_versions(),
        "random_forest": rf_metadata,
        "anomaly_quality_review": anomaly_metadata,
        "mlp": mlp_metadata,
    }
    save_json(output_dir / "metrics_summary.json", run_metadata)

    print("\nAnalysis complete")
    print(f"Data: {data_path}")
    print(f"Rows: {len(frame)}, evaluation groups: {len(np.unique(groups))}")
    print(
        "Random Forest grouped OOF macro F1: "
        f"{rf_metadata['overall_oof_metrics']['f1_macro']:.3f}"
    )
    if mlp_metadata is not None:
        print(
            "MLP independent-test macro F1: "
            f"{mlp_metadata['independent_test_metrics']['f1_macro']:.3f}"
        )
    print(f"Outputs: {output_dir}")
    return run_metadata


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
