from pathlib import Path

import numpy as np

from main import (
    CLASS_LABELS,
    MODEL_FEATURES,
    RunConfig,
    build_evaluation_groups,
    choose_balanced_group_splits,
    load_and_validate_data,
    make_rf_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA = ROOT / "data" / "processed" / "Shear_Wall_Database_clean.xlsx"


def test_clean_data_schema_and_ids() -> None:
    frame = load_and_validate_data(CLEAN_DATA)
    assert len(frame) == 393
    assert frame["RowID"].is_unique
    assert not frame[MODEL_FEATURES].isna().any().any()
    assert set(frame["FailureMode"]) == set(CLASS_LABELS)


def test_evaluation_groups_keep_authors_and_duplicate_vectors_together() -> None:
    frame = load_and_validate_data(CLEAN_DATA)
    groups = build_evaluation_groups(frame)

    assert len(np.unique(groups)) == frame["Author"].nunique()
    assert frame.assign(group=groups).groupby("Author")["group"].nunique().max() == 1
    assert (
        frame.assign(group=groups)
        .groupby(MODEL_FEATURES, dropna=False)["group"]
        .nunique()
        .max()
        == 1
    )


def test_grouped_folds_are_disjoint_and_contain_every_class() -> None:
    frame = load_and_validate_data(CLEAN_DATA)
    groups = build_evaluation_groups(frame)
    X = frame[MODEL_FEATURES]
    y = frame["FailureMode"].to_numpy()
    splits, _ = choose_balanced_group_splits(X, y, groups, n_splits=5, seed=42)

    covered = np.zeros(len(frame), dtype=int)
    for train_index, test_index in splits:
        assert not (set(groups[train_index]) & set(groups[test_index]))
        assert set(y[test_index]) == set(CLASS_LABELS)
        covered[test_index] += 1
    assert np.all(covered == 1)


def test_fold_local_rf_pipeline_predicts_valid_classes() -> None:
    frame = load_and_validate_data(CLEAN_DATA)
    groups = build_evaluation_groups(frame)
    X = frame[MODEL_FEATURES]
    y = frame["FailureMode"].to_numpy()
    splits, _ = choose_balanced_group_splits(X, y, groups, n_splits=5, seed=42)
    train_index, test_index = splits[0]

    pipeline = make_rf_pipeline(
        RunConfig(rf_estimators=20, permutation_repeats=1, bootstrap_repeats=10)
    )
    pipeline.fit(X.iloc[train_index], y[train_index])
    predictions = pipeline.predict(X.iloc[test_index])

    assert len(predictions) == len(test_index)
    assert set(np.unique(predictions)).issubset(set(CLASS_LABELS))
