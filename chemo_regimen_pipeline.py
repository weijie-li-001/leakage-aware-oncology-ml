# -*- coding: utf-8 -*-
"""Leakage-aware evaluation of chemotherapy treatment and regimen labels.

The project is intentionally framed as retrospective classification, not as a
clinical recommendation system.  For each treatment stage it separates:

1. treatment decision: explicit treatment vs explicit no treatment;
2. regimen family: drug family among patients explicitly recorded as treated.

Not-applicable, missing, and not-reported values are excluded from the targets
instead of being relabelled as "no treatment". Treatment fields,
regimen text, outcomes and study membership are never used as predictors.
Study membership is retained only for leave-one-study-out validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import textwrap
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)


RANDOM_STATE = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
MIN_REGIMEN_CLASS_SIZE = 8
PERMUTATION_REPEATS = 5
GROUP_VALIDATION_MODEL = "LogisticRegression"

STUDY_COL = "cohort_id"
NEO_INDICATOR_COL = "neoadjuvant_treatment"
NEO_REGIMEN_COL = "neoadjuvant_regimen"
ADJ_INDICATOR_COL = "adjuvant_treatment"
ADJ_REGIMEN_COL = "adjuvant_regimen"

TREATMENT_AND_REGIMEN_COLS = {
    NEO_INDICATOR_COL,
    NEO_REGIMEN_COL,
    "neoadjuvant_treatment_response",
    ADJ_INDICATOR_COL,
    ADJ_REGIMEN_COL,
    "prior_primary_or_metastasis_adjuvant_treatment",
    "prior_primary_or_metastasis_adjuvant_regimen",
}

OUTCOME_COLS = {
    "progression_status",
    "progression_date",
    "follow_up_date",
    "progression_free_survival",
    "early_progression",
    "vital_status",
    "cancer_related_death",
    "death_date",
    "overall_survival",
}

IDENTIFIER_OR_ADMIN_COLS = {
    "record_id",
    "date_of_birth",
    "surgery_date",
    STUDY_COL,
}

FORBIDDEN_FEATURE_COLS = TREATMENT_AND_REGIMEN_COLS | OUTCOME_COLS | IDENTIFIER_OR_ADMIN_COLS

# Conservative baseline feature set.  For the neoadjuvant tasks, postoperative
# staging/pathology variables are deliberately excluded because the workbook
# does not document when every value was measured.
BASELINE_FEATURE_COLS = [
    "sex",
    "age_at_diagnosis",
    "bmi",
    "primary_diagnosis",
    "sample_type",
    "sample_site",
    "histology",
    "tumor_side",
    "tumor_marker_1",
    "tumor_marker_2",
    "comorbidities",
]

# These pathological/resection variables can plausibly be known before an
# adjuvant decision, which normally follows surgery.  This timing assumption is
# explicit and must be revisited if the data dictionary says otherwise.
ADJUVANT_POSTOPERATIVE_COLS = [
    "t_stage",
    "n_stage",
    "m_stage",
    "venous_invasion",
    "lymphatic_invasion",
    "perineural_invasion",
    "grade",
    "resection_margin",
    "tumor_or_metastasis_diameter",
    "metastasis_timing",
    "intrahepatic_metastasis_count",
    "extrahepatic_metastasis_count",
    "extrahepatic_metastasis_site",
]

NUMERIC_COLS = {
    "age_at_diagnosis",
    "bmi",
    "tumor_or_metastasis_diameter",
}

MISSING_TOKENS = {
    "",
    "missing",
    "not available",
    "n/a",
    "not reported",
    "not determined",
    "unknown",
    "nan",
    "not applicable",
    "not applicable/missing",
}
YES_TOKENS = {"yes", "true", "1"}
NO_TOKENS = {"no", "none", "false", "0"}


@dataclass(frozen=True)
class TaskSpec:
    key: str
    label: str
    stage: str
    target_kind: str
    indicator_col: str
    regimen_col: str


TASKS = [
    TaskSpec(
        "neo_treatment",
        "Neoadjuvant treatment decision",
        "neo",
        "treatment",
        NEO_INDICATOR_COL,
        NEO_REGIMEN_COL,
    ),
    TaskSpec(
        "neo_regimen",
        "Neoadjuvant regimen family among treated records",
        "neo",
        "regimen",
        NEO_INDICATOR_COL,
        NEO_REGIMEN_COL,
    ),
    TaskSpec(
        "adj_treatment",
        "Adjuvant treatment decision",
        "adj",
        "treatment",
        ADJ_INDICATOR_COL,
        ADJ_REGIMEN_COL,
    ),
    TaskSpec(
        "adj_regimen",
        "Adjuvant regimen family among treated records",
        "adj",
        "regimen",
        ADJ_INDICATOR_COL,
        ADJ_REGIMEN_COL,
    ),
]


def normalize_text(raw) -> str | None:
    """Return a stripped lowercase string, preserving German characters."""
    if pd.isna(raw):
        return None
    return str(raw).strip().lower()


def normalize_treatment_indicator(raw) -> str | float:
    """Map only explicit yes/no values; not-applicable remains unknown."""
    value = normalize_text(raw)
    if value in YES_TOKENS:
        return "Treatment"
    if value in NO_TOKENS:
        return "No treatment"
    return np.nan


def classify_regimen(raw) -> str | float:
    """Map free-text drug regimens to transparent, heuristic families.

    The mapping is a modelling convenience and has not been clinically
    validated.  Non-treatment, not-applicable and unknown tokens are excluded.
    """
    value = normalize_text(raw)
    if value is None or value in MISSING_TOKENS or value in NO_TOKENS:
        return np.nan

    if "folfoxiri" in value:
        return "FOLFOXIRI (triplet)"
    if "folfox" in value or "xelox" in value or "oxaliplatin" in value or "oxalipl" in value:
        return "FOLFOX/XELOX family"
    if (
        "folfiri" in value
        or "capiri" in value
        or "irinotecan" in value
        or "cpt-11" in value
        or "cpt 11" in value
        or "cpt11" in value
    ):
        return "FOLFIRI family"
    if "carboplatin" in value or re.search(r"\btch\b", value):
        return "Carboplatin-based"
    if (
        re.search(r"\btac\b", value)
        or "fec" in value
        or re.search(r"\d+x\s*ec\b", value)
        or " ec " in f" {value} "
        or value.endswith(" ec")
        or value == "ec"
        or "pacl" in value
        or "docetaxel" in value
        or re.search(r"\bdoc\b", value)
    ):
        return "Anthracycline/taxane (breast)"
    if "capecitabin" in value:
        return "Capecitabine-based"
    return "Other/rare regimen"


def merge_small_regimen_classes(y: pd.Series) -> pd.Series:
    """Ensure each retained class can participate in five-fold evaluation."""
    counts = y.value_counts()
    small = counts[counts < MIN_REGIMEN_CLASS_SIZE].index
    return y.where(~y.isin(small), "Other/rare regimen")


def to_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", ".", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_marker(raw) -> str | float:
    value = normalize_text(raw)
    if value is None or value in MISSING_TOKENS:
        return np.nan
    if "patholog" in value:
        return "pathological"
    if "normal" in value or "physiological" in value:
        return "normal"
    return np.nan


def normalize_r_status(raw) -> str | float:
    value = normalize_text(raw)
    if value is None or value in MISSING_TOKENS:
        return np.nan
    if value in {"r0", "0"} or "no macroscopic residual tumor" in value:
        return "r0"
    if value in {"r1", "1"}:
        return "r1"
    if value in {"r2", "2"} or "macroscopic residual tumor" in value:
        return "r2"
    return np.nan


def clean_categorical_series(series: pd.Series) -> pd.Series:
    cleaned = series.map(normalize_text)
    return cleaned.where(~cleaned.isin(MISSING_TOKENS), np.nan)


def load_raw_data(data_path: Path) -> tuple[pd.DataFrame, Path]:
    """Load an explicitly supplied local workbook.

    Requiring the path avoids accidentally selecting or packaging a private
    workbook that happens to be present in the repository directory.
    """
    data_path = data_path.resolve()
    raw = pd.read_excel(data_path)
    raw.columns = raw.columns.str.strip()
    return raw, data_path


def prepare_feature_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Create a cleaned analysis copy; never mutate the source workbook."""
    frame = raw.copy(deep=True)

    for column in NUMERIC_COLS & set(frame.columns):
        frame[column] = to_numeric_series(frame[column])

    if "tumor_marker_1" in frame:
        frame["tumor_marker_1"] = frame["tumor_marker_1"].map(normalize_marker)
    if "tumor_marker_2" in frame:
        frame["tumor_marker_2"] = frame["tumor_marker_2"].map(normalize_marker)
    if "resection_margin" in frame:
        frame["resection_margin"] = frame["resection_margin"].map(normalize_r_status)

    possible_features = set(BASELINE_FEATURE_COLS + ADJUVANT_POSTOPERATIVE_COLS)
    for column in possible_features & set(frame.columns):
        if column not in NUMERIC_COLS and column not in {
            "tumor_marker_1",
            "tumor_marker_2",
            "resection_margin",
        }:
            frame[column] = clean_categorical_series(frame[column])

    return frame


def feature_columns_for_stage(frame: pd.DataFrame, stage: str, index: pd.Index) -> list[str]:
    requested = list(BASELINE_FEATURE_COLS)
    if stage == "adj":
        requested.extend(ADJUVANT_POSTOPERATIVE_COLS)

    columns = [column for column in requested if column in frame.columns]
    columns = [column for column in columns if frame.loc[index, column].notna().any()]

    forbidden = sorted(set(columns) & FORBIDDEN_FEATURE_COLS)
    if forbidden:
        raise AssertionError(f"Forbidden leakage columns entered the feature set: {forbidden}")
    if STUDY_COL in columns:
        raise AssertionError("Study membership may only be used as a validation group.")
    return columns


def create_task_dataset(
    raw: pd.DataFrame,
    prepared: pd.DataFrame,
    spec: TaskSpec,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], dict]:
    indicator = raw[spec.indicator_col].map(normalize_treatment_indicator)
    regimen = raw[spec.regimen_col].map(classify_regimen)

    if spec.target_kind == "treatment":
        mask = indicator.notna()
        y = indicator.loc[mask]
    else:
        mask = indicator.eq("Treatment") & regimen.notna()
        y = merge_small_regimen_classes(regimen.loc[mask])

    groups = raw.loc[y.index, STUDY_COL].astype(str).str.strip()
    feature_cols = feature_columns_for_stage(prepared, spec.stage, y.index)
    X = prepared.loc[y.index, feature_cols].copy()

    audit = {
        "task": spec.key,
        "raw_rows": int(len(raw)),
        "usable_rows": int(len(y)),
        "unknown_or_not_applicable_indicator": int(indicator.isna().sum()),
        "explicit_treatment": int(indicator.eq("Treatment").sum()),
        "explicit_no_treatment": int(indicator.eq("No treatment").sum()),
        "known_regimen_family": int(regimen.notna().sum()),
        "treated_with_known_regimen": int((indicator.eq("Treatment") & regimen.notna()).sum()),
        "feature_count": int(len(feature_cols)),
        "features": " | ".join(feature_cols),
    }
    return X, y, groups, feature_cols, audit


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    numeric = [column for column in feature_cols if column in NUMERIC_COLS]
    categorical = [column for column in feature_cols if column not in NUMERIC_COLS]

    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="constant",
                    fill_value="missing",
                    keep_empty_features=True,
                ),
            ),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def candidate_models() -> dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=1.0,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def make_model_pipeline(feature_cols: list[str], model) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(feature_cols)),
            ("classifier", clone(model)),
        ]
    )


def valid_stratified_folds(y: pd.Series, requested: int) -> int:
    min_class_size = int(y.value_counts().min())
    folds = min(requested, min_class_size)
    if folds < 2:
        raise ValueError("At least two observations per class are required for stratified evaluation.")
    return folds


def compare_models(X: pd.DataFrame, y: pd.Series, feature_cols: list[str]) -> pd.DataFrame:
    folds = valid_stratified_folds(y, N_OUTER_FOLDS)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    models = {"DummyMajority": DummyClassifier(strategy="most_frequent")}
    models.update(candidate_models())
    for name, model in models.items():
        pipeline = make_model_pipeline(feature_cols, model)
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=splitter,
            scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
            n_jobs=1,
            error_score="raise",
        )
        rows.append(
            {
                "model": name,
                "is_baseline": name == "DummyMajority",
                "macro_f1_mean": scores["test_macro_f1"].mean(),
                "macro_f1_std": scores["test_macro_f1"].std(),
                "balanced_accuracy_mean": scores["test_balanced_accuracy"].mean(),
                "balanced_accuracy_std": scores["test_balanced_accuracy"].std(),
                "folds": folds,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["is_baseline", "macro_f1_mean"], ascending=[True, False]
    ).reset_index(drop=True)


def select_model_inner_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str],
    outer_fold: int,
) -> tuple[str, dict[str, float]]:
    folds = valid_stratified_folds(y_train, N_INNER_FOLDS)
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE + outer_fold,
    )
    scores_by_model = {}
    for name, model in candidate_models().items():
        pipeline = make_model_pipeline(feature_cols, model)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=splitter,
            scoring="f1_macro",
            n_jobs=1,
            error_score="raise",
        )
        scores_by_model[name] = float(scores["test_score"].mean())
    best_name = max(scores_by_model, key=scores_by_model.get)
    return best_name, scores_by_model


def nested_oof_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict]:
    """Nested model selection with one prediction per record."""
    folds = valid_stratified_folds(y, N_OUTER_FOLDS)
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    predictions = pd.Series(index=y.index, dtype=object, name="prediction")
    fold_rows = []
    importance_rows = []
    model_votes = Counter()

    for fold, (train_pos, test_pos) in enumerate(outer.split(X, y), start=1):
        X_train, X_test = X.iloc[train_pos], X.iloc[test_pos]
        y_train, y_test = y.iloc[train_pos], y.iloc[test_pos]
        best_name, inner_scores = select_model_inner_cv(
            X_train, y_train, feature_cols, outer_fold=fold
        )
        model_votes[best_name] += 1

        pipeline = make_model_pipeline(feature_cols, candidate_models()[best_name])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        predictions.loc[y_test.index] = y_pred

        fold_rows.append(
            {
                "outer_fold": fold,
                "selected_model": best_name,
                "inner_macro_f1": inner_scores[best_name],
                "outer_macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "outer_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "test_rows": len(y_test),
                **{f"inner_{name}_macro_f1": score for name, score in inner_scores.items()},
            }
        )

        perm = permutation_importance(
            pipeline,
            X_test,
            y_test,
            scoring="f1_macro",
            n_repeats=PERMUTATION_REPEATS,
            random_state=RANDOM_STATE + fold,
            n_jobs=1,
        )
        for feature, mean, std in zip(
            feature_cols, perm.importances_mean, perm.importances_std
        ):
            importance_rows.append(
                {
                    "outer_fold": fold,
                    "feature": feature,
                    "importance_mean": mean,
                    "importance_std": std,
                }
            )

    if predictions.isna().any():
        raise AssertionError("Nested CV failed to produce one prediction per record.")

    fold_frame = pd.DataFrame(fold_rows)
    importance_frame = pd.DataFrame(importance_rows)
    importance_summary = (
        importance_frame.groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_between_fold_std=("importance_mean", "std"),
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    summary = {
        "nested_macro_f1": f1_score(y, predictions, average="macro", zero_division=0),
        "nested_balanced_accuracy": balanced_accuracy_score(y, predictions),
        "nested_macro_f1_fold_std": fold_frame["outer_macro_f1"].std(ddof=0),
        "nested_balanced_accuracy_fold_std": fold_frame["outer_balanced_accuracy"].std(ddof=0),
        "outer_folds": folds,
        "selected_model_votes": "; ".join(
            f"{name}={count}" for name, count in sorted(model_votes.items())
        ),
    }
    return predictions, fold_frame, importance_summary, summary


def balanced_accuracy_for_present_classes(y_true: pd.Series, y_pred: np.ndarray) -> float:
    labels = sorted(pd.Series(y_true).unique())
    return float(
        recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )


def group_validation(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    feature_cols: list[str],
    model_name: str,
) -> tuple[pd.DataFrame, dict]:
    """Leave one study out; record folds that cannot train a valid classifier."""
    logo = LeaveOneGroupOut()
    rows = []
    covered_true = []
    covered_pred = []

    for fold, (train_pos, test_pos) in enumerate(logo.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_pos], X.iloc[test_pos]
        y_train, y_test = y.iloc[train_pos], y.iloc[test_pos]
        held_out_group = str(groups.iloc[test_pos[0]])
        train_classes = sorted(y_train.unique())
        test_classes = sorted(y_test.unique())

        row = {
            "fold": fold,
            "held_out_study": held_out_group,
            "model": model_name,
            "train_rows": len(y_train),
            "test_rows": len(y_test),
            "train_classes": " | ".join(train_classes),
            "test_classes": " | ".join(test_classes),
        }
        if len(train_classes) < 2:
            row.update(
                {
                    "status": "not estimable: training studies contain one target class",
                    "macro_f1": np.nan,
                    "balanced_accuracy": np.nan,
                }
            )
            rows.append(row)
            continue

        pipeline = make_model_pipeline(feature_cols, candidate_models()[model_name])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        row.update(
            {
                "status": "evaluated",
                "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_for_present_classes(y_test, y_pred),
            }
        )
        rows.append(row)
        covered_true.extend(y_test.tolist())
        covered_pred.extend(y_pred.tolist())

    frame = pd.DataFrame(rows)
    covered_rows = len(covered_true)
    if covered_rows:
        aggregate_macro_f1 = f1_score(
            covered_true, covered_pred, average="macro", zero_division=0
        )
        aggregate_balanced_accuracy = balanced_accuracy_score(covered_true, covered_pred)
        status = "evaluated" if covered_rows == len(y) else "partial coverage"
    else:
        aggregate_macro_f1 = np.nan
        aggregate_balanced_accuracy = np.nan
        status = "not estimable"

    summary = {
        "group_validation_status": status,
        "group_validation_studies": int(groups.nunique()),
        "group_validation_coverage": covered_rows / len(y),
        "group_macro_f1": aggregate_macro_f1,
        "group_balanced_accuracy": aggregate_balanced_accuracy,
    }
    return frame, summary


def safe_label(label: str, width: int = 22) -> str:
    return textwrap.fill(
        str(label),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def plot_class_distribution(y: pd.Series, title: str, destination: Path) -> None:
    counts = y.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.65 * len(counts) + 2)), constrained_layout=True)
    ax.barh([safe_label(label) for label in counts.index], counts.values, color="#3B82F6")
    ax.set_title(title, pad=12)
    ax.set_xlabel("Records")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str,
    destination: Path,
) -> None:
    labels = sorted(y_true.unique())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display_labels = [safe_label(label, 18) for label in labels]
    size = max(7.5, 1.25 * len(labels) + 3)
    fig, ax = plt.subplots(figsize=(size, size), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=display_labels,
        yticklabels=display_labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "#0F172A",
            )
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_importance(frame: pd.DataFrame, title: str, destination: Path) -> None:
    top = frame.head(15).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.5 * len(top) + 2)), constrained_layout=True)
    ax.barh(
        [safe_label(label, 30) for label in top["feature"]],
        top["importance_mean"],
        xerr=top["importance_between_fold_std"].fillna(0),
        color="#2563EB",
        alpha=0.9,
    )
    ax.axvline(0, color="#475569", linewidth=0.8)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Mean outer-fold macro-F1 decrease when shuffled")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if pd.isna(value):
        return None
    return value


def run_pipeline(data_path: Path, output_dir: Path) -> pd.DataFrame:
    raw, resolved_data_path = load_raw_data(data_path)
    prepared = prepare_feature_dataframe(raw)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    audit_rows = []
    class_count_rows = []

    for spec in TASKS:
        print(f"\n{'=' * 78}\n{spec.label}\n{'=' * 78}")
        X, y, groups, feature_cols, audit = create_task_dataset(raw, prepared, spec)
        audit_rows.append(audit)
        for label, count in y.value_counts().items():
            class_count_rows.append(
                {"task": spec.key, "class": label, "records": int(count)}
            )

        print(f"Usable rows: {len(y)}; classes: {y.nunique()}; features: {len(feature_cols)}")
        print(y.value_counts().to_string())

        comparison = compare_models(X, y, feature_cols)
        comparison.insert(0, "task", spec.key)
        comparison.to_csv(output_dir / f"model_comparison_{spec.key}.csv", index=False)
        best_model_name = comparison.loc[~comparison["is_baseline"], "model"].iloc[0]

        nested_predictions, fold_frame, importance, nested_summary = nested_oof_evaluation(
            X, y, feature_cols
        )
        fold_frame.insert(0, "task", spec.key)
        fold_frame.to_csv(output_dir / f"nested_outer_folds_{spec.key}.csv", index=False)
        importance.insert(0, "task", spec.key)
        importance.to_csv(output_dir / f"feature_importance_{spec.key}.csv", index=False)

        report = pd.DataFrame(
            classification_report(
                y,
                nested_predictions,
                output_dict=True,
                zero_division=0,
            )
        ).T
        report.index.name = "class_or_average"
        report.to_csv(output_dir / f"classification_report_{spec.key}.csv")

        group_frame, group_summary = group_validation(
            X, y, groups, feature_cols, GROUP_VALIDATION_MODEL
        )
        group_frame.insert(0, "task", spec.key)
        group_frame.to_csv(output_dir / f"group_validation_{spec.key}.csv", index=False)

        plot_class_distribution(
            y,
            f"Class distribution - {spec.label}",
            output_dir / f"class_distribution_{spec.key}.png",
        )
        plot_confusion(
            y,
            nested_predictions,
            f"Nested-CV confusion matrix - {spec.label}",
            output_dir / f"confusion_matrix_{spec.key}.png",
        )
        plot_importance(
            importance,
            f"Leakage-audited permutation importance - {spec.label}",
            output_dir / f"feature_importance_{spec.key}.png",
        )

        summary_row = {
            "task": spec.key,
            "task_label": spec.label,
            "target_kind": spec.target_kind,
            "usable_rows": len(y),
            "classes": y.nunique(),
            "features": len(feature_cols),
            "best_model_full_cv": best_model_name,
            "group_validation_model": GROUP_VALIDATION_MODEL,
            **nested_summary,
            **group_summary,
        }
        summary_rows.append(summary_row)
        print(
            f"Nested CV macro-F1={nested_summary['nested_macro_f1']:.3f}; "
            f"balanced accuracy={nested_summary['nested_balanced_accuracy']:.3f}; "
            f"group validation={group_summary['group_validation_status']}"
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "evaluation_summary.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(output_dir / "label_and_feature_audit.csv", index=False)
    pd.DataFrame(class_count_rows).to_csv(output_dir / "class_counts.csv", index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_file": resolved_data_path.name,
        "data_sha256": file_sha256(resolved_data_path),
        "raw_rows": int(raw.shape[0]),
        "raw_columns": int(raw.shape[1]),
        "unique_record_ids": int(raw["record_id"].nunique(dropna=True)),
        "study_count": int(raw[STUDY_COL].nunique(dropna=True)),
        "random_state": RANDOM_STATE,
        "outer_folds": N_OUTER_FOLDS,
        "inner_folds": N_INNER_FOLDS,
        "minimum_regimen_class_size": MIN_REGIMEN_CLASS_SIZE,
        "group_validation_model": GROUP_VALIDATION_MODEL,
        "python_packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "feature_policy": {
            "study_membership": "grouping variable only; never a predictor",
            "treatment_and_regimen_columns": "targets/audit fields only; never predictors",
            "outcomes": "excluded",
            "neoadjuvant": "conservative baseline feature allowlist",
            "adjuvant": "baseline plus explicitly documented postoperative feature allowlist",
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(json_ready(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_dir.resolve()}")
    print(summary.to_string(index=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to a local .xlsx workbook. Data files are intentionally not included in this repository.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts"),
        help="Directory for generated evaluation artifacts (ignored by Git).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.data, args.output)


if __name__ == "__main__":
    main()
