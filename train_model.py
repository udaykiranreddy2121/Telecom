"""Train and evaluate reproducible telecom churn classification models.

Expected input: Telecom Churn.csv in the repository root (or pass --data).
The script compares a majority-class baseline, balanced logistic regression,
and a balanced random forest using a stratified holdout set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET = "churn"
IDENTIFIER_COLUMNS = ["phone number"]


def normalise_target(series: pd.Series) -> pd.Series:
    """Convert common boolean/string/numeric churn labels to 0 and 1."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        values = set(series.dropna().unique())
        if values.issubset({0, 1}):
            return series.astype(int)

    mapping = {
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
        "1": 1, "0": 0,
        "churn": 1, "stay": 0,
    }
    converted = series.astype(str).str.strip().str.lower().map(mapping)
    if converted.isna().any():
        bad_values = sorted(series[converted.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unsupported or missing target labels: {bad_values}")
    return converted.astype(int)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    result: dict[str, Any] = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["not_churn", "churn"], output_dict=True, zero_division=0
        ),
    }
    if scores is not None and y_test.nunique() == 2:
        result["roc_auc"] = float(roc_auc_score(y_test, scores))
        result["average_precision"] = float(average_precision_score(y_test, scores))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train telecom churn classification models.")
    parser.add_argument("--data", default="Telecom Churn.csv", help="Path to the source CSV.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for model and metrics.")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Place Telecom Churn.csv in the project root "
            "or provide --data /path/to/file.csv. The dataset is not committed to this repository."
        )

    data = pd.read_csv(data_path)
    if TARGET not in data.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing.")
    if data[TARGET].isna().any():
        data = data.dropna(subset=[TARGET]).copy()

    y = normalise_target(data[TARGET])
    X = data.drop(columns=[TARGET] + [c for c in IDENTIFIER_COLUMNS if c in data.columns])
    if X.empty:
        raise ValueError("No feature columns remain after removing target and identifiers.")
    if y.nunique() != 2:
        raise ValueError("The churn target must contain both classes (0 and 1).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models: dict[str, Any] = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    results: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", estimator),
        ])
        pipeline.fit(X_train, y_train)
        results.append(evaluate_model(name, pipeline, X_test, y_test))
        fitted_models[name] = pipeline

    # Select by holdout F1 for the churn class; the baseline is retained for comparison.
    candidates = [r for r in results if r["model"] != "majority_baseline"]
    best_name = max(candidates, key=lambda item: item["f1"])["model"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_models[best_name], output_dir / "telecom_churn_model.joblib")

    report = {
        "dataset": str(data_path),
        "rows_used": int(len(data)),
        "feature_count": int(X.shape[1]),
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
        "selected_model_by_holdout_f1": best_name,
        "metrics": results,
        "notes": [
            "Phone number is excluded as an identifier.",
            "Preprocessing is fitted only on the training split inside each pipeline.",
            "The test split is a single holdout; results are estimates, not guarantees.",
            "Review precision and recall together before using a model for retention outreach.",
        ],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Selected model: {best_name}")
    print(pd.DataFrame([
        {key: row.get(key) for key in ("model", "accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")}
        for row in results
    ]).to_string(index=False))
    print(f"Saved model and metrics to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
