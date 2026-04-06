"""Train sklearn / XGBoost models from CSV; persist joblib artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def _build_classifier(algorithm: str, xgb_params: dict[str, Any] | None) -> Any:
    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
    if algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise RuntimeError("xgboost is not installed; pip install xgboost") from e
        params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "random_state": 42}
        if xgb_params:
            params.update(xgb_params)
        return XGBClassifier(**params)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def train_from_csv(
    csv_path: Path,
    target_column: str,
    algorithm: str,
    test_size: float,
    random_state: int,
    xgb_params: dict[str, Any] | None,
    artifact_path: Path,
    *,
    vfl_agent_definitions_path: str | None = None,
    repo_root: Path | None = None,
    storage_root: Path | None = None,
) -> dict[str, Any]:
    if algorithm == "vfl":
        from app.services import ml_vfl

        return ml_vfl.train_vfl_from_csv(
            csv_path,
            target_column,
            test_size,
            random_state,
            artifact_path,
            agent_definitions_path=vfl_agent_definitions_path,
            repo_root=repo_root,
            storage_root=storage_root,
        )

    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in CSV columns: {list(df.columns)}")

    y_raw = df[target_column]
    X = df.drop(columns=[target_column])

    le_y: LabelEncoder | None = None
    if y_raw.dtype == object or str(y_raw.dtype).startswith("category"):
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw.astype(str))
        label_classes = le_y.classes_.tolist()
    else:
        y = y_raw.values
        label_classes = np.unique(y).tolist()

    numeric_features = list(X.select_dtypes(include=[np.number]).columns)
    categorical_features = [c for c in X.columns if c not in numeric_features]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError("No feature columns after removing target")

    preprocess = ColumnTransformer(transformers=transformers)
    clf = _build_classifier(algorithm, xgb_params)
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipe,
        "target_encoder": le_y,
        "target_column": target_column,
        "feature_columns": list(X.columns),
        "label_classes": label_classes,
        "algorithm": algorithm,
    }
    joblib.dump(bundle, artifact_path)

    metrics = {
        "accuracy": acc,
        "f1_weighted": f1,
        "classification_report": report,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    logger.info("Trained %s accuracy=%.4f f1=%.4f -> %s", algorithm, acc, f1, artifact_path)
    return {
        "metrics": metrics,
        "feature_columns": list(X.columns),
        "label_classes": label_classes,
    }


def load_model_bundle(artifact_path: Path) -> dict[str, Any]:
    return joblib.load(artifact_path)


def summarize_bundle_for_llm(bundle: dict[str, Any]) -> str:
    return json.dumps(
        {
            "algorithm": bundle.get("algorithm"),
            "target_column": bundle.get("target_column"),
            "feature_columns": bundle.get("feature_columns"),
            "label_classes": bundle.get("label_classes"),
        },
        indent=2,
    )
