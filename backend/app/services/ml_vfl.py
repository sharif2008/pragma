"""Vertical federated learning (VFL) training with fixed architecture — 3 parties, concat embeddings.

Uses ``app.notebook_runtime.model_utils.VFLModel`` and optional ``storage/agentic_features.json`` for column splits;
otherwise columns are partitioned by ``categorize_feature_by_evidence`` heuristics (IDS-style).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.notebook_runtime.model_utils import VFLModel
from app.notebook_runtime.vfl_utils import (
    categorize_feature_by_evidence,
    get_agent_names,
    load_agent_definitions,
    split_features_by_agent_definitions,
)

logger = logging.getLogger(__name__)

# Fixed VFL hyperparameters (vertical federation layout)
VFL_EMBED_DIM = 64
VFL_HIDDEN_DIM = 128
VFL_EPOCHS = 15
VFL_BATCH_SIZE = 256
VFL_LR = 1e-3


def _finalize_three_parties(
    a1: list[str], a2: list[str], a3: list[str], feature_cols: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """Assign unmapped columns round-robin; ensure each party has >= 1 column for VFLModel."""
    parties = [list(a1), list(a2), list(a3)]
    assigned = set(parties[0]) | set(parties[1]) | set(parties[2])
    pool = [c for c in feature_cols if c not in assigned]
    for i, c in enumerate(pool):
        parties[i % 3].append(c)
    for _ in range(12):
        empty = [i for i in range(3) if len(parties[i]) == 0]
        if not empty:
            break
        donor = max(range(3), key=lambda i: len(parties[i]))
        if len(parties[donor]) > 1:
            parties[empty[0]].append(parties[donor].pop(-1))
        else:
            break
    return parties[0], parties[1], parties[2]


def _split_feature_columns(
    feature_cols: list[str],
    *,
    agent_definitions_path: Path | None,
    repo_root: Path,
) -> tuple[list[str], list[str], list[str]]:
    if agent_definitions_path is not None:
        path = agent_definitions_path if agent_definitions_path.is_absolute() else repo_root / agent_definitions_path
        if not path.is_file():
            raise FileNotFoundError(f"VFL agent definitions not found: {path}")
        defs = load_agent_definitions(path)
        a1, a2, a3, _ = split_features_by_agent_definitions(list(feature_cols), defs)
        a1 = [c for c in a1 if c in feature_cols]
        a2 = [c for c in a2 if c in feature_cols]
        a3 = [c for c in a3 if c in feature_cols]
    else:
        a1, a2, a3 = [], [], []
        for c in feature_cols:
            cat = categorize_feature_by_evidence(str(c))
            if cat == "evidence_volume_rate":
                a1.append(c)
            elif cat == "evidence_packet_size":
                a2.append(c)
            else:
                a3.append(c)

    return _finalize_three_parties(a1, a2, a3, feature_cols)


def train_vfl_from_csv(
    csv_path: Path,
    target_column: str,
    test_size: float,
    random_state: int,
    artifact_path: Path,
    *,
    agent_definitions_path: str | Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo = repo_root or csv_path.resolve().parent
    ad_path = Path(agent_definitions_path) if agent_definitions_path else None

    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in CSV columns: {list(df.columns)}")

    y_raw = df[target_column]
    X = df.drop(columns=[target_column])

    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    if len(numeric_cols) < 3:
        raise ValueError("VFL requires at least 3 numeric feature columns (one per party after split)")

    X = X[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    le_y = LabelEncoder()
    if y_raw.dtype == object or str(y_raw.dtype).startswith("category"):
        y = le_y.fit_transform(y_raw.astype(str))
    else:
        y = le_y.fit_transform(y_raw)
    label_classes = le_y.classes_.tolist()

    num_classes = len(label_classes)
    if num_classes < 2:
        raise ValueError("VFL requires at least 2 classes")

    p1, p2, p3 = _split_feature_columns(list(X.columns), agent_definitions_path=ad_path, repo_root=repo)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    X_test_df = pd.DataFrame(X_test, columns=X.columns)

    idx1 = [X.columns.get_loc(c) for c in p1]
    idx2 = [X.columns.get_loc(c) for c in p2]
    idx3 = [X.columns.get_loc(c) for c in p3]

    scalers = [StandardScaler(), StandardScaler(), StandardScaler()]
    parts_train = [
        scalers[0].fit_transform(X_train_df[p1].values),
        scalers[1].fit_transform(X_train_df[p2].values),
        scalers[2].fit_transform(X_train_df[p3].values),
    ]
    parts_test = [
        scalers[0].transform(X_test_df[p1].values),
        scalers[1].transform(X_test_df[p2].values),
        scalers[2].transform(X_test_df[p3].values),
    ]

    input_dims = [parts_train[0].shape[1], parts_train[1].shape[1], parts_train[2].shape[1]]

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VFLModel(input_dims, embed_dim=VFL_EMBED_DIM, num_classes=num_classes, hidden_dim=VFL_HIDDEN_DIM)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=VFL_LR)

    X_t = [torch.tensor(parts_train[i], dtype=torch.float32, device=device) for i in range(3)]
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)
    n = len(y_train)

    model.train()
    for epoch in range(VFL_EPOCHS):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for start in range(0, n, VFL_BATCH_SIZE):
            batch_idx = perm[start : start + VFL_BATCH_SIZE]
            xb = [X_t[i][batch_idx] for i in range(3)]
            yb = y_t[batch_idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("VFL epoch %s loss=%.4f", epoch + 1, total_loss / max(1, n // VFL_BATCH_SIZE))

    model.eval()
    with torch.no_grad():
        xt = [torch.tensor(parts_test[i], dtype=torch.float32, device=device) for i in range(3)]
        logits = model(xt)
        pred = logits.argmax(dim=1).cpu().numpy()

    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="weighted", zero_division=0))
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {
        "kind": "vfl_torch",
        "algorithm": "vfl",
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dims": input_dims,
        "party_columns": [p1, p2, p3],
        "scalers": scalers,
        "target_column": target_column,
        "target_encoder": le_y,
        "label_classes": label_classes,
        "feature_columns": list(X.columns),
        "num_classes": num_classes,
        "vfl_fixed_config": {
            "embed_dim": VFL_EMBED_DIM,
            "hidden_dim": VFL_HIDDEN_DIM,
            "n_parties": 3,
            "epochs": VFL_EPOCHS,
            "batch_size": VFL_BATCH_SIZE,
            "lr": VFL_LR,
            "party_names": get_agent_names(),
            "fusion": "concat_embeddings",
        },
    }
    joblib.dump(bundle, artifact_path)

    metrics = {
        "accuracy": acc,
        "f1_weighted": f1,
        "classification_report": report,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "algorithm_mode": "vfl_vertical_3party",
    }
    logger.info("Trained VFL accuracy=%.4f f1=%.4f -> %s", acc, f1, artifact_path)
    return {
        "metrics": metrics,
        "feature_columns": list(X.columns),
        "label_classes": label_classes,
    }


def predict_vfl_batch(bundle: dict[str, Any], X_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Return predicted class indices and max softmax probability per row."""
    input_dims: list[int] = bundle["input_dims"]
    party_cols: list[list[str]] = bundle["party_columns"]
    scalers: list = bundle["scalers"]
    num_classes: int = bundle["num_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VFLModel(input_dims, embed_dim=VFL_EMBED_DIM, num_classes=num_classes, hidden_dim=VFL_HIDDEN_DIM)
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device)
    model.eval()

    parts = []
    for i in range(3):
        cols = party_cols[i]
        block = pd.DataFrame({c: X_df[c] if c in X_df.columns else np.nan for c in cols})
        block = block.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float64)
        parts.append(scalers[i].transform(block))

    n = len(X_df)
    probs_out = np.zeros((n, num_classes), dtype=np.float64)

    with torch.no_grad():
        batch = 512
        for start in range(0, n, batch):
            end = min(start + batch, n)
            xb = [
                torch.tensor(parts[i][start:end], dtype=torch.float32, device=device)
                for i in range(3)
            ]
            logits = model(xb)
            pr = torch.softmax(logits, dim=1).cpu().numpy()
            probs_out[start:end] = pr

    pred_idx = probs_out.argmax(axis=1)
    max_p = probs_out.max(axis=1)
    return pred_idx, max_p
