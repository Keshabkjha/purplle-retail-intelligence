import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _ensure_2d(x):
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


class AdaptiveBinaryModel:
    """Tiny online binary classifier with persistence."""

    def __init__(self, state_path: str):
        self.state_path = state_path
        self.clf = SGDClassifier(
            loss="log_loss",
            random_state=42,
            alpha=0.0005,
            max_iter=1,
            tol=None,
            learning_rate="optimal",
        )
        self.fitted = False
        self.update_count = 0
        self._load()

    def _load(self):
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "rb") as f:
                state = pickle.load(f)
            self.clf = state["clf"]
            self.fitted = state["fitted"]
            self.update_count = state.get("update_count", 0)
        except Exception:
            self.fitted = False

    def save(self):
        if not self.state_path:
            return
        try:
            state_dir = os.path.dirname(self.state_path)
            if state_dir:
                os.makedirs(state_dir, exist_ok=True)
            with open(self.state_path, "wb") as f:
                pickle.dump(
                    {
                        "clf": self.clf,
                        "fitted": self.fitted,
                        "update_count": self.update_count,
                    },
                    f,
                )
        except Exception:
            pass

    def update(self, sample: Sequence[float], label: int):
        X = _ensure_2d(sample)
        y = np.asarray([int(label)], dtype=np.int32)

        if not self.fitted:
            self.clf.partial_fit(X, y, classes=np.array([0, 1], dtype=np.int32))
            self.fitted = True
        else:
            self.clf.partial_fit(X, y)
        self.update_count += 1
        if self.update_count % 10 == 0:
            self.save()

    def predict_proba(self, sample: Sequence[float], fallback: float = 0.5) -> float:
        if not self.fitted:
            return float(fallback)
        X = _ensure_2d(sample)
        try:
            return float(self.clf.predict_proba(X)[0, 1])
        except Exception:
            try:
                score = float(self.clf.decision_function(X)[0])
                return float(1.0 / (1.0 + np.exp(-score)))
            except Exception:
                return float(fallback)


class StaticBinaryModel:
    """Loads a supervised binary classifier trained offline."""

    def __init__(self, state_path: str):
        self.state_path = state_path
        self.clf = None
        self.fitted = False
        self.metadata = {}
        self._load()

    def _load(self):
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "rb") as f:
                state = pickle.load(f)
            if isinstance(state, dict) and "model" in state:
                self.clf = state["model"]
                self.metadata = {k: v for k, v in state.items() if k != "model"}
            else:
                self.clf = state
                self.metadata = {}
            self.fitted = self.clf is not None
        except Exception:
            self.clf = None
            self.fitted = False
            self.metadata = {}

    def save(self, metadata: dict | None = None):
        if not self.state_path or self.clf is None:
            return
        try:
            state_dir = os.path.dirname(self.state_path)
            if state_dir:
                os.makedirs(state_dir, exist_ok=True)
            payload = {"model": self.clf, "fitted": True, "metadata": metadata or self.metadata}
            with open(self.state_path, "wb") as f:
                pickle.dump(payload, f)
        except Exception:
            pass

    def predict_proba(self, sample: Sequence[float], fallback: float = 0.5) -> float:
        if not self.fitted or self.clf is None:
            return float(fallback)
        X = _ensure_2d(sample)
        try:
            return float(self.clf.predict_proba(X)[0, 1])
        except Exception:
            try:
                score = float(self.clf.decision_function(X)[0])
                return float(1.0 / (1.0 + np.exp(-score)))
            except Exception:
                return float(fallback)


def build_staff_feature_vector(
    torso_match_ratio: float,
    is_clothing_staff: bool,
    zone_id: str | None,
    wx: float,
    billing_duration_sec: float,
    total_duration_sec: float,
    camera_count: int,
):
    return np.array(
        [
            float(np.clip(torso_match_ratio, 0.0, 1.0)),
            1.0 if is_clothing_staff else 0.0,
            1.0 if zone_id == "BILLING" else 0.0,
            1.0 if wx > 820 else 0.0,
            float(np.clip(billing_duration_sec / 180.0, 0.0, 1.0)),
            float(np.clip(total_duration_sec / 300.0, 0.0, 1.0)),
            float(np.clip(camera_count / 3.0, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def build_identity_feature_vector(
    spatial_score: float,
    temporal_score: float,
    visual_score: float,
    camera_score: float,
    zone_score: float,
    dist_norm: float,
    time_norm: float,
):
    return np.array(
        [
            float(np.clip(spatial_score, 0.0, 1.0)),
            float(np.clip(temporal_score, 0.0, 1.0)),
            float(np.clip(visual_score, -1.0, 1.0)),
            float(np.clip(camera_score, 0.0, 1.0)),
            float(np.clip(zone_score, 0.0, 1.0)),
            float(np.clip(dist_norm, 0.0, 1.0)),
            float(np.clip(time_norm, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def _extract_label(record: dict) -> int:
    for key in ("label", "is_staff", "match", "same_person"):
        if key in record:
            return int(bool(record[key]))
    raise ValueError(
        "Training record must include a label field such as label/is_staff/match/same_person"
    )


def _extract_feature_array(
    record: dict, feature_builder, required_keys: tuple[str, ...]
) -> np.ndarray:
    if "features" in record:
        return np.asarray(record["features"], dtype=np.float32)

    kwargs = {key: record[key] for key in required_keys}
    return np.asarray(feature_builder(**kwargs), dtype=np.float32)


def load_jsonl_records(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            records.append(json.loads(line))
    return records


def train_binary_classifier(X: Sequence[Sequence[float]], y: Sequence[int]):
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int32)
    if len(np.unique(y_arr)) < 2:
        raise ValueError(
            "Need at least one positive and one negative example to train a binary classifier"
        )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    clf.fit(X_arr, y_arr)
    return clf


def train_staff_model_from_records(records: Sequence[dict], output_path: str):
    features = []
    labels = []
    for record in records:
        features.append(
            _extract_feature_array(
                record,
                build_staff_feature_vector,
                (
                    "torso_match_ratio",
                    "is_clothing_staff",
                    "zone_id",
                    "wx",
                    "billing_duration_sec",
                    "total_duration_sec",
                    "camera_count",
                ),
            )
        )
        labels.append(_extract_label(record))

    model = train_binary_classifier(features, labels)
    payload = {
        "model": model,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(records),
        "feature_kind": "staff",
    }
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    return output_path


def train_identity_model_from_records(records: Sequence[dict], output_path: str):
    features = []
    labels = []
    for record in records:
        features.append(
            _extract_feature_array(
                record,
                build_identity_feature_vector,
                (
                    "spatial_score",
                    "temporal_score",
                    "visual_score",
                    "camera_score",
                    "zone_score",
                    "dist_norm",
                    "time_norm",
                ),
            )
        )
        labels.append(_extract_label(record))

    model = train_binary_classifier(features, labels)
    payload = {
        "model": model,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(records),
        "feature_kind": "identity",
    }
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    return output_path


def train_staff_model_from_jsonl(input_path: str, output_path: str):
    return train_staff_model_from_records(load_jsonl_records(input_path), output_path)


def train_identity_model_from_jsonl(input_path: str, output_path: str):
    return train_identity_model_from_records(load_jsonl_records(input_path), output_path)


@dataclass
class AdaptiveModelRegistry:
    """Keeps lightweight learned models for staff detection and identity matching."""

    base_dir: str = "pipeline/model_state"

    def __post_init__(self):
        self.staff_model = AdaptiveBinaryModel(os.path.join(self.base_dir, "staff_online.pkl"))
        self.identity_model = AdaptiveBinaryModel(
            os.path.join(self.base_dir, "identity_online.pkl")
        )
        self.supervised_staff_model = StaticBinaryModel(
            os.getenv(
                "STAFF_SUPERVISED_MODEL_PATH", os.path.join(self.base_dir, "staff_supervised.pkl")
            )
        )
        self.supervised_identity_model = StaticBinaryModel(
            os.getenv(
                "IDENTITY_SUPERVISED_MODEL_PATH",
                os.path.join(self.base_dir, "identity_supervised.pkl"),
            )
        )

    def predict_staff_probability(self, sample: Sequence[float], fallback: float = 0.5) -> float:
        if self.supervised_staff_model.fitted:
            return self.supervised_staff_model.predict_proba(sample, fallback=fallback)
        return self.staff_model.predict_proba(sample, fallback=fallback)

    def predict_identity_probability(self, sample: Sequence[float], fallback: float = 0.5) -> float:
        if self.supervised_identity_model.fitted:
            return self.supervised_identity_model.predict_proba(sample, fallback=fallback)
        return self.identity_model.predict_proba(sample, fallback=fallback)
