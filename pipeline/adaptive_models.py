import os
import pickle
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.linear_model import SGDClassifier


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
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
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


@dataclass
class AdaptiveModelRegistry:
    """Keeps lightweight learned models for staff detection and identity matching."""

    base_dir: str = "pipeline/model_state"

    def __post_init__(self):
        self.staff_model = AdaptiveBinaryModel(os.path.join(self.base_dir, "staff.pkl"))
        self.identity_model = AdaptiveBinaryModel(os.path.join(self.base_dir, "identity.pkl"))
