import json

from pipeline.adaptive_models import (
    AdaptiveModelRegistry,
    build_identity_feature_vector,
    build_staff_feature_vector,
    train_identity_model_from_jsonl,
    train_staff_model_from_jsonl,
)


def test_supervised_artifacts_load_at_runtime(tmp_path, monkeypatch):
    staff_labels = tmp_path / "staff_labels.jsonl"
    staff_labels.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "torso_match_ratio": 0.95,
                        "is_clothing_staff": True,
                        "zone_id": "BILLING",
                        "wx": 860.0,
                        "billing_duration_sec": 180.0,
                        "total_duration_sec": 360.0,
                        "camera_count": 2,
                        "label": 1,
                    }
                ),
                json.dumps(
                    {
                        "torso_match_ratio": 0.05,
                        "is_clothing_staff": False,
                        "zone_id": "EB_KOREAN",
                        "wx": 120.0,
                        "billing_duration_sec": 0.0,
                        "total_duration_sec": 35.0,
                        "camera_count": 1,
                        "label": 0,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    identity_labels = tmp_path / "identity_labels.jsonl"
    identity_labels.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "spatial_score": 0.95,
                        "temporal_score": 0.92,
                        "visual_score": 0.88,
                        "camera_score": 0.90,
                        "zone_score": 0.92,
                        "dist_norm": 0.05,
                        "time_norm": 0.08,
                        "label": 1,
                    }
                ),
                json.dumps(
                    {
                        "spatial_score": 0.10,
                        "temporal_score": 0.18,
                        "visual_score": 0.12,
                        "camera_score": 0.22,
                        "zone_score": 0.20,
                        "dist_norm": 0.88,
                        "time_norm": 0.90,
                        "label": 0,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    staff_model_path = tmp_path / "staff_supervised.pkl"
    identity_model_path = tmp_path / "identity_supervised.pkl"
    train_staff_model_from_jsonl(str(staff_labels), str(staff_model_path))
    train_identity_model_from_jsonl(str(identity_labels), str(identity_model_path))

    monkeypatch.setenv("STAFF_SUPERVISED_MODEL_PATH", str(staff_model_path))
    monkeypatch.setenv("IDENTITY_SUPERVISED_MODEL_PATH", str(identity_model_path))

    registry = AdaptiveModelRegistry(base_dir=str(tmp_path / "online"))

    assert registry.supervised_staff_model.fitted is True
    assert registry.supervised_identity_model.fitted is True
    assert registry.supervised_staff_model.metadata["num_samples"] == 2
    assert registry.supervised_identity_model.metadata["num_samples"] == 2

    staff_features = build_staff_feature_vector(
        torso_match_ratio=0.95,
        is_clothing_staff=True,
        zone_id="BILLING",
        wx=860.0,
        billing_duration_sec=180.0,
        total_duration_sec=360.0,
        camera_count=2,
    )
    identity_features = build_identity_feature_vector(
        spatial_score=0.95,
        temporal_score=0.92,
        visual_score=0.88,
        camera_score=0.90,
        zone_score=0.92,
        dist_norm=0.05,
        time_norm=0.08,
    )

    assert registry.predict_staff_probability(staff_features, fallback=0.0) > registry.staff_model.predict_proba(staff_features, fallback=0.0)
    assert registry.predict_identity_probability(identity_features, fallback=0.0) > registry.identity_model.predict_proba(identity_features, fallback=0.0)
