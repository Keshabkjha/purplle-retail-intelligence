import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.database import Base, DBEvent, SessionLocal, engine
from app.main import app
from app.metrics import get_store_metrics_data
from pipeline.adaptive_models import (
    AdaptiveBinaryModel,
    AdaptiveModelRegistry,
    StaticBinaryModel,
    build_identity_feature_vector,
    build_staff_feature_vector,
    train_identity_model_from_jsonl,
    train_staff_model_from_jsonl,
)
from pipeline.detect import (
    CrossCameraSessionTracker,
    camera_transition_prior,
    compare_appearance,
    compute_appearance_embedding,
    update_staff_status,
    zone_transition_prior,
)


@pytest.fixture
def temp_state_file(tmp_path):
    return os.path.join(tmp_path, "session_state.json")


@pytest.fixture
def db_session():
    # Setup clean sqlite in-memory DB for ingestion/metrics tests
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    return TestClient(app)


def test_cross_camera_tracker_new_session(temp_state_file):
    tracker = CrossCameraSessionTracker(state_file=temp_state_file)
    t = datetime(2026, 4, 10, 16, 40, 0)

    # Registering track_id 1 in entry camera
    vid = tracker.get_unified_id(
        track_id=1,
        camera_id="CAM_ENTRY_01",
        wx=10.0,
        wy=250.0,
        current_time=t,
        is_staff_initial=False,
    )

    assert vid == "ID_60001"
    assert "ID_60001" in tracker.sessions
    assert tracker.sessions["ID_60001"]["last_seen_camera"] == "CAM_ENTRY_01"
    assert tracker.sessions["ID_60001"]["last_seen_x"] == 10.0
    assert tracker.sessions["ID_60001"]["last_seen_y"] == 250.0


def test_cross_camera_tracker_spatial_temporal_match(temp_state_file):
    tracker = CrossCameraSessionTracker(state_file=temp_state_file)
    t1 = datetime(2026, 4, 10, 16, 40, 0)

    # 1. Visitor exits ENTRY camera at time t1
    tracker.get_unified_id(
        track_id=3,
        camera_id="CAM_ENTRY_01",
        wx=145.0,
        wy=300.0,
        current_time=t1,
        is_staff_initial=False,
    )

    # 2. A track appears in MAIN floor 3 seconds later at (150, 305) - very close (within 150px and 30s)
    t2 = t1 + timedelta(seconds=3)
    vid_match = tracker.get_unified_id(
        track_id=2,
        camera_id="CAM_MAIN_01",
        wx=150.0,
        wy=305.0,
        current_time=t2,
        is_staff_initial=False,
    )

    assert vid_match == "ID_60003"  # Should correctly unify to the original VIS_3!
    assert tracker.sessions["ID_60003"]["last_seen_camera"] == "CAM_MAIN_01"
    assert tracker.sessions["ID_60003"]["camera_track_ids"]["CAM_ENTRY_01"] == 3
    assert tracker.sessions["ID_60003"]["camera_track_ids"]["CAM_MAIN_01"] == 2
    assert tracker.is_reentry("ID_60003", "CAM_MAIN_01") is True


def test_cross_camera_tracker_no_match_large_gap(temp_state_file):
    tracker = CrossCameraSessionTracker(state_file=temp_state_file)
    t1 = datetime(2026, 4, 10, 16, 40, 0)

    # 1. Visitor exiting ENTRY camera at time t1
    tracker.get_unified_id(
        track_id=5,
        camera_id="CAM_ENTRY_01",
        wx=10.0,
        wy=200.0,
        current_time=t1,
        is_staff_initial=False,
    )

    # 2. Track appears in MAIN floor 45 seconds later (exceeds 30s temporal threshold)
    t2 = t1 + timedelta(seconds=45)
    vid_no_match_time = tracker.get_unified_id(
        track_id=6,
        camera_id="CAM_MAIN_01",
        wx=15.0,
        wy=205.0,
        current_time=t2,
        is_staff_initial=False,
    )

    assert vid_no_match_time != "ID_60005"
    assert vid_no_match_time == "ID_60006"


def test_hybrid_staff_detection(temp_state_file, tmp_path, monkeypatch):
    import pipeline.detect as detect_module

    monkeypatch.setattr(
        detect_module,
        "MODEL_REGISTRY",
        AdaptiveModelRegistry(base_dir=str(tmp_path / "runtime_state")),
    )
    tracker = CrossCameraSessionTracker(state_file=temp_state_file)
    t = datetime(2026, 4, 10, 16, 40, 0)

    sess = {
        "current_zone": None,
        "enter_time": t,
        "first_seen_time": t,
        "last_seen": t,
        "dwell_sent_count": 0,
        "seq": 1,
        "is_staff": False,
    }
    vid = "ID_600099"
    tracker.sessions[vid] = {
        "unified_id": vid,
        "last_seen_camera": "CAM_ENTRY_01",
        "last_seen_time": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_seen_x": 0.0,
        "last_seen_y": 0.0,
        "is_staff": False,
        "camera_track_ids": {"CAM_ENTRY_01": 99},
    }

    is_staff = update_staff_status(
        vid=vid,
        sess=sess,
        current_time=t + timedelta(seconds=10),
        zone_id="ENTRY",
        wx=50.0,
        wy=250.0,
        is_clothing_staff=False,
        tracker=tracker,
    )
    assert is_staff is False

    is_staff = update_staff_status(
        vid=vid,
        sess=sess,
        current_time=t + timedelta(seconds=20),
        zone_id="ENTRY",
        wx=50.0,
        wy=250.0,
        is_clothing_staff=True,
        tracker=tracker,
    )
    assert is_staff is True


def test_appearance_based_matching(temp_state_file):
    # Mock BGR frames representing same and different visual properties
    frame_red = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_red[:, :, 2] = 255  # Red crop

    frame_blue = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_blue[:, :, 0] = 255  # Blue crop

    hist_red = compute_appearance_embedding(frame_red)
    hist_blue = compute_appearance_embedding(frame_blue)

    # Red should match red perfectly, but not blue
    assert compare_appearance(hist_red, hist_red) > 0.90
    assert compare_appearance(hist_red, hist_blue) < 0.20

    # Test integrated CrossCameraSessionTracker visual matching
    tracker = CrossCameraSessionTracker(state_file=temp_state_file)
    t = datetime(2026, 4, 10, 16, 40, 0)

    # 1. Register a visitor with red clothing in entry camera
    tracker.get_unified_id(
        track_id=1,
        camera_id="CAM_ENTRY_01",
        wx=140.0,
        wy=300.0,
        current_time=t,
        is_staff_initial=False,
        box=[0, 0, 100, 100],
        frame=frame_red,
    )

    # 2. A track appears in MAIN floor with red clothing (should match perfectly)
    vid_match = tracker.get_unified_id(
        track_id=2,
        camera_id="CAM_MAIN_01",
        wx=145.0,
        wy=305.0,
        current_time=t + timedelta(seconds=2),
        is_staff_initial=False,
        box=[0, 0, 100, 100],
        frame=frame_red,
    )
    assert vid_match == "ID_60001"


def test_transition_priors():
    # Should yield higher score for an adjacent layout transition
    assert camera_transition_prior("cam1", "CAM1") > camera_transition_prior(
        "cam1", "CAM5"
    )
    assert camera_transition_prior("CAM_MAIN_01", "CAM_MAIN_01") == 1.0

    # Zone continuity should favor identical or adjacent retail-zone transitions.
    assert zone_transition_prior("EB_KOREAN", "EB_KOREAN") == 1.0
    assert zone_transition_prior("ENTRY", "EB_KOREAN") > zone_transition_prior("EB_KOREAN", "LAKME")


def test_learned_staff_model_separates_examples(tmp_path):
    model = AdaptiveBinaryModel(str(tmp_path / "staff.pkl"))

    pos = build_staff_feature_vector(
        torso_match_ratio=0.92,
        is_clothing_staff=True,
        zone_id="BILLING",
        wx=860.0,
        billing_duration_sec=150.0,
        total_duration_sec=260.0,
        camera_count=2,
    )
    neg = build_staff_feature_vector(
        torso_match_ratio=0.08,
        is_clothing_staff=False,
        zone_id="EB_KOREAN",
        wx=120.0,
        billing_duration_sec=0.0,
        total_duration_sec=40.0,
        camera_count=1,
    )

    model.update(pos, 1)
    model.update(neg, 0)

    assert model.predict_proba(pos, fallback=0.0) > model.predict_proba(neg, fallback=0.0)


def test_learned_identity_model_separates_examples(tmp_path):
    model = AdaptiveBinaryModel(str(tmp_path / "identity.pkl"))

    matched = build_identity_feature_vector(
        spatial_score=0.96,
        temporal_score=0.92,
        visual_score=0.85,
        camera_score=0.95,
        zone_score=0.90,
        dist_norm=0.04,
        time_norm=0.08,
    )
    mismatched = build_identity_feature_vector(
        spatial_score=0.10,
        temporal_score=0.12,
        visual_score=0.15,
        camera_score=0.20,
        zone_score=0.20,
        dist_norm=0.90,
        time_norm=0.88,
    )

    model.update(matched, 1)
    model.update(mismatched, 0)

    assert model.predict_proba(matched, fallback=0.0) > model.predict_proba(
        mismatched, fallback=0.0
    )


def test_supervised_artifact_training_roundtrip(tmp_path):
    staff_jsonl = tmp_path / "staff_labels.jsonl"
    staff_jsonl.write_text(
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

    reid_jsonl = tmp_path / "reid_labels.jsonl"
    reid_jsonl.write_text(
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

    train_staff_model_from_jsonl(str(staff_jsonl), str(tmp_path / "staff_supervised.pkl"))
    train_identity_model_from_jsonl(str(reid_jsonl), str(tmp_path / "identity_supervised.pkl"))

    staff_model = StaticBinaryModel(str(tmp_path / "staff_supervised.pkl"))
    identity_model = StaticBinaryModel(str(tmp_path / "identity_supervised.pkl"))

    staff_pos = build_staff_feature_vector(
        torso_match_ratio=0.95,
        is_clothing_staff=True,
        zone_id="BILLING",
        wx=860.0,
        billing_duration_sec=180.0,
        total_duration_sec=360.0,
        camera_count=2,
    )
    staff_neg = build_staff_feature_vector(
        torso_match_ratio=0.05,
        is_clothing_staff=False,
        zone_id="EB_KOREAN",
        wx=120.0,
        billing_duration_sec=0.0,
        total_duration_sec=35.0,
        camera_count=1,
    )
    identity_pos = build_identity_feature_vector(
        spatial_score=0.95,
        temporal_score=0.92,
        visual_score=0.88,
        camera_score=0.90,
        zone_score=0.92,
        dist_norm=0.05,
        time_norm=0.08,
    )
    identity_neg = build_identity_feature_vector(
        spatial_score=0.10,
        temporal_score=0.18,
        visual_score=0.12,
        camera_score=0.22,
        zone_score=0.20,
        dist_norm=0.88,
        time_norm=0.90,
    )

    assert staff_model.predict_proba(staff_pos, fallback=0.0) > staff_model.predict_proba(
        staff_neg, fallback=0.0
    )
    assert identity_model.predict_proba(identity_pos, fallback=0.0) > identity_model.predict_proba(
        identity_neg, fallback=0.0
    )

    registry = AdaptiveModelRegistry(base_dir=str(tmp_path))
    assert registry.predict_staff_probability(
        staff_pos, fallback=0.0
    ) > registry.predict_staff_probability(staff_neg, fallback=0.0)
    assert registry.predict_identity_probability(
        identity_pos, fallback=0.0
    ) > registry.predict_identity_probability(identity_neg, fallback=0.0)


def test_ingest_batch_size_limit(client):
    # Generating 501 mock events (exceeding cap of 500)
    large_batch = []
    for i in range(501):
        large_batch.append(
            {
                "event_id": f"evt_{i}",
                "store_id": "ST1008",
                "camera_id": "CAM_ENTRY_01",
                "visitor_id": "ID_60001",
                "event_type": "ENTRY",
                "timestamp": "2026-04-10T16:40:00Z",
                "confidence": 0.95,
                "metadata": {"session_seq": 1},
            }
        )

    response = client.post("/events/ingest", json=large_batch)
    assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    assert "exceeds the maximum limit" in response.json()["detail"]


def test_ingest_monotonicity(client, db_session):
    # Post first event with session_seq = 2
    evt1 = {
        "event_id": "a1111111-2222-3333-4444-555555555555",
        "store_id": "ST1008",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": "ID_6000MONO",
        "event_type": "ENTRY",
        "timestamp": "2026-04-10T16:40:00Z",
        "confidence": 0.95,
        "metadata": {"session_seq": 2},
    }

    # Post second event with session_seq = 1 (violates monotonicity)
    evt2 = {
        "event_id": "b1111111-2222-3333-4444-555555555555",
        "store_id": "ST1008",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": "ID_6000MONO",
        "event_type": "ZONE_ENTER",
        "zone_id": "EB_KOREAN",
        "timestamp": "2026-04-10T16:40:10Z",
        "confidence": 0.95,
        "metadata": {"session_seq": 1},
    }

    # Ingest evt1 successfully
    response = client.post("/events/ingest", json=[evt1])
    assert response.status_code == 207
    assert response.json()["ingested"] == 1

    # Ingesting evt2 should fail validation
    response2 = client.post("/events/ingest", json=[evt2])
    assert response2.status_code == 207
    assert response2.json()["failed"] == 1
    assert "Monotonicity violation" in response2.json()["errors"][0]["error"]


def test_reentry_dwell_session_correction(db_session):
    # Seed mock multi-visit events for the same visitor_id (VIS_RE)
    # Customer enters, spends 30s in store, exits. Re-enters 2 mins later, spends 40s in store, exits.
    events = [
        DBEvent(
            event_id="e1",
            store_id="ST1008",
            camera_id="CAM_ENTRY_01",
            visitor_id="ID_6000RE",
            event_type="ENTRY",
            timestamp="2026-04-10T16:40:00Z",
            zone_id="ENTRY",
            confidence=0.95,
            metadata_json={"session_seq": 1},
        ),
        DBEvent(
            event_id="e2",
            store_id="ST1008",
            camera_id="CAM_ENTRY_01",
            visitor_id="ID_6000RE",
            event_type="EXIT",
            timestamp="2026-04-10T16:40:30Z",
            zone_id="ENTRY",
            confidence=0.95,
            metadata_json={"session_seq": 2},
        ),
        DBEvent(
            event_id="e3",
            store_id="ST1008",
            camera_id="CAM_ENTRY_01",
            visitor_id="ID_6000RE",
            event_type="REENTRY",
            timestamp="2026-04-10T16:42:30Z",
            zone_id="ENTRY",
            confidence=0.95,
            metadata_json={"session_seq": 3},
        ),
        DBEvent(
            event_id="e4",
            store_id="ST1008",
            camera_id="CAM_ENTRY_01",
            visitor_id="ID_6000RE",
            event_type="EXIT",
            timestamp="2026-04-10T16:43:10Z",
            zone_id="ENTRY",
            confidence=0.95,
            metadata_json={"session_seq": 4},
        ),
    ]
    db_session.bulk_save_objects(events)
    db_session.commit()

    # Calculate metrics
    res = get_store_metrics_data("ST1008", db_session)

    # Unique visitors count must be 1 (single physical person)
    assert res["unique_visitors"] == 1

    # Dwell minutes calculation should segment visits (30s + 40s = 70s total -> 1.17 mins)
    # NOT 190s total (which would include the 2 min gap outside the store)
    expected_dwell_min = round((70.0 / 60.0), 2)
    assert res["average_dwell_minutes"] == expected_dwell_min


def test_stale_feed_latency(client, db_session):
    # Ingest a clean event to establish the store in DB
    evt = {
        "event_id": "c1111111-2222-3333-4444-555555555555",
        "store_id": "ST1008",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": "ID_6000STALE",
        "event_type": "ENTRY",
        "timestamp": "2026-04-10T16:40:00Z",
        "confidence": 0.95,
        "metadata": {"session_seq": 1},
    }
    client.post("/events/ingest", json=[evt])

    # Mock last_ingest_time on the app state directly to simulate a 15-minute lag
    app.state.last_ingest_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        minutes=15
    )

    res = client.get("/health")
    assert res.status_code == 200
    # The store feed should be flagged as stale due to high latency lag
    assert res.json()["stores"]["ST1008"]["stale_feed"] is True
