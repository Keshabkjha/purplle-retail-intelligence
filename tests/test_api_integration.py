import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import RateLimiter, app
from pipeline.detect import post_event

engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(name="db_session")
def fixture_db_session():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(name="client")
def fixture_client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_rate_limit_blocks_excess(client):
    original_limiter = app.state.rate_limiter
    app.state.rate_limiter = RateLimiter(limit=2, window_seconds=60)
    try:
        assert client.get("/stores/ST1008/metrics").status_code == 200
        assert client.get("/stores/ST1008/metrics").status_code == 200
        response = client.get("/stores/ST1008/metrics")
        assert response.status_code == 429
        assert response.json()["detail"] == "Rate limit exceeded"
    finally:
        app.state.rate_limiter = original_limiter


def test_ingest_batch_limit(client):
    original_max = app.state.max_ingest_batch
    app.state.max_ingest_batch = 1
    try:
        payload = [
            {
                "event_id": str(uuid.uuid4()),
                "store_id": "ST1008",
                "camera_id": "CAM_ENTRY_01",
                "visitor_id": "VIS_200",
                "event_type": "ENTRY",
                "timestamp": "2026-04-10T10:00:00Z",
                "zone_id": "ENTRY",
                "dwell_ms": 0,
                "is_staff": False,
                "confidence": 0.95,
                "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1},
            },
            {
                "event_id": str(uuid.uuid4()),
                "store_id": "ST1008",
                "camera_id": "CAM_ENTRY_01",
                "visitor_id": "VIS_201",
                "event_type": "ENTRY",
                "timestamp": "2026-04-10T10:00:02Z",
                "zone_id": "ENTRY",
                "dwell_ms": 0,
                "is_staff": False,
                "confidence": 0.95,
                "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1},
            },
        ]
        response = client.post("/events/ingest", json=payload)
        assert response.status_code == 413
        assert "Batch too large" in response.json()["detail"]
    finally:
        app.state.max_ingest_batch = original_max


def test_pipeline_post_event_payload(monkeypatch):
    captured = {}

    class DummyResponse:
        status_code = 207
        text = ""

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        return DummyResponse()

    monkeypatch.setattr("pipeline.detect.requests.post", fake_post)
    event = {
        "event_id": str(uuid.uuid4()),
        "store_id": "ST1008",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": "42",
        "event_type": "ENTRY",
        "timestamp": "2026-04-10T10:00:00Z",
        "zone_id": "ENTRY",
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.9,
        "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1},
    }
    post_event(event)

    assert captured["url"].endswith("/events/ingest")
    assert captured["payload"] == [event]
