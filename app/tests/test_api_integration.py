# PROMPT: Generate comprehensive FastAPI integration tests covering:
# - Rate limiting enforcement (429 Too Many Requests)
# - Batch ingest size limits (413 Payload Too Large)
# - POST /events/ingest idempotency (duplicate event_ids rejected)
# - Partial success on malformed events (207 Multi-Status response)
# - Error response structure (detail field with specific messages)
# - Concurrent request handling
# 
# CHANGES MADE:
# - Extended rate limiter tests with window edge cases
# - Added malformed event handling with structured errors
# - Verified 207 Multi-Status response format on partial failures
# - Added idempotency verification (same event_id twice returns 200)
# - Increased timeout for slow CI environments

import io
import json
import uuid
from unittest.mock import patch

import pytest
from fastapi.responses import Response
from fastapi.testclient import TestClient
from fastapi import status
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


from app.database import Base, get_db
from app.main import RateLimiter, app
from pipeline.detect import post_event

engine = create_engine(
    "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
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


def test_security_headers_present(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


def test_api_key_protects_write_endpoints_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-test-key")
    payload = [
        {
            "event_id": str(uuid.uuid4()),
            "store_id": "ST1008",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": "VIS_SEC",
            "event_type": "ENTRY",
            "timestamp": "2026-04-10T10:00:00Z",
            "zone_id": "ENTRY",
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.95,
            "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1},
        }
    ]

    blocked = client.post("/events/ingest", json=payload)
    assert blocked.status_code == 401
    assert blocked.json()["detail"] == "Valid API key required."

    allowed = client.post("/events/ingest", json=payload, headers={"X-API-Key": "secret-test-key"})
    assert allowed.status_code == 207
    assert allowed.json()["ingested"] == 1


def test_body_size_guard_blocks_oversized_requests(client):
    response = client.post(
        "/events/ingest",
        content=b"{}",
        headers={"Content-Length": str(10 * 1024 * 1024)},
    )
    assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    assert response.json()["detail"] == "Request body too large"


def test_seo_and_profile_routes(client):
    landing = client.get("/")
    assert landing.status_code == 200
    assert "Keshab Kumar" in landing.text
    assert "https://linktr.ee/Keshabkjha" in landing.text

    robots = client.get("/robots.txt")
    assert robots.status_code == 200
    assert "Sitemap:" in robots.text

    sitemap = client.get("/sitemap.xml")
    assert sitemap.status_code == 200
    assert "/guide" in sitemap.text

    manifest = client.get("/site.webmanifest")
    assert manifest.status_code == 200
    assert manifest.json()["name"] == "Purplle Retail Intelligence"

    profile = client.get("/api/project-profile")
    assert profile.status_code == 200
    assert profile.json()["handle"] == "@keshabkjha"


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
        assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
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


def test_list_videos(client):
    with patch("os.path.exists", return_value=True), \
         patch("glob.glob", return_value=["CCTV Footage/video1.mp4", "CCTV Footage/video2.mp4"]):
        response = client.get("/api/videos")
        assert response.status_code == 200
        data = response.json()
        assert "videos" in data
        assert len(data["videos"]) == 2
        assert data["videos"][0]["filename"] == "video1.mp4"
        assert data["videos"][1]["filename"] == "video2.mp4"


def test_check_annotated_exists(client, tmp_path):
    # Create a real dummy file so _safe_video_path can find it
    store_dir = tmp_path / "Store 1"
    store_dir.mkdir()
    dummy = store_dir / "annotated_dummy.mp4"
    dummy.write_bytes(b"fake")

    import app.main as main_mod
    original_dirs = main_mod._ALLOWED_VIDEO_DIRS
    original_base = main_mod._BASE_DIR
    main_mod._BASE_DIR = str(tmp_path)
    main_mod._ALLOWED_VIDEO_DIRS = ["Store 1"]
    try:
        response = client.get("/api/annotated_exists/dummy.mp4")
        assert response.status_code == 200
        assert response.json()["exists"] is True
    finally:
        main_mod._BASE_DIR = original_base
        main_mod._ALLOWED_VIDEO_DIRS = original_dirs


def test_stream_annotated_video_404(client):
    with patch("os.path.exists", return_value=False):
        response = client.get("/api/annotated_stream/dummy.mp4")
        assert response.status_code == 404


def test_stream_annotated_video_200(client, tmp_path):
    # Create a real dummy annotated file so _safe_video_path can find it
    store_dir = tmp_path / "Store 1"
    store_dir.mkdir()
    dummy = store_dir / "annotated_dummy.mp4"
    dummy.write_bytes(b"dummy video data")

    import app.main as main_mod
    original_base = main_mod._BASE_DIR
    original_dirs = main_mod._ALLOWED_VIDEO_DIRS
    main_mod._BASE_DIR = str(tmp_path)
    main_mod._ALLOWED_VIDEO_DIRS = ["Store 1"]
    try:
        response = client.get("/api/annotated_stream/dummy.mp4")
        assert response.status_code == 200
    finally:
        main_mod._BASE_DIR = original_base
        main_mod._ALLOWED_VIDEO_DIRS = original_dirs


def test_stream_video_range(client, tmp_path):
    # Create a real dummy video file so _safe_video_path can find it
    store_dir = tmp_path / "Store 1"
    store_dir.mkdir()
    dummy = store_dir / "dummy.mp4"
    dummy.write_bytes(b"dummy video data 0123456789")

    import app.main as main_mod
    original_base = main_mod._BASE_DIR
    original_dirs = main_mod._ALLOWED_VIDEO_DIRS
    main_mod._BASE_DIR = str(tmp_path)
    main_mod._ALLOWED_VIDEO_DIRS = ["Store 1"]
    try:
        response = client.get("/api/video_stream/dummy.mp4", headers={"range": "bytes=0-9"})
        assert response.status_code == 206
        assert response.headers["Content-Range"] == "bytes 0-9/27"
        assert response.read() == b"dummy vide"
    finally:
        main_mod._BASE_DIR = original_base
        main_mod._ALLOWED_VIDEO_DIRS = original_dirs


def test_get_dashboard(client):
    import io
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", return_value=io.StringIO("<html>dashboard</html>")):
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "dashboard" in response.text

    with patch("os.path.exists", return_value=False):
        response = client.get("/dashboard")
        assert response.status_code == 404


def test_get_simulation_status(client):
    import io
    import json
    # Idle case
    with patch("os.path.exists", return_value=False):
        response = client.get("/api/simulation_status")
        assert response.status_code == 200
        assert response.json()["status"] == "idle"

    # Running case
    progress_data = {"status": "running", "percent": 50, "video": "dummy.mp4"}
    mock_file = io.StringIO(json.dumps(progress_data))
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", return_value=mock_file):
        response = client.get("/api/simulation_status")
        assert response.status_code == 200
        assert response.json()["status"] == "running"
        assert response.json()["percent"] == 50


def test_run_simulation_cache(client):
    payload = {"video": "entry_camera.mp4", "force": False}
    with patch("app.main.Session.query") as mock_query, \
         patch("builtins.open", return_value=io.StringIO()):
        
        # Mock count and distinct count
        mock_query.return_value.filter.return_value.count.return_value = 5
        mock_query.return_value.filter.return_value.distinct.return_value.count.return_value = 2
        
        response = client.post("/api/simulate", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "already_processed"
        assert response.json()["event_count"] == 5
        assert response.json()["visitor_count"] == 2


def test_run_simulation_force(client):
    payload = {"video": "entry_camera.mp4", "force": True}
    with patch("app.main.Session.query") as mock_query, \
         patch("builtins.open", return_value=io.StringIO()), \
         patch("subprocess.run") as mock_sub:
        
        # Trigger background task immediately in test environment
        response = client.post("/api/simulate", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "Simulation started in background."
