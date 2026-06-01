# PROMPT: Generate standard Pytest unit tests for a FastAPI store intelligence application. Cover event ingestion idempotency, store metrics (/stores/{id}/metrics), visitor funnel (/stores/{id}/funnel) including entry, zone visits, and purchase drop-offs, and store heatmaps (/stores/{id}/heatmap) under varying test scenarios (empty database, staff events exclusion, and re-entry). Use an in-memory SQLite database.
# CHANGES MADE: Refactored the database engine to use an in-memory SQLite URL with StaticPool, allowing reliable connection sharing across all test client calls without encountering file-locking or read-only database errors.

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import DBPOS, Base, DBEvent, get_db
from app.main import app

# Setup an in-memory SQLite database with StaticPool for connection persistence
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


def test_health_check_empty_db(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "healthy"
    assert data["last_event_timestamp"] is None
    assert data["stale_feed"] is False


def test_event_ingestion_and_idempotency(client):
    event_id = str(uuid.uuid4())
    event_payload = [
        {
            "event_id": event_id,
            "store_id": "ST1008",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": "VIS_TEST01",
            "event_type": "ENTRY",
            "timestamp": "2026-04-10T10:00:00Z",
            "zone_id": "ENTRY",
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.95,
            "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1},
        }
    ]

    # Ingest first time
    response = client.post("/events/ingest", json=event_payload)
    assert response.status_code == 207
    assert response.json()["ingested"] == 1
    assert response.json()["failed"] == 0

    # Ingest second time (idempotency check)
    response = client.post("/events/ingest", json=event_payload)
    assert response.status_code == 207
    assert response.json()["ingested"] == 1
    assert response.json()["failed"] == 0


def test_store_metrics_calculations(client, db_session):
    # Setup customer and staff events
    vid_cust = "VIS_CUST01"
    vid_staff = "VIS_STAFF01"
    store_id = "ST1008"

    # Add DBPOS transaction
    db_session.add(
        DBPOS(
            order_id="TXN001",
            store_id=store_id,
            timestamp="2026-04-10T10:08:00Z",
            brand_name="LAKME",
            total_amount=500.0,
        )
    )

    # Add customer events
    # ENTRY
    db_session.add(
        DBEvent(
            event_id="e1",
            store_id=store_id,
            camera_id="CAM_ENTRY_01",
            visitor_id=vid_cust,
            event_type="ENTRY",
            timestamp="2026-04-10T10:00:00Z",
            zone_id="ENTRY",
            dwell_ms=0,
            is_staff=False,
            confidence=0.9,
            metadata_json={"session_seq": 1},
        )
    )
    # Enter retail zone
    db_session.add(
        DBEvent(
            event_id="e2",
            store_id=store_id,
            camera_id="CAM_MAIN_01",
            visitor_id=vid_cust,
            event_type="ZONE_ENTER",
            timestamp="2026-04-10T10:01:00Z",
            zone_id="EB_KOREAN",
            dwell_ms=0,
            is_staff=False,
            confidence=0.9,
            metadata_json={"session_seq": 2},
        )
    )
    # Billing queue join
    db_session.add(
        DBEvent(
            event_id="e3",
            store_id=store_id,
            camera_id="CAM_BILLING_01",
            visitor_id=vid_cust,
            event_type="BILLING_QUEUE_JOIN",
            timestamp="2026-04-10T10:05:00Z",
            zone_id="BILLING",
            dwell_ms=60000,
            is_staff=False,
            confidence=0.9,
            metadata_json={"session_seq": 3, "queue_depth": 1},
        )
    )
    # EXIT
    db_session.add(
        DBEvent(
            event_id="e4",
            store_id=store_id,
            camera_id="CAM_ENTRY_01",
            visitor_id=vid_cust,
            event_type="EXIT",
            timestamp="2026-04-10T10:10:00Z",
            zone_id="ENTRY",
            dwell_ms=0,
            is_staff=False,
            confidence=0.9,
            metadata_json={"session_seq": 4},
        )
    )

    # Add staff events (should be excluded)
    db_session.add(
        DBEvent(
            event_id="s1",
            store_id=store_id,
            camera_id="CAM_ENTRY_01",
            visitor_id=vid_staff,
            event_type="ENTRY",
            timestamp="2026-04-10T10:00:00Z",
            zone_id="ENTRY",
            dwell_ms=0,
            is_staff=True,
            confidence=0.9,
            metadata_json={"session_seq": 1},
        )
    )

    db_session.commit()

    # Query metrics
    response = client.get(f"/stores/{store_id}/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert metrics["unique_visitors"] == 1
    assert (
        metrics["conversion_rate"] == 100.0
    )  # Converted since billing was 10:05:00 and purchase was 10:08:00 (within 5 mins)
    assert metrics["average_dwell_minutes"] == 10.0  # 10:00:00 to 10:10:00
    assert metrics["current_queue_depth"] == 1
    assert metrics["abandonment_rate"] == 0.0


def test_funnel_and_heatmap_values(client, db_session):
    vid = "VIS_CUST02"
    store_id = "ST1008"

    # ENTRY
    db_session.add(
        DBEvent(
            event_id="f1",
            store_id=store_id,
            camera_id="CAM_ENTRY_01",
            visitor_id=vid,
            event_type="ENTRY",
            timestamp="2026-04-10T11:00:00Z",
            zone_id="ENTRY",
            dwell_ms=0,
            is_staff=False,
            confidence=0.95,
            metadata_json={"session_seq": 1},
        )
    )
    # Retails Zone visit
    db_session.add(
        DBEvent(
            event_id="f2",
            store_id=store_id,
            camera_id="CAM_MAIN_01",
            visitor_id=vid,
            event_type="ZONE_ENTER",
            timestamp="2026-04-10T11:02:00Z",
            zone_id="EB_KOREAN",
            dwell_ms=30000,
            is_staff=False,
            confidence=0.95,
            metadata_json={"session_seq": 2},
        )
    )
    db_session.commit()

    # Test Funnel
    response = client.get(f"/stores/{store_id}/funnel")
    assert response.status_code == 200
    funnel = response.json()
    assert funnel["funnel"]["entry"] == 1
    assert funnel["funnel"]["zone_visit"] == 1
    assert funnel["funnel"]["billing_queue"] == 0
    assert funnel["funnel"]["purchase"] == 0
    assert funnel["dropoff_percentages"]["entry_to_zone"] == 0.0
    assert funnel["dropoff_percentages"]["zone_to_billing"] == 100.0

    # Test Heatmap
    response = client.get(f"/stores/{store_id}/heatmap")
    assert response.status_code == 200
    heatmap = response.json()
    assert len(heatmap["zones"]) == 1
    assert heatmap["zones"][0]["zone_id"] == "EB_KOREAN"
    assert heatmap["zones"][0]["visit_count"] == 1
    assert heatmap["zones"][0]["intensity"] == 100.0
    assert heatmap["data_confidence"] is False  # unique_sessions is 1 < 20


def test_partial_ingestion_malformed_event(client):
    event_id_valid = str(uuid.uuid4())
    event_payload = [
        {
            "event_id": event_id_valid,
            "store_id": "ST1008",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": "VIS_TEST02",
            "event_type": "ENTRY",
            "timestamp": "2026-04-10T10:00:00Z",
            "zone_id": "ENTRY",
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.95,
            "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1},
        },
        {
            "event_id": "malformed_id",
            "store_id": "ST1008",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": "VIS_TEST03",
            "event_type": "ENTRY",
            "timestamp": "2026-04-10T10:00:00Z",
            "zone_id": "ENTRY",
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": "MALFORMED_STRING_NOT_FLOAT",
            "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1},
        },
    ]

    response = client.post("/events/ingest", json=event_payload)
    assert response.status_code == 207
    data = response.json()
    assert data["ingested"] == 1
    assert data["failed"] == 1
    assert len(data["errors"]) == 1
    assert data["errors"][0]["event_id"] == "malformed_id"
    assert "Schema validation failed" in data["errors"][0]["error"]
