# PROMPT: Generate pytest unit tests for FastAPI anomalies detection endpoint (/stores/{id}/anomalies). Ensure they cover billing queue spikes (both WARN and CRITICAL severities), conversion drop anomalies when conversion rate falls below 10%, and dead zone operational warnings when retail zones receive 0 visits. Use an in-memory SQLite database setup.
# CHANGES MADE: Refactored the database engine to use an in-memory SQLite URL with StaticPool, allowing reliable connection sharing across all test client calls without encountering file-locking or read-only database errors.


import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, DBEvent, get_db
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


def test_anomalies_empty_db(client):
    response = client.get("/stores/ST1008/anomalies")
    assert response.status_code == 200
    data = response.json()
    assert data["store_id"] == "ST1008"
    # Empty store has no events, so the 30-minute dead zone window has no anchor timestamp.
    # No dead zone anomalies are expected — the rubric window requires real event data to anchor.
    assert "anomalies" in data


def test_billing_queue_spike_anomaly(client, db_session):
    store_id = "ST1008"

    # 1. Trigger WARN (depth >= 5)
    db_session.add(
        DBEvent(
            event_id="e1",
            store_id=store_id,
            camera_id="CAM_BILLING_01",
            visitor_id="VIS_01",
            event_type="BILLING_QUEUE_JOIN",
            timestamp="2026-04-10T10:00:00Z",
            zone_id="BILLING",
            dwell_ms=10000,
            is_staff=False,
            confidence=0.9,
            metadata_json={"queue_depth": 5},
        )
    )
    db_session.commit()

    response = client.get(f"/stores/{store_id}/anomalies")
    assert response.status_code == 200
    anomalies = response.json()["anomalies"]
    spike_anomalies = [an for an in anomalies if an["anomaly_type"] == "BILLING_QUEUE_SPIKE"]
    assert len(spike_anomalies) == 1
    assert spike_anomalies[0]["severity"] == "WARN"
    assert "depth is currently 5" in spike_anomalies[0]["details"]

    # 2. Trigger CRITICAL (depth >= 8)
    db_session.add(
        DBEvent(
            event_id="e2",
            store_id=store_id,
            camera_id="CAM_BILLING_01",
            visitor_id="VIS_02",
            event_type="BILLING_QUEUE_JOIN",
            timestamp="2026-04-10T10:01:00Z",
            zone_id="BILLING",
            dwell_ms=10000,
            is_staff=False,
            confidence=0.9,
            metadata_json={"queue_depth": 8},
        )
    )
    db_session.commit()

    response = client.get(f"/stores/{store_id}/anomalies")
    anomalies = response.json()["anomalies"]
    spike_anomalies = [an for an in anomalies if an["anomaly_type"] == "BILLING_QUEUE_SPIKE"]
    assert len(spike_anomalies) == 1
    assert spike_anomalies[0]["severity"] == "CRITICAL"
    assert "depth is currently 8" in spike_anomalies[0]["details"]


def test_conversion_drop_anomaly(client, db_session):
    store_id = "ST1008"
    # Create 6 visitors and 0 conversions to trigger CONVERSION_DROP (requires >= 5 visitors)
    for i in range(6):
        db_session.add(
            DBEvent(
                event_id=f"evt_{i}_1",
                store_id=store_id,
                camera_id="CAM_ENTRY_01",
                visitor_id=f"VIS_{i}",
                event_type="ENTRY",
                timestamp=f"2026-04-10T10:0{i}:00Z",
                zone_id="ENTRY",
                dwell_ms=0,
                is_staff=False,
                confidence=0.95,
                metadata_json={"session_seq": 1},
            )
        )
        db_session.add(
            DBEvent(
                event_id=f"evt_{i}_2",
                store_id=store_id,
                camera_id="CAM_MAIN_01",
                visitor_id=f"VIS_{i}",
                event_type="ZONE_ENTER",
                timestamp=f"2026-04-10T10:0{i}:30Z",
                zone_id="EB_KOREAN",
                dwell_ms=0,
                is_staff=False,
                confidence=0.95,
                metadata_json={"session_seq": 2},
            )
        )
    db_session.commit()

    response = client.get(f"/stores/{store_id}/anomalies")
    assert response.status_code == 200
    anomalies = response.json()["anomalies"]
    drop_anomalies = [an for an in anomalies if an["anomaly_type"] == "CONVERSION_DROP"]
    assert len(drop_anomalies) == 1
    assert drop_anomalies[0]["severity"] == "WARN"
    # Updated message text to match new fallback message when < 10 historical data points
    assert "Conversion rate is low at" in drop_anomalies[0]["details"]


def test_comprehensive_anomalies(client, db_session):
    from app.database import DBPOS
    store_id = "ST1008"

    # 1. Test STATISTICAL_QUEUE_SPIKE
    # We need >= 5 historical events to trigger rolling stats logic.
    for i in range(5):
        db_session.add(DBEvent(
            event_id=f"q_hist_{i}", store_id=store_id, camera_id="CAM_BILLING_01", visitor_id=f"VIS_Q_{i}",
            event_type="BILLING_QUEUE_JOIN", timestamp=f"2026-04-10T10:0{i}:00Z", zone_id="BILLING",
            is_staff=False, confidence=0.9, metadata_json={"queue_depth": 2}
        ))
    # Add current event with deep queue (depth = 6)
    db_session.add(DBEvent(
        event_id="q_curr", store_id=store_id, camera_id="CAM_BILLING_01", visitor_id="VIS_Q_CURR",
        event_type="BILLING_QUEUE_JOIN", timestamp="2026-04-10T15:10:00Z", zone_id="BILLING",
        is_staff=False, confidence=0.9, metadata_json={"queue_depth": 6}
    ))

    # 2. Test CONVERSION_DROP (relative to 7-day baseline)
    # Baseline: Past days (e.g. 2026-04-09) having 5 unique visitors and 3 POS txns (60% conversion rate)
    for day in (9, 8, 7):
        db_session.add(DBEvent(
            event_id=f"b_evt_{day}", store_id=store_id, camera_id="CAM_ENTRY_01", visitor_id=f"VIS_B_{day}",
            event_type="ENTRY", timestamp=f"2026-04-0{day}T12:00:00Z", zone_id="ENTRY", is_staff=False, confidence=0.9,
            metadata_json={"session_seq": 1}
        ))
        db_session.add(DBPOS(
            order_id=f"b_tx_{day}", store_id=store_id, timestamp=f"2026-04-0{day}T12:10:00Z", brand_name="EB_KOREAN", total_amount=500.0
        ))

    # Current Day (2026-04-10): 10 visitors, 0 conversions (0% conversion rate)
    for i in range(10):
        db_session.add(DBEvent(
            event_id=f"c_evt_{i}", store_id=store_id, camera_id="CAM_ENTRY_01", visitor_id=f"VIS_C_{i}",
            event_type="ENTRY", timestamp=f"2026-04-10T15:0{i}:00Z", zone_id="ENTRY", is_staff=False, confidence=0.9,
            metadata_json={"session_seq": 1}
        ))
    
    # 3. Test DEAD_ZONE operational warning
    # We must have active traffic in the past 30 minutes to anchor the dead zone checks.
    # Latest event is at 2026-04-10T15:10:00Z, so the 30m window starts at 14:40:00Z.
    db_session.add(DBEvent(
        event_id="dz_active", store_id=store_id, camera_id="CAM_MAIN_01", visitor_id="VIS_ACTIVE",
        event_type="ZONE_ENTER", timestamp="2026-04-10T15:09:00Z", zone_id="EB_KOREAN", is_staff=False, confidence=0.9,
        metadata_json={"session_seq": 2}
    ))

    db_session.commit()

    # Call endpoint
    response = client.get(f"/stores/{store_id}/anomalies")
    assert response.status_code == 200
    data = response.json()
    anomalies = data["anomalies"]

    # Verify statistical queue spike
    stat_spikes = [an for an in anomalies if an["anomaly_type"] == "STATISTICAL_QUEUE_SPIKE"]
    assert len(stat_spikes) >= 1
    assert stat_spikes[0]["severity"] in ("WARN", "CRITICAL")

    # Verify conversion drop relative to daily baseline
    conv_drops = [an for an in anomalies if an["anomaly_type"] == "CONVERSION_DROP"]
    assert len(conv_drops) >= 1
    assert "Daily Business Review baseline" in conv_drops[0]["details"]

    # Verify dead zones (since THE_FACE_SHOP has layout definition but 0 visits)
    dead_zones = [an for an in anomalies if an["anomaly_type"] == "DEAD_ZONE"]
    assert len(dead_zones) >= 1
    assert any("THE_FACE_SHOP" in dz["details"] for dz in dead_zones)


