from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
from app.database import Base, DBEvent, get_db
from app.main import app

engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)
db = TestingSessionLocal()
def override_get_db():
    try:
        yield db
    finally:
        pass
app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

store_id = "ST1008"
for i in range(5):
    db.add(DBEvent(
        event_id=f"q_hist_{i}", store_id=store_id, camera_id="CAM_BILLING_01", visitor_id=f"VIS_Q_{i}",
        event_type="BILLING_QUEUE_JOIN", timestamp=f"2026-04-10T10:0{i}:00Z", zone_id="BILLING",
        is_staff=False, confidence=0.9, metadata_json={"queue_depth": 2}
    ))
db.add(DBEvent(
    event_id="q_curr", store_id=store_id, camera_id="CAM_BILLING_01", visitor_id="VIS_Q_CURR",
    event_type="BILLING_QUEUE_JOIN", timestamp="2026-04-10T10:10:00Z", zone_id="BILLING",
    is_staff=False, confidence=0.9, metadata_json={"queue_depth": 6}
))
db.commit()

response = client.get(f"/stores/{store_id}/anomalies")
print(response.json())
