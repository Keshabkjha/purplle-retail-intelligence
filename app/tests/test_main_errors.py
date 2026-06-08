import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_stream_video_range_416():
    response = client.get("/api/video_stream/test.mp4", headers={"Range": "bytes=1000000000-2000000000"})
    assert response.status_code in [404, 416]

def test_get_cameras_invalid_store():
    response = client.get("/stores/INVALID_STORE/cameras")
    assert response.status_code == 200
    assert response.json() == []

def test_simulate_invalid_action():
    response = client.post("/api/simulate", json={"action": "invalid"})
    assert response.status_code == 422
    
