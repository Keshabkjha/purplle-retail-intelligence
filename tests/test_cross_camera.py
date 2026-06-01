import os
from datetime import datetime, timedelta

import pytest

from pipeline.detect import CrossCameraSessionTracker, update_staff_status


@pytest.fixture
def temp_state_file(tmp_path):
    return os.path.join(tmp_path, "session_state.json")

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
        is_staff_initial=False
    )
    
    assert vid == "VIS_1"
    assert "VIS_1" in tracker.sessions
    assert tracker.sessions["VIS_1"]["last_seen_camera"] == "CAM_ENTRY_01"
    assert tracker.sessions["VIS_1"]["last_seen_x"] == 10.0
    assert tracker.sessions["VIS_1"]["last_seen_y"] == 250.0

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
        is_staff_initial=False
    )
    
    # 2. A track appears in MAIN floor 3 seconds later at (150, 305) - very close (within 150px and 30s)
    t2 = t1 + timedelta(seconds=3)
    vid_match = tracker.get_unified_id(
        track_id=2,
        camera_id="CAM_MAIN_01",
        wx=150.0,
        wy=305.0,
        current_time=t2,
        is_staff_initial=False
    )
    
    assert vid_match == "VIS_3"  # Should correctly unify to the original VIS_3!
    assert tracker.sessions["VIS_3"]["last_seen_camera"] == "CAM_MAIN_01"
    assert tracker.sessions["VIS_3"]["camera_track_ids"]["CAM_ENTRY_01"] == 3
    assert tracker.sessions["VIS_3"]["camera_track_ids"]["CAM_MAIN_01"] == 2
    assert tracker.is_reentry("VIS_3", "CAM_MAIN_01") is True

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
        is_staff_initial=False
    )
    
    # 2. Track appears in MAIN floor 45 seconds later (exceeds 30s temporal threshold)
    t2 = t1 + timedelta(seconds=45)
    vid_no_match_time = tracker.get_unified_id(
        track_id=6,
        camera_id="CAM_MAIN_01",
        wx=15.0,
        wy=205.0,
        current_time=t2,
        is_staff_initial=False
    )
    
    # Should get a brand new ID because the time difference was too high
    assert vid_no_match_time != "VIS_5"
    assert vid_no_match_time == "VIS_6"
    
    # 3. Track appears in BILLING camera (wx=800, wy=300) within 5 seconds but far away (exceeds 150px spatial threshold)
    t3 = t1 + timedelta(seconds=5)
    vid_no_match_space = tracker.get_unified_id(
        track_id=7,
        camera_id="CAM_BILLING_01",
        wx=800.0,
        wy=300.0,
        current_time=t3,
        is_staff_initial=False
    )
    
    # Should get a brand new ID because spatial gap was too high
    assert vid_no_match_space != "VIS_5"
    assert vid_no_match_space == "VIS_7"

def test_hybrid_staff_detection(temp_state_file):
    tracker = CrossCameraSessionTracker(state_file=temp_state_file)
    t = datetime(2026, 4, 10, 16, 40, 0)
    
    # Register session
    sess = {
        "current_zone": None,
        "enter_time": t,
        "first_seen_time": t,
        "last_seen": t,
        "dwell_sent_count": 0,
        "seq": 1,
        "is_staff": False
    }
    vid = "VIS_99"
    tracker.sessions[vid] = {
        "unified_id": vid,
        "last_seen_camera": "CAM_ENTRY_01",
        "last_seen_time": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_seen_x": 0.0,
        "last_seen_y": 0.0,
        "is_staff": False,
        "camera_track_ids": {"CAM_ENTRY_01": 99}
    }
    
    # Case 1: Simple visitor, no clothing match, short duration -> not staff
    is_staff = update_staff_status(
        vid=vid,
        sess=sess,
        current_time=t + timedelta(seconds=10),
        zone_id="ENTRY",
        wx=50.0,
        wy=250.0,
        is_clothing_staff=False,
        tracker=tracker
    )
    assert is_staff is False
    
    # Case 2: Torso color heuristic matches -> classified as staff
    is_staff = update_staff_status(
        vid=vid,
        sess=sess,
        current_time=t + timedelta(seconds=20),
        zone_id="ENTRY",
        wx=50.0,
        wy=250.0,
        is_clothing_staff=True,
        tracker=tracker
    )
    assert is_staff is True
    
    # Case 3: Behavior-only: stays near cash register behind billing counter (>820) for a long time
    sess2 = {
        "current_zone": "BILLING",
        "enter_time": t,
        "first_seen_time": t,
        "last_seen": t,
        "dwell_sent_count": 0,
        "seq": 1,
        "is_staff": False
    }
    vid2 = "VIS_100"
    tracker.sessions[vid2] = {
        "unified_id": vid2,
        "last_seen_camera": "CAM_BILLING_01",
        "last_seen_time": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_seen_x": 850.0,
        "last_seen_y": 300.0,
        "is_staff": False,
        "camera_track_ids": {"CAM_BILLING_01": 100}
    }
    
    is_staff = update_staff_status(
        vid=vid2,
        sess=sess2,
        current_time=t + timedelta(seconds=120),  # Spent 2 minutes behind register
        zone_id="BILLING",
        wx=850.0,
        wy=300.0,
        is_clothing_staff=False,
        tracker=tracker
    )
    assert is_staff is True  # Unified behavioral flags identify them as staff!
