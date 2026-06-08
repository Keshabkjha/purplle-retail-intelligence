import numpy as np
import pytest
from pipeline.detect import (
    age_to_bucket,
    camera_transition_prior,
    compare_appearance,
    compute_appearance_embedding,
    is_staff_heuristic,
    resolve_camera_id_and_store,
    zone_transition_prior,
)

def test_age_to_bucket():
    assert age_to_bucket(17) == "18-24"
    assert age_to_bucket(22) == "18-24"
    assert age_to_bucket(30) == "25-34"
    assert age_to_bucket(40) == "35-44"
    assert age_to_bucket(50) == "45-54"
    assert age_to_bucket(60) == "55+"

def test_resolve_camera_id_and_store_fallback():
    # Should resolve correctly with fallback patterns
    cam_id, sid, s_code, zones, roles, role = resolve_camera_id_and_store("Store 1/CAM 3 - entry.mp4")
    assert sid == "ST1008"
    assert "cam" in cam_id.lower() or "cam1" in cam_id.lower()
    
    cam_id, sid, s_code, zones, roles, role = resolve_camera_id_and_store("Store 2/billing_area.mp4")
    assert sid == "ST1076"
    assert cam_id == "PURPLLE_MUM_1076_CAM6"

def test_compute_appearance_embedding():
    # Valid crop
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, :] = [255, 0, 0] # Blue
    emb = compute_appearance_embedding(crop)
    assert emb is not None
    assert len(emb) > 0
    
    # Invalid crop
    assert compute_appearance_embedding(None) is None
    assert compute_appearance_embedding(np.array([])) is None

def test_compare_appearance():
    emb1 = [0.1, 0.2, 0.3]
    emb2 = [0.1, 0.2, 0.3]
    emb3 = [0.9, 0.1, 0.0]
    
    # Identical
    assert compare_appearance(emb1, emb2) > 0.99
    # Different
    assert compare_appearance(emb1, emb3) < 1.0
    # Invalid
    assert compare_appearance(None, emb1) == 0.0

def test_camera_transition_prior():
    # Same camera should return 1.0
    assert camera_transition_prior("CAM1", "CAM1") == 1.0
    # Known transitions
    assert camera_transition_prior("cam1", "CAM1") == 0.95
    # Unknown combinations should fallback
    assert camera_transition_prior("unknown_1", "unknown_2") == 0.60
    assert camera_transition_prior(None, "CAM1") == 0.0

def test_zone_transition_prior():
    # Same zone
    assert zone_transition_prior("ZONE_1", "ZONE_1") == 1.0
    # Entry
    assert zone_transition_prior("ENTRY_PORTAL", "ZONE_1") == 0.85
    # Billing
    assert zone_transition_prior("ZONE_1", "BILLING_COUNTER") == 0.90
    # Unknown/generic
    assert zone_transition_prior("ZONE_A", "ZONE_B") == 0.80
    assert zone_transition_prior(None, "ZONE_1") == 0.50

def test_is_staff_heuristic():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Give it a black torso
    frame[0:50, 0:100] = [0, 0, 0]
    # Bbox around the whole frame
    bbox = [0, 0, 100, 100]
    
    assert is_staff_heuristic(1, bbox, frame) == True
    
    # White torso
    frame[0:50, 0:100] = [255, 255, 255]
    assert is_staff_heuristic(2, bbox, frame) == False
    
    # Invalid bbox
    assert is_staff_heuristic(3, [-10, -10, -5, -5], frame) == False
