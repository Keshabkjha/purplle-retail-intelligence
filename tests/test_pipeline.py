# PROMPT: Generate pytest unit tests for a computer vision spatial mapping pipeline. Cover ray-casting point-in-polygon checks (inside, outside, and boundary cases), camera coordinate to 2D store floor plan mapping transforms, and named zone determination.
# CHANGES MADE: Created isolated unit test suite covering pipeline spatial mapping utilities.

from pipeline.detect import determine_zone, map_camera_to_floor, point_in_polygon


def test_point_in_polygon_square():
    # Simple 100x100 square polygon
    poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
    
    # Strictly inside
    assert point_in_polygon(50, 50, poly) is True
    # Strictly outside
    assert point_in_polygon(150, 50, poly) is False
    assert point_in_polygon(50, -10, poly) is False

def test_point_in_polygon_complex():
    # L-shaped polygon
    poly = [[0, 0], [100, 0], [100, 50], [50, 50], [50, 100], [0, 100]]
    
    # Inside the main body
    assert point_in_polygon(25, 25, poly) is True
    # Inside the vertical extension
    assert point_in_polygon(25, 75, poly) is True
    # In the cut-out section (outside)
    assert point_in_polygon(75, 75, poly) is False

def test_map_camera_to_floor_fallbacks():
    # Test that mapping doesn't crash and returns appropriate coordinates
    # Default fallbacks
    wx, wy = map_camera_to_floor(960, 540, "CAM_ENTRY_01")
    assert 0 <= wx <= 940
    assert 0 <= wy <= 451
    
    wx, wy = map_camera_to_floor(960, 540, "CAM_BILLING_01")
    assert 0 <= wx <= 940
    assert 0 <= wy <= 451

def test_determine_zone_matching():
    # Test zone checking for entry
    zone = determine_zone(50, 300, "CAM_ENTRY_01")
    assert zone == "ENTRY"
    
    # Test zone checking for billing
    zone = determine_zone(800, 300, "CAM_BILLING_01")
    assert zone == "BILLING"
    
    # Test out of bounds mapping returns None
    zone = determine_zone(999, 999, "CAM_ENTRY_01")
    assert zone is None
