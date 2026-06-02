# PROMPT: Generate pytest unit tests for a computer vision spatial mapping pipeline. Cover ray-casting point-in-polygon checks (inside, outside, and boundary cases), camera coordinate to 2D store floor plan mapping transforms, and named zone determination.
# CHANGES MADE: Created isolated unit test suite covering pipeline spatial mapping utilities.

from unittest.mock import MagicMock, patch
import cv2
import numpy as np
from pipeline.detect import determine_zone, map_camera_to_floor, point_in_polygon, run_detection


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


@patch("pipeline.detect.cv2.VideoCapture")
@patch("pipeline.detect.cv2.VideoWriter")
@patch("pipeline.detect.YOLO")
@patch("pipeline.detect.post_event")
@patch("subprocess.run")
def test_run_detection_pipeline(mock_sub, mock_post, mock_yolo_class, mock_writer_class, mock_cap_class):
    # 1. Mock VideoCapture
    mock_cap = MagicMock()
    mock_cap.isOpened.side_effect = [True, True, True, True, False]  # read 4 frames
    mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, mock_frame)
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_COUNT: 4,
        cv2.CAP_PROP_FRAME_WIDTH: 1920,
        cv2.CAP_PROP_FRAME_HEIGHT: 1080
    }.get(prop, 0)
    mock_cap_class.return_value = mock_cap

    # 2. Mock VideoWriter
    mock_writer = MagicMock()
    mock_writer_class.return_value = mock_writer

    # 3. Mock YOLO
    mock_model = MagicMock()
    mock_results = MagicMock()
    
    # Mock box predictions
    mock_box = MagicMock()
    mock_box.xyxy.cpu.return_value.numpy.return_value = np.array([[100, 100, 200, 200]], dtype=np.float32)
    mock_box.id.cpu.return_value.numpy.return_value = np.array([42])
    mock_box.cls.cpu.return_value.numpy.return_value = np.array([0])  # person
    mock_box.conf.cpu.return_value.numpy.return_value = np.array([0.95])
    
    mock_results.boxes = mock_box
    mock_model.track.return_value = [mock_results]
    mock_yolo_class.return_value = mock_model

    # 4. Run detection
    with patch("os.path.exists", return_value=True), \
         patch("os.remove") as mock_remove:
        run_detection("CCTV Footage/entry_camera.mp4", model_path="yolo11n.pt")

    # Assertions
    mock_cap.read.assert_called()
    mock_model.track.assert_called()
    mock_writer.write.assert_called()
    mock_sub.assert_called_once()

