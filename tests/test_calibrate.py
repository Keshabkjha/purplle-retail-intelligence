import numpy as np
from unittest.mock import MagicMock, patch
import pytest
import cv2
import pipeline.calibrate
from pipeline.calibrate import click_event, main


def test_click_event():
    with patch("cv2.circle") as mock_circle, patch("cv2.imshow") as mock_imshow:
        # Mock global img in calibrate module
        pipeline.calibrate.img = np.zeros((400, 400, 3), dtype=np.uint8)
        pipeline.calibrate.points.clear()

        click_event(cv2.EVENT_LBUTTONDOWN, 100, 200, None, None)

        assert (100, 200) in pipeline.calibrate.points
        mock_circle.assert_called_once()
        mock_imshow.assert_called_once()


@patch("os.path.exists", return_value=False)
def test_main_file_not_found(mock_exists):
    main("nonexistent.mp4")


@patch("os.path.exists", return_value=True)
@patch("cv2.imread")
@patch("cv2.imshow")
@patch("cv2.setMouseCallback")
@patch("cv2.waitKey", side_effect=[ord("q")])
@patch("cv2.destroyAllWindows")
def test_main_image(mock_destroy, mock_waitkey, mock_mouse, mock_imshow, mock_imread, mock_exists):
    mock_img = np.zeros((400, 400, 3), dtype=np.uint8)
    mock_imread.return_value = mock_img
    main("dummy_image.png")
    mock_imread.assert_called_once_with("dummy_image.png")
    mock_imshow.assert_called_once()
    mock_destroy.assert_called_once()


@patch("os.path.exists", return_value=True)
@patch("cv2.VideoCapture")
@patch("cv2.imshow")
@patch("cv2.setMouseCallback")
@patch("cv2.waitKey", side_effect=[ord("q")])
@patch("cv2.destroyAllWindows")
def test_main_video(mock_destroy, mock_waitkey, mock_mouse, mock_imshow, mock_cap_class, mock_exists):
    mock_cap = MagicMock()
    mock_img = np.zeros((400, 400, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, mock_img)
    mock_cap_class.return_value = mock_cap
    main("dummy_video.mp4")
    mock_cap.read.assert_called_once()
    mock_cap.release.assert_called_once()
    mock_destroy.assert_called_once()


@patch("os.path.exists", return_value=True)
@patch("cv2.VideoCapture")
def test_main_video_failed_read(mock_cap_class, mock_exists):
    mock_cap = MagicMock()
    mock_cap.read.return_value = (False, None)
    mock_cap_class.return_value = mock_cap
    main("dummy_video.mp4")
    mock_cap.read.assert_called_once()
    mock_cap.release.assert_called_once()


@patch("os.path.exists", return_value=True)
@patch("cv2.imread")
@patch("cv2.imshow")
@patch("cv2.setMouseCallback")
@patch("cv2.waitKey", side_effect=[ord("c"), ord("q")])
@patch("cv2.destroyAllWindows")
def test_main_image_clear_points(mock_destroy, mock_waitkey, mock_mouse, mock_imshow, mock_imread, mock_exists):
    mock_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    mock_imread.return_value = mock_img
    main("dummy_image.png")
    # Verify that points was cleared when ord("c") was processed
    assert len(pipeline.calibrate.points) == 0

