"""Tests for the video analytics app."""

import argparse
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app import Analytics, draw_overlay, open_source, run


# ---------------------------------------------------------------------------
# Analytics tests
# ---------------------------------------------------------------------------


class TestAnalytics:
    def test_initial_state(self):
        a = Analytics()
        assert a.total_frames == 0
        assert a.total_detections == 0
        assert a.avg_detections_per_frame == 0.0
        assert a.max_detections_per_frame == 0

    def test_update_single_frame(self):
        a = Analytics()
        a.update(["person", "car"])
        assert a.total_frames == 1
        assert a.total_detections == 2
        assert a.detection_counts["person"] == 1
        assert a.detection_counts["car"] == 1

    def test_update_multiple_frames(self):
        a = Analytics()
        a.update(["person", "person"])
        a.update(["car"])
        a.update([])
        assert a.total_frames == 3
        assert a.total_detections == 3
        assert a.detection_counts["person"] == 2
        assert a.detection_counts["car"] == 1

    def test_avg_detections_per_frame(self):
        a = Analytics()
        a.update(["person", "car"])  # 2 detections
        a.update(["person"])  # 1 detection
        assert a.avg_detections_per_frame == pytest.approx(1.5)

    def test_max_detections_per_frame(self):
        a = Analytics()
        a.update(["person"])
        a.update(["car", "truck", "bus"])
        assert a.max_detections_per_frame == 3

    def test_summary_contains_key_info(self):
        a = Analytics()
        a.update(["person", "car"])
        s = a.summary()
        assert "Total frames processed" in s
        assert "Total detections" in s
        assert "person" in s
        assert "car" in s

    def test_summary_empty(self):
        a = Analytics()
        s = a.summary()
        assert "0" in s


# ---------------------------------------------------------------------------
# draw_overlay tests
# ---------------------------------------------------------------------------


def test_draw_overlay_modifies_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    original = frame.copy()
    draw_overlay(frame, fps=29.9, det_count=3)
    # Frame should be modified (text was drawn)
    assert not np.array_equal(frame, original)


# ---------------------------------------------------------------------------
# open_source tests
# ---------------------------------------------------------------------------


def test_open_source_invalid_raises():
    with pytest.raises(RuntimeError, match="Cannot open video source"):
        open_source("/nonexistent/path/video.mp4")


@patch("app.cv2.VideoCapture")
def test_open_source_camera_index(mock_cap_cls):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap_cls.return_value = mock_cap
    cap = open_source("0")
    mock_cap_cls.assert_called_once_with(0)
    assert cap is mock_cap


@patch("app.cv2.VideoCapture")
def test_open_source_file_path(mock_cap_cls):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap_cls.return_value = mock_cap
    cap = open_source("video.mp4")
    mock_cap_cls.assert_called_once_with("video.mp4")
    assert cap is mock_cap


# ---------------------------------------------------------------------------
# run() integration test (fully mocked)
# ---------------------------------------------------------------------------


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        source="0",
        model="yolov8n.pt",
        output=None,
        conf=0.25,
        iou=0.45,
        track=False,
        show=False,
        device="",
        classes=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_mock_result(class_indices):
    """Build a minimal mock YOLO result."""
    result = MagicMock()
    annotated = np.zeros((480, 640, 3), dtype=np.uint8)
    result.plot.return_value = annotated
    if class_indices:
        boxes = MagicMock()
        boxes.cls.cpu.return_value.numpy.return_value = np.array(
            class_indices, dtype=float
        )
        result.boxes = boxes
        # Make len() work
        boxes.__len__ = MagicMock(return_value=len(class_indices))
    else:
        result.boxes = None
    return result


@patch("app.YOLO")
@patch("app.cv2.VideoCapture")
def test_run_processes_frames(mock_cap_cls, mock_yolo_cls):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0
    # Two frames then stop
    mock_cap.read.side_effect = [
        (True, frame.copy()),
        (True, frame.copy()),
        (False, None),
    ]
    mock_cap_cls.return_value = mock_cap

    mock_model = MagicMock()
    mock_model.names = {0: "person", 2: "car"}
    result1 = _make_mock_result([0, 0])  # two persons
    result2 = _make_mock_result([2])  # one car
    mock_model.predict.side_effect = [[result1], [result2]]
    mock_yolo_cls.return_value = mock_model

    args = _make_args()
    analytics = run(args)

    assert analytics.total_frames == 2
    assert analytics.total_detections == 3
    assert analytics.detection_counts["person"] == 2
    assert analytics.detection_counts["car"] == 1


@patch("app.YOLO")
@patch("app.cv2.VideoCapture")
def test_run_tracking_uses_track_method(mock_cap_cls, mock_yolo_cls):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0
    mock_cap.read.side_effect = [(True, frame.copy()), (False, None)]
    mock_cap_cls.return_value = mock_cap

    mock_model = MagicMock()
    mock_model.names = {0: "person"}
    result = _make_mock_result([0])
    mock_model.track.return_value = [result]
    mock_yolo_cls.return_value = mock_model

    args = _make_args(track=True)
    analytics = run(args)

    mock_model.track.assert_called_once()
    mock_model.predict.assert_not_called()
    assert analytics.total_frames == 1
