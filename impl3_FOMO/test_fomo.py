"""Geometry and heatmap checks. Run: python test_fomo.py"""

import numpy as np
import torch

from geom import cell_box_to_frame, cell_center_to_frame, frame_point_to_cell, letterbox
from post import HeatmapGate, detections_from_logits
from tracker import CentroidTracker, _Axis, smooth_size


def test_cell_center_roundtrip():
    # 854x480 letterboxed into 192. A cell center mapped out and back is that cell.
    image = np.zeros((480, 854, 3), np.uint8)
    _canvas, scale, pad_x, pad_y = letterbox(image, 192)
    stride = 8
    for col, row in ((3, 4), (10, 12), (20, 8)):
        cx, cy = cell_center_to_frame(col, row, scale, pad_x, pad_y, stride)
        back_col, back_row = frame_point_to_cell(cx, cy, scale, pad_x, pad_y, stride)
        assert (back_col, back_row) == (col, row), (col, row, back_col, back_row)


def test_one_cell_is_one_stride_box():
    image = np.zeros((480, 854, 3), np.uint8)
    _canvas, scale, pad_x, pad_y = letterbox(image, 192)
    x1, y1, x2, y2 = cell_box_to_frame(5, 6, 5, 6, scale, pad_x, pad_y, 8)
    # One cell is 8 input pixels, undoing the letterbox scale.
    assert abs((x2 - x1) * scale - 8) < 1e-6
    assert abs((y2 - y1) * scale - 8) < 1e-6


def test_two_adjacent_cells_make_one_detection():
    # Logits: background 0, person channel high on cells (4,5) and (4,6).
    logits = torch.full((3, 12, 12), -4.0)
    logits[0] = 2.0
    logits[1, 5, 4] = 6.0
    logits[1, 6, 4] = 6.0
    dets = detections_from_logits(logits, ["person", "pole"], 0.5, scale=1.0, pad_x=0, pad_y=0, stride=8)
    assert len(dets) == 1, dets
    assert dets[0]["class_name"] == "person"
    # Two vertical cells: height 16, width 8, in input pixels since scale is 1 and pad is 0.
    assert abs((dets[0]["x2"] - dets[0]["x1"]) - 8) < 1e-4
    assert abs((dets[0]["y2"] - dets[0]["y1"]) - 16) < 1e-4


def test_tracker_keeps_id_when_centroid_steps():
    tracker = CentroidTracker(match_px=40, max_missed=3)
    first = tracker.update([_det(100, 100)])
    second = tracker.update([_det(112, 104)])
    assert first[0]["track_id"] == second[0]["track_id"]


def test_tracker_does_not_merge_two_classes():
    tracker = CentroidTracker(match_px=40, max_missed=3)
    ids = tracker.update([_det(50, 50, class_id=0), _det(52, 50, class_id=1)])
    assert ids[0]["track_id"] != ids[1]["track_id"]


def test_one_hot_frame_does_not_light_a_cell():
    gate = HeatmapGate(momentum=0.7, on=0.55, off=0.35)
    quiet = np.zeros((2, 1, 1), np.float32)
    quiet[0, 0, 0] = 1.0
    gate.update(quiet)
    spike = np.zeros_like(quiet)
    spike[1, 0, 0] = 1.0
    _state, active, _cls = gate.update(spike)
    # 0.7 * background + 0.3 * person = 0.3, under 0.55.
    assert float(gate.state[1, 0, 0]) == np.float32(0.3) or abs(float(gate.state[1, 0, 0]) - 0.3) < 1e-5
    assert not bool(active[0, 0])


def test_kalman_follows_a_steady_step():
    axis = _Axis(0.0, process=1.0, measure=1.0)
    for position in (10.0, 20.0, 30.0, 40.0):
        axis.predict()
        axis.update(position)
    upcoming = axis.predict()
    assert abs(upcoming - 50.0) < 8.0, upcoming
    assert axis.p00 >= 0.0 and axis.p11 >= 0.0


def test_size_cannot_double_in_one_frame():
    updated = smooth_size(100.0, 200.0, max_ratio=1.08, blend=0.35)
    # Clipped measurement is 108. Blend: 0.65 * 100 + 0.35 * 108 = 102.8.
    assert abs(updated - 102.8) < 1e-6, updated


def _det(cx, cy, class_id=0, width=20.0, height=40.0):
    return {
        "class_name": "person" if class_id == 0 else "pole",
        "class_id": class_id,
        "confidence": 0.9,
        "cx": cx,
        "cy": cy,
        "x1": cx - 10,
        "y1": cy - 20,
        "x2": cx + 10,
        "y2": cy + 20,
    }


if __name__ == "__main__":
    test_cell_center_roundtrip()
    test_one_cell_is_one_stride_box()
    test_two_adjacent_cells_make_one_detection()
    test_tracker_keeps_id_when_centroid_steps()
    test_tracker_does_not_merge_two_classes()
    test_one_hot_frame_does_not_light_a_cell()
    test_kalman_follows_a_steady_step()
    test_size_cannot_double_in_one_frame()
    print("all fomo checks passed")
