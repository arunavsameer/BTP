"""Small checks for the three detectors. Run: python test_methods.py"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FelzenszwalbConfig, FlowConfig, PavementConfig
from felzenszwalb import find_regions as find_felzenszwalb
from felzenszwalb import segment_labels
from flow import FlowBoxes
from pavement import find_regions as find_pavement
from regions import Region, RegionTracker


def test_felzenszwalb_splits_two_flat_colors():
    frame = np.zeros((48, 96, 3), np.uint8)
    frame[:, :48] = (0, 0, 210)  # red in BGR
    frame[:, 48:] = (210, 0, 0)  # blue

    for backend in ("python", "ximgproc"):
        labels = segment_labels(frame, k=200, min_size=10, sigma=0.0, backend=backend)
        left = np.unique(labels[:, :40])
        right = np.unique(labels[:, 56:])
        assert len(left) == 1, (backend, left)
        assert len(right) == 1, (backend, right)
        assert int(left[0]) != int(right[0]), backend

    cfg = FelzenszwalbConfig(
        k=200,
        min_size=10,
        sigma=0.0,
        min_area=50,
        max_area_frac=0.8,
        min_fill=0.5,
        max_boxes=5,
        process_width=0,
    )
    boxes = find_felzenszwalb(frame, cfg)
    assert len(boxes) == 2, boxes


def test_python_merges_a_gentle_gradient():
    # A step of 2 is a weak edge. With k/|C| still above that step, FH
    # absorbs the ramp. Exact-color flood-fill would leave one column each.
    gradient = np.zeros((4, 24, 3), np.uint8)
    for x in range(24):
        gradient[:, x] = (x * 2, x * 2, x * 2)
    labels = segment_labels(gradient, k=30, min_size=2, sigma=0.0, backend="python")
    assert len(np.unique(labels)) == 1, np.unique(labels)

    step = np.zeros((8, 20, 3), np.uint8)
    step[:, :10] = 20
    step[:, 10:] = 220
    split = segment_labels(step, k=30, min_size=2, sigma=0.0, backend="python")
    assert len(np.unique(split[:, :8])) == 1
    assert len(np.unique(split[:, 12:])) == 1
    assert int(split[0, 0]) != int(split[0, -1])


def test_pavement_mask_drops_a_road_colored_box():
    frame = np.zeros((90, 140, 3), np.uint8)
    frame[:, :] = (180, 90, 40)  # saturated sky
    road = (150, 150, 150)
    frame[62:, :] = road
    frame[10:38, 12:42] = (40, 40, 210)  # red obstacle
    frame[10:38, 96:126] = road  # same paint as the road, not connected to it

    cfg = PavementConfig(
        k=150,
        min_size=15,
        sigma=0.0,
        min_area=200,
        max_area_frac=0.95,
        min_fill=0.4,
        max_boxes=8,
        process_width=0,
        trap_top=0.75,
        trap_top_width=0.85,
        hist_bins=8,
        match_threshold=0.02,
        reject_fraction=0.5,
        gray_saturation=40,
    )
    boxes = find_pavement(frame, cfg)

    def center_bgr(box):
        x = int(min(frame.shape[1] - 1, max(0, (box.x1 + box.x2) / 2)))
        y = int(min(frame.shape[0] - 1, max(0, (box.y1 + box.y2) / 2)))
        return tuple(int(value) for value in frame[y, x])

    colors = [center_bgr(box) for box in boxes]
    assert (40, 40, 210) in colors, colors
    assert road not in colors, colors


def test_sparse_flow_static_frame_does_not_crash():
    blank = np.zeros((80, 100, 3), np.uint8)
    blank_finder = FlowBoxes(FlowConfig())
    assert blank_finder.boxes(blank) == []
    assert blank_finder.boxes(blank) == []

    grid = np.full((180, 240, 3), 40, np.uint8)
    for x in range(12, 230, 16):
        cv2.rectangle(grid, (x, 24), (x + 7, 150), (230, 230, 230), -1)
    for y in range(24, 160, 18):
        cv2.line(grid, (8, y), (230, y), (20, 20, 20), 2)
    finder = FlowBoxes(FlowConfig())
    assert finder.boxes(grid) == []
    again = finder.boxes(grid.copy())
    assert again == [], again


def test_tracker_keeps_id_when_boxes_overlap():
    tracker = RegionTracker(iou_match=0.3, max_missed=2)

    def box(x1, y1, x2, y2):
        return Region(x1, y1, x2, y2, area=(x2 - x1) * (y2 - y1), name="seg")

    first = tracker.update([box(0, 0, 20, 20)])
    second = tracker.update([box(2, 2, 22, 22)])
    assert first[0][1] == second[0][1]


if __name__ == "__main__":
    test_felzenszwalb_splits_two_flat_colors()
    test_python_merges_a_gentle_gradient()
    test_pavement_mask_drops_a_road_colored_box()
    test_sparse_flow_static_frame_does_not_crash()
    test_tracker_keeps_id_when_boxes_overlap()
    print("all method checks passed")
