"""Synthetic checks for edge regions. Run: python test_segment.py

A blank frame has no edges, so the whole picture is one component.
max_area_frac below 1 drops that component.
"""

import cv2
import numpy as np

from config import Config
from segment import find_regions


def _cfg(**kwargs) -> Config:
    base = dict(
        blur_ksize=0,
        canny_low=50,
        canny_high=150,
        canny_aperture=3,
        edge_dilate=0,
        connectivity=4,
        min_area=10,
        max_area_frac=1.0,
        min_fill=0.0,
        max_boxes=30,
        process_width=0,
    )
    base.update(kwargs)
    return Config(**base)


def test_rectangle_outline_has_an_interior():
    frame = np.zeros((120, 160, 3), np.uint8)
    cv2.rectangle(frame, (40, 30), (120, 90), (255, 255, 255), thickness=3)
    regions = find_regions(
        frame,
        _cfg(edge_dilate=1, min_area=50, min_fill=0.4, max_area_frac=0.9),
    )
    # The centre of the drawn rectangle should sit in an open region,
    # and that region should be the inside, not the whole frame.
    interior = [
        region
        for region in regions
        if region.x1 < 80 < region.x2 and region.y1 < 60 < region.y2 and region.x1 > 15 and region.y1 > 10
    ]
    assert interior, regions
    assert interior[0].x2 < 145 and interior[0].y2 < 110


def test_specks_under_min_area_disappear():
    frame = np.zeros((120, 160, 3), np.uint8)
    cv2.rectangle(frame, (40, 30), (120, 90), (255, 255, 255), thickness=3)
    cv2.rectangle(frame, (8, 8), (18, 18), (255, 255, 255), thickness=2)
    with_specks = find_regions(frame, _cfg(min_area=1, min_fill=0.0, max_area_frac=1.0, max_boxes=50))
    kept = find_regions(frame, _cfg(min_area=400, min_fill=0.0, max_area_frac=1.0, max_boxes=50))
    assert any(region.area < 400 for region in with_specks), with_specks
    assert kept, with_specks
    assert all(region.area >= 400 for region in kept), kept


def test_blank_frame_is_one_region_until_max_area():
    frame = np.zeros((48, 64, 3), np.uint8)
    one = find_regions(frame, _cfg(min_area=1, max_area_frac=1.0, min_fill=0.0))
    assert len(one) == 1, one
    assert abs(one[0].area - 48 * 64) < 1
    dropped = find_regions(frame, _cfg(min_area=1, max_area_frac=0.5, min_fill=0.0))
    assert dropped == []


if __name__ == "__main__":
    test_rectangle_outline_has_an_interior()
    test_specks_under_min_area_disappear()
    test_blank_frame_is_one_region_until_max_area()
    print("all segment checks passed")
