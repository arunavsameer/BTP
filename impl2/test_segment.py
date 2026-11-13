"""Small checks for color blobs and id matching. Run: python test_segment.py"""

import numpy as np

from config import Config
from motion import background_ids, median
from segment import find_blobs
from tracker import BlobTracker, _iou


def _cfg(**kwargs) -> Config:
    base = dict(
        color_tolerance=0.0,
        min_area=10,
        max_area_frac=1.0,
        min_fill=0.0,
        max_boxes=20,
        connectivity=4,
        compare_to_seed=True,
        blur_ksize=0,
        process_width=0,
        merge_gap=-1,
    )
    base.update(kwargs)
    return Config(**base)


def _blob(x1, y1, x2, y2, name="red", color=(0, 0, 180)):
    from segment import Blob

    return Blob(x1, y1, x2, y2, area=(x2 - x1) * (y2 - y1), name=name, color=color)


def test_exact_colors_stay_apart():
    frame = np.zeros((40, 80, 3), np.uint8)
    frame[:, :40] = (0, 0, 220)  # red in BGR
    frame[:, 40:] = (220, 0, 0)  # blue
    blobs = find_blobs(frame, _cfg(color_tolerance=0.0))
    assert len(blobs) == 2, blobs
    names = sorted(blob.name for blob in blobs)
    assert names == ["blue", "red"], names


def test_full_tolerance_is_one_blob():
    frame = np.zeros((40, 80, 3), np.uint8)
    frame[:, :40] = (0, 0, 220)
    frame[:, 40:] = (220, 0, 0)
    blobs = find_blobs(frame, _cfg(color_tolerance=1.0))
    assert len(blobs) == 1, blobs
    assert blobs[0].area > 40 * 80 * 0.9


def test_small_gap_joins_only_past_the_dial():
    # Right half is 10 levels brighter on blue. Tolerance 0 must not merge.
    # Tolerance 0.04 → gap round(10.2) = 10, so the two halves become one blob.
    frame = np.zeros((30, 40, 3), np.uint8)
    frame[:, :] = (100, 100, 100)
    frame[:, 20:] = (110, 100, 100)
    apart = find_blobs(frame, _cfg(color_tolerance=0.0, min_area=50))
    joined = find_blobs(frame, _cfg(color_tolerance=0.04, min_area=50))
    assert len(apart) == 2, apart
    assert len(joined) == 1, joined


def test_speck_is_dropped():
    frame = np.zeros((80, 80, 3), np.uint8)
    frame[10:16, 10:16] = (0, 0, 200)  # 36 pixels on a black field
    kept = find_blobs(frame, _cfg(min_area=10))
    dropped = find_blobs(frame, _cfg(min_area=500))
    assert any(blob.area < 100 for blob in kept), kept
    assert all(blob.area >= 500 for blob in dropped), dropped


def test_thin_diagonal_fails_fill():
    frame = np.zeros((40, 40, 3), np.uint8)
    for i in range(40):
        frame[i, i] = (0, 0, 255)
    # 8-connected, so the diagonal is one component. Its box is almost empty.
    # The diagonal is one 8-connected component, but its box is almost empty.
    # The black background is solid, so a fill cutoff removes only the diagonal.
    sparse = find_blobs(
        frame,
        _cfg(connectivity=8, min_area=1, min_fill=0.4, max_boxes=50),
    )
    kept = find_blobs(
        frame,
        _cfg(connectivity=8, min_area=1, min_fill=0.0, max_boxes=50),
    )
    assert all(blob.area > 100 for blob in sparse), sparse
    assert any(blob.area < 100 for blob in kept), kept


def test_tracker_keeps_id_when_box_overlaps():
    tracker = BlobTracker(iou_match=0.3, max_missed=2)
    first = tracker.update([_blob(0, 0, 20, 20)])
    second = tracker.update([_blob(2, 2, 22, 22)])
    assert first[0][1] == second[0][1]


def test_euclidean_allows_a_single_channel_shift_the_cube_rejects():
    # +40 on blue only. At tolerance 0.10 the cube allows 26 per channel,
    # the ball allows about 44 in any direction.
    frame = np.full((20, 40, 3), 80, np.uint8)
    frame[:, 20:] = (120, 80, 80)
    common = dict(color_tolerance=0.10, min_area=20, max_boxes=10)
    channel = find_blobs(frame, _cfg(color_metric="channel", **common))
    euclid = find_blobs(frame, _cfg(color_metric="euclidean", **common))
    assert len(channel) == 2, channel
    assert len(euclid) == 1, euclid


def test_thin_box_is_dropped_by_aspect():
    frame = np.zeros((40, 40, 3), np.uint8)
    frame[:, 18:20] = (0, 0, 220)
    blobs = find_blobs(frame, _cfg(min_area=1, max_aspect=3.0, max_boxes=20))
    assert all(blob.name != "red" for blob in blobs), blobs


def test_gray_is_dropped_by_saturation():
    frame = np.zeros((20, 40, 3), np.uint8)
    frame[:, :20] = (100, 100, 100)
    frame[:, 20:] = (0, 0, 200)
    blobs = find_blobs(frame, _cfg(min_area=20, min_saturation=0.3, color_tolerance=0.0))
    assert [blob.name for blob in blobs] == ["red"], blobs


def test_tracker_waits_for_min_frames():
    tracker = BlobTracker(iou_match=0.3, max_missed=2, min_track_frames=3)
    assert tracker.update([_blob(0, 0, 20, 20)]) == []
    assert tracker.update([_blob(1, 1, 21, 21)]) == []
    third = tracker.update([_blob(1, 1, 21, 21)])
    assert len(third) == 1
    assert third[0][1] == 1


def test_tracker_new_id_after_the_box_leaves():
    # max_missed=1 and memory_frames=1, so two empty frames drop the id
    # completely. The defaults coast and then remember for much longer.
    tracker = BlobTracker(iou_match=0.3, max_missed=1, memory_frames=1)
    first = tracker.update([_blob(0, 0, 10, 10)])
    tracker.update([])  # missed once
    tracker.update([])  # dropped
    third = tracker.update([_blob(0, 0, 10, 10)])
    assert third[0][1] != first[0][1]


def test_tracker_keeps_id_when_box_jumps_half_its_width():
    # 80 px box steps 40 px in both axes. Overlap IoU is about 0.14, under
    # the 0.20 gate, and the area does not change.
    old = _blob(0, 0, 80, 80)
    jumped = _blob(40, 40, 120, 120)
    assert _iou(old, jumped) < 0.2
    tracker = BlobTracker()
    first = tracker.update([old])
    second = tracker.update([jumped])
    assert len(second) == 1
    assert second[0][1] == first[0][1]


def test_memory_reattaches_same_color_after_the_coast():
    # Five empty frames with max_missed=2 puts the id in memory, not the
    # live list. A nearby blob of the same red takes that id back.
    red = (0, 0, 200)
    tracker = BlobTracker(max_missed=2, memory_frames=12, memory_color=40, memory_radius_px=80, memory_radius_cap=100)
    first = tracker.update([_blob(0, 0, 30, 30, color=red)])
    for _ in range(5):
        assert tracker.update([]) == []
    assert tracker._tracks == []
    assert len(tracker._memory) == 1
    back = tracker.update([_blob(6, 4, 36, 34, color=(0, 0, 188))])
    assert len(back) == 1
    assert back[0][1] == first[0][1]
    assert tracker._memory == []


def test_memory_rejects_a_different_color_and_a_far_blob():
    red = (0, 0, 200)
    tracker = BlobTracker(max_missed=2, memory_frames=12, memory_color=40, memory_radius_px=80, memory_radius_cap=100)
    first = tracker.update([_blob(0, 0, 30, 30, color=red)])
    for _ in range(5):
        tracker.update([])
    other = tracker.update([_blob(4, 4, 34, 34, color=(200, 0, 0))])
    assert other[0][1] != first[0][1]

    tracker = BlobTracker(max_missed=2, memory_frames=12, memory_color=40, memory_radius_px=80, memory_radius_cap=100)
    first = tracker.update([_blob(0, 0, 30, 30, color=red)])
    for _ in range(5):
        tracker.update([])
    far = tracker.update([_blob(400, 400, 430, 430, color=red)])
    assert far[0][1] != first[0][1]


def test_median_odd_and_even():
    assert median([1.0, 3.0, 2.0]) == 2.0
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_motion_keeps_the_clump_and_flags_the_departure():
    # Four tracks crawl at 1 px/frame. One jumps to 10.
    # Median is 1. Deviations are 0,0,0,0,9 so the MAD is 0 and the floor
    # of 1.5 is the scale. 9 / 1.5 = 6, past 2.5. The clump scores 0.
    clump = {i: (1.0, 0.0, 0.0) for i in range(4)}
    clump[10] = (10.0, 0.0, 0.0)
    background = background_ids(clump, outlier_mad=2.5, floor=1.5, min_tracks=5)
    assert background == {0, 1, 2, 3}

    # 0, 2, 4, 6, 8. Median 4, MAD 2. The ends score 4/2 = 2.0, still inside 2.5.
    spread = {i: (float(2 * i), 0.0, 0.0) for i in range(5)}
    assert background_ids(spread, outlier_mad=2.5, floor=1.5, min_tracks=5) == set(spread)
    spread[20] = (20.0, 0.0, 0.0)
    # |20 - 4| / 2 = 8.
    assert 20 not in background_ids(spread, outlier_mad=2.5, floor=1.5, min_tracks=5)
    assert 2 in background_ids(spread, outlier_mad=2.5, floor=1.5, min_tracks=5)


def test_motion_does_not_guess_from_too_few_tracks():
    motions = {1: (0.0, 0.0, 0.0), 2: (50.0, 0.0, 0.0)}
    assert background_ids(motions, outlier_mad=2.5, floor=1.5, min_tracks=5) == set()


def test_nearby_shades_merge_before_the_fill_test():
    # Two green blocks 10 px apart. Each is solid. Together they are one
    # object once merge_gap is on. A far-away blue block stays separate.
    frame = np.zeros((50, 120, 3), np.uint8)
    frame[8:42, 4:36] = (40, 170, 50)
    frame[8:42, 46:78] = (55, 185, 70)
    frame[8:42, 96:116] = (180, 40, 40)
    blobs = find_blobs(
        frame,
        _cfg(min_area=40, min_fill=0.3, merge_gap=16, merge_color=80, max_boxes=10),
    )
    greens = [b for b in blobs if b.name == "green"]
    blues = [b for b in blobs if b.name == "blue"]
    assert len(greens) == 1, [(b.name, b.x1, b.x2) for b in blobs]
    assert len(blues) == 1
    assert greens[0].x2 - greens[0].x1 > 60


def test_split_shirt_stays_one_id():
    # One red box becomes two overlapping red pieces. They must not mint
    # a second id, and the drawn box must cover both pieces.
    tracker = BlobTracker()
    red = (0, 0, 200)
    first = tracker.update([_blob(0, 0, 40, 40, color=red)])
    left = _blob(0, 0, 24, 40, color=(0, 0, 190))
    right = _blob(16, 0, 40, 40, color=(0, 0, 210))
    second = tracker.update([left, right])
    assert len(second) == 1, second
    assert second[0][1] == first[0][1]
    box = second[0][0]
    assert box.x1 <= 1 and box.x2 >= 39


def test_tracker_keeps_id_across_a_two_frame_gap():
    # 20 px/frame to the right. After two misses the predicted center is
    # three steps on from the last sighting: (40, 20) + (20, 0) * 3 = (100, 20).
    # The box comes back a few pixels from that spot and must keep the id.
    tracker = BlobTracker()
    first = tracker.update([_blob(0, 0, 40, 40)])
    tracker.update([_blob(20, 0, 60, 40)])
    assert tracker.update([]) == []
    assert tracker.update([]) == []
    back = tracker.update([_blob(84, 2, 124, 42)])
    assert len(back) == 1
    assert back[0][1] == first[0][1]
    assert tracker._tracks[0].hits == 3
    assert tracker._tracks[0].missed == 0


if __name__ == "__main__":
    test_exact_colors_stay_apart()
    test_full_tolerance_is_one_blob()
    test_small_gap_joins_only_past_the_dial()
    test_speck_is_dropped()
    test_thin_diagonal_fails_fill()
    test_tracker_keeps_id_when_box_overlaps()
    test_tracker_waits_for_min_frames()
    test_euclidean_allows_a_single_channel_shift_the_cube_rejects()
    test_thin_box_is_dropped_by_aspect()
    test_gray_is_dropped_by_saturation()
    test_tracker_new_id_after_the_box_leaves()
    test_tracker_keeps_id_when_box_jumps_half_its_width()
    test_tracker_keeps_id_across_a_two_frame_gap()
    test_median_odd_and_even()
    test_motion_keeps_the_clump_and_flags_the_departure()
    test_motion_does_not_guess_from_too_few_tracks()
    test_nearby_shades_merge_before_the_fill_test()
    test_split_shirt_stays_one_id()
    test_memory_reattaches_same_color_after_the_coast()
    test_memory_rejects_a_different_color_and_a_far_blob()
    print("all segment checks passed")
