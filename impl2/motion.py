"""Decide which tracks are the walking camera, and which are not.

Most boxes are walls, road, and sky. Their image motion clumps together.
A person close to the lens, or someone crossing the street, sits away from
that clump.

For a track seen from frame t0 to t1:

    vx = (cx1 - cx0) / (t1 - t0)
    vy = (cy1 - cy0) / (t1 - t0)
    g  = (sqrt(area1) - sqrt(area0)) / (t1 - t0)

vx and vy are pixels per frame. g is the change in the box's linear size
per frame. The camera estimate is the median of each, across tracks.
A track is background when every component stays within outlier_mad robust
steps of that median:

    score = max_i  |m_i - median_i| / max(MAD_i, floor)

MAD_i is the median of |m_i - median_i| over the tracks. The floor stops a
tiny MAD from turning one pixel of jitter into an outlier. Fewer than
min_tracks motions means we do not know the camera yet, so nobody is
called background.
"""

from __future__ import annotations


def median(values: list[float]) -> float:
    """Middle value. For an even count, the average of the two middle ones."""
    if not values:
        raise ValueError("median of an empty list")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return 0.5 * (float(ordered[mid - 1]) + float(ordered[mid]))


def track_motion(samples) -> tuple[float, float, float] | None:
    """(vx, vy, g) from the oldest and newest box. None if time did not advance."""
    t0, box0 = samples[0][0], samples[0][1]
    t1, box1 = samples[-1][0], samples[-1][1]
    dt = float(t1 - t0)
    if dt < 1.0:
        return None
    vx = (box1.cx - box0.cx) / dt
    vy = (box1.cy - box0.cy) / dt
    g = (box1.area ** 0.5 - box0.area ** 0.5) / dt
    return vx, vy, g


def background_ids(
    motions: dict[int, tuple[float, float, float]],
    outlier_mad: float,
    floor: float,
    min_tracks: int,
) -> set[int]:
    """Ids whose motion matches the median. Empty when there are too few tracks."""
    if len(motions) < min_tracks:
        return set()
    columns = [[motion[axis] for motion in motions.values()] for axis in range(3)]
    centers = [median(column) for column in columns]
    deviations = [[abs(value - center) for value in column] for column, center in zip(columns, centers)]
    scales = [max(median(deviation), float(floor), 1e-6) for deviation in deviations]
    background = set()
    for track_id, motion in motions.items():
        score = max(abs(motion[axis] - centers[axis]) / scales[axis] for axis in range(3))
        if score <= outlier_mad:
            background.add(track_id)
    return background
