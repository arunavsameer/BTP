"""Keep one id per walkway object with a constant-velocity Kalman filter.

State on each axis is [position, velocity]. Each frame predicts one step,
then a matched detection updates the state. Width and height are blended in
separately and cannot jump by more than size_max_ratio in one frame, so a
flickering grid cell cannot double the box the collision fit sees.
"""

from __future__ import annotations

import math


class _Axis:
    """Constant-velocity Kalman filter for one number. dt is one frame.

    State is position and velocity. Covariance is
    [[p00, p01], [p01, p11]]. The measurement is the position.
    """

    def __init__(self, position: float, process: float = 4.0, measure: float = 36.0):
        self.x = float(position)
        self.v = 0.0
        self.p00 = 25.0
        self.p01 = 0.0
        self.p11 = 25.0
        self.process = float(process)
        self.measure = float(measure)

    def predict(self) -> float:
        dt = 1.0
        self.x = self.x + self.v * dt
        q = self.process
        p00 = self.p00 + 2.0 * dt * self.p01 + dt * dt * self.p11 + 0.25 * q * dt * dt
        p01 = self.p01 + dt * self.p11 + 0.5 * q * dt
        p11 = self.p11 + q
        self.p00, self.p01, self.p11 = p00, p01, p11
        return self.x

    def update(self, measurement: float) -> float:
        p00, p01, p11 = self.p00, self.p01, self.p11
        innovation = float(measurement) - self.x
        scale = p00 + self.measure
        k0 = p00 / scale
        k1 = p01 / scale
        self.x = self.x + k0 * innovation
        self.v = self.v + k1 * innovation
        # (I - K H) P with H = [1, 0].
        self.p00 = (1.0 - k0) * p00
        self.p01 = (1.0 - k0) * p01
        self.p11 = p11 - k1 * p01
        return self.x


def smooth_size(previous: float, measured: float, max_ratio: float, blend: float) -> float:
    """Move previous toward measured, but not by more than max_ratio in one frame."""
    lo = previous / max_ratio
    hi = previous * max_ratio
    clipped = min(max(measured, lo), hi)
    return (1.0 - blend) * previous + blend * clipped


class _Track:
    def __init__(self, track_id: int, det: dict):
        self.track_id = track_id
        self.class_id = int(det["class_id"])
        self.class_name = det["class_name"]
        self.kx = _Axis(float(det["cx"]))
        self.ky = _Axis(float(det["cy"]))
        self.w = max(1.0, float(det["x2"]) - float(det["x1"]))
        self.h = max(1.0, float(det["y2"]) - float(det["y1"]))
        self.missed = 0

    def predicted(self) -> tuple[float, float]:
        return self.kx.x, self.ky.x


class CentroidTracker:
    def __init__(
        self,
        match_px: float = 48.0,
        max_missed: int = 15,
        size_max_ratio: float = 1.08,
        size_blend: float = 0.35,
    ):
        self.match_px = float(match_px)
        self.max_missed = int(max_missed)
        self.size_max_ratio = float(size_max_ratio)
        self.size_blend = float(size_blend)
        self._next_id = 1
        self._tracks: list[_Track] = []

    def update(self, dets: list[dict]) -> list[dict]:
        """Return detections with a stable id, Kalman center, and smoothed size."""
        for track in self._tracks:
            track.kx.predict()
            track.ky.predict()

        pairs = []
        for det_i, det in enumerate(dets):
            for track in self._tracks:
                if track.class_id != det["class_id"]:
                    continue
                px, py = track.predicted()
                dist = math.hypot(float(det["cx"]) - px, float(det["cy"]) - py)
                gate = max(self.match_px, 0.6 * max(track.w, track.h))
                if dist <= gate:
                    pairs.append((dist, det_i, track))
        pairs.sort(key=lambda item: item[0])

        used_dets = set()
        used_ids = set()
        matched: dict[int, _Track] = {}
        for _dist, det_i, track in pairs:
            if det_i in used_dets or track.track_id in used_ids:
                continue
            used_dets.add(det_i)
            used_ids.add(track.track_id)
            det = dets[det_i]
            track.kx.update(float(det["cx"]))
            track.ky.update(float(det["cy"]))
            measured_w = max(1.0, float(det["x2"]) - float(det["x1"]))
            measured_h = max(1.0, float(det["y2"]) - float(det["y1"]))
            track.w = smooth_size(track.w, measured_w, self.size_max_ratio, self.size_blend)
            track.h = smooth_size(track.h, measured_h, self.size_max_ratio, self.size_blend)
            track.missed = 0
            matched[det_i] = track

        for track in self._tracks:
            if track.track_id not in used_ids:
                track.missed += 1
        self._tracks = [track for track in self._tracks if track.missed <= self.max_missed]

        output = []
        for det_i, det in enumerate(dets):
            track = matched.get(det_i)
            if track is None:
                track = _Track(self._next_id, det)
                self._next_id += 1
                self._tracks.append(track)
            seen = dict(det)
            seen["track_id"] = track.track_id
            seen["cx"] = track.kx.x
            seen["cy"] = track.ky.x
            seen["x1"] = track.kx.x - 0.5 * track.w
            seen["y1"] = track.ky.x - 0.5 * track.h
            seen["x2"] = track.kx.x + 0.5 * track.w
            seen["y2"] = track.ky.x + 0.5 * track.h
            output.append(seen)
        return output
