"""Collision forecast from a linear fit over the last k boxes.

Idea (no extra sensors):
  An object getting closer looks bigger in the camera. For each tracked object we
  keep the last k boxes and fit a straight line (least squares) to centre x/y and
  to width/height vs frame number. That line is the average trend — one noisy
  frame cannot yank the forecast the way a two-point slope can.

  We then walk the fitted size until one side would fill the screen, read the
  fitted centre at that same time, and call it a possible collision if that
  centre still lands inside a hit zone that can be taller than the picture
  (vertical margin) and narrower than the picture (horizontal margin).

All tunables live in CollisionConfig so they are easy to tweak.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2


@dataclass
class CollisionConfig:
    """Knobs. Change these; leave the math functions alone unless the model changes."""

    # How many recent boxes go into the linear fit (k).
    # 2 is the old two-point slope. 5–10 smooths tracker jitter.
    frame_gap: int = 15

    # How far off-screen the predicted centre may be and still count as a hit.
    # 1.0 = must land inside the picture. >1 expands that axis; <1 shrinks it.
    # Vertical 1.5 = allow 25% extra above/below. Horizontal 0.8 = centre must
    # stay in the middle 80% of the width (side-passers are ignored).
    frame_margin_vertical: float = 10.0
    frame_margin_horizontal: float = 0.8

    # Which side of the box we grow until it matches the screen.
    # "first_to_fill" = whichever of width/height would hit the screen first.
    # "width", "height", or "max" (larger box side vs larger screen side).
    size_axis: str = "first_to_fill"

    # Box must grow at least this much between the two samples (e.g. 1.03 = +3%).
    # Stops parked objects and tracker jitter from looking like a crash.
    min_growth_ratio: float = 1.03

    # Ignore forecasts further away than this (seconds).
    # 4s was too tight for a small floor object: the "fill the screen" TTC
    # is pessimistic, so a real hit a couple of seconds away got discarded.
    max_ttc_seconds: float = 20.0

    # How many dots to draw on the future centre path.
    trajectory_points: int = 16

    # Skip tiny detections (pixels).
    min_box_area: float = 400.0

    # Detection confidence passed to YOLO.
    conf: float = 0.25

    # Need at least this many boxes in the window before fitting.
    min_samples: int = 3

    # Mark HIT only if the last k forecasts for that track were all collisions.
    # 1 = no extra wait. 3 kills one-frame flicker without eating the whole approach.
    confirm_hits: int = 3


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def width(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @classmethod
    def from_xyxy(cls, xyxy) -> Box:
        x1, y1, x2, y2 = xyxy
        return cls(float(x1), float(y1), float(x2), float(y2))

    def interpolated(self, other: Box, lam: float) -> Box:
        """Straight-line mix: lam=0 is this box, lam=1 is other."""
        cx = self.cx + lam * (other.cx - self.cx)
        cy = self.cy + lam * (other.cy - self.cy)
        w = self.width + lam * (other.width - self.width)
        h = self.height + lam * (other.height - self.height)
        return Box(cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)


def box_from_center(cx: float, cy: float, width: float, height: float) -> Box:
    width = max(1.0, width)
    height = max(1.0, height)
    return Box(cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height)


@dataclass
class Line:
    """value(t) = intercept + slope * t, with t = frame index."""

    intercept: float
    slope: float

    def at(self, t: float) -> float:
        return self.intercept + self.slope * t


def fit_line(times: list[float], values: list[float]) -> Line:
    """Ordinary least squares. If time does not change, return a flat average."""
    n = len(times)
    mean_t = sum(times) / n
    mean_v = sum(values) / n
    var_t = sum((t - mean_t) ** 2 for t in times)
    if var_t < 1e-9:
        return Line(mean_v, 0.0)
    cov = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values))
    slope = cov / var_t
    return Line(mean_v - slope * mean_t, slope)


@dataclass
class Forecast:
    track_id: int
    class_name: str
    will_collide: bool
    reason: str
    ttc_seconds: float | None
    impact_cx: float | None
    impact_cy: float | None
    path: list[tuple[float, float]] = field(default_factory=list)
    ghost_boxes: list[Box] = field(default_factory=list)
    # Geometric hit this frame, before the consecutive-hit gate.
    raw_collide: bool = False
    hit_streak: int = 0
    confirm_hits: int = 1


def _time_until_fill(size_line: Line, screen_size: float) -> float | None:
    """Frame index when fitted size reaches the screen. None if not growing."""
    if size_line.slope <= 1e-3:
        return None
    return (screen_size - size_line.intercept) / size_line.slope


def _pick_hit_time(
    width_line: Line,
    height_line: Line,
    frame_w: int,
    frame_h: int,
    now: float,
    axis: str,
) -> tuple[float | None, str]:
    """Soonest fill time. If that time is already past, the side is already bigger than the screen."""
    t_w = _time_until_fill(width_line, frame_w)
    t_h = _time_until_fill(height_line, frame_h)

    if axis == "width":
        return t_w, "width"
    if axis == "height":
        return t_h, "height"
    if axis == "max":
        if width_line.at(now) >= height_line.at(now):
            return t_w, "width"
        return t_h, "height"

    candidates = [(t_w, "width"), (t_h, "height")]
    valid = [(t, name) for t, name in candidates if t is not None]
    if not valid:
        return None, "none"
    t_hit, name = min(valid, key=lambda item: item[0])
    return t_hit, name


def centre_in_margin_frame(
    cx: float,
    cy: float,
    frame_w: int,
    frame_h: int,
    margin_horizontal: float,
    margin_vertical: float,
) -> bool:
    """True if (cx, cy) lies in a rectangle scaled independently on each axis, centred on the image."""
    extra_w = 0.5 * (margin_horizontal - 1.0) * frame_w
    extra_h = 0.5 * (margin_vertical - 1.0) * frame_h
    return (-extra_w <= cx <= frame_w + extra_w) and (-extra_h <= cy <= frame_h + extra_h)


def forecast_from_window(
    samples: list[tuple[int, Box, str]],
    now: int,
    fps: float,
    frame_w: int,
    frame_h: int,
    cfg: CollisionConfig,
    track_id: int,
    class_name: str,
) -> Forecast:
    """Fit cx, cy, width, height vs frame index, then extrapolate. See module docstring."""
    if len(samples) < cfg.min_samples:
        return Forecast(track_id, class_name, False, "not enough samples", None, None, None)

    newest = samples[-1][1]
    if newest.area < cfg.min_box_area:
        return Forecast(track_id, class_name, False, "box too small", None, None, None)

    times = [float(t) for t, _, _ in samples]
    span = times[-1] - times[0]
    min_span = max(2.0, cfg.frame_gap / 3.0)
    if span < min_span:
        return Forecast(track_id, class_name, False, "window too short for a stable fit", None, None, None)

    cx_line = fit_line(times, [b.cx for _, b, _ in samples])
    cy_line = fit_line(times, [b.cy for _, b, _ in samples])
    w_line = fit_line(times, [b.width for _, b, _ in samples])
    h_line = fit_line(times, [b.height for _, b, _ in samples])

    t_old, t_new = times[0], times[-1]
    w_growth = w_line.at(t_new) / max(w_line.at(t_old), 1.0)
    h_growth = h_line.at(t_new) / max(h_line.at(t_old), 1.0)
    if w_growth < cfg.min_growth_ratio and h_growth < cfg.min_growth_ratio:
        path = _sample_fitted_path(cx_line, cy_line, now, now + cfg.frame_gap, cfg.trajectory_points)
        return Forecast(
            track_id,
            class_name,
            False,
            "not getting bigger (likely passing or far)",
            None,
            None,
            None,
            path,
        )

    t_hit, axis = _pick_hit_time(w_line, h_line, frame_w, frame_h, float(now), cfg.size_axis)
    if t_hit is None:
        return Forecast(track_id, class_name, False, "no growing side to extrapolate", None, None, None)

    # If the line says the box already filled the screen, the fill *time* is in the
    # past. Collision *now* depends on where the centre is now, not where it was
    # when the size first crossed the screen.
    t_eval = max(t_hit, float(now))
    frames_from_now = t_eval - now
    ttc = frames_from_now / fps if fps > 0 else None
    impact = box_from_center(cx_line.at(t_eval), cy_line.at(t_eval), w_line.at(t_eval), h_line.at(t_eval))
    path_end = max(t_eval, float(now) + 1.0)
    path = _sample_fitted_path(cx_line, cy_line, float(now), path_end, cfg.trajectory_points)
    ghosts = [
        box_from_center(
            cx_line.at(float(now) + (path_end - now) * frac),
            cy_line.at(float(now) + (path_end - now) * frac),
            w_line.at(float(now) + (path_end - now) * frac),
            h_line.at(float(now) + (path_end - now) * frac),
        )
        for frac in (0.35, 0.7, 1.0)
        if path_end > now
    ]

    if ttc is not None and ttc <= 1e-6:
        ttc = 0.0
        reason_time = "already filling the view"
    elif ttc is not None and ttc > cfg.max_ttc_seconds:
        return Forecast(
            track_id,
            class_name,
            False,
            f"would fill {axis} only in {ttc:.1f}s (beyond max_ttc)",
            ttc,
            impact.cx,
            impact.cy,
            path,
            ghosts,
        )
    else:
        reason_time = f"fills {axis} in {ttc:.2f}s" if ttc is not None else "fills screen"

    hits_zone = centre_in_margin_frame(
        impact.cx,
        impact.cy,
        frame_w,
        frame_h,
        cfg.frame_margin_horizontal,
        cfg.frame_margin_vertical,
    )
    zone = f"{cfg.frame_margin_horizontal:.2f}x W, {cfg.frame_margin_vertical:.2f}x H"
    if hits_zone:
        return Forecast(
            track_id,
            class_name,
            True,
            f"COLLISION likely — {reason_time}, centre stays in {zone}",
            ttc,
            impact.cx,
            impact.cy,
            path,
            ghosts,
            raw_collide=True,
        )
    return Forecast(
        track_id,
        class_name,
        False,
        f"passing aside — {reason_time}, centre leaves the {zone} frame",
        ttc,
        impact.cx,
        impact.cy,
        path,
        ghosts,
        raw_collide=False,
    )


def _sample_fitted_path(
    cx_line: Line,
    cy_line: Line,
    t_start: float,
    t_end: float,
    n: int,
) -> list[tuple[float, float]]:
    n = max(2, n)
    span = t_end - t_start
    if abs(span) < 1e-6:
        return [(cx_line.at(t_start), cy_line.at(t_start))]
    return [
        (cx_line.at(t_start + span * i / (n - 1)), cy_line.at(t_start + span * i / (n - 1)))
        for i in range(n)
    ]


class HitLatch:
    """A track is a collision only after `confirm_hits` consecutive geometric hits.

    One noisy frame cannot flip the red HIT label. A miss, or the track dropping
    out of view, resets the streak to zero.
    """

    def __init__(self, confirm_hits: int):
        self.confirm_hits = max(1, confirm_hits)
        self._recent: dict[int, list[bool]] = {}

    def apply(self, forecasts: list[Forecast], live_ids: set[int]) -> list[Forecast]:
        forecast_ids = {f.track_id for f in forecasts}
        for tid in list(self._recent):
            if tid not in live_ids or tid not in forecast_ids:
                del self._recent[tid]

        gated: list[Forecast] = []
        for forecast in forecasts:
            hist = self._recent.setdefault(forecast.track_id, [])
            hist.append(bool(forecast.will_collide))
            self._recent[forecast.track_id] = hist[-self.confirm_hits :]
            flags = self._recent[forecast.track_id]
            streak = 0
            for bit in reversed(flags):
                if not bit:
                    break
                streak += 1
            confirmed = len(flags) >= self.confirm_hits and all(flags)
            gated.append(
                replace(
                    forecast,
                    will_collide=confirmed,
                    raw_collide=forecast.will_collide,
                    hit_streak=streak,
                    confirm_hits=self.confirm_hits,
                    reason=(
                        forecast.reason
                        if confirmed or not forecast.will_collide
                        else f"pending {streak}/{self.confirm_hits} — {forecast.reason}"
                    ),
                )
            )
        return gated


class TrackMemory:
    """Keep recent boxes per track id so we can fit a line over the last k frames."""

    def __init__(self, frame_gap: int):
        self.frame_gap = max(2, frame_gap)
        self._history: dict[int, list[tuple[int, Box, str]]] = {}

    def update(self, frame_idx: int, track_id: int, box: Box, class_name: str) -> None:
        hist = self._history.setdefault(track_id, [])
        hist.append((frame_idx, box, class_name))
        cutoff = frame_idx - 4 * self.frame_gap
        self._history[track_id] = [item for item in hist if item[0] >= cutoff]

    def window(self, track_id: int, current_frame: int) -> list[tuple[int, Box, str]]:
        """Last k frames: [current-(k-1), current], if the object was seen on each."""
        hist = self._history.get(track_id, [])
        start = current_frame - (self.frame_gap - 1)
        return [item for item in hist if item[0] >= start]


def _clip_pt(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    return int(max(0, min(w - 1, x))), int(max(0, min(h - 1, y)))


def draw_dashed_rect(frame, box: Box, color, thickness: int = 2, dash: int = 8) -> None:
    x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)

    for (a, b, c, d) in (
        (x1, y1, x2, y1),
        (x2, y1, x2, y2),
        (x2, y2, x1, y2),
        (x1, y2, x1, y1),
    ):
        length = max(abs(c - a), abs(d - b))
        if length == 0:
            continue
        steps = max(1, length // dash)
        for i in range(0, steps, 2):
            t0 = i / steps
            t1 = min(1.0, (i + 1) / steps)
            p0 = (int(a + (c - a) * t0), int(b + (d - b) * t0))
            p1 = (int(a + (c - a) * t1), int(b + (d - b) * t1))
            cv2.line(frame, p0, p1, color, thickness)


def draw_margin_frame(
    frame,
    margin_horizontal: float,
    margin_vertical: float,
    color=(80, 80, 80),
) -> None:
    h, w = frame.shape[:2]
    extra_w = 0.5 * (margin_horizontal - 1.0) * w
    extra_h = 0.5 * (margin_vertical - 1.0) * h
    x1 = int(round(-extra_w))
    y1 = int(round(-extra_h))
    x2 = int(round(w + extra_w))
    y2 = int(round(h + extra_h))
    vis_x1 = max(0, min(w - 1, x1))
    vis_y1 = max(0, min(h - 1, y1))
    vis_x2 = max(0, min(w - 1, x2))
    vis_y2 = max(0, min(h - 1, y2))
    cv2.rectangle(frame, (vis_x1, vis_y1), (vis_x2, vis_y2), color, 1)
    cv2.putText(
        frame,
        f"hit zone = {margin_horizontal:.2f}x W, {margin_vertical:.2f}x H",
        (12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
    )


def draw_forecasts(frame, forecasts: list[Forecast], current_boxes: list[tuple[Box, int, str]]) -> None:
    h, w = frame.shape[:2]
    by_id = {f.track_id: f for f in forecasts}

    for box, track_id, class_name in current_boxes:
        forecast = by_id.get(track_id)
        if forecast is None:
            color = (200, 200, 0)
            label = f"{class_name} #{track_id}"
        elif forecast.will_collide:
            color = (0, 0, 255)
            ttc = f"  ttc {forecast.ttc_seconds:.1f}s" if forecast.ttc_seconds is not None else ""
            label = f"HIT {class_name} #{track_id}{ttc}"
        elif forecast.raw_collide:
            color = (0, 165, 255)
            label = f"WAIT {forecast.hit_streak}/{forecast.confirm_hits} {class_name} #{track_id}"
        else:
            color = (0, 200, 255)
            label = f"{class_name} #{track_id}"

        p1 = (int(box.x1), int(box.y1))
        p2 = (int(box.x2), int(box.y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        cv2.putText(frame, label, (p1[0], max(16, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if forecast is None:
            continue

        path = forecast.path
        for i in range(1, len(path)):
            a = _clip_pt(path[i - 1][0], path[i - 1][1], w, h)
            b = _clip_pt(path[i][0], path[i][1], w, h)
            cv2.line(frame, a, b, color, 2)
            cv2.circle(frame, b, 3, color, -1)

        for ghost in forecast.ghost_boxes:
            draw_dashed_rect(frame, ghost, color, thickness=1, dash=6)

        if forecast.impact_cx is not None and forecast.impact_cy is not None:
            ix, iy = forecast.impact_cx, forecast.impact_cy
            if 0 <= ix < w and 0 <= iy < h:
                cv2.drawMarker(frame, (int(ix), int(iy)), color, cv2.MARKER_CROSS, 14, 2)

        if forecast.will_collide:
            tag = "COLLISION"
        elif forecast.raw_collide:
            tag = f"wait {forecast.hit_streak}/{forecast.confirm_hits}"
        else:
            tag = "no hit"
        cv2.putText(
            frame,
            tag,
            (int(box.x1), min(h - 8, int(box.y2) + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

