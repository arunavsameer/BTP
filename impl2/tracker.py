"""Keep one id on a blob while it walks across the frame.

Overlap still wins when the boxes sit on top of each other. If a shirt jumps
or misses a frame or two, the id follows the box we expect from the last
step. After that coast, the id is remembered for a few seconds: a later blob
of a similar color, near the last place, takes the same id.
"""

from collections import deque
import math

from segment import Blob


def _iou(a: Blob, b: Blob) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if overlap <= 0.0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - overlap
    if union <= 0.0:
        return 0.0
    return overlap / union


def _average(history: deque, latest: Blob) -> Blob:
    if len(history) == 1:
        return latest
    n = float(len(history))
    return Blob(
        x1=sum(blob.x1 for blob in history) / n,
        y1=sum(blob.y1 for blob in history) / n,
        x2=sum(blob.x2 for blob in history) / n,
        y2=sum(blob.y2 for blob in history) / n,
        area=latest.area,
        name=latest.name,
        color=latest.color,
    )


def _center(blob: Blob) -> tuple[float, float]:
    return 0.5 * (blob.x1 + blob.x2), 0.5 * (blob.y1 + blob.y2)


def _wh(blob: Blob) -> tuple[float, float]:
    return max(0.0, blob.x2 - blob.x1), max(0.0, blob.y2 - blob.y1)


def _color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def _union(a: Blob, b: Blob) -> Blob:
    """One box around two pieces of the same track."""
    return Blob(
        x1=min(a.x1, b.x1),
        y1=min(a.y1, b.y1),
        x2=max(a.x2, b.x2),
        y2=max(a.y2, b.y2),
        area=a.area + b.area,
        name=a.name,
        color=a.color,
    )


class _Track:
    def __init__(self, track_id: int, blob: Blob, smooth_frames: int):
        self.track_id = track_id
        self.blob = blob
        self.missed = 0
        self.hits = 1
        self.history: deque = deque([blob], maxlen=smooth_frames)
        self.cx, self.cy = _center(blob)
        self.w, self.h = _wh(blob)
        self.vx = 0.0
        self.vy = 0.0

    def predict(self) -> Blob:
        """Last box, shifted by the last step. A miss coasts one step further."""
        step = self.missed + 1
        cx = self.cx + self.vx * step
        cy = self.cy + self.vy * step
        return Blob(
            x1=cx - self.w / 2.0,
            y1=cy - self.h / 2.0,
            x2=cx + self.w / 2.0,
            y2=cy + self.h / 2.0,
            area=self.w * self.h,
            name=self.blob.name,
            color=self.blob.color,
        )

    def observe(self, blob: Blob) -> None:
        cx, cy = _center(blob)
        step = self.missed + 1
        measured_vx = (cx - self.cx) / step
        measured_vy = (cy - self.cy) / step
        # The first real step is taken as-is. After that, one noisy frame
        # cannot swing the prediction as hard as the raw step would.
        if self.hits <= 1:
            self.vx, self.vy = measured_vx, measured_vy
        else:
            self.vx = 0.65 * measured_vx + 0.35 * self.vx
            self.vy = 0.65 * measured_vy + 0.35 * self.vy
        self.cx, self.cy = cx, cy
        self.w, self.h = _wh(blob)
        self.history.append(blob)
        self.blob = _average(self.history, blob)
        self.hits += 1
        self.missed = 0


class BlobTracker:
    def __init__(
        self,
        iou_match: float = 0.2,
        max_missed: int = 15,
        min_track_frames: int = 1,
        smooth_frames: int = 1,
        center_match_frac: float = 0.75,
        center_match_px: float = 40.0,
        size_ratio: float = 2.5,
        color_absorb: float = 80.0,
        absorb_iou: float = 0.15,
        memory_frames: int = 90,
        memory_color: float = 50.0,
        memory_radius_px: float = 160.0,
        memory_radius_cap: float = 240.0,
        memory_size_ratio: float = 4.0,
    ):
        self.iou_match = float(iou_match)
        self.max_missed = int(max_missed)
        self.min_track_frames = max(1, int(min_track_frames))
        self.smooth_frames = max(1, int(smooth_frames))
        self.center_match_frac = float(center_match_frac)
        self.center_match_px = float(center_match_px)
        self.size_ratio = float(size_ratio)
        self.color_absorb = float(color_absorb)
        self.absorb_iou = float(absorb_iou)
        self.memory_frames = int(memory_frames)
        self.memory_color = float(memory_color)
        self.memory_radius_px = float(memory_radius_px)
        self.memory_radius_cap = float(memory_radius_cap)
        self.memory_size_ratio = float(memory_size_ratio)
        self._next_id = 1
        self._tracks: list[_Track] = []
        self._memory: list[_Track] = []

    def _center_gate(self, blob: Blob, predicted: Blob) -> tuple[bool, float]:
        dist = math.hypot(*[a - b for a, b in zip(_center(blob), _center(predicted))])
        width, height = _wh(predicted)
        limit = max(self.center_match_frac * max(width, height), self.center_match_px)
        return dist <= limit, dist

    def _size_ok(self, blob: Blob, predicted: Blob) -> bool:
        """Area may grow or shrink, but only up to size_ratio."""
        pred_w, pred_h = _wh(predicted)
        blob_w, blob_h = _wh(blob)
        pred_area = pred_w * pred_h
        blob_area = blob_w * blob_h
        if pred_area <= 0.0 or blob_area <= 0.0 or self.size_ratio < 1.0:
            return False
        ratio = blob_area / pred_area
        low = 1.0 / self.size_ratio
        return low <= ratio <= self.size_ratio

    def _memory_limit(self, track: _Track) -> float:
        long_side = max(track.w, track.h, 1.0)
        grown = self.memory_radius_px + 1.5 * min(track.missed, 40)
        return min(self.memory_radius_cap, max(grown, long_side * 1.5))

    def _memory_hit(self, blob: Blob, track: _Track) -> tuple[float, float] | None:
        """(color distance, place distance) when this blob can resume the track."""
        color = _color_distance(blob.color, track.blob.color)
        if color > self.memory_color:
            return None
        pred_area = max(track.w, 1.0) * max(track.h, 1.0)
        blob_w, blob_h = _wh(blob)
        ratio = (blob_w * blob_h) / pred_area
        if self.memory_size_ratio < 1.0:
            return None
        if ratio < 1.0 / self.memory_size_ratio or ratio > self.memory_size_ratio:
            return None
        bx, by = _center(blob)
        dist_last = math.hypot(bx - track.cx, by - track.cy)
        px, py = _center(track.predict())
        dist_pred = math.hypot(bx - px, by - py)
        dist = min(dist_last, dist_pred)
        if dist > self._memory_limit(track):
            return None
        return color, dist

    def _host_track(self, blob: Blob, predicted: dict[int, Blob], used_ids: set[int]) -> _Track | None:
        """Track that should keep this leftover piece, if any.

        A piece of a box we already matched this frame joins that id.
        A piece that still overlaps a coasting track wakes that id up.
        """
        best: _Track | None = None
        best_key: tuple[int, float, float] | None = None
        for track in self._tracks:
            pred = predicted[track.track_id]
            color = _color_distance(blob.color, pred.color)
            if color > self.color_absorb:
                continue
            if track.track_id in used_ids:
                score = _iou(blob, track.blob)
                if score < self.absorb_iou or blob.area > track.blob.area * 1.2:
                    continue
                key = (0, -score, color)
            else:
                score = _iou(blob, pred)
                if score < self.absorb_iou or not self._size_ok(blob, pred):
                    continue
                key = (1, -score, color)
            if best_key is None or key < best_key:
                best_key = key
                best = track
        return best

    def update(self, blobs: list[Blob]) -> list[tuple[Blob, int]]:
        """Return (blob, track_id) for blobs that have been seen enough times.

        Coasting tracks stay in memory and keep their id, but they are not
        drawn until they match again. A miss does not clear the hit count.
        """
        predicted = {track.track_id: track.predict() for track in self._tracks}

        # Stage 1: highest overlap against the predicted box wins.
        overlap_pairs = []
        for blob_i, blob in enumerate(blobs):
            for track in self._tracks:
                pred = predicted[track.track_id]
                score = _iou(blob, pred)
                if score >= self.iou_match:
                    color = _color_distance(blob.color, pred.color)
                    overlap_pairs.append((score, color, blob_i, track))
        # Same-color box wins when two overlaps are close.
        overlap_pairs.sort(key=lambda item: (-item[0], item[1]))

        blob_track: dict[int, _Track] = {}
        used_ids: set[int] = set()
        for _score, _color, blob_i, track in overlap_pairs:
            if blob_i in blob_track or track.track_id in used_ids:
                continue
            blob_track[blob_i] = track
            used_ids.add(track.track_id)
            track.observe(blobs[blob_i])

        # Stage 2: leftovers whose center is close and whose size is similar.
        # Color sorts pairs that already passed; it is not a gate.
        center_pairs = []
        for blob_i, blob in enumerate(blobs):
            if blob_i in blob_track:
                continue
            for track in self._tracks:
                if track.track_id in used_ids:
                    continue
                pred = predicted[track.track_id]
                close, dist = self._center_gate(blob, pred)
                if not close or not self._size_ok(blob, pred):
                    continue
                color = _color_distance(blob.color, pred.color)
                center_pairs.append((dist, color, blob_i, track))
        center_pairs.sort(key=lambda item: (item[0], item[1]))

        for _dist, _color, blob_i, track in center_pairs:
            if blob_i in blob_track or track.track_id in used_ids:
                continue
            blob_track[blob_i] = track
            used_ids.add(track.track_id)
            track.observe(blobs[blob_i])

        # A shirt often splits into two blobs for one frame. The second piece
        # stays on the id that already took the first piece, instead of
        # starting a new track. A coasting track can also wake up on a piece
        # that still overlaps where we expected it.
        absorbed: set[int] = set()
        leftovers = [i for i in range(len(blobs)) if i not in blob_track]
        leftovers.sort(key=lambda i: blobs[i].area, reverse=True)
        for blob_i in leftovers:
            blob = blobs[blob_i]
            host = self._host_track(blob, predicted, used_ids)
            if host is None:
                continue
            if host.track_id in used_ids:
                host.blob = _union(host.blob, blob)
                absorbed.add(blob_i)
            else:
                host.observe(blob)
                used_ids.add(host.track_id)
                blob_track[blob_i] = host

        for track in self._tracks:
            if track.track_id not in used_ids:
                track.missed += 1

        # Resume an id that already left the live list, when a blob of the
        # same color shows up near the place we last saw it.
        remembered = list(self._memory)
        resumed: set[int] = set()
        still_open = [i for i in leftovers if i not in blob_track and i not in absorbed]
        still_open.sort(key=lambda i: blobs[i].area, reverse=True)
        for blob_i in still_open:
            blob = blobs[blob_i]
            best: _Track | None = None
            best_key: tuple[float, float] | None = None
            for track in remembered:
                if track.track_id in resumed:
                    continue
                hit = self._memory_hit(blob, track)
                if hit is None:
                    continue
                if best_key is None or hit < best_key:
                    best_key = hit
                    best = track
            if best is None:
                continue
            best.observe(blob)
            resumed.add(best.track_id)
            blob_track[blob_i] = best
            self._tracks.append(best)

        self._memory = [track for track in remembered if track.track_id not in resumed]
        for track in self._memory:
            track.missed += 1
        self._memory = [track for track in self._memory if track.missed <= self.memory_frames]

        still_live = []
        for track in self._tracks:
            if track.missed <= self.max_missed:
                still_live.append(track)
            elif track.missed <= self.memory_frames:
                self._memory.append(track)
        self._tracks = still_live

        assigned = []
        for blob_i, blob in enumerate(blobs):
            if blob_i in absorbed:
                continue
            track = blob_track.get(blob_i)
            if track is None:
                track = _Track(self._next_id, blob, self.smooth_frames)
                self._next_id += 1
                self._tracks.append(track)
            if track.hits < self.min_track_frames:
                continue
            assigned.append((track.blob, track.track_id))
        return assigned
