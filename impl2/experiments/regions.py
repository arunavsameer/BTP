"""A box, and matching boxes from one frame to the next by overlap.

No appearance model. If a shirt's box still overlaps its box from the last
frame, it keeps the same id, and the collision fit can follow it.
"""

from dataclasses import dataclass


@dataclass
class Region:
    x1: float
    y1: float
    x2: float
    y2: float
    area: float
    name: str


def _iou(a: Region, b: Region) -> float:
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


class _Track:
    def __init__(self, track_id: int, region: Region):
        self.track_id = track_id
        self.region = region
        self.missed = 0


class RegionTracker:
    def __init__(self, iou_match: float = 0.2, max_missed: int = 8):
        self.iou_match = float(iou_match)
        self.max_missed = int(max_missed)
        self._next_id = 1
        self._tracks: list[_Track] = []

    def update(self, regions: list[Region]) -> list[tuple[Region, int]]:
        """Return (region, track_id) for this frame."""
        pairs = []
        for index, region in enumerate(regions):
            for track in self._tracks:
                score = _iou(region, track.region)
                if score >= self.iou_match:
                    pairs.append((score, index, track))
        pairs.sort(key=lambda item: item[0], reverse=True)

        # Best overlap wins. Each region and each old track is used at most once.
        assigned: dict[int, _Track] = {}
        used_ids = set()
        for _score, index, track in pairs:
            if index in assigned or track.track_id in used_ids:
                continue
            assigned[index] = track
            used_ids.add(track.track_id)
            track.region = regions[index]
            track.missed = 0

        for track in self._tracks:
            if track.track_id not in used_ids:
                track.missed += 1
        self._tracks = [track for track in self._tracks if track.missed <= self.max_missed]

        result = []
        for index, region in enumerate(regions):
            track = assigned.get(index)
            if track is None:
                track = _Track(self._next_id, region)
                self._next_id += 1
                self._tracks.append(track)
            result.append((track.region, track.track_id))
        return result
