"""Discover complete clips. Incomplete directories are skipped, never crash the run."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .heatmap_annotations import HeatmapError, load_heatmaps
from .video_reader import VideoReadError, probe_video

logger = logging.getLogger("collision_jepa3.manifest")

OUTCOME_NAMES = ("collision", "miss_left", "miss_right")
REQUIRED_FILES = (
    "preview.mp4",
    "episode.json",
    "spatial_annotations/spatial_annotations.json",
)


@dataclass
class ClipRecord:
    clip_id: str
    sample_id: str
    scenario_group_id: str
    scenario_type: str
    outcome_type: str
    background_id: str
    video_path: str
    episode_json_path: str
    heatmap_json_path: str
    frame_count: int
    fps: float
    width: int
    height: int
    valid: bool
    invalid_reason: str | None = None
    overlay_path: str | None = None
    annotations_path: str | None = None
    lighting: str | None = None
    biome: str | None = None
    bg_style: str | None = None
    diverge_frame: int | None = None
    hit_s: float | None = None
    outcome_group_id: str | None = None
    background_pair_id: str | None = None
    fork_frame: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scenario_type(sample_dir: str) -> str:
    name = sample_dir
    if name.startswith("sample_") and "_" in name[7:]:
        rest = name.split("_", 2)
        if len(rest) >= 3:
            return rest[2]
    return name


def _read_episode(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inspect_clip_dir(
    clip_dir: Path,
    dataset_root: Path,
    cfg,
) -> ClipRecord:
    rel = clip_dir.relative_to(dataset_root).as_posix()
    parts = Path(rel).parts
    sample_dir = parts[0] if parts else clip_dir.name
    outcome = parts[1] if len(parts) > 1 else "unknown"
    background = parts[2] if len(parts) > 2 else "unknown"
    sample_id = sample_dir
    scenario_type = _scenario_type(sample_dir)
    video = clip_dir / "preview.mp4"
    episode_p = clip_dir / "episode.json"
    hm_p = clip_dir / "spatial_annotations" / "spatial_annotations.json"
    overlay = clip_dir / "spatial_overlay.mp4"
    ann_p = clip_dir / "annotations" / "annotations.json"

    missing = [name for name in REQUIRED_FILES if not (clip_dir / name).exists()]
    lighting = biome = bg_style = None
    diverge = hit_s = None
    fork = int(cfg.get("sampling.fork_frame", 365))
    width = height = 0
    fps = 0.0
    frame_count = 0
    extra: dict[str, Any] = {}

    if missing:
        return ClipRecord(
            clip_id=rel,
            sample_id=sample_id,
            scenario_group_id=sample_dir,
            scenario_type=scenario_type,
            outcome_type=outcome,
            background_id=background,
            video_path=str(video),
            episode_json_path=str(episode_p),
            heatmap_json_path=str(hm_p),
            frame_count=0,
            fps=0.0,
            width=0,
            height=0,
            valid=False,
            invalid_reason=f"incomplete: missing {missing}",
            overlay_path=str(overlay) if overlay.exists() else None,
            annotations_path=str(ann_p) if ann_p.exists() else None,
            outcome_group_id=f"{sample_dir}|{background}",
            background_pair_id=f"{sample_dir}|{outcome}",
            fork_frame=fork,
        )

    try:
        ep = _read_episode(episode_p)
        lighting = ep.get("bg_lighting") or (ep.get("environment") or {}).get("lighting")
        biome = ep.get("biome") or (ep.get("environment") or {}).get("biome")
        bg_style = ep.get("bg_style")
        diverge = ep.get("diverge_frame", fork)
        hit_s = ep.get("hit_s")
        extra = {
            "scenario": ep.get("scenario"),
            "ego_mode": (ep.get("ego") or {}).get("mode"),
            "label_histogram": ep.get("label_histogram"),
        }
        if ep.get("spatial_overlay") and "spatial_overlay" in str(ep.get("video", "")):
            extra["warn_overlay_as_video"] = True
        if str(ep.get("video", "preview.mp4")) == "spatial_overlay.mp4":
            raise VideoReadError("episode.json points at overlay video; refusing leak")
    except Exception as exc:
        return _invalid(
            rel, sample_id, sample_dir, scenario_type, outcome, background,
            video, episode_p, hm_p, overlay, ann_p, fork, f"episode.json: {exc}"
        )

    try:
        info = probe_video(video)
        width, height, fps, frame_count = info.width, info.height, info.fps, info.reported_frames
    except VideoReadError as exc:
        return _invalid(
            rel, sample_id, sample_dir, scenario_type, outcome, background,
            video, episode_p, hm_p, overlay, ann_p, fork, str(exc),
            lighting, biome, bg_style, diverge, hit_s,
        )

    try:
        heatmaps = load_heatmaps(
            hm_p,
            expected_frames=int(cfg.get("data.expected_frames", 450)),
            grid=int(cfg.get("data.target_grid", 5)),
        )
        extra["heatmap_shape"] = list(heatmaps.shape)
        extra["heatmap_peak"] = float(heatmaps.max())
    except HeatmapError as exc:
        return _invalid(
            rel, sample_id, sample_dir, scenario_type, outcome, background,
            video, episode_p, hm_p, overlay, ann_p, fork, f"heatmaps: {exc}",
            lighting, biome, bg_style, diverge, hit_s, width, height, fps, frame_count,
        )

    expected_frames = int(cfg.get("data.expected_frames", 450))
    min_frames = int(cfg.get("data.min_decoded_frames", expected_frames))
    if frame_count < min_frames:
        reason = f"reported_frames={frame_count} < min_decoded_frames={min_frames}"
        return _invalid(
            rel, sample_id, sample_dir, scenario_type, outcome, background,
            video, episode_p, hm_p, overlay, ann_p, fork, reason,
            lighting, biome, bg_style, diverge, hit_s, width, height, fps, frame_count,
        )

    return ClipRecord(
        clip_id=rel,
        sample_id=sample_id,
        scenario_group_id=sample_dir,
        scenario_type=scenario_type,
        outcome_type=outcome,
        background_id=background,
        video_path=str(video.resolve()),
        episode_json_path=str(episode_p.resolve()),
        heatmap_json_path=str(hm_p.resolve()),
        frame_count=int(frame_count),
        fps=float(fps),
        width=int(width),
        height=int(height),
        valid=True,
        invalid_reason=None,
        overlay_path=str(overlay.resolve()) if overlay.exists() else None,
        annotations_path=str(ann_p.resolve()) if ann_p.exists() else None,
        lighting=lighting,
        biome=biome,
        bg_style=bg_style,
        diverge_frame=int(diverge) if diverge is not None else fork,
        hit_s=float(hit_s) if hit_s is not None else None,
        outcome_group_id=f"{sample_dir}|{background}",
        background_pair_id=f"{sample_dir}|{outcome}",
        fork_frame=int(diverge) if diverge is not None else fork,
        extra=extra,
    )


def _invalid(
    rel, sample_id, sample_dir, scenario_type, outcome, background,
    video, episode_p, hm_p, overlay, ann_p, fork, reason,
    lighting=None, biome=None, bg_style=None, diverge=None, hit_s=None,
    width=0, height=0, fps=0.0, frame_count=0,
) -> ClipRecord:
    logger.warning("skipping %s: %s", rel, reason)
    return ClipRecord(
        clip_id=rel,
        sample_id=sample_id,
        scenario_group_id=sample_dir,
        scenario_type=scenario_type,
        outcome_type=outcome,
        background_id=background,
        video_path=str(video),
        episode_json_path=str(episode_p),
        heatmap_json_path=str(hm_p),
        frame_count=int(frame_count),
        fps=float(fps),
        width=int(width),
        height=int(height),
        valid=False,
        invalid_reason=str(reason),
        overlay_path=str(overlay) if overlay.exists() else None,
        annotations_path=str(ann_p) if Path(ann_p).exists() else None,
        lighting=lighting,
        biome=biome,
        bg_style=bg_style,
        diverge_frame=int(diverge) if diverge is not None else fork,
        hit_s=float(hit_s) if hit_s is not None else None,
        outcome_group_id=f"{sample_dir}|{background}",
        background_pair_id=f"{sample_dir}|{outcome}",
        fork_frame=fork,
    )


def iter_clip_dirs(dataset_root: Path) -> Iterable[Path]:
    root = Path(dataset_root)
    if not root.exists():
        return []
    for sample in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("sample_")):
        for outcome_dir in sorted(p for p in sample.iterdir() if p.is_dir()):
            if outcome_dir.name not in OUTCOME_NAMES:
                continue
            for bg_dir in sorted(p for p in outcome_dir.iterdir() if p.is_dir() and p.name.startswith("bg_")):
                yield bg_dir
    return []


def build_manifest(
    cfg,
    dataset_root: Path | None = None,
    *,
    max_samples: int | None = None,
) -> list[ClipRecord]:
    root = Path(dataset_root) if dataset_root is not None else cfg.dataset_root()
    clip_dirs = list(iter_clip_dirs(root))
    if max_samples is not None and max_samples > 0:
        allowed = set()
        for d in clip_dirs:
            sample_name = d.relative_to(root).parts[0]
            allowed.add(sample_name)
            if len(allowed) >= max_samples:
                break
        clip_dirs = [d for d in clip_dirs if d.relative_to(root).parts[0] in allowed]
        logger.info("max_samples=%d → keeping %d sample dirs (%d clip dirs)", max_samples, len(allowed), len(clip_dirs))
    records = [inspect_clip_dir(d, root, cfg) for d in clip_dirs]
    n_valid = sum(1 for r in records if r.valid)
    logger.info("manifest: %d clip dirs, %d valid, %d invalid", len(records), n_valid, len(records) - n_valid)
    return records


def save_manifest(records: list[ClipRecord], path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.to_dict(), default=str) for r in records]
    text = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_manifest(path: str | Path) -> list[ClipRecord]:
    path = Path(path)
    records: list[ClipRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(ClipRecord(**json.loads(line)))
    return records


def manifest_stats(records: list[ClipRecord]) -> dict[str, Any]:
    from collections import Counter

    valid = [r for r in records if r.valid]
    invalid = [r for r in records if not r.valid]
    return {
        "n_total": len(records),
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "invalid_reasons": dict(Counter(r.invalid_reason or "unknown" for r in invalid)),
        "outcomes": dict(Counter(r.outcome_type for r in valid)),
        "backgrounds": dict(Counter(r.background_id for r in valid)),
        "scenarios": dict(Counter(r.scenario_type for r in valid)),
        "biomes": dict(Counter(r.biome or "unknown" for r in valid)),
        "lighting": dict(Counter(r.lighting or "unknown" for r in valid)),
        "bg_style": dict(Counter(r.bg_style or "unknown" for r in valid)),
        "groups": len({r.scenario_group_id for r in valid}),
    }


def valid_clips(records: list[ClipRecord]) -> list[ClipRecord]:
    return [r for r in records if r.valid]
