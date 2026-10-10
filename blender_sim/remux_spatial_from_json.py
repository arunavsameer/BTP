"""Recompute k×k heat from annotations.json and remux spatial_overlay.mp4.

No Blender / depsgraph. Scores come from current ``spatial_threat.py`` plus
JSON poses, boxes, and occlusion flags. Does not reuse baked overlay matrices.

Streetlamp / tree paint a pinhole-projected pole strip (not the cobra-arm /
canopy AABB). Other classes splat the JSON outer box.

    python remux_spatial_from_json.py --pack datasets/groups15_check_s61_n3
    python remux_spatial_from_json.py --clip datasets/.../miss_left/bg_00
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spatial_overlay import overlay_filter_complex, write_heat_sequence
from spatial_threat import (
    empty_grid,
    episode_spatial_payload,
    frame_spatial_entry,
    heading_frame,
    splat_object,
)
from threat_math import vdot, vnorm, vscale, vsub, yup_to_blender_zup


# Threat-scoring hulls (not reservation capsules). Floor is 0.12 in the scorer.
_HALF = {
    "person": (0.28, 0.38),
    "vehicle": (2.20, 0.95),
    "bicycle": (0.85, 0.40),
    "scooter": (0.70, 0.35),
    "streetlamp": (0.12, 0.12),
    "tree": (0.22, 0.22),
    "pothole": (0.50, 0.50),
    "crater": (0.50, 0.50),
    "broken_slab": (0.50, 0.50),
    "debris": (0.40, 0.40),
}
_POLE = frozenset({"streetlamp", "tree"})
_INFLIGHT_S = 45.0


def _half(class_name: str) -> tuple[float, float]:
    c = str(class_name or "").strip().lower()
    if c.startswith("threat_"):
        return _HALF["person"]
    if c.startswith("vehicle"):
        return _HALF["vehicle"]
    return _HALF.get(c, (0.35, 0.35))


def _heading(cam: dict, prev: tuple[float, float, float] | None) -> tuple[float, float, float]:
    vel = yup_to_blender_zup(cam.get("velocity") or (0.0, 0.0, 0.0))
    h = (float(vel[0]), float(vel[1]), 0.0)
    n = vnorm(h)
    if n < 0.08:
        return prev or (0.0, 1.0, 0.0)
    return vscale(h, 1.0 / n)


def _pole_box(
    p_cam: tuple[float, float, float],
    p_obj: tuple[float, float, float],
    heading: tuple[float, float, float],
    hfov_deg: float,
    bbox: tuple[float, float, float, float],
    res_x: int,
    res_y: int,
) -> list[tuple[float, float, float, float]]:
    """Thin vertical strip at the projected pole origin. Empty if off-screen."""
    s_hat, r_hat = heading_frame(heading)
    d = vsub(p_obj, p_cam)
    z_fwd = vdot((d[0], d[1], 0.0), s_hat)
    if z_fwd <= 1e-3:
        return []
    x_right = vdot((d[0], d[1], 0.0), r_hat)
    half_w = math.tan(math.radians(max(float(hfov_deg), 1.0)) * 0.5)
    u = (x_right / z_fwd / half_w + 1.0) * (res_x * 0.5)
    y0 = float(bbox[1])
    y1 = float(bbox[3])
    if y1 - y0 < 4.0:
        y0, y1 = 0.0, float(res_y)
    half = 10.0
    return [(u - half, y0, u + half, y1)]


def recompute_frames(
    object_frames: list[dict],
    *,
    k: int,
    res_x: int,
    res_y: int,
) -> list[dict]:
    heading: tuple[float, float, float] | None = None
    out: list[dict] = []
    kk = max(1, int(k))
    for rec in object_frames:
        cam = rec.get("camera_data") or {}
        heading = _heading(cam, heading)
        p_cam = yup_to_blender_zup(cam.get("world_position") or (0.0, 0.0, 0.0))
        walk = float(cam.get("ego_speed") or 0.0)
        hfov = float(cam.get("hfov_deg") or 54.71)
        grid = empty_grid(kk)
        for obj in rec.get("objects") or []:
            cls = str(obj.get("class_name") or "")
            kin = obj.get("kinematics") or {}
            p_obj = yup_to_blender_zup(kin.get("world_position") or (0.0, 0.0, 0.0))
            v_obj = yup_to_blender_zup(kin.get("velocity") or (0.0, 0.0, 0.0))
            flags = obj.get("flags") or {}
            occ = bool(flags.get("occluded"))
            bb = obj.get("bounding_box_2d") or {}
            bbox = (
                float(bb.get("xmin", 0.0)),
                float(bb.get("ymin", 0.0)),
                float(bb.get("xmax", 0.0)),
                float(bb.get("ymax", 0.0)),
            )
            hs, hl = _half(cls)
            delta = vsub(p_obj, p_cam)
            look_dot = vdot((delta[0], delta[1], 0.0), heading)
            z_cam = -look_dot
            behind = bool(z_cam >= -1e-3 or look_dot < -0.05)
            if cls in _POLE:
                boxes = _pole_box(p_cam, p_obj, heading, hfov, bbox, res_x, res_y)
            else:
                boxes = [bbox]
            splat_object(
                grid,
                p_cam=p_cam,
                p_obj=p_obj,
                v_obj=v_obj,
                heading=heading,
                walk_speed=walk,
                half_s=hs,
                half_lat=hl,
                bbox=bbox,
                res_x=res_x,
                res_y=res_y,
                boxes=boxes,
                class_name=cls,
                occluded=occ,
                behind_camera=behind,
                z_cam=z_cam,
            )
        fid = str(rec.get("frame_id", f"{len(out):06d}"))
        out.append(frame_spatial_entry(fid, grid))
    return out


def _count_mp4_frames(path: Path) -> int | None:
    exe = shutil.which("ffprobe")
    if exe is None or not path.is_file():
        return None
    proc = subprocess.run(
        [
            exe, "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets,nb_frames",
            "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    nums = [int(p) for p in (proc.stdout or "").replace("\n", ",").split(",") if p.strip().isdigit()]
    return max(nums) if nums else None


def _extract_pngs(preview: Path, rgb_dir: Path, n_frames: int, fps: int) -> None:
    rgb_dir.mkdir(parents=True, exist_ok=True)
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not on PATH")
    proc = subprocess.run(
        [
            exe, "-y", "-i", str(preview),
            "-start_number", "0",
            "-frames:v", str(int(n_frames)),
            str(rgb_dir / "%06d.png"),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(err[-800:] or "ffmpeg extract failed")
    first = rgb_dir / "000000.png"
    last = rgb_dir / f"{int(n_frames) - 1:06d}.png"
    if not first.is_file() or not last.is_file():
        got = len(list(rgb_dir.glob("*.png")))
        raise RuntimeError(f"extract wrote {got} pngs, expected {n_frames}")


def _mux_overlay(
    *,
    rgb_input: list[str],
    heat_dir: Path,
    n_frames: int,
    fps: int,
    k: int,
    res_x: int,
    res_y: int,
    out_path: Path,
) -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not on PATH")
    filt = overlay_filter_complex(k, res_x, res_y)
    tmp = out_path.with_name(out_path.stem + ".__tmp__.mp4")
    common = (
        [exe, "-y"]
        + rgb_input
        + [
            "-framerate", str(int(fps)), "-start_number", "0",
            "-i", str(heat_dir / "%06d.ppm"),
            "-frames:v", str(int(n_frames)),
            "-filter_complex", filt, "-map", "[out]",
            "-r", str(int(fps)), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        ]
    )

    def _run(cmd: list[str]) -> tuple[bool, str]:
        proc = subprocess.run(cmd + [str(tmp)], capture_output=True, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            return False, err[-800:] or "ffmpeg overlay failed"
        return True, ""

    ok, err = _run(common + ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "constqp", "-qp", "18"])
    via = "h264_nvenc"
    if not ok:
        ok, err = _run(common + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-threads", "0"])
        via = "libx264"
        if not ok:
            head, sep, _ = filt.partition(";[mix]drawgrid")
            if sep:
                filt_plain = f"{head};[mix]copy[out]"
                common[common.index(filt)] = filt_plain
                ok, err = _run(common + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-threads", "0"])
                via = "libx264-nodrawgrid"
            if not ok:
                raise RuntimeError(err)
    tmp.replace(out_path)
    return via


def is_inflight(clip: Path) -> bool:
    rgb = clip / "rgb"
    if not rgb.is_dir():
        return False
    pngs = list(rgb.glob("*.png"))
    if not pngs:
        return False
    newest = max(p.stat().st_mtime for p in pngs)
    if (time.time() - newest) < _INFLIGHT_S:
        return True
    # Render still writing the sequence (preview not muxed yet).
    return not (clip / "preview.mp4").is_file()


def remux_clip(clip: Path, *, res_x: int = 1920, res_y: int = 1080) -> dict:
    preview = clip / "preview.mp4"
    ann_path = clip / "annotations" / "annotations.json"
    spat_path = clip / "spatial_annotations" / "spatial_annotations.json"
    overlay = clip / "spatial_overlay.mp4"
    if is_inflight(clip):
        return {"clip": str(clip), "status": "skip_inflight"}
    if not preview.is_file() or not ann_path.is_file():
        return {"clip": str(clip), "status": "skip_missing"}
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    object_frames = list(ann.get("frames") or [])
    n_frames = len(object_frames)
    if n_frames < 2:
        return {"clip": str(clip), "status": "skip_short", "n": n_frames}
    old_k = 5
    old_frames = []
    if spat_path.is_file():
        old = json.loads(spat_path.read_text(encoding="utf-8"))
        old_k = int(old.get("k") or 5)
        old_frames = list(old.get("frames") or [])
    k = old_k
    fps = 30
    ep = clip / "episode.json"
    if ep.is_file():
        meta = json.loads(ep.read_text(encoding="utf-8"))
        fps = int(meta.get("fps") or fps)
        k = int(meta.get("spatial_annotations_k") or k)
    new_frames = recompute_frames(object_frames, k=k, res_x=res_x, res_y=res_y)
    payload = episode_spatial_payload(k, new_frames)

    n_video = _count_mp4_frames(preview)
    if n_video is not None and n_video != n_frames:
        return {
            "clip": str(clip),
            "status": "skip_framecount",
            "json_n": n_frames,
            "mp4_n": n_video,
        }

    work = Path(tempfile.mkdtemp(prefix="spatial_remux_", dir=str(clip)))
    heat_dir = work / "heat"
    rgb_dir = work / "rgb"
    try:
        written = write_heat_sequence(new_frames, heat_dir, k=k)
        if written < n_frames:
            raise RuntimeError(f"heat frames {written} < {n_frames}")
        def _from_png() -> str:
            _extract_pngs(preview, rgb_dir, n_frames, fps)
            return _mux_overlay(
                rgb_input=[
                    "-framerate", str(int(fps)), "-start_number", "0",
                    "-i", str(rgb_dir / "%06d.png"),
                ],
                heat_dir=heat_dir, n_frames=n_frames, fps=fps,
                k=k, res_x=res_x, res_y=res_y, out_path=overlay,
            ) + "+png"

        try:
            via = _mux_overlay(
                rgb_input=["-i", str(preview)],
                heat_dir=heat_dir, n_frames=n_frames, fps=fps,
                k=k, res_x=res_x, res_y=res_y, out_path=overlay,
            ) + "+preview.mp4"
            got = _count_mp4_frames(overlay)
            if got is not None and got != n_frames:
                via = _from_png()
        except RuntimeError:
            via = _from_png()
        spat_path.parent.mkdir(parents=True, exist_ok=True)
        spat_path.write_text(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    def _mx(frames: list, i: int) -> list | None:
        if 0 <= i < len(frames):
            return frames[i].get("matrix")
        return None

    return {
        "clip": str(clip),
        "status": "ok",
        "n": n_frames,
        "k": k,
        "encoder": via,
        "overlay": str(overlay),
        "old_f260": _mx(old_frames, 260),
        "new_f260": _mx(new_frames, 260),
        "old_f180": _mx(old_frames, 180),
        "new_f180": _mx(new_frames, 180),
    }


def iter_pack_clips(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.glob("sample_*/*/*/preview.mp4")})


def extract_spotcheck(overlay: Path, frame_ids: list[int], dest_dir: Path) -> list[Path]:
    exe = shutil.which("ffmpeg")
    if exe is None or not overlay.is_file():
        return []
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fid in frame_ids:
        out = dest_dir / f"overlay_spotcheck_{fid:04d}.png"
        proc = subprocess.run(
            [
                exe, "-y", "-i", str(overlay),
                "-vf", f"select=eq(n\\,{int(fid)})",
                "-vframes", "1", str(out),
            ],
            capture_output=True, text=True,
        )
        if proc.returncode == 0 and out.is_file():
            written.append(out)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, help="groups pack root")
    ap.add_argument("--clip", type=Path, action="append", default=[], help="one clip dir (repeatable)")
    ap.add_argument("--spotcheck-frames", default="180,260")
    args = ap.parse_args()
    clips: list[Path] = [p.resolve() for p in args.clip]
    if args.pack:
        clips.extend(iter_pack_clips(args.pack.resolve()))
    # unique, keep order
    seen: set[str] = set()
    ordered: list[Path] = []
    for c in clips:
        s = str(c)
        if s not in seen:
            seen.add(s)
            ordered.append(c)
    if not ordered:
        print("no clips", file=sys.stderr)
        return 2
    results = []
    for clip in ordered:
        print(f"remux {clip} ...", flush=True)
        try:
            r = remux_clip(clip)
        except Exception as exc:
            r = {"clip": str(clip), "status": "error", "error": str(exc)}
        results.append(r)
        print(f"  {r.get('status')}  {r.get('encoder', '')}  {r.get('error', '')}", flush=True)
        if r.get("status") == "ok" and r.get("old_f260") is not None:
            print(f"  f260 old {r['old_f260']}", flush=True)
            print(f"  f260 new {r['new_f260']}", flush=True)
    # Spot-check PNGs only for the first successful jaywalker miss_left/bg_00, else first ok.
    want = None
    for r in results:
        if r.get("status") != "ok":
            continue
        p = Path(r["clip"])
        if p.as_posix().endswith("sample_0000_jaywalker/miss_left/bg_00"):
            want = r
            break
        if want is None:
            want = r
    if want is not None:
        ids = [int(x) for x in str(args.spotcheck_frames).split(",") if x.strip().isdigit()]
        paths = extract_spotcheck(Path(want["overlay"]), ids, Path(want["clip"]))
        want["spotcheck"] = [str(p) for p in paths]
        print("spotcheck", want.get("spotcheck"), flush=True)
    print(json.dumps([{"clip": r["clip"], "status": r["status"], "encoder": r.get("encoder"), "error": r.get("error")} for r in results], indent=2))
    return 0 if any(r.get("status") == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
