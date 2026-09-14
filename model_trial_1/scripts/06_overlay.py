"""Write Blender-style overlay videos: model pred (and GT when available).

Examples
--------
  # A few balanced test episodes (side-by-side GT | pred)
  python scripts/06_overlay.py --split test --limit 8

  # One episode from the catalog
  python scripts/06_overlay.py --episode mixed/episode_0000_jaywalker_turn_toward

  # Arbitrary mp4 (prediction only, wearable view)
  python scripts/06_overlay.py --video /path/to/clip.mp4

  python scripts/06_overlay.py --config configs/no_shuffle.yaml \\
      --ckpt checkpoints_noshuffle/student.pt --split test
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rs_jepa.config import Config, resolve_device  # noqa: E402
from rs_jepa.data import load_json  # noqa: E402
from rs_jepa.data.video import decode_all_frames, decode_video_full  # noqa: E402
from rs_jepa.models.student import RSJEPA  # noqa: E402
from rs_jepa.viz.overlay import stack_panel, write_mp4  # noqa: E402
from rs_jepa.warning import HitLatch, classify, severity_name  # noqa: E402


def _resize_rgb(frames: np.ndarray, size: int) -> np.ndarray:
    import cv2

    out = np.empty((frames.shape[0], size, size, 3), dtype=np.uint8)
    for i, fr in enumerate(frames):
        out[i] = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
    return out


@torch.no_grad()
def predict_heatmaps(
    model: RSJEPA,
    frames_small: np.ndarray,
    offsets: list[int],
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """frames_small [N,S,S,3] uint8 RGB → pred k×k at each t (zeros before history)."""
    n = int(frames_small.shape[0])
    grid = int(model.feature_grid)
    pred = np.zeros((n, grid, grid), dtype=np.float32)
    t_lo = -min(offsets)
    times = list(range(t_lo, n))
    x_all = np.transpose(frames_small.astype(np.float32) / 255.0, (0, 3, 1, 2))
    for i0 in range(0, len(times), batch_size):
        chunk = times[i0 : i0 + batch_size]
        clips = np.stack([x_all[[t + o for o in offsets]] for t in chunk], axis=0)
        tensor = torch.from_numpy(np.ascontiguousarray(clips)).to(device, non_blocking=True)
        out = model.predict_future_heatmap(tensor).float().cpu().numpy()
        for t, hm in zip(chunk, out):
            pred[t] = hm
    return pred


def _pick_episodes(keys: list[str], split: dict, limit: int, seed: int) -> list[str]:
    if limit <= 0 or limit >= len(keys):
        return list(keys)
    by_fam: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        by_fam[split["families"].get(k, "other")].append(k)
    rng = random.Random(seed)
    for v in by_fam.values():
        rng.shuffle(v)
    picked: list[str] = []
    fams = sorted(by_fam)
    i = 0
    while len(picked) < limit:
        fam = fams[i % len(fams)]
        if by_fam[fam]:
            picked.append(by_fam[fam].pop())
        i += 1
        if i > limit * 8:
            break
    return picked


def _load_display_and_student(video: Path | None, cache_frames: Path | None, student_size: int):
    if video is not None and video.is_file():
        display, fps = decode_video_full(video)
        small = _resize_rgb(display, student_size)
        return display, small, fps
    if cache_frames is not None and cache_frames.is_file():
        small = np.load(cache_frames)
        return small, small, 30.0
    raise FileNotFoundError("no preview.mp4 or cached frames")


def overlay_sequence(
    display: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray | None,
    warn: dict,
    *,
    mode: str,
    tau: int,
    latch: bool,
) -> list[np.ndarray]:
    n = display.shape[0]
    t_lo = 0
    for t in range(n):
        if float(np.abs(pred[t]).sum()) > 1e-6:
            t_lo = t
            break
    hit = HitLatch(int(warn["confirm_frames"]), int(warn["release_frames"])) if latch else None
    frames: list[np.ndarray] = []
    t0 = t_lo if mode == "wearable" else min(n - 1, t_lo + tau)
    for t in range(t0, n):
        rgb = display[t]
        if mode == "wearable":
            p = pred[t]
            g = gt[t] if gt is not None and t < len(gt) else None
            extra = "forecast +1.0s on NOW"
        else:
            src = t - tau
            p = pred[src] if src >= 0 else pred[t]
            g = gt[t] if gt is not None and t < len(gt) else None
            extra = "aligned at t+1.0s"

        pd, ps, _ = classify(p, warn)
        if hit is not None:
            pd, ps = hit.update(ps, pd)
        p_name = severity_name(ps)
        p_panel = stack_panel(rgb, p, "PRED", pd, p_name, extra)

        if g is None:
            frames.append(p_panel)
            continue
        gd, gs, _ = classify(g, warn)
        g_panel = stack_panel(rgb, g, "GT", gd, severity_name(gs), extra)
        # pad heights if banners match (they do)
        if g_panel.shape[0] != p_panel.shape[0]:
            h = max(g_panel.shape[0], p_panel.shape[0])
            g_panel = np.pad(g_panel, ((0, h - g_panel.shape[0]), (0, 0), (0, 0)))
            p_panel = np.pad(p_panel, ((0, h - p_panel.shape[0]), (0, 0), (0, 0)))
        gap = np.zeros((g_panel.shape[0], 8, 3), dtype=np.uint8)
        frames.append(np.concatenate([g_panel, gap, p_panel], axis=1))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--split", default="", choices=("", "val", "test", "train"))
    parser.add_argument("--episode", default="", help="Catalog key, e.g. mixed/episode_0000_...")
    parser.add_argument("--video", default="", help="Arbitrary mp4 (pred-only overlay).")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--mode", default="aligned", choices=("aligned", "wearable"))
    parser.add_argument("--latch", action="store_true", help="Apply SAFE/CAUTION/STOP hit latch.")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    device = resolve_device(cfg.get("train.device", "auto"))
    ckpt = Path(args.ckpt) if args.ckpt else Path(cfg.get("paths.ckpt_dir")) / "student.pt"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    model = RSJEPA.from_config(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    model.eval()
    print(f"[overlay] device={device}  ckpt={ckpt}")

    offsets = list(cfg.get("student.frame_offsets"))
    tau = int(cfg.get("horizon.tau_frames"))
    size = int(cfg.get("student.img_size"))
    warn = cfg.get("warning")
    out_dir = Path(args.out) if args.out else Path(cfg.get("paths.overlay_dir", ROOT / "overlays"))
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path | None, Path | None, Path | None]] = []
    # name, video, cache_frames, heatmap_npy
    if args.video:
        jobs.append((Path(args.video).stem, Path(args.video), None, None))
    else:
        split = load_json(cfg.get("data.split_file"))
        cache = Path(cfg.get("data.cache_dir"))
        if args.episode:
            keys = [args.episode]
        else:
            name = args.split or ("test" if split.get("test") else "val")
            keys = list(split[name])
            keys = _pick_episodes(keys, split, args.limit, args.seed)
            print(f"[overlay] split={name}  episodes={len(keys)}")
        for key in keys:
            ep_dir = Path(split["paths"][key])
            video = ep_dir / "preview.mp4"
            hm = cache / key / "heatmaps.npy"
            fr = cache / key / f"frames_{size}.npy"
            jobs.append((key.replace("/", "__"), video if video.is_file() else None, fr, hm if hm.is_file() else None))

    for i, (name, video, cache_fr, hm_path) in enumerate(jobs, start=1):
        display, small, fps = _load_display_and_student(video, cache_fr, size)
        n = min(len(display), len(small))
        display, small = display[:n], small[:n]
        pred = predict_heatmaps(model, small, offsets, device)
        gt = np.load(hm_path)[:n] if hm_path is not None else None
        if gt is not None and len(gt) < n:
            n = len(gt)
            display, small, pred = display[:n], small[:n], pred[:n]
        mode = args.mode
        if gt is None:
            mode = "wearable"
        vis = overlay_sequence(display, pred, gt, warn, mode=mode, tau=tau, latch=args.latch)
        dest = out_dir / f"{name}.mp4"
        write_mp4(dest, vis, fps=fps or 30.0)
        print(f"[overlay] {i}/{len(jobs)}  {dest}  frames={len(vis)}", flush=True)

    print(f"[overlay] wrote {len(jobs)} videos → {out_dir}")


if __name__ == "__main__":
    main()
