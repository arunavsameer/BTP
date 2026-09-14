"""Render student 5x5 future-heatmap overlays on the original preview videos.

At each time t the student sees I(t-10), I(t-5), I(t) and predicts H(t+1s).
That prediction is drawn on the *current* frame (what a wearable would show now).
A ground-truth future 5x5 inset is shown for comparison.

Run:  python scripts/08_render_overlay.py
      python scripts/08_render_overlay.py --split val
      python scripts/08_render_overlay.py --episodes episode_0011_car_near_miss_lane
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa.config import Config, resolve_device  # noqa: E402
from collision_jepa.data.splits import load_split  # noqa: E402
from collision_jepa.data.unzip import find_episode_dirs  # noqa: E402
from collision_jepa.models.student import Student  # noqa: E402
from collision_jepa.warning import HitLatch, classify, severity_name  # noqa: E402

_SEV_BGR = {
    "SAFE": (80, 200, 80),
    "CAUTION": (0, 200, 255),
    "STOP": (0, 40, 255),
}


def episode_dir_map(cfg: Config) -> dict[str, Path]:
    raw_dir = Path(cfg.get("data.raw_dir"))
    roots = [p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("dataset")]
    root = roots[0] if roots else raw_dir
    return {p.name: p for p in find_episode_dirs(root)}


def colorize_heatmap(hm: np.ndarray, width: int, height: int) -> np.ndarray:
    """5x5 [0,1] -> BGR image of size (width, height) with crisp cells."""
    hm_u8 = np.clip(hm * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    return cv2.resize(color, (width, height), interpolation=cv2.INTER_NEAREST)


def draw_grid(img: np.ndarray, grid: int, color=(255, 255, 255), thickness=1) -> None:
    h, w = img.shape[:2]
    for i in range(1, grid):
        x = int(round(i * w / grid))
        y = int(round(i * h / grid))
        cv2.line(img, (x, 0), (x, h), color, thickness)
        cv2.line(img, (0, y), (w, y), color, thickness)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, thickness)


def inset_panel(hm: np.ndarray, size: int, label: str) -> np.ndarray:
    panel = colorize_heatmap(hm, size, size)
    draw_grid(panel, hm.shape[0], (220, 220, 220), 1)
    bar = np.zeros((28, size, 3), dtype=np.uint8)
    cv2.putText(bar, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def draw_banner(frame: np.ndarray, text: str, origin: tuple[int, int], sev_name: str) -> None:
    x, y = origin
    color = _SEV_BGR.get(sev_name, (255, 255, 255))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(frame, (x - 8, y - th - 12), (x + tw + 12, y + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def blend_overlay(frame_bgr: np.ndarray, hm: np.ndarray, alpha: float = 0.38) -> np.ndarray:
    heat = colorize_heatmap(hm, frame_bgr.shape[1], frame_bgr.shape[0])
    out = cv2.addWeighted(frame_bgr, 1.0 - alpha, heat, alpha, 0)
    draw_grid(out, hm.shape[0], (255, 255, 255), 1)
    return out


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    w, h = size
    for fourcc in ("mp4v", "avc1", "XVID"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open video writer for {path}")


@torch.no_grad()
def render_episode(
    ep: str,
    ep_dir: Path,
    cache_dir: Path,
    model: Student,
    device: str,
    cfg: Config,
    out_path: Path,
) -> None:
    size = int(cfg.get("student.img_size"))
    offsets = list(cfg.get("student.frame_offsets"))
    tau = int(cfg.get("horizon.tau_frames"))
    warn_cfg = cfg.get("warning")
    fps = float(cfg.get("data.fps", 30))
    t_lo = -min(offsets)

    video_path = ep_dir / "preview.mp4"
    frames_small = np.load(cache_dir / ep / f"frames_{size}.npy", mmap_mode="r")
    heatmaps = np.load(cache_dir / ep / "heatmaps.npy")
    n = min(len(heatmaps), frames_small.shape[0])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = open_writer(out_path, fps, (src_w, src_h))
    latch = HitLatch(int(warn_cfg["confirm_frames"]), int(warn_cfg["release_frames"]))

    inset = 180
    for t in range(n):
        ok, frame = cap.read()
        if not ok:
            break

        pred = None
        if t >= t_lo:
            clip = np.stack([np.array(frames_small[t + o]) for o in offsets], axis=0)
            clip = np.transpose(clip.astype(np.float32) / 255.0, (0, 3, 1, 2))
            x = torch.from_numpy(np.ascontiguousarray(clip)).unsqueeze(0).float().to(device)
            pred = model.predict_future_heatmap(x)[0].detach().cpu().numpy()

        gt_future = heatmaps[t + tau] if t + tau < n else None
        gt_now = heatmaps[t]

        vis = frame
        if pred is not None:
            vis = blend_overlay(frame, pred)
            p_dir, p_sev, _ = classify(pred, warn_cfg)
            s_dir, s_sev = latch.update(p_sev, p_dir)
            draw_banner(vis, f"PRED  {s_dir}  {severity_name(s_sev)}", (24, 48), severity_name(s_sev))
        else:
            draw_banner(vis, "PRED  (warming up)", (24, 48), "SAFE")

        if gt_future is not None:
            g_dir, g_sev, _ = classify(gt_future, warn_cfg)
            draw_banner(vis, f"GT+1s {g_dir}  {severity_name(g_sev)}", (24, 96), severity_name(g_sev))
        else:
            g_dir, g_sev, _ = classify(gt_now, warn_cfg)
            draw_banner(vis, f"GT now {g_dir}  {severity_name(g_sev)}", (24, 96), severity_name(g_sev))

        cv2.putText(
            vis,
            f"{ep}   t={t:03d}/{n - 1:03d}   overlay = predicted H(t+1s)",
            (24, src_h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        pred_panel = inset_panel(pred if pred is not None else np.zeros((5, 5), np.float32), inset, "PRED +1s")
        gt_src = gt_future if gt_future is not None else gt_now
        gt_panel = inset_panel(gt_src, inset, "GT +1s" if gt_future is not None else "GT now")
        panels = np.hstack([pred_panel, gt_panel])
        ph, pw = panels.shape[:2]
        vis[16 : 16 + ph, src_w - pw - 16 : src_w - 16] = panels

        writer.write(vis)

    cap.release()
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--split", default="val", choices=["val", "train", "all"])
    parser.add_argument("--episodes", nargs="*", default=None, help="Optional episode dir names.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    device = resolve_device(cfg.get("train.device", "auto"))
    split = load_split(cfg.get("data.split_file"))
    dir_map = episode_dir_map(cfg)

    if args.episodes:
        episodes = list(args.episodes)
    elif args.split == "all":
        episodes = sorted(set(split["train"]) | set(split["val"]))
    else:
        episodes = list(split[args.split])
    if args.limit:
        episodes = episodes[: args.limit]

    model = Student(
        int(cfg.get("student.cnn_width", 32)),
        int(cfg.get("student.feature_grid", 5)),
        int(cfg.get("student.z_channels", 16)),
    ).to(device)
    ckpt = Path(cfg.get("paths.ckpt_dir")) / "student.pt"
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[overlay] loaded {ckpt}  epoch={state.get('epoch')}  device={device}")

    out_dir = Path(cfg.get("paths.export_dir")) / "overlays"
    cache_dir = Path(cfg.get("data.cache_dir"))
    print(f"[overlay] writing {len(episodes)} videos -> {out_dir}")

    for i, ep in enumerate(episodes):
        if ep not in dir_map:
            print(f"[overlay] skip missing {ep}")
            continue
        out_path = out_dir / f"{ep}.mp4"
        render_episode(ep, dir_map[ep], cache_dir, model, device, cfg, out_path)
        print(f"[overlay] {i + 1}/{len(episodes)}  {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    print(f"[overlay] done. videos in {out_dir}")


if __name__ == "__main__":
    main()
