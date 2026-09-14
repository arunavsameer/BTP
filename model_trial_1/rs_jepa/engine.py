"""Shared training / evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .data.gpu_preprocess import prepare_batch
from .metrics import MetricAccumulator, WearableAccumulator, wearable_from_cfg


def move_batch(batch: dict, device: str) -> dict:
    return prepare_batch(batch, device, augment=False)


@torch.no_grad()
def predict_episode_heatmaps(
    model,
    frames_uint8: np.ndarray,
    offsets: list[int],
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """uint8 RGB [N,H,W,3] → pred k×k at each t (zeros before history)."""
    n = int(frames_uint8.shape[0])
    grid = int(model.feature_grid)
    pred = np.zeros((n, grid, grid), dtype=np.float32)
    t_lo = -min(offsets)
    times = list(range(t_lo, n))
    x_all = np.transpose(frames_uint8.astype(np.float32) / 255.0, (0, 3, 1, 2))
    for i0 in range(0, len(times), batch_size):
        chunk = times[i0 : i0 + batch_size]
        clips = np.stack([x_all[[t + o for o in offsets]] for t in chunk], axis=0)
        tensor = torch.from_numpy(np.ascontiguousarray(clips)).to(device, non_blocking=True)
        out = model.predict_future_heatmap(tensor).float().cpu().numpy()
        for t, hm in zip(chunk, out):
            pred[t] = hm
    return pred


@torch.no_grad()
def evaluate_future_heatmap(
    predict_fn: Callable[[dict], torch.Tensor],
    loader,
    cfg,
    device: str,
    episodes: list[str],
    has_rare: dict,
) -> dict:
    recall_threshold = float(cfg.get("heatmap.recall_threshold", 0.5))
    warn_cfg = cfg.get("warning")
    overall = MetricAccumulator(recall_threshold, warn_cfg)
    rare = MetricAccumulator(recall_threshold, warn_cfg)

    for batch in loader:
        batch = move_batch(batch, device)
        pred = predict_fn(batch).detach().float().cpu().numpy()
        true = batch["h_future"].detach().float().cpu().numpy()
        ep_idx = batch["ep_idx"].detach().cpu().numpy()
        for b in range(pred.shape[0]):
            overall.update(pred[b], true[b])
            ep_name = episodes[int(ep_idx[b])]
            if has_rare.get(ep_name, False):
                rare.update(pred[b], true[b])

    return {"overall": overall.summary(), "rare": rare.summary()}


def _score_heatmaps_wearable(
    pred: np.ndarray,
    true: np.ndarray,
    tau: int,
    t_lo: int,
    acc: WearableAccumulator,
    rare_acc: WearableAccumulator | None,
    is_rare: bool,
) -> None:
    t_hi = true.shape[0] - 1 - tau
    acc.start_episode()
    if rare_acc is not None and is_rare:
        rare_acc.start_episode()
    for t in range(t_lo, t_hi + 1):
        acc.update_heatmaps(pred[t], true[t + tau])
        if rare_acc is not None and is_rare:
            rare_acc.update_heatmaps(pred[t], true[t + tau])


@torch.no_grad()
def evaluate_wearable(
    cfg,
    episodes: list[str],
    has_rare: dict,
    device: str,
    model=None,
    copy_baseline: bool = False,
) -> dict:
    """Latched nuisance/miss on every frame of each episode (wearable beep)."""
    cache_dir = Path(cfg.get("data.cache_dir"))
    size = int(cfg.get("student.img_size"))
    offsets = list(cfg.get("student.frame_offsets"))
    tau = int(cfg.get("horizon.tau_frames"))
    t_lo = -min(offsets)
    overall = wearable_from_cfg(cfg)
    rare = wearable_from_cfg(cfg)
    if model is not None:
        model.eval()
    for ep in episodes:
        hms = np.load(cache_dir / ep / "heatmaps.npy")
        is_rare = bool(has_rare.get(ep, False))
        if copy_baseline:
            pred = hms
        else:
            if model is None:
                raise ValueError("model required unless copy_baseline=True")
            frames = np.load(cache_dir / ep / f"frames_{size}.npy", mmap_mode="r")
            pred = predict_episode_heatmaps(model, np.asarray(frames), offsets, device)
        _score_heatmaps_wearable(pred, hms, tau, t_lo, overall, rare, is_rare)
    return {"overall": overall.summary(), "rare": rare.summary()}


def wearable_better(cur: dict, best: dict | None) -> bool:
    """Prefer eligible checkpoints; among the same gate class, higher score."""
    if best is None:
        return True
    c_el, b_el = bool(cur.get("eligible")), bool(best.get("eligible"))
    if c_el != b_el:
        return c_el
    return float(cur["score"]) > float(best["score"])


def save_checkpoint(state: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime() -> None:
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
