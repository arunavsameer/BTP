"""Shared training / evaluation helpers.

Kept model-agnostic: callers pass a ``predict_fn(batch) -> future_heatmap`` so the
same evaluation path works for the Stage-0 baseline, the teacher (via decoded H),
and the full student.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

from .metrics import MetricAccumulator


def move_batch(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def evaluate_future_heatmap(
    predict_fn: Callable[[dict], torch.Tensor],
    loader,
    cfg,
    device: str,
    episodes: list[str],
    has_rare: dict,
) -> dict:
    """Run ``predict_fn`` over ``loader`` and summarize metrics (overall + rare).

    ``predict_fn`` returns predicted FUTURE heatmaps [B, grid, grid]. The batch must
    carry ``h_future`` and ``ep_idx``.
    """
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


def train_one_epoch(
    model,
    loader,
    optimizer,
    device: str,
    loss_step: Callable[[dict], torch.Tensor],
) -> float:
    """Generic train loop. ``loss_step(batch) -> scalar loss`` (model already set)."""
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_step(batch)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * batch["h_future"].shape[0]
        n += batch["h_future"].shape[0]
    return total / max(n, 1)


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
    """Set thread/cuDNN options for fast, stable training on this machine."""
    # Fixed input sizes -> let cuDNN pick the fastest kernels.
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        for name in ("enable_flash_sdp", "enable_mem_efficient_sdp"):
            fn = getattr(torch.backends.cuda, name, None)
            if callable(fn):
                try:
                    fn(True)
                except Exception:
                    pass


def prefetch_items(names, load_fn) -> Iterator:
    """Decode/load the next item on a CPU thread while the GPU works."""
    names = list(names)
    if not names:
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(load_fn, names[0])
        for i in range(len(names)):
            item = fut.result()
            if i + 1 < len(names):
                fut = pool.submit(load_fn, names[i + 1])
            if item is not None:
                yield item


def uint8_clips_to_device(
    clips_u8: np.ndarray,
    device: str,
    targets: np.ndarray | None = None,
):
    """Pin and move packed uint8 clips ``[B,T,3,H,W]`` to GPU, normalize to [0,1]."""
    x = torch.from_numpy(np.ascontiguousarray(clips_u8))
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    x = x.float().div_(255.0)
    if targets is None:
        return x
    y = torch.from_numpy(np.ascontiguousarray(targets))
    if device == "cuda":
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        y = y.to(device)
    return x, y
