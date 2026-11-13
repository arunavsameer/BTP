"""Build a small centroid set from the street video, outside the test clips.

The open-vocabulary model is used once, offline, to mark centroids. FOMO
then trains on those grid cells. The three demo clips are not in this set.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from config import CLASSES, LABEL_PROMPTS, Config
from geom import frame_point_to_cell, letterbox
from model import Fomo


def video_duration(path: str) -> float:
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path,
        ]
    )
    return float(json.loads(raw)["format"]["duration"])


def sample_times(duration: float, cfg: Config) -> list[float]:
    blocked = [(start, end) for _name, start, end in cfg.clips]
    times = []
    t = 8.0
    while t < duration - 8.0 and len(times) < cfg.train_frames:
        if not any(a - 1.0 <= t <= b + 1.0 for a, b in blocked):
            times.append(t)
        t += cfg.train_every_seconds
    return times


def read_frame_at(path: str, seconds: float) -> np.ndarray | None:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{seconds:.3f}",
            "-i",
            path,
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-an",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    # Width and height are fixed for this video. 854x480.
    frame_bytes = 854 * 480 * 3
    if len(proc.stdout) < frame_bytes:
        return None
    return np.frombuffer(proc.stdout[:frame_bytes], np.uint8).reshape(480, 854, 3).copy()


def _load_labeler(cfg: Config):
    from ultralytics import YOLOE

    model = YOLOE(cfg.yolo_weights)
    model.eval()
    model.set_classes(list(LABEL_PROMPTS))
    return model


def centroids_from_yolo(result) -> list[tuple[str, float, float, float, float]]:
    """(fomo class, cx, cy, width, height) in the original frame."""
    found = []
    if result.boxes is None or len(result.boxes) == 0:
        return found
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.int().cpu().tolist()
    names = result.names
    for box, class_index in zip(xyxy, cls):
        prompt = names[int(class_index)]
        fomo_name = LABEL_PROMPTS.get(str(prompt))
        if fomo_name is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        found.append((fomo_name, 0.5 * (x1 + x2), 0.5 * (y1 + y2), max(1.0, x2 - x1), max(1.0, y2 - y1)))
    return found


def build_cache(cfg: Config) -> Path:
    cache = Path(cfg.cache)
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / "train_grids_v3.pt"
    if out.exists():
        return out

    times = sample_times(video_duration(cfg.video), cfg)
    print(f"labeling {len(times)} frames outside the demo clips")
    labeler = _load_labeler(cfg)
    fomo = Fomo(num_classes=len(CLASSES), input_size=cfg.input_size, torch_home=cfg.torch_home)
    fomo.eval()
    grid = cfg.input_size // cfg.stride
    features = []
    targets = []
    sizes = []
    kept = 0
    for index, seconds in enumerate(times):
        frame = read_frame_at(cfg.video, seconds)
        if frame is None:
            continue
        result = labeler.predict(frame, conf=cfg.label_conf, verbose=False)[0]
        points = centroids_from_yolo(result)
        canvas, scale, pad_x, pad_y = letterbox(frame, cfg.input_size)
        target = np.zeros((grid, grid), np.int64)
        size = np.zeros((2, grid, grid), np.float32)
        area_at = np.zeros((grid, grid), np.float32)
        for name, cx, cy, width, height in points:
            col, row = frame_point_to_cell(cx, cy, scale, pad_x, pad_y, cfg.stride)
            if not (0 <= col < grid and 0 <= row < grid):
                continue
            area = width * height
            if area < area_at[row, col]:
                continue
            area_at[row, col] = area
            # Row is y, column is x. Class ids are 1..N so 0 stays background.
            target[row, col] = CLASSES.index(name) + 1
            size[0, row, col] = float(np.log(width))
            size[1, row, col] = float(np.log(height))
        tensor = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            feat = fomo.features_from_images(tensor)[0].cpu()
        features.append(feat)
        targets.append(torch.from_numpy(target))
        sizes.append(torch.from_numpy(size))
        kept += 1
        if kept % 10 == 0:
            positives = int((target > 0).sum())
            print(f"  {kept} frames, last one has {positives} centroids")
    if not features:
        raise RuntimeError("no training frames were labeled")
    payload = {
        "features": torch.stack(features),
        "targets": torch.stack(targets),
        "sizes": torch.stack(sizes),
    }
    torch.save(payload, out)
    print(f"saved {kept} frames to {out}")
    return out


def train_head(cfg: Config, grids: Path) -> Path:
    cache = Path(cfg.cache)
    checkpoint = cache / "fomo_v3.pt"
    data = torch.load(grids, weights_only=False)
    features = data["features"]
    targets = data["targets"]
    sizes = data["sizes"]
    model = Fomo(num_classes=len(CLASSES), input_size=cfg.input_size, torch_home=cfg.torch_home)
    opt = torch.optim.Adam(model.head.parameters(), lr=cfg.lr)
    count = features.shape[0]
    print(f"training the FOMO head on {count} frames for {cfg.epochs} epochs")
    model.train()
    for epoch in range(cfg.epochs):
        order = torch.randperm(count)
        total = 0.0
        seen = 0
        for start in range(0, count, cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            logits = model.head(features[batch])
            target = targets[batch]
            loss = _balanced_loss(logits, target, sizes[batch], cfg.background_loss_weight, cfg.size_loss_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(batch)
            seen += len(batch)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:02d}  loss {total / max(seen, 1):.4f}")
    threshold = _choose_threshold(model, features, targets, cfg.conf_threshold)
    torch.save({"head": model.head.state_dict(), "threshold": threshold, "classes": CLASSES}, checkpoint)
    print(f"saved {checkpoint} with threshold {threshold:.2f}")
    return checkpoint


def _balanced_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    sizes: torch.Tensor,
    background_weight: float,
    size_weight: float,
) -> torch.Tensor:
    """Class loss on logits[:, :-2], smooth-L1 on log width/height at object cells."""
    class_logits = logits[:, :-2]
    flat_logits = class_logits.permute(0, 2, 3, 1).reshape(-1, class_logits.shape[1])
    flat_target = target.reshape(-1)
    loss = torch.nn.functional.cross_entropy(flat_logits, flat_target, reduction="none")
    weights = torch.where(flat_target == 0, flat_logits.new_tensor(background_weight), flat_logits.new_tensor(1.0))
    class_loss = (loss * weights).sum() / weights.sum().clamp(min=1.0)
    mask = target > 0
    if bool(mask.any()):
        pred = logits[:, -2:].permute(0, 2, 3, 1)[mask]
        true = sizes.permute(0, 2, 3, 1)[mask]
        size_loss = torch.nn.functional.smooth_l1_loss(pred, true)
    else:
        size_loss = class_loss * 0.0
    return class_loss + size_weight * size_loss


def _choose_threshold(model: Fomo, features: torch.Tensor, targets: torch.Tensor, default: float) -> float:
    """Pick a high threshold. False person cells are worse than a missed one."""
    model.eval()
    with torch.no_grad():
        logits = model.head(features)[:, :-2]
        prob = torch.softmax(logits, dim=1)
    background = prob[:, 0]
    obj_prob, obj_cls = prob[:, 1:].max(dim=1)
    true = targets
    best_score = -1.0
    best = default
    # beta=0.5 weights precision twice as much as recall.
    beta2 = 0.25
    for threshold in (0.45, 0.55, 0.65, 0.75, 0.85):
        pred_active = (obj_prob >= threshold) & (obj_prob >= background)
        pred_cls = torch.where(pred_active, obj_cls + 1, torch.zeros_like(obj_cls))
        tp = int(((pred_cls == true) & (true > 0)).sum())
        fp = int(((pred_cls > 0) & (pred_cls != true)).sum())
        fn = int(((true > 0) & (pred_cls != true)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        score = 0.0 if prec + rec == 0 else (1 + beta2) * prec * rec / (beta2 * prec + rec)
        print(f"  threshold {threshold:.2f}  precision {prec:.2f}  recall {rec:.2f}  score {score:.2f}")
        # Prefer the higher threshold when two scores are nearly the same.
        if score > best_score + 0.01 or (abs(score - best_score) <= 0.01 and threshold > best):
            best_score = score
            best = threshold
    print(f"  using threshold {best:.2f}")
    return best
