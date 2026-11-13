#!/usr/bin/env python3
"""Frame- and cell-wise SAFE/THREAT/STOP confusion on the group-random val set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset, build_dataloader  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.evaluation.operating_point import operating_metrics  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone  # noqa: E402
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state  # noqa: E402
from collision_jepa3.training.engine import move_batch, teacher_forward  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402

LABELS = ("SAFE", "THREAT", "STOP")


def band(peak: np.ndarray, caution: float, stop: float) -> np.ndarray:
    out = np.zeros(peak.shape, dtype=np.int64)
    out[peak >= caution] = 1
    out[peak >= stop] = 2
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int = 3) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true.ravel(), y_pred.ravel()):
        cm[int(t), int(p)] += 1
    return cm


def cm_report(cm: np.ndarray) -> dict:
    labels = list(LABELS)
    rec = {}
    prec = {}
    for i, name in enumerate(labels):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        rec[name] = tp / (tp + fn) if (tp + fn) else None
        prec[name] = tp / (tp + fp) if (tp + fp) else None
    # 2-class: not-SAFE vs SAFE
    t_pos = int(cm[1:, :].sum())
    t_pos_hit = int(cm[1:, 1:].sum())
    return {
        "labels": labels,
        "counts_true_row_pred_col": cm.tolist(),
        "support": {name: int(cm[i, :].sum()) for i, name in enumerate(labels)},
        "recall": rec,
        "precision": prec,
        "threat_or_stop_recall": t_pos_hit / t_pos if t_pos else None,
        "n": int(cm.sum()),
    }


@torch.no_grad()
def collect(model, loader, device: str) -> dict[str, np.ndarray]:
    model.eval()
    pred, gt, valid, pre, outcome, anchor = [], [], [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        out = teacher_forward(model, batch)
        p = out["Hhat_1s"].float().cpu().numpy()
        g = batch["h_1s"].float().cpu().numpy()
        amb = batch["amb_1s"].detach().cpu().numpy().astype(bool)
        pred.append(p)
        gt.append(g)
        valid.append(np.broadcast_to((~amb)[:, None, None], p.shape).copy())
        pre.append(batch["pre_fork"].detach().cpu().numpy().astype(bool))
        outcome.extend(list(batch["outcome_type"]))
        anchor.append(batch["anchor"].detach().cpu().numpy().astype(np.int64))
    return {
        "pred": np.concatenate(pred, axis=0),
        "gt": np.concatenate(gt, axis=0),
        "valid": np.concatenate(valid, axis=0),
        "pre_fork": np.concatenate(pre, axis=0),
        "outcome": np.asarray(outcome, dtype=object),
        "anchor": np.concatenate(anchor, axis=0),
    }


def slice_cm(pack: dict, caution: float, stop: float, mask: np.ndarray) -> dict:
    pred = pack["pred"][mask]
    gt = pack["gt"][mask]
    valid = pack["valid"][mask]
    keep = valid.reshape(len(pred), -1).all(axis=1)
    pred, gt = pred[keep], gt[keep]
    if len(pred) == 0:
        return {"n": 0}
    gt_peak = gt.reshape(len(gt), -1).max(axis=1)
    pr_peak = pred.reshape(len(pred), -1).max(axis=1)
    gt_f = band(gt_peak, caution, stop)
    pr_f = band(pr_peak, caution, stop)
    cell_keep = np.ones_like(gt, dtype=bool)
    gt_c = band(gt, caution, stop)[cell_keep]
    pr_c = band(pred, caution, stop)[cell_keep]
    op = operating_metrics(pred, gt, np.ones_like(pred, dtype=bool))
    return {
        "n_frames": int(len(pred)),
        "n_cells": int(gt_c.size),
        "gt_frame_mix": {n: int((gt_f == i).sum()) for i, n in enumerate(LABELS)},
        "gt_cell_mix": {n: int((gt_c == i).sum()) for i, n in enumerate(LABELS)},
        "mean_pred_peak": float(pr_peak.mean()),
        "mean_gt_peak": float(gt_peak.mean()),
        "operating": op,
        "frame": cm_report(confusion(gt_f, pr_f)),
        "cell": cm_report(confusion(gt_c, pr_c)),
        "mean_pred_gt80": op.get("mean_pred_gt80"),
        "recall_at_5fa": op.get("recall_at_5fa"),
        "recall_at_0.5": op.get("recall_at_0.5"),
        "ap_high": op.get("ap_high"),
    }


def _dummy(cfg) -> DummyVJEPA2Backbone:
    return DummyVJEPA2Backbone(
        hidden_size=1408,
        crop_size=int(cfg.get("preprocessing.input_size", 256)),
        patch_size=16,
        tubelet_size=2,
        num_frames=int(cfg.get("sampling.input_frames", 16)),
        num_layers=1,
        mean=tuple(cfg.get("preprocessing.mean")),
        std=tuple(cfg.get("preprocessing.std")),
    )


def score_ckpt(cfg, loader, device: str, path: Path) -> dict:
    model = CollisionTeacher3(cfg, backbone=_dummy(cfg))
    blob = load_checkpoint(path, map_location=device)
    load_teacher_state(model, blob)
    model.to(device).eval()
    pack = collect(model, loader, device)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    caution = float(cfg.get("warning.caution_threshold", 0.25))
    stop = float(cfg.get("warning.stop_threshold", 0.60))
    pre = pack["pre_fork"]
    coll = np.array([str(o) == "collision" for o in pack["outcome"]])
    return {
        "checkpoint": str(path),
        "epoch": blob.get("epoch"),
        "caution_threshold": caution,
        "stop_threshold": stop,
        "n_windows": int(len(pre)),
        "n_pre_fork": int(pre.sum()),
        "n_post_fork": int((~pre).sum()),
        "all": slice_cm(pack, caution, stop, np.ones(len(pre), dtype=bool)),
        "pre_fork": slice_cm(pack, caution, stop, pre),
        "post_fork": slice_cm(pack, caution, stop, ~pre),
        "collision": slice_cm(pack, caution, stop, coll),
        "miss": slice_cm(pack, caution, stop, ~coll),
    }


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="reports/direct_h1s_appear_blob_gr40b/warning_confusion.json")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "gr40b=checkpoints/direct_h1s_appear_blob_gr40b/teacher_phase_a_best.pt",
            "appear_blob=checkpoints/direct_h1s_appear_blob/teacher_phase_a_best.pt",
        ],
    )
    args = parser.parse_args()
    cfg = load_cfg(args)
    cfg.set("heads.mode", "direct_h1s")
    cfg.set("evaluation.val_subset_fraction", 1.0)
    device = resolve_device(args.device)
    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    ds = TeacherDataset(cfg, records, "val", splits, dummy_backbone=False, use_token_cache=True)
    loader = build_dataloader(ds, cfg, shuffle=False)
    print(
        f"val clips={len(ds.clip_ids())} windows={len(ds)} token_cache={getattr(ds._token_cache, 'root', None)}",
        flush=True,
    )
    report = {
        "split": "val",
        "anchor_mode": cfg.get("sampling.anchor_mode"),
        "n_clips": len(ds.clip_ids()),
        "n_windows": len(ds),
        "runs": {},
    }
    for item in args.checkpoints:
        name, path = item.split("=", 1)
        print(f"scoring {name} {path}", flush=True)
        report["runs"][name] = score_ckpt(cfg, loader, device, Path(path))
        print(json.dumps(report["runs"][name]["all"]["frame"]["recall"], indent=2), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    def _json(o):
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(type(o))

    out.write_text(json.dumps(report, indent=2, default=_json), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
