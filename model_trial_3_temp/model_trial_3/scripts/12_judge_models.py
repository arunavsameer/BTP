#!/usr/bin/env python3
"""Score H_1s checkpoints with the multi-test judge (token cache, no Giant)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset, build_dataloader  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.evaluation.evaluator import collect_h1s_pack  # noqa: E402
from collision_jepa3.evaluation.judge import HeatmapPack, compare_models, mean_heatmap_prior  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone  # noqa: E402
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not math.isfinite(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if obj is None:
        return None
    return str(obj)


def _parse_spec(item: str) -> tuple[str, Path, str]:
    raw = item
    mode = "direct_h1s"
    if raw.count(":") >= 1 and raw.rsplit(":", 1)[-1] in {"direct_h1s", "residual"}:
        raw, mode = raw.rsplit(":", 1)
    if "=" in raw:
        name, path = raw.split("=", 1)
    else:
        path = raw
        name = Path(path).parent.name if Path(path).name.startswith("teacher_") else Path(path).stem
    return name, Path(path), mode


def _giant_dummy(cfg) -> DummyVJEPA2Backbone:
    # Token cache already has Giant tokens; the dummy encoder is never run.
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


def _load_model(cfg, path: Path, mode: str, device: str) -> CollisionTeacher3:
    cfg.set("heads.mode", mode)
    model = CollisionTeacher3(cfg, backbone=_giant_dummy(cfg))
    blob = load_checkpoint(str(path), map_location=device)
    load_teacher_state(model, blob)
    model.to(device).eval()
    return model, blob.get("epoch")


def _train_mean_map(ds_train: TeacherDataset, max_n: int = 4000) -> np.ndarray:
    maps = []
    for i, (cid, t) in enumerate(ds_train.index):
        if bool(ds_train.ambiguity.get((cid, int(t)), {}).get("1s", False)):
            continue
        hm = ds_train._heatmaps[cid]
        fut = int(t) + int(ds_train.horizons[0])
        if fut >= hm.shape[0]:
            continue
        maps.append(np.asarray(hm[fut], dtype=np.float64))
        if len(maps) >= max_n:
            break
    if not maps:
        return np.zeros((5, 5), dtype=np.float64)
    return np.mean(np.stack(maps, axis=0), axis=0)


def _pack_from_npz(path: Path) -> HeatmapPack:
    z = np.load(path, allow_pickle=True)
    return HeatmapPack(
        pred=z["pred"],
        gt=z["gt"],
        valid=z["valid"],
        clip_id=z["clip_id"],
        scenario=z["scenario"] if "scenario" in z else None,
        background=z["background"] if "background" in z else None,
        outcome=z["outcome"] if "outcome" in z else None,
        pre_fork=z["pre_fork"] if "pre_fork" in z else None,
        anchor=z["anchor"] if "anchor" in z else None,
    )


def _save_npz(path: Path, pack: HeatmapPack) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pred=pack.pred.astype(np.float32),
        gt=pack.gt.astype(np.float32),
        valid=pack.valid,
        clip_id=pack.clip_id,
        scenario=pack.scenario,
        background=pack.background,
        outcome=pack.outcome,
        pre_fork=pack.pre_fork,
        anchor=pack.anchor,
    )


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--models", nargs="+", default=None, help="name=path[:direct_h1s|residual]")
    parser.add_argument("--from-npz-dir", default=None, help="Skip GPU collect; load packs from this dir")
    parser.add_argument("--dump-npz-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()
    cfg = load_cfg(args)
    cfg.set("evaluation.val_subset_fraction", 1.0)
    out = Path(args.out)
    warning = cfg.get("warning")

    packs: dict[str, HeatmapPack] = {}
    meta: dict[str, dict] = {}
    if args.from_npz_dir:
        npz_dir = Path(args.from_npz_dir)
        for path in sorted(npz_dir.glob("*.npz")):
            packs[path.stem] = _pack_from_npz(path)
            meta[path.stem] = {"npz": str(path)}
        if "mean_prior" in packs:
            meta["mean_prior"]["source"] = "npz"
        gt_pack = next(p for n, p in packs.items() if n != "mean_prior")
    else:
        if not args.models:
            raise SystemExit("pass --models or --from-npz-dir")
        device = resolve_device(args.device)
        records = load_manifest(cfg.path("paths.manifest"))
        splits = load_splits(cfg.path("paths.split_file"))
        ds_val = TeacherDataset(cfg, records, "val", splits, dummy_backbone=False, use_token_cache=True)
        ds_train = TeacherDataset(cfg, records, "train", splits, dummy_backbone=False, use_token_cache=True)
        loader = build_dataloader(ds_val, cfg, shuffle=False)
        print(f"val clips={len(ds_val.clip_ids())} windows={len(ds_val)} device={device}", flush=True)
        train_mean = _train_mean_map(ds_train)
        gt_pack = None
        for spec in args.models:
            name, path, mode = _parse_spec(spec)
            print(f"collect {name} mode={mode} {path}", flush=True)
            model, epoch = _load_model(cfg, path, mode, device)
            pack = collect_h1s_pack(model, loader, device)
            packs[name] = pack
            meta[name] = {"checkpoint": str(path), "epoch": epoch, "heads_mode": mode}
            if args.dump_npz_dir:
                _save_npz(Path(args.dump_npz_dir) / f"{name}.npz", pack)
            gt_pack = pack
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        prior_pred = np.broadcast_to(train_mean, gt_pack.gt.shape).copy()
        packs["mean_prior"] = gt_pack.copy_pred(prior_pred)
        meta["mean_prior"] = {"source": "train_H_1s_mean", "n_train_maps_capped": 4000}
        if args.dump_npz_dir:
            _save_npz(Path(args.dump_npz_dir) / "mean_prior.npz", packs["mean_prior"])

    if "mean_prior" not in packs:
        packs["mean_prior"] = mean_heatmap_prior(gt_pack.gt, gt_pack.valid, gt_pack)
        meta["mean_prior"] = {"source": "val_gt_mean_fallback"}

    report = compare_models(
        packs,
        prior_name="mean_prior",
        warning=warning,
        seed=int(cfg.get("project.seed", 1337)),
        n_folds=int(args.n_folds),
        n_boot=int(args.n_boot),
    )
    report["models"] = meta
    report["n_windows"] = int(gt_pack.n)
    report["n_clips"] = int(len(set(gt_pack.clip_id.tolist())))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonify(report), indent=2), encoding="utf-8")
    print("winner", report["winner"], "unique", report["unique"], file=sys.stderr)
    print(report["reason"], file=sys.stderr)
    print("wrote", out)


if __name__ == "__main__":
    main()
