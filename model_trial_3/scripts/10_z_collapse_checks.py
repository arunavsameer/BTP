#!/usr/bin/env python3
"""Cheap Z/H diagnostics for direct H_1s checkpoints (token cache, no Giant)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.pairing import background_pair_valid  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.evaluation.metrics import direction_scores_np  # noqa: E402
from collision_jepa3.models.heatmap_decoder import HeatmapDecoder  # noqa: E402
from collision_jepa3.models.spatial_adapter import SpatialTemporalAdapter  # noqa: E402
from collision_jepa3.training.checkpoint import load_checkpoint  # noqa: E402


PROBE_ANCHORS = [90, 180, 270, 360, 405]


def cell_cosine(z_a: np.ndarray, z_b: np.ndarray) -> float:
    a = z_a.astype(np.float64).transpose(1, 2, 0).reshape(-1, z_a.shape[0])
    b = z_b.astype(np.float64).transpose(1, 2, 0).reshape(-1, z_b.shape[0])
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    return float((a * b).sum(axis=1).mean())


def summarize(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "mean_1_minus_cos": float(1.0 - arr.mean()),
    }


def pair_cosines(zs: dict, recs, ds, centered_mean: np.ndarray | None = None) -> dict[str, list[float]]:
    def z_of(key):
        z = zs[key]
        return z - centered_mean if centered_mean is not None else z

    bg_pair = []
    unrelated = []
    by_pair: dict[tuple[str, int], list[str]] = defaultdict(list)
    for cid, t in zs:
        pid = recs[cid].background_pair_id or ""
        if pid:
            by_pair[(pid, int(t))].append(cid)
    for (_pid, t), cids in by_pair.items():
        if len(cids) < 2:
            continue
        a, b = cids[0], cids[1]
        ok = background_pair_valid(
            recs[a], recs[b], ds._heatmaps[a], ds._heatmaps[b], int(t), ds.horizons, ds.bg_tol
        )
        if not ok:
            continue
        bg_pair.append(cell_cosine(z_of((a, t)), z_of((b, t))))

    by_t: dict[int, list[str]] = defaultdict(list)
    for cid, t in zs:
        by_t[int(t)].append(cid)
    rng = np.random.RandomState(1337)
    bag = []
    for t, cids in by_t.items():
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                ra, rb = recs[cids[i]], recs[cids[j]]
                if ra.scenario_group_id == rb.scenario_group_id:
                    continue
                if ra.background_pair_id and ra.background_pair_id == rb.background_pair_id:
                    continue
                bag.append(((cids[i], t), (cids[j], t)))
    if len(bag) > 4000:
        idx = rng.choice(len(bag), size=4000, replace=False)
        bag = [bag[i] for i in idx]
    for ka, kb in bag:
        unrelated.append(cell_cosine(z_of(ka), z_of(kb)))
    return {"bg_pair": bg_pair, "unrelated": unrelated}


def participation_ratio(evals: np.ndarray) -> float:
    lam = np.clip(evals.astype(np.float64), 0.0, None)
    s = lam.sum()
    if s <= 0:
        return 0.0
    return float((s * s) / np.square(lam).sum())


def load_heads(ckpt: Path, device: str):
    blob = load_checkpoint(ckpt, map_location="cpu")
    adapter = SpatialTemporalAdapter(
        hidden_size=1408, temporal_groups=8, spatial=16, z_channels=32, feature_grid=5,
        projection_channels=128, temporal_blocks=2, temporal_kernel=3, temporal_dilation=1,
        group_norm_groups=8,
    )
    decoder = HeatmapDecoder(z_channels=32)
    adapter.load_state_dict(blob["adapter"])
    decoder.load_state_dict(blob["decoder"])
    adapter.to(device).eval()
    decoder.to(device).eval()
    return adapter, decoder, blob.get("epoch")


@torch.no_grad()
def encode_split(ds, keys, adapter, decoder, device: str, batch=16):
    z_out = {}
    h_out = {}
    for start in range(0, len(keys), batch):
        chunk = keys[start : start + batch]
        toks = np.stack([ds._load_tokens(cid, t) for cid, t in chunk], axis=0)
        x = torch.from_numpy(toks).to(device)
        z = adapter(x)
        h = decoder(z)
        z_np = z.float().cpu().numpy()
        h_np = h.float().cpu().numpy()
        for j, key in enumerate(chunk):
            z_out[key] = z_np[j]
            h_out[key] = h_np[j]
    return z_out, h_out


def threat_dir(hm: np.ndarray, warn: dict) -> str:
    scores = direction_scores_np(hm, warn["row_weights"], warn["left_cols"], warn["center_cols"], warn["right_cols"])
    return max(scores, key=scores.get)


def linear_probe(x_tr, y_tr, x_va, y_va) -> dict:
    n_cls = len(set(y_tr) | set(y_va))
    maj = max(set(y_tr), key=list(y_tr).count)
    base = float(np.mean(np.asarray(y_va) == maj))
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_tr)
    xva = scaler.transform(x_va)
    clf = LogisticRegression(max_iter=400)
    clf.fit(xtr, y_tr)
    pred = clf.predict(xva)
    return {
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_classes": int(n_cls),
        "majority_baseline": base,
        "accuracy": float(accuracy_score(y_va, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_va, pred)),
    }


def diagnose(name, ckpt, ds_val, ds_train, probe_keys_val, mse_keys_val, probe_keys_train, device, warn):
    print(f"== {name} {ckpt} ==", flush=True)
    adapter, decoder, epoch = load_heads(ckpt, device)
    recs = ds_val.records_by_id

    z_val, h_val = encode_split(ds_val, probe_keys_val, adapter, decoder, device)
    z_mse, h_mse = encode_split(ds_val, mse_keys_val, adapter, decoder, device)
    z_tr, _ = encode_split(ds_train, probe_keys_train, adapter, decoder, device)

    raw = pair_cosines(z_val, recs, ds_val, None)
    z_stack = np.stack([z_val[k] for k in probe_keys_val], axis=0)
    mean_z = z_stack.mean(axis=0)
    cen = pair_cosines(z_val, recs, ds_val, mean_z)

    # prediction variance + mean heatmap baseline
    h_pred = np.stack([h_mse[k] for k in mse_keys_val], axis=0)
    h_true = []
    mask = []
    for cid, t in mse_keys_val:
        s = None
        # pull GT from dataset cache
        idx = ds_val.index.index((cid, t)) if (cid, t) in ds_val.index else None
        sample = ds_val[ds_val.index.index((cid, int(t)))]
        h_true.append(np.asarray(sample["h_1s"], dtype=np.float32))
        mask.append(not bool(sample["amb_1s"]))
    h_true = np.stack(h_true, axis=0)
    valid = np.asarray(mask, dtype=bool)
    pred_std = h_pred[valid].std(axis=0)
    train_maps = []
    for cid, t in ds_train.index:
        sample = ds_train[ds_train.index.index((cid, int(t)))]
        if bool(sample["amb_1s"]):
            continue
        train_maps.append(np.asarray(sample["h_1s"], dtype=np.float32))
        if len(train_maps) >= 8000:
            break
    train_mean = np.mean(np.stack(train_maps, axis=0), axis=0)
    model_mse = float(np.mean((h_pred[valid] - h_true[valid]) ** 2))
    base_mse = float(np.mean((train_mean[None] - h_true[valid]) ** 2))
    zero_mse = float(np.mean(h_true[valid] ** 2))

    # Z spread / PCA
    z_flat = z_stack.reshape(z_stack.shape[0], -1)
    z_flat = z_flat - z_flat.mean(axis=0, keepdims=True)
    ch_std = z_stack.std(axis=0).mean(axis=(1, 2))
    # SVD on centered flattened Z
    u, s, vt = np.linalg.svd(z_flat, full_matrices=False)
    evals = (s ** 2) / max(z_flat.shape[0] - 1, 1)
    pr = participation_ratio(evals)
    cum = np.cumsum(evals) / max(evals.sum(), 1e-12)
    rank95 = int(np.searchsorted(cum, 0.95) + 1)

    # linear probes
    def pack(zs, ds, keys):
        x, y_sc, y_dir = [], [], []
        for cid, t in keys:
            sample = ds[ds.index.index((cid, int(t)))]
            x.append(zs[(cid, int(t))].reshape(-1))
            y_sc.append(str(sample["scenario_type"]))
            y_dir.append(threat_dir(np.asarray(sample["h_1s"]), warn))
        return np.stack(x), np.asarray(y_sc), np.asarray(y_dir)

    xtr, ytr_s, ytr_d = pack(z_tr, ds_train, probe_keys_train)
    xva, yva_s, yva_d = pack(z_val, ds_val, probe_keys_val)

    out = {
        "checkpoint": str(ckpt),
        "epoch_in_ckpt": epoch,
        "raw_cosine": {"bg_pair": summarize(raw["bg_pair"]), "unrelated": summarize(raw["unrelated"])},
        "centered_cosine": {"bg_pair": summarize(cen["bg_pair"]), "unrelated": summarize(cen["unrelated"])},
        "prediction": {
            "n_val": int(valid.sum()),
            "hhat_cell_std_mean": float(pred_std.mean()),
            "hhat_cell_std": pred_std.astype(float).tolist(),
            "model_mse": model_mse,
            "mean_heatmap_baseline_mse": base_mse,
            "zero_baseline_mse": zero_mse,
            "mse_improvement_vs_mean": float(base_mse - model_mse),
        },
        "z_spread": {
            "per_channel_std_mean": float(ch_std.mean()),
            "per_channel_std": [float(x) for x in ch_std],
            "pca_participation_ratio": pr,
            "pca_rank_95": rank95,
            "pca_top5_var_frac": [float(x) for x in (evals[:5] / max(evals.sum(), 1e-12))],
            "n": int(z_stack.shape[0]),
            "dim": int(z_flat.shape[1]),
        },
        "linear_probe": {
            "scenario_type": linear_probe(xtr, ytr_s, xva, yva_s),
            "threat_direction": linear_probe(xtr, ytr_d, xva, yva_d),
        },
    }
    print(json.dumps({k: out[k] for k in ("raw_cosine", "centered_cosine", "prediction", "z_spread", "linear_probe")}, indent=2), flush=True)
    return out


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path pairs or raw paths")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_cfg(args)
    cfg.set("heads.mode", "direct_h1s")
    device = args.device
    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    ds_train = TeacherDataset(cfg, records, "train", splits, dummy_backbone=False, use_token_cache=True)
    ds_val = TeacherDataset(cfg, records, "val", splits, dummy_backbone=False, use_token_cache=True)
    ds_val.subsample_clips(0.25, int(cfg.get("project.seed", 1337)))
    probe_anchors = [t for t in PROBE_ANCHORS if t in set(ds_val.anchors)]
    val_clips = ds_val.clip_ids()
    train_clips = ds_train.clip_ids()
    probe_val = [(c, t) for c in val_clips for t in probe_anchors]
    mse_val = [(c, int(t)) for c, t in ds_val.index]
    probe_tr = [(c, t) for c in train_clips for t in probe_anchors]
    print(
        f"val clips {len(val_clips)} probe {len(probe_val)} mse {len(mse_val)} train probe {len(probe_tr)}",
        flush=True,
    )
    warn = cfg.get("warning")
    named = []
    for item in args.checkpoints:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            name, path = Path(item).stem, item
        named.append((name, Path(path)))
    report = {"val_subset_clips": len(val_clips), "probe_anchors": probe_anchors, "runs": {}}
    for name, path in named:
        report["runs"][name] = diagnose(
            name, path, ds_val, ds_train, probe_val, mse_val, probe_tr, device, warn
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
