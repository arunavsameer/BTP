"""Data integrity checks for F_dec pilot."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from collision_jepa3.student_research.parent import load_pair_cf_parent, resolve_parent_ckpt


def _pair_indices(rows: list[dict], row_ids: list[int]) -> list[tuple[int, int, tuple]]:
    by_key: dict[tuple, list[int]] = defaultdict(list)
    for rid in row_ids:
        r = rows[rid]
        pid = str(r.get("background_pair_id") or "")
        if not pid:
            continue
        key = (pid, int(r["frame"]))
        by_key[key].append(int(rid))
    out = []
    for key, ids in by_key.items():
        if len(ids) == 2:
            out.append((ids[0], ids[1], key))
    return out


def compute_fdec_norm_stats(pred, train_row_ids: list[int]) -> dict[str, Any]:
    f = np.asarray(pred["F_dec"][train_row_ids], dtype=np.float32)
    # per-channel mean/std over N×H×W
    mu = f.mean(axis=(0, 2, 3))
    sigma = f.std(axis=(0, 2, 3))
    sigma = np.maximum(sigma, 1e-6)
    return {"mu": mu.astype(np.float32), "sigma": sigma.astype(np.float32)}


@torch.no_grad()
def decoder_replay_sample(cfg, pred, row_ids: list[int], device: str, n: int = 64) -> dict[str, float]:
    model, _ = load_pair_cf_parent(cfg, resolve_parent_ckpt(cfg, None), device=device)
    idx = row_ids[:n]
    fdec = pred["F_dec"]
    href = pred["H"]
    mses, corrs = [], []
    for start in range(0, len(idx), 16):
        sl = idx[start : start + 16]
        feat = torch.from_numpy(np.asarray(fdec[sl], dtype=np.float32)).to(device)
        h_rep, _ = model.decoder.forward_with_logits(feat)
        ht = torch.from_numpy(np.asarray(href[sl], dtype=np.float32)).to(device)
        mses.append(float(F.mse_loss(h_rep, ht).item()))
        for j in range(len(sl)):
            a = h_rep[j].reshape(-1).float().cpu().numpy()
            b = ht[j].reshape(-1).float().cpu().numpy()
            if np.std(a) > 1e-8:
                corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "mse_mean": float(np.mean(mses)),
        "correlation_mean": float(np.mean(corrs)) if corrs else float("nan"),
        "n_samples": len(idx),
    }


def build_integrity_report(
    cfg,
    cache_dir: Path,
    train_ids: list[int],
    dev_ids: list[int],
    device: str = "cpu",
) -> dict[str, Any]:
    rows = [json.loads(l) for l in (cache_dir / "index.jsonl").open()]
    pred = np.load(cache_dir / "teacher_pred.npz", mmap_mode="r")
    gt = np.load(cache_dir / "gt_heatmaps.npz", mmap_mode="r")
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    train_set, dev_set = set(train_ids), set(dev_ids)
    fdec_miss_tr = int(np.isnan(pred["F_dec"][train_ids]).any(axis=(1, 2, 3)).sum())
    fdec_miss_dev = int(np.isnan(pred["F_dec"][dev_ids]).any(axis=(1, 2, 3)).sum())
    def _audit_pairs(row_ids: list[int], *, split: str) -> tuple[list[tuple], list[dict], int]:
        pairs = _pair_indices(rows, row_ids)
        verified: list[tuple] = []
        mismatch: list[dict] = []
        for a, b, key in pairs:
            gta = np.asarray(gt["gt_1s"][a], dtype=np.float32)
            gtb = np.asarray(gt["gt_1s"][b], dtype=np.float32)
            same = float(np.max(np.abs(gta - gtb))) < 0.02
            if same:
                verified.append(key)
            else:
                mismatch.append(
                    {"split": split, "key": key, "row_a": a, "row_b": b, "max_abs_diff": float(np.max(np.abs(gta - gtb)))}
                )
        return verified, mismatch, len(pairs)

    verified_train, mismatch_train, pairs_tr_n = _audit_pairs(train_ids, split="train")
    verified_dev, mismatch_dev, pairs_dev_n = _audit_pairs(dev_ids, split="dev")
    verified_pair_keys = verified_train
    mismatch_pairs = mismatch_train + mismatch_dev
    replay = decoder_replay_sample(cfg, pred, train_ids, device, n=128)
    fdec_norm = compute_fdec_norm_stats(pred, train_ids)
    h = hashlib.sha256(json.dumps(train_ids).encode()).hexdigest()[:12]
    return {
        "train_rows": len(train_ids),
        "dev_rows": len(dev_ids),
        "fdec_missing_train": fdec_miss_tr,
        "fdec_missing_dev": fdec_miss_dev,
        "fdec_shape": list(pred["F_dec"].shape[1:]),
        "teacher_checkpoint": meta.get("checkpoint"),
        "pair_train_total": pairs_tr_n,
        "pair_dev_total": pairs_dev_n,
        "pair_gt_verified_train": len(verified_train),
        "pair_gt_verified_dev": len(verified_dev),
        "pair_gt_verified": len(verified_pair_keys),
        "pair_gt_mismatch": len(mismatch_pairs),
        "pair_gt_mismatch_train": len(mismatch_train),
        "pair_gt_mismatch_dev": len(mismatch_dev),
        "mismatch_pairs": mismatch_pairs,
        "verified_pair_keys": [list(k) for k in verified_pair_keys],
        "verified_pair_keys_dev": [list(k) for k in verified_dev],
        "pair_eval_note": "Dev metrics must use verified_pair_keys_dev; train keys on dev yield n_pairs_verified_evaluated=0.",
        "decoder_replay_pilot": replay,
        "fdec_norm_stats": {"mu_shape": list(fdec_norm["mu"].shape), "sigma_shape": list(fdec_norm["sigma"].shape)},
        "manifest_hash_prefix": h,
        "ok_for_s1": fdec_miss_tr == 0 and fdec_miss_dev == 0,
    }, fdec_norm
