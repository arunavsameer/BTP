"""PCA on teacher Z_t pair differences (Parts 6–7)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _pair_rows(rows: list[dict], z: np.ndarray, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    by_pair: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if r.get("split") != split:
            continue
        pid = r.get("background_pair_id") or ""
        if not pid:
            continue
        by_pair.setdefault(pid, []).append(i)
    d_ch, d_sp = [], []
    for ids in by_pair.values():
        if len(ids) < 2:
            continue
        a, b = ids[0], ids[1]
        za, zb = z[a], z[b]
        d = (za - zb) / 2.0
        d_ch.append(d.mean(axis=(1, 2)))
        d_sp.append(d.reshape(-1))
    return np.stack(d_ch, axis=0), np.stack(d_sp, axis=0)


def fit_pca(x: np.ndarray, k: int) -> dict:
    x = x.astype(np.float64)
    mu = x.mean(0)
    xc = x - mu
    cov = (xc.T @ xc) / max(len(x) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    k = min(k, evecs.shape[1])
    U = evecs[:, :k]
    explained = (evals[:k].sum() / max(evals.sum(), 1e-12)).item()
    return {"k": k, "explained_variance_ratio": explained, "eigenvalues": evals[:k].tolist(), "U": U, "mu": mu}


def run_part6(cache_dir: Path, out_dir: Path) -> dict:
    rows = [json.loads(l) for l in (cache_dir / "index.jsonl").open()]
    pred = np.load(cache_dir / "teacher_pred.npz", mmap_mode="r")
    z = np.asarray(pred["Z"], dtype=np.float32)
    splits_in = {str(r.get("split", "")) for r in rows}
    pca_split = "train" if "train" in splits_in else "val"
    d_ch, d_sp = _pair_rows(rows, z, pca_split)
    if d_ch.size == 0 and pca_split != "val":
        d_ch, d_sp = _pair_rows(rows, z, "val")
    report = {"pca_split": pca_split, "n_pair_diffs_channel": int(len(d_ch)), "channel_pooled": {}, "spatial_flat": {}}
    bases = {}
    for k in (2, 4, 8, 16):
        if len(d_ch) >= k + 1:
            fit = fit_pca(d_ch, k)
            report["channel_pooled"][str(k)] = {kk: fit[kk] for kk in ("k", "explained_variance_ratio", "eigenvalues")}
            bases[f"channel_k{k}"] = {"U": fit["U"].astype(np.float32), "mu": fit["mu"].astype(np.float32)}
    for k in (4, 8, 16, 32):
        if len(d_sp) >= k + 1:
            fit = fit_pca(d_sp, k)
            report["spatial_flat"][str(k)] = {kk: fit[kk] for kk in ("k", "explained_variance_ratio", "eigenvalues")}
            bases[f"spatial_k{k}"] = {"U": fit["U"].astype(np.float32), "mu": fit["mu"].astype(np.float32)}
    pca_dir = out_dir / "pca_bases"
    pca_dir.mkdir(parents=True, exist_ok=True)
    for name, blob in bases.items():
        np.savez_compressed(pca_dir / f"{name}.npz", **blob)
    (out_dir / "part6_latent_pca_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
