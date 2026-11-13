#!/usr/bin/env python3
"""P4: channel stability ICC on train matched-background pairs (frozen Giant tokens)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.analysis.channel_icc import (  # noqa: E402
    anchor_feature_vec,
    icc_percentiles,
    pair_w_b_icc,
    select_low_icc_channels,
)
from collision_jepa3.data.cache import (  # noqa: E402
    TokenCache,
    frame_cache_path,
    load_cached_frames,
    token_cache_id,
    token_fingerprint,
)
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.preprocessing import preprocess_clip, signature_from_cfg, uint8_to_nchw01  # noqa: E402
from collision_jepa3.data.sampling import FRAME_OFFSETS, input_indices  # noqa: E402
from collision_jepa3.data.video_reader import decode_all_rgb  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.training.pair_signals import DEFAULT_PAIR_GT_TOL  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402
from scripts._common import add_config_args, load_cfg  # noqa: E402


def build_gt_valid_pairs(idx: list[dict], gt: np.ndarray, gt_tol: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Same pairing rules as manual P3 script (background_pair_id + GT match)."""
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(idx):
        groups.setdefault((r["sample_id"], r["outcome"], int(r["frame"])), []).append(i)
    a_list, b_list = [], []
    mismatched = 0
    for ids in groups.values():
        if len(ids) != 2:
            continue
        ia, ib = ids
        ra, rb = idx[ia], idx[ib]
        if ra.get("background_pair_id") != rb.get("background_pair_id"):
            continue
        if ra.get("background_id") == rb.get("background_id"):
            continue
        if np.abs(gt[ia].astype(np.float64) - gt[ib].astype(np.float64)).max() >= gt_tol:
            mismatched += 1
            continue
        a_list.append(ia)
        b_list.append(ib)
    return (
        np.asarray(a_list, dtype=np.int64),
        np.asarray(b_list, dtype=np.int64),
        mismatched,
    )


def _try_load_token(caches: list[TokenCache], clip_id: str, t: int) -> np.ndarray | None:
    for cache in caches:
        try:
            arr, mapping = cache.load_clip(clip_id, mmap=True)
        except FileNotFoundError:
            continue
        if int(t) not in mapping:
            continue
        row = np.asarray(arr[mapping[int(t)]], dtype=np.float32)
        return anchor_feature_vec(row)
    return None


def _encode_features(
    cfg,
    idx: list[dict],
    row_ids: np.ndarray,
    device: str,
    batch: int,
    cache_path: Path | None,
) -> tuple[np.ndarray, dict]:
    """Return ``F[row_id]`` for each unique row in ``row_ids``."""
    row_ids = np.unique(row_ids.astype(np.int64))
    n_rows = len(idx)
    n_c = 1408
    if cache_path and cache_path.exists():
        data = np.load(cache_path)
        F = data["F"].astype(np.float32)
        meta = json.loads(data["meta"].item()) if "meta" in data else {}
        if F.shape[0] == n_rows and F.shape[1] == n_c:
            return F, meta
    F = np.full((n_rows, n_c), np.nan, dtype=np.float32)
    keys: list[tuple[str, int, int]] = []
    for rid in row_ids:
        r = idx[int(rid)]
        keys.append((r["clip_id"], int(r["frame"]), int(rid)))

    fp = token_fingerprint(cfg, dummy=False)
    fp_gr = token_cache_id(cfg, dummy=False)
    caches = [
        TokenCache(cfg.path("paths.cache_root"), fp_gr),
        TokenCache(cfg.path("paths.cache_root"), fp),
    ]
    stats = {"cache_hits": 0, "encoded": 0, "token_cache_fps": [fp_gr, fp]}
    need: list[tuple[str, int, int]] = []
    for clip_id, t, rid in keys:
        vec = _try_load_token(caches, clip_id, t)
        if vec is not None:
            F[rid] = vec.astype(np.float32)
            stats["cache_hits"] += 1
        else:
            need.append((clip_id, t, rid))

    encoder = None
    offsets = list(cfg.get("sampling.frame_offsets", FRAME_OFFSETS))
    sig = signature_from_cfg(cfg)
    records = {r.clip_id: r for r in load_manifest(cfg.path("paths.manifest")) if r.valid}
    frame_mem: dict[str, np.ndarray] = {}
    by_clip: dict[str, list[tuple[int, int]]] = {}
    for clip_id, t, rid in need:
        by_clip.setdefault(clip_id, []).append((t, rid))

    import torch

    def _frames_u8(clip_id: str) -> np.ndarray:
        if clip_id in frame_mem:
            return frame_mem[clip_id]
        npy = frame_cache_path(cfg.path("paths.cache_root"), clip_id)
        if npy.exists():
            frame_mem[clip_id] = load_cached_frames(cfg.path("paths.cache_root"), clip_id, mmap=True)
        else:
            rec = records.get(clip_id)
            if rec is None:
                raise KeyError(clip_id)
            raw, _ = decode_all_rgb(rec.video_path)
            frame_mem[clip_id] = preprocess_clip(raw, sig)
        return frame_mem[clip_id]

    for ci, (clip_id, items) in enumerate(by_clip.items(), start=1):
        frames = _frames_u8(clip_id)
        for start in range(0, len(items), batch):
            chunk = items[start : start + batch]
            windows = []
            rids = []
            for t, rid in chunk:
                idx_f = input_indices(t, offsets)
                window = np.stack([np.asarray(frames[k]) for k in idx_f], axis=0)
                windows.append(uint8_to_nchw01(window))
                rids.append(rid)
            x = torch.from_numpy(np.stack(windows, axis=0)).to(device)
            if encoder is None:
                print(f"loading Giant encoder on {device} ({len(need)} windows to encode)", flush=True)
                encoder = CollisionTeacher3(cfg, dummy_backbone=False).to(device).eval()
            with torch.no_grad():
                if str(device).startswith("cuda"):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        tok = encoder.encode_tokens(x)
                else:
                    tok = encoder.encode_tokens(x)
            tok = tok.float().cpu().numpy()
            for j, rid in enumerate(rids):
                F[rid] = anchor_feature_vec(tok[j]).astype(np.float32)
            stats["encoded"] += len(rids)
        if ci % 50 == 0:
            print(f"  encoded clips {ci}/{len(by_clip)}", flush=True)

    if np.isnan(F[row_ids]).any():
        bad = int(np.isnan(F[row_ids]).sum())
        raise RuntimeError(f"{bad} pair rows still missing features after encode")

    meta = {
        **stats,
        "feature_pool": "mean over [n_tokens, C] at anchor (tokens shape typically [2048, 1408])",
        "n_pair_rows": int(len(row_ids)),
    }
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, F=F.astype(np.float16), meta=np.array(json.dumps(meta)))
        print(f"wrote feature cache {cache_path}", flush=True)
    return F, meta


def _save_histogram(path: Path, values: np.ndarray, title: str, xlabel: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=80, color="#4472C4", edgecolor="white", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--dump", default="reports/dense_dump_train")
    parser.add_argument("--out-dir", default="reports/p4_channel_icc")
    parser.add_argument("--gt-tol", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--encode-batch", type=int, default=8)
    parser.add_argument("--feature-cache", default="reports/p4_channel_icc/row_features.npz")
    parser.add_argument("--no-encode", action="store_true", help="require full feature cache hit")
    args = parser.parse_args()
    cfg = load_cfg(args)
    device = args.device or resolve_device(cfg.get("project.device", None))

    dump = Path(args.dump)
    if not dump.is_absolute():
        dump = ROOT / dump
    idx = [json.loads(l) for l in (dump / "index.jsonl").open()]
    gt = np.load(dump / "gt_heatmaps.npz")["gt_1s"]

    a, b, n_mismatch = build_gt_valid_pairs(idx, gt, args.gt_tol)
    print(f"pairs={len(a)} gt_mismatched_excluded={n_mismatch}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = np.unique(np.concatenate([a, b]))
    cache_path = Path(args.feature_cache)
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path

    if args.no_encode and not cache_path.exists():
        raise SystemExit(f"--no-encode requires {cache_path}")

    F, feat_meta = _encode_features(cfg, idx, pair_rows, device, args.encode_batch, cache_path)

    fa = F[a]
    fb = F[b]
    w, b_var, icc = pair_w_b_icc(fa, fb)
    selection = select_low_icc_channels(icc, w)
    pct = icc_percentiles(icc)

    # Verify token layout from one cached clip
    fp_gr = token_cache_id(cfg, dummy=False)
    sample_meta_path = cfg.path("paths.cache_root") / "tokens" / fp_gr
    sample_clip = idx[int(a[0])]["clip_id"].replace("/", "__")
    layout_note = {}
    meta_file = sample_meta_path / sample_clip / "meta.json"
    if meta_file.exists():
        layout_note = json.loads(meta_file.read_text(encoding="utf-8"))

    report = {
        "dump": str(dump),
        "n_pairs": int(len(a)),
        "n_gt_mismatch_excluded": int(n_mismatch),
        "gt_tol": float(args.gt_tol),
        "n_channels": int(icc.shape[0]),
        "token_cache_root": str(cfg.path("paths.cache_root") / "tokens"),
        "token_cache_id_group_random": fp_gr,
        "example_clip_meta": layout_note,
        "feature_extraction": feat_meta,
        "W_percentiles": {f"p{int(p)}": float(np.percentile(w, p)) for p in (10, 25, 50, 75, 90)},
        "ICC_percentiles": pct,
        "ICC_all_channels": icc_percentiles(icc, (10, 25, 50, 75, 90)),
        "selection": selection,
    }
    np.savez_compressed(out_dir / "channel_icc.npz", W=w.astype(np.float64), B=b_var.astype(np.float64), ICC=icc.astype(np.float64))
    (out_dir / "p4_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== P4 channel ICC (train, GT-matched pairs) ===")
    print("eligible channels (W >= median W):", selection["n_eligible"], "/", icc.shape[0])
    print("selected lowest-ICC 20% among eligible:", selection.get("n_selected", 0))
    print("ICC percentiles:", ", ".join(f"{k}={v:.4f}" for k, v in pct.items()))
    print("W median:", float(np.median(w)))

    _save_histogram(out_dir / "icc_histogram.png", icc, "ICC per channel (train pairs)", "ICC")
    _save_histogram(out_dir / "w_histogram.png", w, "Background variation W (train pairs)", "W")
    print(f"wrote {out_dir / 'p4_report.json'}")


if __name__ == "__main__":
    main()
