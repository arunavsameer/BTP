#!/usr/bin/env python3
"""Read-only channel-wise affine calibration audit of epoch-38 student F vs Pair-CF F_dec."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
TEMP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp")
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402
from collision_jepa3.evaluation.metrics import CAUTION, SAFE, STOP  # noqa: E402
from collision_jepa3.student_research.cache_access import ParentTargetCache  # noqa: E402
from collision_jepa3.student_research.fdec_student import build_parent_decoder, decoder_state_hash  # noqa: E402
from collision_jepa3.utils.device import resolve_device_safe  # noqa: E402

spec = importlib.util.spec_from_file_location("s75", ROOT / "scripts/75_train_student_lat_paircf.py")
s75 = importlib.util.module_from_spec(spec)
sys.modules["s75"] = s75
spec.loader.exec_module(s75)

EXP = TEMP / "student_lat_paircf"
OUT = EXP / "fdec_affine_audit"
RGB_ROOT = TEMP / "shared_rgb_cache_256_v1"
PARENT_ROOT = TEMP / "parent_target_cache_v2"
EXPECTED_HASH = "77f78c8714667617"
CKPT = EXP / "checkpoint_epoch_38.pt"
C = 128
HW = 16 * 16


def peak_class(h, stop_thr=0.60, caution_thr=0.25):
    p = float(np.max(h))
    if p >= stop_thr:
        return STOP
    if p >= caution_thr:
        return CAUTION
    return SAFE


def summarize(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).ravel()
    qs = np.quantile(x, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(qs[0]),
        "p10": float(qs[1]),
        "p25": float(qs[2]),
        "p50": float(qs[3]),
        "p75": float(qs[4]),
        "p90": float(qs[5]),
        "max": float(qs[6]),
    }


class ChanAcc:
    """Online per-channel moments over (N,C,H,W) without storing maps."""

    def __init__(self):
        self.n = 0
        self.sum_s = np.zeros(C, np.float64)
        self.sum_t = np.zeros(C, np.float64)
        self.sum_s2 = np.zeros(C, np.float64)
        self.sum_t2 = np.zeros(C, np.float64)
        self.sum_st = np.zeros(C, np.float64)
        self.sum_abs_s = np.zeros(C, np.float64)
        self.sum_abs_t = np.zeros(C, np.float64)
        self.norms_s = []
        self.norms_t = []
        self.cos = []
        self.l1 = []

    def update(self, fs: torch.Tensor, ft: torch.Tensor):
        # fs, ft: [B,C,H,W] float32
        b = fs.shape[0]
        s = fs.float().reshape(b, C, HW)
        t = ft.float().reshape(b, C, HW)
        self.n += b * HW
        self.sum_s += s.sum(dim=(0, 2)).double().cpu().numpy()
        self.sum_t += t.sum(dim=(0, 2)).double().cpu().numpy()
        self.sum_s2 += (s * s).sum(dim=(0, 2)).double().cpu().numpy()
        self.sum_t2 += (t * t).sum(dim=(0, 2)).double().cpu().numpy()
        self.sum_st += (s * t).sum(dim=(0, 2)).double().cpu().numpy()
        self.sum_abs_s += s.abs().sum(dim=(0, 2)).double().cpu().numpy()
        self.sum_abs_t += t.abs().sum(dim=(0, 2)).double().cpu().numpy()
        ns = torch.linalg.vector_norm(s.reshape(b, -1), dim=1).cpu().numpy()
        nt = torch.linalg.vector_norm(t.reshape(b, -1), dim=1).cpu().numpy()
        self.norms_s.append(ns)
        self.norms_t.append(nt)
        self.cos.append(
            torch.nn.functional.cosine_similarity(s.reshape(b, -1), t.reshape(b, -1), dim=1).cpu().numpy()
        )
        self.l1.append((s - t).abs().mean(dim=(1, 2)).cpu().numpy())

    def channel_table(self) -> dict:
        n = max(self.n, 1)
        mu_s = self.sum_s / n
        mu_t = self.sum_t / n
        var_s = np.maximum(self.sum_s2 / n - mu_s**2, 0)
        var_t = np.maximum(self.sum_t2 / n - mu_t**2, 0)
        std_s = np.sqrt(var_s)
        std_t = np.sqrt(var_t)
        cov = self.sum_st / n - mu_s * mu_t
        rms_s = np.sqrt(self.sum_s2 / n)
        rms_t = np.sqrt(self.sum_t2 / n)
        a = np.where(var_s > 1e-12, cov / var_s, 0.0)
        b = mu_t - a * mu_s
        corr = np.where((std_s > 1e-12) & (std_t > 1e-12), cov / (std_s * std_t), 0.0)
        return {
            "n_locations": int(n),
            "mean_s": mu_s,
            "mean_t": mu_t,
            "std_s": std_s,
            "std_t": std_t,
            "rms_s": rms_s,
            "rms_t": rms_t,
            "mae": (self.sum_abs_s / n),  # placeholder overwritten
            "a": a,
            "b": b,
            "corr": corr,
            "mean_abs_s": self.sum_abs_s / n,
            "mean_abs_t": self.sum_abs_t / n,
        }

    def pack(self) -> dict:
        ch = self.channel_table()
        ns = np.concatenate(self.norms_s)
        nt = np.concatenate(self.norms_t)
        return {
            "n_windows": int(ns.size),
            "n_locations": ch["n_locations"],
            "cosine": summarize(np.concatenate(self.cos)),
            "l1": summarize(np.concatenate(self.l1)),
            "norm_student": summarize(ns),
            "norm_teacher": summarize(nt),
            "norm_ratio_s_over_t": summarize(ns / np.maximum(nt, 1e-8)),
            "channel_mean_student": summarize(ch["mean_s"]),
            "channel_mean_teacher": summarize(ch["mean_t"]),
            "channel_std_student": summarize(ch["std_s"]),
            "channel_std_teacher": summarize(ch["std_t"]),
            "channel_rms_student": summarize(ch["rms_s"]),
            "channel_rms_teacher": summarize(ch["rms_t"]),
            "channel_corr": summarize(ch["corr"]),
            "affine_a": summarize(ch["a"]),
            "affine_b": summarize(ch["b"]),
            "n_channels_corr_lt_0.2": int((ch["corr"] < 0.2).sum()),
            "n_channels_corr_gt_0.8": int((ch["corr"] > 0.8).sum()),
            "n_channels_a_lt_0.5": int((ch["a"] < 0.5).sum()),
            "n_channels_a_gt_1.5": int((ch["a"] > 1.5).sum()),
        }, ch


def metrics_from(hs, hg, pref):
    n = len(hs)
    gc = np.array([peak_class(hg[i]) for i in range(n)])
    sc = np.array([peak_class(hs[i]) for i in range(n)])
    pref = np.asarray(pref, dtype=bool)
    peaks = np.array([float(np.max(hs[i])) for i in range(n)])

    def rec(g, p):
        d = int((gc == g).sum())
        return float(np.logical_and(gc == g, sc == p).sum() / d) if d else float("nan")

    stop_pred = sc == STOP
    tp = int(np.logical_and(gc == STOP, stop_pred).sum())
    fp = int(np.logical_and(gc != STOP, stop_pred).sum())
    out = {
        "n": n,
        "safe_recall": rec(SAFE, SAFE),
        "threat_recall": rec(CAUTION, CAUTION),
        "stop_recall": rec(STOP, STOP),
        "stop_precision": tp / max(1, tp + fp),
        "threat_precision": int(np.logical_and(gc == CAUTION, sc == CAUTION).sum())
        / max(1, int((sc == CAUTION).sum())),
        "safe_precision": int(np.logical_and(gc == SAFE, sc == SAFE).sum()) / max(1, int((sc == SAFE).sum())),
        "pre_fork_stop_recall": float(
            np.logical_and(np.logical_and(pref, gc == STOP), sc == STOP).sum()
            / max(1, int(np.logical_and(pref, gc == STOP).sum()))
        ),
        "post_fork_stop_recall": float(
            np.logical_and(np.logical_and(~pref, gc == STOP), sc == STOP).sum()
            / max(1, int(np.logical_and(~pref, gc == STOP).sum()))
        ),
        "n_gt_stop_prefork": int(np.logical_and(pref, gc == STOP).sum()),
        "n_gt_stop_postfork": int(np.logical_and(~pref, gc == STOP).sum()),
        "peak_all": summarize(peaks),
        "peak_gt_safe": summarize(peaks[gc == SAFE]),
        "peak_gt_threat": summarize(peaks[gc == CAUTION]),
        "peak_gt_stop": summarize(peaks[gc == STOP]),
        "frac_peak_ge_0.25": float(np.mean(peaks >= 0.25)),
        "frac_peak_ge_0.60": float(np.mean(peaks >= 0.60)),
        "frac_gt_stop_peak_ge_0.25": float(np.mean(peaks[gc == STOP] >= 0.25)) if (gc == STOP).any() else None,
        "frac_gt_stop_peak_lt_0.25": float(np.mean(peaks[gc == STOP] < 0.25)) if (gc == STOP).any() else None,
    }
    return out


@torch.no_grad()
def run_split(model, loader, device, amp, a=None, b=None, collect_stats=False):
    acc = ChanAcc() if collect_stats else None
    hs_raw, hs_cal, hg, pref, ht = [], [], [], [], []
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp else torch.nullcontext()
    a_t = b_t = None
    if a is not None:
        a_t = torch.tensor(a, dtype=torch.float32, device=device).view(1, C, 1, 1)
        b_t = torch.tensor(b, dtype=torch.float32, device=device).view(1, C, 1, 1)
    for batch in loader:
        clip = batch["clip"].to(device, non_blocking=True)
        ft = batch["F_teacher"].to(device, non_blocking=True)
        with ctx:
            fs = model.encode_context(clip)
            h0, _ = model.decoder.forward_with_logits(fs)
            if a_t is not None:
                fcal = a_t * fs.float() + b_t
                hc, _ = model.decoder.forward_with_logits(fcal)
            else:
                hc = None
        if acc is not None:
            acc.update(fs, ft)
        hs_raw.append(h0.float().cpu().numpy())
        hg.append(batch["gt_1s"].numpy())
        ht.append(batch["H_teacher"].numpy())
        pref.append(batch["pre_fork"].numpy())
        if hc is not None:
            hs_cal.append(hc.float().cpu().numpy())
    pack = {
        "H_raw": np.concatenate(hs_raw),
        "H_gt": np.concatenate(hg),
        "H_teacher": np.concatenate(ht),
        "pre_fork": np.concatenate(pref),
        "H_cal": np.concatenate(hs_cal) if hs_cal else None,
        "stats": None,
        "channels": None,
    }
    if acc is not None:
        pack["stats"], pack["channels"] = acc.pack()
    return pack


def ch_summary_json(ch):
    keys = ["mean_s", "mean_t", "std_s", "std_t", "rms_s", "rms_t", "a", "b", "corr"]
    return {k: [float(x) for x in ch[k]] for k in keys}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--batch", type=int, default=8)
    add_config_args(parser)
    args = parser.parse_args()
    cfg = load_cfg(args)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (PARENT_ROOT / "index.jsonl").open()]
    gt = np.load(PARENT_ROOT / "gt_1s.npy").astype(np.float32)
    train_ids = [i for i, r in enumerate(rows) if r["split"] == "train"]
    sel_ids = [i for i, r in enumerate(rows) if r["split"] == "selection"]
    assert len(train_ids) == 22680 and len(sel_ids) == 5120
    parent = ParentTargetCache(PARENT_ROOT)
    rgb = s75.RGBCache256(RGB_ROOT)
    train_loader = DataLoader(
        s75.WinDS(rows, train_ids, gt, rgb, parent),
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        collate_fn=s75.collate,
        pin_memory=True,
    )
    sel_loader = DataLoader(
        s75.WinDS(rows, sel_ids, gt, rgb, parent),
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        collate_fn=s75.collate,
        pin_memory=True,
    )
    device = resolve_device_safe(str(args.device))
    decoder = build_parent_decoder(cfg, device="cpu")
    dec_h = decoder_state_hash(decoder)
    if dec_h != EXPECTED_HASH:
        raise SystemExit(f"hash {dec_h}")
    model = s75.StudentLatPairCF(decoder).to(device).eval()
    blob = torch.load(CKPT, map_location=device, weights_only=False)
    model.encoder.load_state_dict(blob["encoder"])
    if blob.get("predictor"):
        model.predictor.load_state_dict(blob["predictor"])
    amp = str(device).startswith("cuda")
    print(f"device={device} epoch=38 train pass", flush=True)
    tr = run_split(model, train_loader, device, amp, collect_stats=True)
    a, b = tr["channels"]["a"], tr["channels"]["b"]
    np.savez(OUT / "affine_train.npz", a=a.astype(np.float32), b=b.astype(np.float32), corr=tr["channels"]["corr"])
    print("selection pass", flush=True)
    sel = run_split(model, sel_loader, device, amp, a=a, b=b, collect_stats=True)
    raw_m = metrics_from(sel["H_raw"], sel["H_gt"], sel["pre_fork"])
    cal_m = metrics_from(sel["H_cal"], sel["H_gt"], sel["pre_fork"])
    par_m = metrics_from(sel["H_teacher"], sel["H_gt"], sel["pre_fork"])
    # residual after affine on selection
    ch = sel["channels"]
    report = {
        "checkpoint": str(CKPT),
        "decoder_hash": dec_h,
        "affine_fit": "ordinary least squares per channel on all train spatial locations; F'_c = a_c F_c + b_c",
        "train_feature_match": tr["stats"],
        "selection_feature_match_uncalibrated": sel["stats"],
        "affine_from_train": {
            "a": summarize(a),
            "b": summarize(b),
            "corr_train": summarize(tr["channels"]["corr"]),
        },
        "selection_original": raw_m,
        "selection_calibrated": cal_m,
        "selection_parent": par_m,
        "delta_cal_minus_raw": {k: (cal_m[k] - raw_m[k]) for k in (
            "safe_recall", "threat_recall", "stop_recall", "stop_precision",
            "pre_fork_stop_recall", "post_fork_stop_recall",
        )},
        "n_test_accessed": 0,
    }
    (OUT / "channels_train.json").write_text(json.dumps(ch_summary_json(tr["channels"]), indent=2), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = []
    md.append("# Epoch-38 F_dec affine calibration audit\n")
    md.append("Train-only OLS `F'_c = a_c F_c + b_c`. Frozen Pair-CF decoder. Selection eval. No test.\n")
    md.append("## Feature match (student F vs cached F_dec)\n")
    md.append("| split | cosine mean | L1 | ||F_s|| mean | ||F_t|| mean | corr mean | a mean |\n|---|---:|---:|---:|---:|---:|---:|")
    md.append(
        f"| train | {tr['stats']['cosine']['mean']:.4f} | {tr['stats']['l1']['mean']:.4f} | {tr['stats']['norm_student']['mean']:.1f} | {tr['stats']['norm_teacher']['mean']:.1f} | {tr['stats']['channel_corr']['mean']:.3f} | {tr['stats']['affine_a']['mean']:.3f} |"
    )
    md.append(
        f"| selection | {sel['stats']['cosine']['mean']:.4f} | {sel['stats']['l1']['mean']:.4f} | {sel['stats']['norm_student']['mean']:.1f} | {sel['stats']['norm_teacher']['mean']:.1f} | {sel['stats']['channel_corr']['mean']:.3f} | — |"
    )
    md.append("\nPer-channel (train):")
    md.append(f"- a: mean {float(a.mean()):.3f} p10 {np.quantile(a,0.1):.3f} p90 {np.quantile(a,0.9):.3f}")
    md.append(f"- b: mean {float(b.mean()):.4f} p10 {np.quantile(b,0.1):.4f} p90 {np.quantile(b,0.9):.4f}")
    md.append(f"- corr: mean {float(tr['channels']['corr'].mean()):.3f}; <0.2: {int((tr['channels']['corr']<0.2).sum())}/128; >0.8: {int((tr['channels']['corr']>0.8).sum())}/128")
    md.append("\n## Selection operating point 0.25/0.60\n")
    md.append("| | SAFE rec | THREAT rec | STOP rec | STOP prec | pre-STOP | post-STOP |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, m in (("parent", par_m), ("epoch38 original", raw_m), ("epoch38 calibrated", cal_m)):
        md.append(
            f"| {name} | {m['safe_recall']:.3f} | {m['threat_recall']:.3f} | {m['stop_recall']:.3f} | {m['stop_precision']:.3f} | {m['pre_fork_stop_recall']:.3f} | {m['post_fork_stop_recall']:.3f} |"
        )
    md.append("\nPeak distributions (heatmap max):\n")
    md.append("| model | GT-SAFE mean/p50 | GT-THREAT mean/p50 | GT-STOP mean/p50 | GT-STOP frac<0.25 |")
    md.append("|---|---|---|---|---:|")
    for name, m in (("original", raw_m), ("calibrated", cal_m), ("parent", par_m)):
        md.append(
            f"| {name} | {m['peak_gt_safe']['mean']:.3f}/{m['peak_gt_safe']['p50']:.3f} | {m['peak_gt_threat']['mean']:.3f}/{m['peak_gt_threat']['p50']:.3f} | {m['peak_gt_stop']['mean']:.3f}/{m['peak_gt_stop']['p50']:.3f} | {m['frac_gt_stop_peak_lt_0.25']:.3f} |"
        )
    d = report["delta_cal_minus_raw"]
    md.append("\n## Recommendation\n")
    stop_d = d["stop_recall"]
    safe_d = d["safe_recall"]
    md.append(
        f"Calibration deltas on selection: STOP {stop_d:+.3f}, SAFE {safe_d:+.3f}, THREAT {d['threat_recall']:+.3f}, post-STOP {d['post_fork_stop_recall']:+.3f}."
    )
    (OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (TEMP / "reports/student_lat_paircf_fdec_affine_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"train_cos": tr["stats"]["cosine"], "sel_raw": raw_m, "sel_cal": cal_m, "delta": d}, indent=2, default=str), flush=True)
    print("WROTE", OUT / "report.md", flush=True)


if __name__ == "__main__":
    main()
