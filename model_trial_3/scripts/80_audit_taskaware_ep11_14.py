#!/usr/bin/env python3
"""Read-only selection+train-batch audit of task-aware student epochs 11–14. No test. No training."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
TEMP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp")
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402
from collision_jepa3.evaluation.metrics import CAUTION, SAFE, STOP  # noqa: E402
from collision_jepa3.student_research.cache_access import ParentTargetCache  # noqa: E402
from collision_jepa3.student_research.fdec_student import build_parent_decoder, decoder_state_hash  # noqa: E402
from collision_jepa3.utils.device import resolve_device_safe  # noqa: E402

spec79 = importlib.util.spec_from_file_location("s79", ROOT / "scripts/79_student_lat_paircf_taskaware.py")
s79 = importlib.util.module_from_spec(spec79)
sys.modules["s79"] = s79
spec79.loader.exec_module(s79)
s75 = s79.s75

EXP = TEMP / "student_lat_paircf_taskaware"
OUT = EXP / "audit_ep11_14"
RGB_ROOT = TEMP / "shared_rgb_cache_256_v1"
PARENT_ROOT = TEMP / "parent_target_cache_v2"
EXPECTED_HASH = "77f78c8714667617"
STATS = TEMP / "student_lat_paircf" / "fdec_channel_stats.npz"
EPOCHS = (11, 12, 13, 14)
LOSS_NAMES = ("feat", "cos", "rel", "mgd", "h", "peak", "focal", "rank")
CLS = {SAFE: "SAFE", CAUTION: "THREAT", STOP: "STOP"}


def peak_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return {"n": 0}
    qs = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(qs[2]),
        "std": float(x.std()),
        "p10": float(qs[0]),
        "p25": float(qs[1]),
        "p75": float(qs[3]),
        "p90": float(qs[4]),
        "frac_lt_0.25": float(np.mean(x < 0.25)),
        "frac_0.25_0.60": float(np.mean((x >= 0.25) & (x < 0.60))),
        "frac_ge_0.60": float(np.mean(x >= 0.60)),
    }


def prf(cm, k):
    tp = cm[k, k]
    fp = cm[:, k].sum() - tp
    fn = cm[k, :].sum() - tp
    rec = tp / max(1, cm[k].sum())
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * rec / max(1e-12, prec + rec) if (prec + rec) else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(cm[k].sum())}


def flatten_grads(params):
    parts = []
    for p in params:
        if p.grad is None:
            parts.append(torch.zeros(p.numel(), device=p.device))
        else:
            parts.append(p.grad.detach().float().reshape(-1))
    return torch.cat(parts)


@torch.no_grad()
def collect_sel(model, loader, device):
    model.eval()
    recs = []
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else torch.nullcontext()
    d_f, d_p, abs_b = [], [], []
    for batch in loader:
        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with ctx:
            out = model(b["clip"])
        h = out["Hhat_1s"].float().cpu().numpy()
        gt = b["gt_1s"].float().cpu().numpy()
        ht = b["H_teacher"].float().cpu().numpy()
        aux = F.softmax(out["aux_logits"].float(), dim=1).cpu().numpy()
        aux_y = aux.argmax(1)
        f = out["F"].float()
        ctxf = out["F_ctx"].float()
        d_f.append((f - ctxf).abs().mean(dim=(1, 2, 3)).cpu().numpy())
        W = model.proj.conv.weight.detach().float()
        I = torch.eye(128, device=W.device).view(128, 128, 1, 1)
        d_p.append(float((W - I).pow(2).sum().sqrt()))
        abs_b.append(float(model.proj.conv.bias.detach().abs().mean()))
        pref = b["pre_fork"].cpu().numpy()
        frames = b["frame"].cpu().numpy()
        for i in range(h.shape[0]):
            recs.append(
                {
                    "clip_id": b["clip_id"][i],
                    "sample_id": b["sample_id"][i],
                    "anchor_frame": int(frames[i]),
                    "pre_fork": bool(pref[i]),
                    "H": h[i],
                    "gt": gt[i],
                    "Hp": ht[i],
                    "aux_p": aux[i],
                    "aux_cls": int(aux_y[i]),
                    "hm_cls": int(s75.peak_class(h[i])),
                    "gt_cls": int(s75.peak_class(gt[i])),
                    "par_cls": int(s75.peak_class(ht[i])),
                    "st_peak": float(h[i].max()),
                    "gt_peak": float(gt[i].max()),
                    "par_peak": float(ht[i].max()),
                    "gt_argmax": tuple(int(x) for x in np.unravel_index(int(np.argmax(gt[i])), (5, 5))),
                }
            )
    return recs, float(np.mean(d_f[0] if False else np.concatenate(d_f))), float(np.mean(d_p)), float(np.mean(abs_b))


def analyze_peaks(recs):
    out = {}
    for name, pred_key, mask_fn in (
        ("student", "st_peak", lambda r, g: r["gt_cls"] == g),
        ("gt", "gt_peak", lambda r, g: r["gt_cls"] == g),
        ("parent", "par_peak", lambda r, g: r["gt_cls"] == g),
    ):
        out[name] = {}
        for g, gn in ((SAFE, "GT-SAFE"), (CAUTION, "GT-THREAT"), (STOP, "GT-STOP")):
            vals = np.array([r[pred_key] for r in recs if mask_fn(r, g)])
            out[name][gn] = peak_stats(vals)
    stop = [r for r in recs if r["gt_cls"] == STOP]
    out["student_STOP_prefork"] = peak_stats(np.array([r["st_peak"] for r in stop if r["pre_fork"]]))
    out["student_STOP_postfork"] = peak_stats(np.array([r["st_peak"] for r in stop if not r["pre_fork"]]))
    out["gt_STOP_prefork"] = peak_stats(np.array([r["gt_peak"] for r in stop if r["pre_fork"]]))
    out["gt_STOP_postfork"] = peak_stats(np.array([r["gt_peak"] for r in stop if not r["pre_fork"]]))
    return out


def analyze_aux(recs):
    n = len(recs)
    cm_aux = np.zeros((3, 3), np.int64)
    cm_hm = np.zeros((3, 3), np.int64)
    agree = 0
    mean_p = {g: [] for g in range(3)}
    disagree = Counter()
    for r in recs:
        g, a, h = r["gt_cls"], r["aux_cls"], r["hm_cls"]
        cm_aux[g, a] += 1
        cm_hm[g, h] += 1
        mean_p[g].append(r["aux_p"])
        if a == h:
            agree += 1
        else:
            disagree[f"aux={CLS[a]}|hm={CLS[h]}|gt={CLS[g]}"] += 1
    mp = {}
    for g in range(3):
        arr = np.stack(mean_p[g]) if mean_p[g] else np.zeros((0, 3))
        mp[CLS[g]] = {
            "P_SAFE": float(arr[:, 0].mean()) if arr.size else None,
            "P_THREAT": float(arr[:, 1].mean()) if arr.size else None,
            "P_STOP": float(arr[:, 2].mean()) if arr.size else None,
        }
    return {
        "aux_confusion": cm_aux.tolist(),
        "heatmap_confusion": cm_hm.tolist(),
        "aux_prf": {CLS[k]: prf(cm_aux, k) for k in range(3)},
        "heatmap_prf": {CLS[k]: prf(cm_hm, k) for k in range(3)},
        "aux_mean_probs": mp,
        "aux_heatmap_agreement": agree / n,
        "n_disagree": n - agree,
        "disagree_top": dict(disagree.most_common(12)),
        "aux_acc": float(np.trace(cm_aux) / n),
        "heatmap_acc": float(np.trace(cm_hm) / n),
    }


def stop_buckets(recs):
    stop = [r for r in recs if r["gt_cls"] == STOP]
    groups = {"lt_0.25": [], "mid": [], "ge_0.60": []}
    for r in stop:
        if r["st_peak"] < 0.25:
            groups["lt_0.25"].append(r)
        elif r["st_peak"] < 0.60:
            groups["mid"].append(r)
        else:
            groups["ge_0.60"].append(r)
    out = {}
    for k, rs in groups.items():
        par_stop = sum(r["par_cls"] == STOP for r in rs)
        cells = Counter(f"r{r['gt_argmax'][0]}_c{r['gt_argmax'][1]}" for r in rs)
        samples = Counter(r["sample_id"] for r in rs)
        out[k] = {
            "n": len(rs),
            "parent_STOP": par_stop,
            "parent_nonSTOP": len(rs) - par_stop,
            "parent_STOP_frac": par_stop / max(1, len(rs)),
            "n_prefork": sum(r["pre_fork"] for r in rs),
            "n_postfork": sum(not r["pre_fork"] for r in rs),
            "gt_cells": dict(cells.most_common()),
            "top_samples": dict(samples.most_common(12)),
        }
    # parent-correct student-miss
    miss = [r for r in stop if r["st_peak"] < 0.60]
    out["student_miss_n"] = len(miss)
    out["student_miss_parent_was_STOP"] = sum(r["par_cls"] == STOP for r in miss)
    out["student_miss_parent_was_nonSTOP"] = sum(r["par_cls"] != STOP for r in miss)
    return out


def loss_bundle(model, batch, mu, sigma, cw_t, alpha, margin):
    out = model(batch["clip"])
    out["predictor_ref"] = model.predictor
    lf, lc, lr, lm = s79.latent_terms(out, batch, mu, sigma, model.proj)
    y = s79.peak_class_t(batch["gt_1s"])
    gw = cw_t[y]
    lh, lp = s79.task_heatmap(out["Hhat_1s"].float(), batch["gt_1s"].float(), gw)
    lfo = s79.focal_ce(out["aux_logits"].float(), y, alpha)
    lrk = s79.ranking_margin(out["Hhat_1s"].float().flatten(1).max(1).values, y, margin)
    return {"feat": lf, "cos": lc, "rel": lr, "mgd": lm, "h": lh, "peak": lp, "focal": lfo, "rank": lrk}, out


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad-batches", type=int, default=8)
    add_config_args(parser)
    args = parser.parse_args()
    cfg = load_cfg(args)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (PARENT_ROOT / "index.jsonl").open()]
    gt = np.load(PARENT_ROOT / "gt_1s.npy").astype(np.float32)
    train_ids = [i for i, r in enumerate(rows) if r["split"] == "train"]
    sel_ids = [i for i, r in enumerate(rows) if r["split"] == "selection"]
    assert len(sel_ids) == 5120
    rgb = s75.RGBCache256(RGB_ROOT)
    parent = ParentTargetCache(PARENT_ROOT)
    sel_loader = DataLoader(s75.WinDS(rows, sel_ids, gt, rgb, parent), batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=s75.collate)
    # fixed first 64 train windows (dataset order), not weighted sampler
    n_fixed = args.batch * args.grad_batches
    train_ds = s75.WinDS(rows, train_ids[:n_fixed], gt, rgb, parent)
    grad_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=s75.collate)
    device = resolve_device_safe(str(args.device))
    decoder = build_parent_decoder(cfg, device="cpu")
    assert decoder_state_hash(decoder) == EXPECTED_HASH
    model = s79.StudentTaskAware(decoder).to(device)
    st = np.load(STATS)
    mu = torch.tensor(st["mu"], dtype=torch.float32, device=device)
    sigma = torch.tensor(st["sigma"], dtype=torch.float32, device=device)
    peaks = gt[np.array(train_ids)].reshape(len(train_ids), -1).max(1)
    cls = np.zeros(len(train_ids), np.int64)
    cls[peaks >= 0.25] = CAUTION
    cls[peaks >= 0.60] = STOP
    counts = np.bincount(cls, minlength=3).astype(np.float64)
    inv = (len(train_ids) / np.maximum(counts, 1)) / 3.0
    class_w_vec = inv / inv.mean()
    cw_t = torch.tensor(class_w_vec, dtype=torch.float32, device=device)
    alpha = cw_t.clone()
    scales = json.loads((EXP / "loss_calibration.json").read_text())["scales"]
    amp = str(device).startswith("cuda")
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.nullcontext()
    backbone = list(model.encoder.parameters())
    print(f"device={device} decoder_ok n_sel={len(sel_ids)} n_grad_windows={n_fixed}", flush=True)

    all_ep = {}
    for ep in EPOCHS:
        blob = torch.load(EXP / f"checkpoint_epoch_{ep:02d}.pt", map_location=device, weights_only=False)
        model.encoder.load_state_dict(blob["encoder"])
        model.predictor.load_state_dict(blob["predictor"])
        model.proj.load_state_dict(blob["proj"])
        model.aux.load_state_dict(blob["aux"])
        assert decoder_state_hash(model.decoder) == EXPECTED_HASH
        recs, dF, dW, dB = collect_sel(model, sel_loader, device)
        peaks = analyze_peaks(recs)
        aux = analyze_aux(recs)
        buckets = stop_buckets(recs)
        # gradients on fixed train batches
        model.train()
        model.decoder.eval()
        g_abs = {k: [] for k in LOSS_NAMES}
        g_vec = {k: [] for k in LOSS_NAMES}
        l_mag = {k: [] for k in LOSS_NAMES}
        for batch in grad_loader:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with autocast:
                losses, _ = loss_bundle(model, b, mu, sigma, cw_t, alpha, 0.20)
            for name in LOSS_NAMES:
                model.zero_grad(set_to_none=True)
                lval = losses[name].float()
                l_mag[name].append(float(lval.detach()))
                if lval.requires_grad and float(lval.detach()) != 0.0:
                    lval.backward(retain_graph=True)
                    vec = flatten_grads(backbone)
                else:
                    vec = torch.zeros(sum(p.numel() for p in backbone))
                g_vec[name].append(vec.cpu())
                g_abs[name].append(float(vec.norm()))
                with autocast:
                    losses, _ = loss_bundle(model, b, mu, sigma, cw_t, alpha, 0.20)
        # mean over batches
        mean_abs = {k: float(np.mean(g_abs[k])) for k in LOSS_NAMES}
        tot = sum(mean_abs.values()) + 1e-12
        mean_share = {k: mean_abs[k] / tot for k in LOSS_NAMES}
        # pairwise cosine of concatenated batch grads
        cat = {k: torch.cat(g_vec[k]) for k in LOSS_NAMES}
        pair = {}
        for a in LOSS_NAMES:
            pair[a] = {}
            for bnm in LOSS_NAMES:
                va, vb = cat[a], cat[bnm]
                pair[a][bnm] = float(F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item())
        met = json.loads((EXP / f"metrics_epoch_{ep:02d}.json").read_text())
        all_ep[str(ep)] = {
            "logged": {
                "phase": met["train"]["phase"],
                "wts_used_this_epoch": met["train"]["wts"],
                "ema_cos": met["train"]["ema_cos"],
                "mean_train_cos": met["train"]["mean_cos"],
                "sel_cos": met["selection"]["fdec_cosine"],
                "SAFE": met["selection"]["safe_recall"],
                "THREAT": met["selection"]["threat_recall"],
                "STOP": met["selection"]["stop_recall"],
                "losses": {k: met["train"][k] for k in ("L_feat", "L_cos", "L_rel", "L_mgd", "L_h", "L_peak", "L_focal", "L_rank")},
                "logged_step0_gnorms": {k: met["train"].get(k) for k in ("g_feat", "g_h", "g_peak", "g_focal", "g_rank")},
            },
            "peaks": peaks,
            "aux": aux,
            "stop_buckets": buckets,
            "proj": {"mean_abs_F_minus_Fctx": dF, "weight_frobenius_to_I": dW, "mean_abs_bias": dB},
            "grad_abs": mean_abs,
            "grad_share": mean_share,
            "grad_pair_cosine": pair,
            "loss_mag_fixed_batches": {k: float(np.mean(l_mag[k])) for k in LOSS_NAMES},
        }
        print(f"epoch {ep} STOP={met['selection']['stop_recall']:.3f} collapse={peaks['student']['GT-STOP']['frac_lt_0.25']:.3f} aux_acc={aux['aux_acc']:.3f} hm_acc={aux['heatmap_acc']:.3f} agree={aux['aux_heatmap_agreement']:.3f}", flush=True)

    sched = []
    for ep in range(1, 15):
        m = json.loads((EXP / f"metrics_epoch_{ep:02d}.json").read_text())
        sched.append(
            {
                "epoch": ep,
                "logged_phase_after_epoch": m["train"]["phase"],
                "triggered": m.get("triggered"),
                "wts": m["train"]["wts"],
                "ema_cos": m["train"]["ema_cos"],
                "sel_cos": m["selection"]["fdec_cosine"],
                "SAFE": m["selection"]["safe_recall"],
                "THREAT": m["selection"]["threat_recall"],
                "STOP": m["selection"]["stop_recall"],
                "collapse": m["selection"]["peaks"]["collapsed_stop_frac"],
                "L": {k: m["train"][k] for k in ("L_feat", "L_cos", "L_h", "L_peak", "L_focal", "L_rank")},
            }
        )
    report = {
        "n_selection": 5120,
        "n_test": 0,
        "decoder_hash": EXPECTED_HASH,
        "grad_windows": n_fixed,
        "scales_from_calibration": scales,
        "schedule_epochs_1_14": sched,
        "epochs": all_ep,
        "schedule_note": (
            "Phase label is written AFTER the epoch's phase machine. "
            "Epoch 12 train still used Phase-A weights; failsafe ep>=12 set phase=B at the end of 12. "
            "Epoch 13 still used A weights (trans_t=0). Epoch 14 is the first mixed B lerp."
        ),
    }
    (OUT / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("WROTE", OUT / "audit.json", flush=True)


if __name__ == "__main__":
    main()
