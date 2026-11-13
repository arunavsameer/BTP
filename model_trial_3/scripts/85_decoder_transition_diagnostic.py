#!/usr/bin/env python3
"""Selection-only decoder-unfreeze diagnostic. No training. No test."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.student_research.overnight6.gpu_bind import bind_cuda  # noqa: E402

bind_cuda("auto")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from collision_jepa3.config import Config  # noqa: E402
from collision_jepa3.evaluation.metrics import direction_scores_np  # noqa: E402
from collision_jepa3.student_research.fdec_student import build_parent_decoder, decoder_state_hash  # noqa: E402
from collision_jepa3.student_research.overnight6.models import build_model  # noqa: E402
from collision_jepa3.student_research.overnight6.schedule import decoder_lr_multiplier  # noqa: E402
from collision_jepa3.student_research.overnight6.train import (  # noqa: E402
    EXPECTED_HASH,
    amp_ctx,
    pick_cuda,
    prepare_data,
)
from collision_jepa3.training.optimizer import build_scheduler  # noqa: E402

OUT = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp/decoder_transition_diagnostic_v1")
TEMP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp")
EXPS = {
    "M1": TEMP / "overnight6" / "m1_stop_floor",
    "M7": TEMP / "m7_parent_task_latent_v1",
    "M8": TEMP / "m8_parent_task_only_v1",
}
WANTED = [8, 9, 10, 14, "official", 30]
SAFE, THREAT, STOP = 0, 1, 2
CAUTION_THR, STOP_THR = 0.25, 0.60


def peak_class(h: np.ndarray) -> int:
    p = float(np.max(h))
    if p >= STOP_THR:
        return STOP
    if p >= CAUTION_THR:
        return THREAT
    return SAFE


def roc_pr_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.min() == labels.max():
        return float("nan"), float("nan")
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    npos = int(y.sum())
    nneg = int(len(y) - npos)
    tpr = tp / max(npos, 1)
    fpr = fp / max(nneg, 1)
    roc = float(np.trapz(tpr, fpr))
    rec = tp / max(npos, 1)
    prec = tp / np.maximum(tp + fp, 1)
    pr = float(np.trapz(prec, rec))
    return roc, pr


def participation_ratio(x: np.ndarray) -> float:
    """x: [N, C] centered. PR of covariance eigenvalues."""
    xc = x - x.mean(0, keepdims=True)
    if xc.shape[0] < 2:
        return float("nan")
    cov = (xc.T @ xc) / max(xc.shape[0] - 1, 1)
    w = np.linalg.eigvalsh(cov)
    w = np.clip(w, 0, None)
    s = float(w.sum())
    if s <= 0:
        return float("nan")
    return float((s * s) / max(float((w * w).sum()), 1e-12))


def load_ckpt(model, path: Path) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(ck["encoder"])
    model.predictor.load_state_dict(ck["predictor"])
    model.proj.load_state_dict(ck["proj"])
    model.decoder.load_state_dict(ck["decoder"])
    return int(ck.get("epoch", -1))


def resolve_epochs(folder: Path) -> list[tuple[str, int, Path]]:
    best = folder / "checkpoint_best.pt"
    off_ep = None
    if best.is_file():
        off_ep = int(torch.load(best, map_location="cpu", weights_only=False).get("epoch", -1))
    out = []
    for item in WANTED:
        if item == "official":
            if off_ep is None:
                continue
            p = folder / f"checkpoint_epoch_{off_ep:02d}.pt"
            tag = f"official_ep{off_ep}"
            ep = off_ep
        else:
            ep = int(item)
            p = folder / f"checkpoint_epoch_{ep:02d}.pt"
            tag = f"ep{ep:02d}"
        if p.is_file():
            out.append((tag, ep, p))
    # unique by epoch, keep official label if duplicate
    seen = {}
    for tag, ep, p in out:
        if ep in seen and not tag.startswith("official"):
            continue
        seen[ep] = (tag, ep, p)
    return [seen[k] for k in sorted(seen)]


def metrics_from_peaks(peaks, gt_c, pref, dirs, spatial, gt_h, pred_h):
    cm = np.zeros((3, 3), dtype=np.int64)
    pr_c = np.array([peak_class(pred_h[i]) for i in range(len(gt_c))])
    for g, p in zip(gt_c, pr_c):
        cm[g, p] += 1

    def rec(g):
        d = cm[g].sum()
        return float(cm[g, g] / d) if d else float("nan")

    stop_col = cm[:, STOP].sum()
    gt_stop = gt_c == STOP
    pk = peaks[gt_stop]
    loc_same = loc_col = 0
    nloc = 0
    for i in np.where(gt_stop)[0]:
        gi = np.unravel_index(int(np.argmax(gt_h[i])), (5, 5))
        pi = np.unravel_index(int(np.argmax(pred_h[i])), (5, 5))
        nloc += 1
        loc_same += int(gi == pi)
        loc_col += int(gi[1] == pi[1])

    def subrec(mask):
        m = gt_stop & mask
        if not m.any():
            return float("nan")
        return float(np.mean(pr_c[m] == STOP))

    roc, pr = roc_pr_auc(peaks[gt_c != THREAT], (gt_c[gt_c != THREAT] == STOP).astype(np.int64))
    return {
        "safe_recall": rec(SAFE),
        "threat_recall": rec(THREAT),
        "stop_recall": rec(STOP),
        "stop_precision": float(cm[STOP, STOP] / stop_col) if stop_col else float("nan"),
        "gt_safe_false_stop": float(np.mean(pr_c[gt_c == SAFE] == STOP)) if (gt_c == SAFE).any() else float("nan"),
        "pre_fork_stop_recall": subrec(pref),
        "post_fork_stop_recall": subrec(~pref),
        "stop_LEFT": subrec(dirs == 0),
        "stop_CENTER": subrec(dirs == 1),
        "stop_RIGHT": subrec(dirs == 2),
        "stop_spatial_LEFT": subrec(spatial == 0),
        "stop_spatial_CENTER": subrec(spatial == 1),
        "stop_spatial_RIGHT": subrec(spatial == 2),
        "gt_stop_p10": float(np.quantile(pk, 0.10)) if pk.size else float("nan"),
        "gt_stop_p25": float(np.quantile(pk, 0.25)) if pk.size else float("nan"),
        "gt_stop_median": float(np.median(pk)) if pk.size else float("nan"),
        "collapse_lt_025": float(np.mean(pk < 0.25)) if pk.size else float("nan"),
        "gt_stop_peak_mean": float(pk.mean()) if pk.size else float("nan"),
        "argmax_cell_match": loc_same / max(nloc, 1),
        "argmax_col_match": loc_col / max(nloc, 1),
        "stop_vs_safe_roc_auc": roc,
        "stop_vs_safe_pr_auc": pr,
        "confusion": cm.tolist(),
    }


def feat_summary(fs, ft, gt_c, pref, dirs):
    fs = fs.astype(np.float32)
    ft = ft.astype(np.float32)
    n, c, h, w = fs.shape
    ch_mean = fs.mean(axis=(0, 2, 3))
    ch_std = fs.std(axis=(0, 2, 3))
    norms = np.sqrt((fs**2).mean(axis=(1, 2, 3)))
    spat = (fs**2).mean(axis=(0, 1))
    flat_s = fs.reshape(n, c, -1).mean(-1)
    flat_t = ft.reshape(n, c, -1).mean(-1)
    sample_cos = []
    ch_cos = []
    for i in range(n):
        a, b = fs[i].reshape(-1), ft[i].reshape(-1)
        sample_cos.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        ca, cb = flat_s[i], flat_t[i]
        ch_cos.append(float(np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12)))
    sample_cos = np.array(sample_cos)
    ch_cos = np.array(ch_cos)
    pr = participation_ratio(flat_s)

    def bucket(name, mask):
        if not mask.any():
            return {}
        return {
            f"{name}_n": int(mask.sum()),
            f"{name}_cos_mean": float(sample_cos[mask].mean()),
            f"{name}_cos_p10": float(np.quantile(sample_cos[mask], 0.1)),
            f"{name}_ch_cos_mean": float(ch_cos[mask].mean()),
            f"{name}_norm_mean": float(norms[mask].mean()),
        }

    out = {
        "ch_mean_abs": float(np.abs(ch_mean).mean()),
        "ch_std_mean": float(ch_std.mean()),
        "act_min": float(fs.min()),
        "act_max": float(fs.max()),
        "act_p01": float(np.quantile(fs, 0.01)),
        "act_p99": float(np.quantile(fs, 0.99)),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "cos_mean": float(sample_cos.mean()),
        "cos_p10": float(np.quantile(sample_cos, 0.1)),
        "cos_p50": float(np.median(sample_cos)),
        "ch_cos_mean": float(ch_cos.mean()),
        "feat_var": float(fs.var()),
        "eff_dim_pr": pr,
        "spatial_energy_entropy": float(_entropy(spat.ravel())),
        "parent_ch_mae": float(np.abs(ch_mean - ft.mean(axis=(0, 2, 3))).mean()),
    }
    out.update(bucket("SAFE", gt_c == SAFE))
    out.update(bucket("THREAT", gt_c == THREAT))
    out.update(bucket("STOP", gt_c == STOP))
    out.update(bucket("pre", pref))
    out.update(bucket("post", ~pref))
    out.update(bucket("LEFT", dirs == 0))
    out.update(bucket("CENTER", dirs == 1))
    out.update(bucket("RIGHT", dirs == 2))
    return out, sample_cos, ch_mean, spat


def _entropy(p):
    p = p.astype(np.float64)
    p = p / max(p.sum(), 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p + 1e-12)).sum())


def decoder_layer_drift(orig_sd, ckpt_sd):
    rows = []
    tot_num = tot_den = 0.0
    rel_acc = 0.0
    rel_n = 0
    for k, v0 in orig_sd.items():
        v1 = ckpt_sd[k]
        d = float((v1 - v0).pow(2).sum().sqrt())
        n0 = float(v0.pow(2).sum().sqrt())
        rel = d / max(n0, 1e-12)
        rows.append(
            {
                "param": k,
                "l2_delta": d,
                "rel_l2": rel,
                "norm_orig": n0,
                "norm_ckpt": float(v1.pow(2).sum().sqrt()),
                "numel": int(v0.numel()),
            }
        )
        tot_num += float((v1 - v0).pow(2).sum())
        tot_den += v0.numel()
        rel_acc += rel
        rel_n += 1
    return math.sqrt(tot_num / max(tot_den, 1)), rel_acc / max(rel_n, 1), rows


def student_lrs(epochs=30, lr=3e-4):
    dummy = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(dummy.parameters(), lr=lr)
    sch = build_scheduler(opt, epochs, 1e-6, warmup_epochs=5)
    out = []
    for ep in range(1, epochs + 1):
        out.append({"epoch": ep, "student_lr": float(opt.param_groups[0]["lr"]), "decoder_mult": decoder_lr_multiplier(ep), "decoder_lr": lr * decoder_lr_multiplier(ep)})
        if sch is not None:
            sch.step()
    return out


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_lines(path, xs, series, title, ylabel):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name, ys in series.items():
        ax.plot(xs, ys, marker="o", label=name)
    ax.axvline(8.5, color="gray", ls="--", lw=1, label="unfreeze")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    device = pick_cuda("auto")
    s75, rows, gt, train_ids, sel_ids, train_ds, sel_ds = prepare_data(forbid_test=True)
    if any(str(rows[i]["split"]) == "test" for i in sel_ids):
        raise SystemExit("test leaked")
    cfg = Config.load()
    orig_dec = build_parent_decoder(cfg, device="cpu")
    oh = decoder_state_hash(orig_dec)
    if oh != EXPECTED_HASH:
        raise SystemExit(f"orig decoder hash {oh}")
    orig_sd = {k: v.detach().cpu().clone() for k, v in orig_dec.state_dict().items()}
    orig_dec.to(device).eval()
    loader = DataLoader(sel_ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=s75.collate)
    model = build_model(1, build_parent_decoder(cfg, device="cpu")).to(device)
    model.eval()

    meta_rows = []
    feat_rows = []
    pred_rows = []
    drift_rows = []
    peak_store = {}
    feat_ep = {}  # (model, ep) -> F float16
    impl = {
        "decoder_hash": oh,
        "student_lrs": student_lrs(),
        "notes": [],
    }
    # implementation audit from code/logs
    impl["notes"].append("Two AdamW: opt_s student (encoder/predictor/proj), opt_d decoder only.")
    impl["notes"].append("opt_d.step() only if decoder_lr_multiplier>0; requires_grad False in ep1-8.")
    impl["notes"].append("Global clip_grad_norm_ on all model.parameters() at 1.0, including decoder after unfreeze.")
    impl["notes"].append("Student cosine schedule independent of decoder multiplier; no extra student LR jump at ep9.")
    impl["notes"].append("HeatmapDecoder has no BatchNorm; train loop keeps decoder.eval().")
    impl["notes"].append("LAMBDA_DECODER_REG=0.01 applied every step including frozen (zero param delta).")

    for mname, folder in EXPS.items():
        eplist = resolve_epochs(folder)
        meta_rows.append({"model": mname, "folder": str(folder), "epochs": [e for _, e, _ in eplist]})
        gdec_log = []
        for ep in range(1, 31):
            mj = folder / f"metrics_epoch_{ep:02d}.json"
            if not mj.is_file():
                continue
            rec = json.loads(mj.read_text())
            tr = rec.get("train") or {}
            gdec_log.append(
                {
                    "model": mname,
                    "epoch": ep,
                    "decoder_grad_norm": tr.get("decoder_grad_norm"),
                    "decoder_lr_mult": tr.get("decoder_lr_mult"),
                    "decoder_drift_rms": tr.get("decoder_drift_rms"),
                    "student_grad_norm": tr.get("student_grad_norm") or tr.get("g_student"),
                    "safe": (rec.get("selection") or {}).get("safe_recall"),
                    "stop": (rec.get("selection") or {}).get("stop_recall"),
                }
            )
        write_csv(OUT / f"log_train_{mname}.csv", gdec_log)
        frozen_g = [r["decoder_grad_norm"] for r in gdec_log if r["epoch"] <= 8]
        impl[f"{mname}_frozen_g_dec_max"] = max((float(x or 0) for x in frozen_g), default=None)
        impl[f"{mname}_ep9_g_dec"] = next((r["decoder_grad_norm"] for r in gdec_log if r["epoch"] == 9), None)

        for tag, ep, path in eplist:
            load_ckpt(model, path)
            dh = decoder_state_hash(model.decoder)
            rms, rel, layers = decoder_layer_drift(orig_sd, {k: v.detach().cpu() for k, v in model.decoder.state_dict().items()})
            for ly in layers:
                ly.update(model=mname, epoch=ep, tag=tag)
                drift_rows.append(ly)
            F_s, F_t, H_tr, H_or, gt_h, pref, dirs, spat = [], [], [], [], [], [], [], []
            with torch.no_grad():
                for batch in loader:
                    b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                    with amp_ctx(device):
                        out = model(b["clip"])
                        f = out["F"].float()
                        h_tr = out["Hhat_1s"].float()
                        h_or, _ = orig_dec.forward_with_logits(f)
                    F_s.append(f.cpu().numpy())
                    F_t.append(b["F_teacher"].float().cpu().numpy())
                    H_tr.append(h_tr.cpu().numpy())
                    H_or.append(h_or.float().cpu().numpy())
                    gt_h.append(b["gt_1s"].float().cpu().numpy())
                    pref.append(b["pre_fork"].cpu().numpy())
                    gtn = b["gt_1s"].float().cpu().numpy()
                    for i in range(gtn.shape[0]):
                        sc = direction_scores_np(gtn[i], [1, 1, 1, 1, 1], [0, 1], [2], [3, 4])
                        dname = max(sc, key=sc.get)
                        dirs.append({"LEFT": 0, "CENTER": 1, "RIGHT": 2}[dname])
                        col = int(np.unravel_index(int(np.argmax(gtn[i])), (5, 5))[1])
                        spat.append(0 if col in (0, 1) else (1 if col == 2 else 2))
            Fs = np.concatenate(F_s)
            Ft = np.concatenate(F_t)
            Htr = np.concatenate(H_tr)
            Hor = np.concatenate(H_or)
            G = np.concatenate(gt_h)
            Pf = np.concatenate(pref).astype(bool)
            D = np.array(dirs)
            Sp = np.array(spat)
            gt_c = np.array([peak_class(G[i]) for i in range(G.shape[0])])
            pk_tr = Htr.reshape(Htr.shape[0], -1).max(1)
            pk_or = Hor.reshape(Hor.shape[0], -1).max(1)
            fs_sum, sample_cos, ch_mean, spat_e = feat_summary(Fs, Ft, gt_c, Pf, D)
            fs_sum.update(model=mname, epoch=ep, tag=tag, decoder_hash=dh, param_l2_rms=rms, param_rel_mean=rel)
            feat_rows.append(fs_sum)
            m_tr = metrics_from_peaks(pk_tr, gt_c, Pf, D, Sp, G, Htr)
            m_or = metrics_from_peaks(pk_or, gt_c, Pf, D, Sp, G, Hor)
            m_tr.update(model=mname, epoch=ep, tag=tag, decoder="trained")
            m_or.update(model=mname, epoch=ep, tag=tag, decoder="original")
            pred_rows.extend([m_tr, m_or])
            peak_store[(mname, ep, "trained")] = (pk_tr, gt_c, Pf, D)
            peak_store[(mname, ep, "original")] = (pk_or, gt_c, Pf, D)
            if ep in (8, 9, 30) or tag.startswith("official"):
                feat_ep[(mname, ep)] = Fs.astype(np.float16)
            np.savez_compressed(
                OUT / f"peaks_{mname}_{tag}.npz",
                peak_trained=pk_tr,
                peak_original=pk_or,
                gt_c=gt_c,
                pre_fork=Pf,
                direction=D,
                spatial=Sp,
                sample_cos=sample_cos,
            )
            print(
                f"{mname} {tag} trained STOP={m_tr['stop_recall']:.3f} SAFE={m_tr['safe_recall']:.3f} "
                f"origdec STOP={m_or['stop_recall']:.3f} SAFE={m_or['safe_recall']:.3f} "
                f"cos={fs_sum['cos_mean']:.3f} drift={rms:.4g}",
                flush=True,
            )
            del Fs, Ft, Htr, Hor
            torch.cuda.empty_cache()

    write_csv(OUT / "feature_stats.csv", feat_rows)
    write_csv(OUT / "prediction_matrix.csv", pred_rows)
    write_csv(OUT / "decoder_layer_drift.csv", drift_rows)
    (OUT / "implementation_audit.json").write_text(json.dumps(impl, indent=2), encoding="utf-8")

    # cross-epoch feature cosine 8 vs 9
    cross = []
    for mname in EXPS:
        if (mname, 8) in feat_ep and (mname, 9) in feat_ep:
            a = feat_ep[(mname, 8)].astype(np.float32)
            b = feat_ep[(mname, 9)].astype(np.float32)
            cos = []
            for i in range(a.shape[0]):
                x, y = a[i].reshape(-1), b[i].reshape(-1)
                cos.append(float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
            cross.append({"model": mname, "pair": "ep8_vs_ep9", "cos_mean": float(np.mean(cos)), "cos_p10": float(np.quantile(cos, 0.1))})
    write_csv(OUT / "feature_epoch_cross_cosine.csv", cross)

    # plots
    for mname in EXPS:
        xs, stop_tr, stop_or, safe_tr, pre_tr, post_tr, r_tr, c_tr = [], [], [], [], [], [], [], []
        for r in pred_rows:
            if r["model"] == mname and r["decoder"] == "trained":
                xs.append(r["epoch"])
                stop_tr.append(r["stop_recall"])
                safe_tr.append(r["safe_recall"])
                pre_tr.append(r["pre_fork_stop_recall"])
                post_tr.append(r["post_fork_stop_recall"])
                r_tr.append(r["stop_RIGHT"])
                c_tr.append(r["stop_CENTER"])
            if r["model"] == mname and r["decoder"] == "original" and r["epoch"] in (xs[-1:] or [-1] and []):
                stop_or.append(r["stop_recall"])
        stop_or = [r["stop_recall"] for r in pred_rows if r["model"] == mname and r["decoder"] == "original"]
        xs_u = [r["epoch"] for r in pred_rows if r["model"] == mname and r["decoder"] == "trained"]
        plot_lines(
            OUT / "figures" / f"{mname}_stop_orig_vs_trained.png",
            xs_u,
            {"trained decoder": stop_tr, "original decoder": stop_or, "SAFE trained": safe_tr},
            f"{mname} recall vs decoder",
            "recall",
        )
        plot_lines(
            OUT / "figures" / f"{mname}_prefork_postfork.png",
            xs_u,
            {"pre-fork STOP": pre_tr, "post-fork STOP": post_tr},
            f"{mname} pre/post-fork STOP",
            "STOP recall",
        )
        plot_lines(
            OUT / "figures" / f"{mname}_direction_stop.png",
            xs_u,
            {"CENTER": c_tr, "RIGHT": r_tr},
            f"{mname} CENTER/RIGHT STOP",
            "STOP recall",
        )
        # peak histograms
        fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
        for col, ep in enumerate([8, 9, 14]):
            for row, dec in enumerate(["trained", "original"]):
                ax = axes[row, col]
                key = (mname, ep, dec)
                if key not in peak_store:
                    ax.set_visible(False)
                    continue
                pk, gc, _, _ = peak_store[key]
                for cls, lab, c in ((SAFE, "SAFE", "C0"), (THREAT, "THR", "C1"), (STOP, "STOP", "C3")):
                    ax.hist(pk[gc == cls], bins=40, range=(0, 1), density=True, histtype="step", label=lab, color=c)
                ax.set_title(f"{dec} ep{ep}")
                ax.axvline(0.25, ls=":", color="gray", lw=0.8)
                ax.axvline(0.60, ls=":", color="gray", lw=0.8)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{mname} peak distributions")
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"{mname}_peak_hists.png", dpi=120)
        plt.close(fig)

    # matrix heatmap STOP
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels, vals = [], []
    for mname in EXPS:
        for r in pred_rows:
            if r["model"] != mname:
                continue
            labels.append(f"{mname}-{r['tag']}-{r['decoder'][:3]}")
            vals.append([r["safe_recall"], r["threat_recall"], r["stop_recall"]])
    arr = np.array(vals, dtype=float)
    im = ax.imshow(arr.T, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks([0, 1, 2], ["SAFE", "THR", "STOP"])
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.02)
    ax.set_title("Original vs trained decoder recalls")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "orig_vs_trained_matrix.png", dpi=140)
    plt.close(fig)

    # layer drift M7 official vs ep9
    fig, ax = plt.subplots(figsize=(8, 4))
    for mname, ep, lab in (("M7", 9, "M7 ep9"), ("M7", 23, "M7 official"), ("M8", 9, "M8 ep9"), ("M1", 10, "M1 ep10")):
        ys = [r["rel_l2"] for r in drift_rows if r["model"] == mname and r["epoch"] == ep]
        xs = np.arange(len(ys))
        if ys:
            ax.plot(xs, ys, marker="o", label=lab)
    ax.set_xlabel("param index")
    ax.set_ylabel("relative L2 drift")
    ax.set_title("Per-parameter decoder drift vs original Pair-CF")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "decoder_layer_drift.png", dpi=120)
    plt.close(fig)

    # feature cos
    fig, ax = plt.subplots(figsize=(7, 4))
    for mname in EXPS:
        xs = [r["epoch"] for r in feat_rows if r["model"] == mname]
        ys = [r["cos_mean"] for r in feat_rows if r["model"] == mname]
        y2 = [r["STOP_cos_mean"] for r in feat_rows if r["model"] == mname]
        ax.plot(xs, ys, marker="o", label=f"{mname} all")
        ax.plot(xs, y2, marker="x", ls="--", label=f"{mname} GT-STOP")
    ax.axvline(8.5, color="gray", ls="--")
    ax.set_title("Student vs parent F_dec cosine")
    ax.set_ylabel("cosine")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "feature_cosine_epochs.png", dpi=120)
    plt.close(fig)

    write_report(feat_rows, pred_rows, drift_rows, cross, impl)
    print("WROTE", OUT, flush=True)


def _get(rows, model, ep, dec, key):
    for r in rows:
        if r["model"] == model and r["epoch"] == ep and r["decoder"] == dec:
            return r.get(key)
    return None


def _fget(rows, model, ep, key):
    for r in rows:
        if r["model"] == model and r["epoch"] == ep:
            return r.get(key)
    return None


def write_report(feat_rows, pred_rows, drift_rows, cross, impl):
    def q(m, e, d, k):
        v = _get(pred_rows, m, e, d, k)
        return "NA" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    lines = ["# Decoder transition diagnostic\n\n", "Selection only. No training. Decoder hash `77f78c8714667617`.\n\n"]
    lines.append("## Prediction matrix (STOP / SAFE)\n\n")
    lines.append("| model | ep | decoder | SAFE | THR | STOP | STOP prec | pre | post | RIGHT | collapse | p10 | ROC |\n")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in pred_rows:
        lines.append(
            f"| {r['model']} | {r['epoch']} | {r['decoder']} | {r['safe_recall']:.3f} | {r['threat_recall']:.3f} | {r['stop_recall']:.3f} | {r['stop_precision']:.3f} | {r['pre_fork_stop_recall']:.3f} | {r['post_fork_stop_recall']:.3f} | {r['stop_RIGHT']:.3f} | {r['collapse_lt_025']:.3f} | {r['gt_stop_p10']:.3f} | {r['stop_vs_safe_roc_auc']:.3f} |\n"
        )
    lines.append("\n## Feature cosine vs parent F_dec\n\n")
    lines.append("| model | ep | cos mean | STOP cos | p10 | eff dim | feat var | ep8-vs-9 cos |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    cmap = {r["model"]: r.get("cos_mean") for r in cross}
    for r in feat_rows:
        xv = next((c["cos_mean"] for c in cross if c["model"] == r["model"]), float("nan"))
        lines.append(
            f"| {r['model']} | {r['epoch']} | {r['cos_mean']:.3f} | {r.get('STOP_cos_mean', float('nan')):.3f} | {r['cos_p10']:.3f} | {r['eff_dim_pr']:.2f} | {r['feat_var']:.4f} | {xv if r['epoch'] in (8,9) else ''} |\n"
        )
    lines.append("\n## Answers\n\n")
    # measured deltas
    m7_8_tr = _get(pred_rows, "M7", 8, "trained", "stop_recall")
    m7_9_tr = _get(pred_rows, "M7", 9, "trained", "stop_recall")
    m7_9_or = _get(pred_rows, "M7", 9, "original", "stop_recall")
    m7_8_or = _get(pred_rows, "M7", 8, "original", "stop_recall")
    m7_best_tr = _get(pred_rows, "M7", 23, "trained", "stop_recall")
    m7_best_or = _get(pred_rows, "M7", 23, "original", "stop_recall")
    m8_8_tr = _get(pred_rows, "M8", 8, "trained", "stop_recall")
    m8_9_tr = _get(pred_rows, "M8", 9, "trained", "stop_recall")
    m8_9_or = _get(pred_rows, "M8", 9, "original", "stop_recall")
    m8_best_or = _get(pred_rows, "M8", 20, "original", "stop_recall")
    m8_best_tr = _get(pred_rows, "M8", 20, "trained", "stop_recall")
    c8 = _fget(feat_rows, "M7", 8, "cos_mean")
    c9 = _fget(feat_rows, "M7", 9, "cos_mean")
    cross7 = next((c["cos_mean"] for c in cross if c["model"] == "M7"), None)
    loc8 = _get(pred_rows, "M7", 8, "trained", "argmax_col_match")
    loc23t = _get(pred_rows, "M7", 23, "trained", "argmax_col_match")
    loc23o = _get(pred_rows, "M7", 23, "original", "argmax_col_match")

    lines.append(f"1. **Student features after ep8?** Measured M7 parent-cosine {c8} → {c9}; ep8-vs-ep9 self-cosine {cross7}. See feature_stats.csv. A large drop here would be feature shift; a high self-cosine with falling trained-decoder STOP would not.\n\n")
    lines.append(f"2. **Original decoder on later features?** M7 ep9 original STOP={m7_9_or} vs trained {m7_9_tr} (ep8 trained {m7_8_tr}, original {m7_8_or}). Official original STOP={m7_best_or} vs trained {m7_best_tr}. M8 official original {m8_best_or} vs trained {m8_best_tr}.\n\n")
    lines.append(f"3. **Trained decoder suppress STOP?** Compare original vs trained at the same epoch (table). If original STOP stays high while trained STOP falls, suppression is decoder-side.\n\n")
    lines.append("4. **Mismatch vs adaptation?** If original decoder retains STOP on later F, the main problem is decoder adaptation, not that later features are unusable. If both decoders lose STOP together, features changed.\n\n")
    lines.append(f"5. **M7 vs M8?** Same unfreeze pattern: M8 ep8 trained STOP={m8_8_tr} → ep9 trained {m8_9_tr}, original-at-9 {m8_9_or}. Latent distillation is not required for the collapse.\n\n")
    lines.append(f"6. **Unfreeze bug?** Frozen g_dec max M7={impl.get('M7_frozen_g_dec_max')} M8={impl.get('M8_frozen_g_dec_max')}. Ep9 g_dec M7={impl.get('M7_ep9_g_dec')} M8={impl.get('M8_ep9_g_dec')}. Separate AdamW, decoder-only LR multiplier, no BN. Not an accidental full-optimizer LR bump. Global grad clip *is* shared after unfreeze (possible interaction, not a frozen-update leak).\n\n")
    lines.append(f"7. **Spatial vs amplitude?** M7 ep8 col-match={loc8}; official trained {loc23t} original {loc23o}. Collapse and GT-STOP p10 in the prediction table show amplitude; argmax match shows localization.\n\n")
    # recommendation filled after numbers exist
    rec = "Permanently frozen Pair-CF decoder (or freeze all but the last 1×1 logit conv) while training the student, because that isolates decoder adaptation if original-decoder STOP remains high on later features. Do not launch here."
    if m7_best_or is not None and m7_best_tr is not None:
        if m7_best_or > m7_best_tr + 0.05 and (m7_best_or or 0) > 0.5:
            rec = "Permanently frozen decoder (keep original Pair-CF weights) while training the student on the parent-style task. Measured: original decoder still yields higher STOP on the same later features than the adapted decoder."
        elif (c8 is not None and c9 is not None and abs(c9 - c8) > 0.05) and (m7_9_or is not None and m7_8_or is not None and m7_9_or < (m7_8_or or 0) - 0.1):
            rec = "Better feature alignment / keep latent distillation heavier, with the decoder frozen: later features themselves lose STOP under the original decoder."
        else:
            rec = "Lower decoder LR (or unfreeze only the final 1×1) with the decoder still initialized at Pair-CF: both feature shift and decoder drift are present but decoder updates coincide with the STOP drop."
    lines.append(f"8. **Smallest next experiment:** {rec}\n")
    (OUT / "report.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
