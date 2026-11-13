#!/usr/bin/env python3
"""Experiment A: task-aware fine-tune of student_lat from raw epoch 38. No OLS, no 1x1, no parent H KD."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
TEMP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp")
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402
from collision_jepa3.evaluation.metrics import CAUTION, SAFE, STOP  # noqa: E402
from collision_jepa3.student_research.cache_access import ParentTargetCache  # noqa: E402
from collision_jepa3.student_research.fdec_student import build_parent_decoder, decoder_state_hash  # noqa: E402
from collision_jepa3.training.optimizer import build_scheduler  # noqa: E402
from collision_jepa3.utils.device import resolve_device_safe  # noqa: E402
from student_model.loss import blob_weights, relation_loss, spatial_cosine, weighted_l1  # noqa: E402

spec = importlib.util.spec_from_file_location("s75", ROOT / "scripts/75_train_student_lat_paircf.py")
s75 = importlib.util.module_from_spec(spec)
sys.modules["s75"] = s75
spec.loader.exec_module(s75)

SRC38 = TEMP / "student_lat_paircf" / "checkpoint_epoch_38.pt"
EXP = TEMP / "student_lat_paircf_expA"
RGB_ROOT = TEMP / "shared_rgb_cache_256_v1"
PARENT_ROOT = TEMP / "parent_target_cache_v2"
EXPECTED_HASH = "77f78c8714667617"
STATS = TEMP / "student_lat_paircf" / "fdec_channel_stats.npz"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def peak_class_t(gt: torch.Tensor) -> torch.Tensor:
    p = gt.flatten(1).max(dim=1).values
    out = torch.zeros(gt.size(0), dtype=torch.long, device=gt.device)
    out = torch.where(p >= 0.25, torch.full_like(out, CAUTION), out)
    out = torch.where(p >= 0.60, torch.full_like(out, STOP), out)
    return out


def ramp(epoch: int, start: int = 1, full: int = 5) -> float:
    if epoch <= start:
        return 0.0
    if epoch >= full:
        return 1.0
    return (epoch - start) / float(full - start)


def gnorm(params, loss):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    s = 0.0
    for g in grads:
        if g is not None:
            s += float(g.detach().pow(2).sum())
    return s ** 0.5


def task_losses(h, gt, class_w):
    """Occupied-weighted Smooth-L1 + peak Smooth-L1. Gradients flow through frozen decoder."""
    w_cell = 0.10 + 4.0 * gt.clamp(0, 1).pow(2) + 2.0 * (gt >= 0.50).to(gt.dtype)
    sl1 = F.smooth_l1_loss(h, gt, reduction="none", beta=0.05)
    gw = class_w.view(-1, 1, 1)
    l_h = (sl1 * w_cell * gw).mean()
    l_peak = (
        F.smooth_l1_loss(
            h.flatten(1).max(dim=1).values,
            gt.flatten(1).max(dim=1).values,
            reduction="none",
            beta=0.05,
        )
        * class_w
    ).mean()
    return l_h, l_peak


def latent_terms(out, batch, mu, sigma):
    fs, ft = out["F"], batch["F_teacher"]
    w = blob_weights(ft, None, batch["gt_1s"], energy_alpha=0.0, appear_boost=0.0, stop_boost=0.0)
    fs_n = (fs - mu) / (sigma + 1e-6)
    ft_n = (ft - mu) / (sigma + 1e-6)
    l_feat = weighted_l1(fs_n, ft_n, w)
    l_cos = spatial_cosine(fs, ft, w)
    l_rel = relation_loss(fs, ft)
    mask = s75.random_spatial_mask(out["F_ctx"], keep=0.5)
    f_mgd = out["predictor_ref"](out["F_ctx"], mask=mask)
    l_mgd = weighted_l1((f_mgd - mu) / (sigma + 1e-6), ft_n, w)
    return l_feat, l_cos, l_rel, l_mgd


def peak_dist(hs, hg):
    peaks = np.max(hs, axis=(1, 2))
    gc = np.array([s75.peak_class(hg[i]) for i in range(len(hg))])

    def s(x):
        x = np.asarray(x, np.float64)
        if x.size == 0:
            return {}
        return {"n": int(x.size), "mean": float(x.mean()), "p50": float(np.median(x)), "p10": float(np.quantile(x, 0.1)), "p90": float(np.quantile(x, 0.9))}

    return {
        "all": s(peaks),
        "gt_safe": s(peaks[gc == SAFE]),
        "gt_threat": s(peaks[gc == CAUTION]),
        "gt_stop": s(peaks[gc == STOP]),
        "gt_stop_frac_lt_0.25": float(np.mean(peaks[gc == STOP] < 0.25)) if (gc == STOP).any() else None,
    }


@torch.no_grad()
def eval_sel(model, loader, device, amp):
    model.eval()
    rows = []
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.nullcontext()
    cos, l1 = [], []
    hs, hg = [], []
    for batch in loader:
        b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with ctx:
            out = model(b["clip"])
        fs, ft = out["F"].float(), b["F_teacher"].float()
        cos.append(F.cosine_similarity(fs.flatten(1), ft.flatten(1)).cpu().numpy())
        l1.append((fs - ft).abs().mean(dim=(1, 2, 3)).cpu().numpy())
        h = out["Hhat_1s"].float().cpu().numpy()
        gt = b["gt_1s"].float().cpu().numpy()
        ht = b["H_teacher"].float().cpu().numpy()
        pref = b["pre_fork"].cpu().numpy()
        hs.append(h)
        hg.append(gt)
        for i in range(h.shape[0]):
            rows.append(
                {
                    "H_student": h[i],
                    "H_teacher": ht[i],
                    "gt_1s": gt[i],
                    "pre_fork": bool(pref[i]),
                    "sample_id": b["sample_id"][i],
                    "frame": int(b["frame"][i]),
                    "background_id": b["background_id"][i],
                    "clip_id": b["clip_id"][i],
                }
            )
    m = s75.metrics_from_rows(rows, "H_student")
    m["fdec_cosine"] = float(np.concatenate(cos).mean())
    m["fdec_l1"] = float(np.concatenate(l1).mean())
    m["peaks"] = peak_dist(np.concatenate(hs), np.concatenate(hg))
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--num-workers", type=int, default=4)
    add_config_args(parser)
    args = parser.parse_args()
    cfg = load_cfg(args)
    EXP.mkdir(parents=True, exist_ok=True)
    log_path = EXP / "run.log"

    def log(m: str) -> None:
        print(m, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(m + "\n")

    try:
        src_hash = sha256(SRC38)
        shutil.copy2(SRC38, EXP / "init_from_epoch38.pt")
        assert sha256(SRC38) == src_hash, "epoch-38 source mutated"
        (EXP / "source_epoch38.sha256").write_text(src_hash + "\n")
        rows = [json.loads(l) for l in (PARENT_ROOT / "index.jsonl").open()]
        gt = np.load(PARENT_ROOT / "gt_1s.npy").astype(np.float32)
        train_ids = [i for i, r in enumerate(rows) if r["split"] == "train"]
        sel_ids = [i for i, r in enumerate(rows) if r["split"] == "selection"]
        if (len(train_ids), len(sel_ids)) != (22680, 5120):
            raise SystemExit("split mismatch")
        peaks = gt[np.array(train_ids)].reshape(len(train_ids), -1).max(1)
        cls = np.zeros(len(train_ids), np.int64)
        cls[peaks >= 0.25] = CAUTION
        cls[peaks >= 0.60] = STOP
        counts = np.bincount(cls, minlength=3).astype(np.float64)
        inv = (len(train_ids) / np.maximum(counts, 1.0)) / 3.0
        sample_w = inv[cls]
        class_w_vec = inv / inv.mean()
        log(f"class counts SAFE/THREAT/STOP={counts.tolist()} sampler_inv={inv.tolist()}")

        rgb = s75.RGBCache256(RGB_ROOT)
        parent = ParentTargetCache(PARENT_ROOT)
        train_ds = s75.WinDS(rows, train_ids, gt, rgb, parent)
        sel_ds = s75.WinDS(rows, sel_ids, gt, rgb, parent)
        sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double), num_samples=len(train_ids), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, sampler=sampler, num_workers=args.num_workers, collate_fn=s75.collate, pin_memory=True
        )
        sel_loader = DataLoader(sel_ds, batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=s75.collate)

        device = resolve_device_safe(str(args.device))
        decoder = build_parent_decoder(cfg, device="cpu")
        dec_h = decoder_state_hash(decoder)
        if dec_h != EXPECTED_HASH:
            raise SystemExit(f"decoder hash {dec_h}")
        model = s75.StudentLatPairCF(decoder).to(device)
        blob = torch.load(EXP / "init_from_epoch38.pt", map_location=device, weights_only=False)
        model.encoder.load_state_dict(blob["encoder"])
        model.predictor.load_state_dict(blob["predictor"])
        for p in model.decoder.parameters():
            p.requires_grad_(False)
        model.decoder.eval()
        dec_h2 = decoder_state_hash(model.decoder)
        if dec_h2 != EXPECTED_HASH:
            raise SystemExit("decoder changed after load")
        st = np.load(STATS)
        mu = torch.tensor(st["mu"], dtype=torch.float32, device=device)
        sigma = torch.tensor(st["sigma"], dtype=torch.float32, device=device)
        enc_p = [p for p in model.encoder.parameters() if p.requires_grad]
        pred_p = [p for p in model.predictor.parameters() if p.requires_grad]
        params = enc_p + pred_p
        opt = torch.optim.AdamW(
            [{"params": enc_p, "lr": args.lr, "weight_decay": 1e-4}, {"params": pred_p, "lr": args.lr, "weight_decay": 1e-4}]
        )
        sch = build_scheduler(opt, args.epochs, max(args.lr * 0.05, 1e-7), warmup_epochs=2)
        amp = str(device).startswith("cuda")
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.nullcontext()
        feat_w, cos_w, mgd_w, rel_w = 1.0, 0.2, 0.5, 0.2
        cw_t = torch.tensor(class_w_vec, dtype=torch.float32, device=device)

        # gradient calibration of heatmap/peak vs L_feat (train batches, no param update)
        model.train()
        model.decoder.eval()
        g_store = defaultdict(list)
        ncal = 0
        for batch in train_loader:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with autocast:
                out = model(b["clip"])
                out["predictor_ref"] = model.predictor
                lf, lc, lr, lm = latent_terms(out, b, mu, sigma)
                labels = peak_class_t(b["gt_1s"])
                gw = cw_t[labels]
                lh, lp = task_losses(out["Hhat_1s"].float(), b["gt_1s"].float(), gw)
            g_store["feat"].append(gnorm(enc_p, lf.float()))
            g_store["h"].append(gnorm(enc_p, lh.float()))
            g_store["peak"].append(gnorm(enc_p, lp.float()))
            ncal += 1
            if ncal >= 4:
                break
        gf = float(np.median(g_store["feat"]) + 1e-8)
        gh = float(np.median(g_store["h"]) + 1e-8)
        gp = float(np.median(g_store["peak"]) + 1e-8)
        # heatmap ~ 0.35 * L_feat grad; peak ~ 0.20 * L_feat grad at full ramp
        lam_h = 0.35 * gf / gh
        lam_p = 0.20 * gf / gp
        calib = {"g_feat": gf, "g_h": gh, "g_peak": gp, "lam_h": lam_h, "lam_peak": lam_p}
        log(f"grad_calib {calib}")
        (EXP / "loss_calibration.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
        cfg_out = {
            "experiment": "A_task_aware_finetune",
            "init": str(SRC38),
            "init_sha256": src_hash,
            "ols_calibration": False,
            "parent_heatmap_kd": False,
            "learned_1x1": False,
            "decoder_hash": dec_h,
            "decoder_frozen": True,
            "grad_through_decoder": True,
            "lr": args.lr,
            "epochs": args.epochs,
            "latent_weights": {"feat": feat_w, "cos": cos_w, "mgd": mgd_w, "rel": rel_w},
            "lam_h": lam_h,
            "lam_peak": lam_p,
            "ramp": "heatmap/peak 0 at epoch 1, linear to 1 by epoch 5",
            "class_balanced_sampler": True,
            "class_loss_weights": class_w_vec.tolist(),
            "no_test": True,
        }
        (EXP / "config.json").write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")

        history = []
        best_rank = None
        best_ep = None
        decoder_hash_ok = True
        for ep in range(1, args.epochs + 1):
            r = ramp(ep, 1, 5)
            model.train()
            model.decoder.eval()
            t0 = time.perf_counter()
            acc = defaultdict(float)
            n_b = 0
            g_epoch = defaultdict(float)
            g_n = 0
            for step, batch in enumerate(train_loader):
                b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                with autocast:
                    out = model(b["clip"])
                    out["predictor_ref"] = model.predictor
                    lf, lc, lr_, lm = latent_terms(out, b, mu, sigma)
                    labels = peak_class_t(b["gt_1s"])
                    gw = cw_t[labels]
                    lh, lp = task_losses(out["Hhat_1s"].float(), b["gt_1s"].float(), gw)
                    l_lat = feat_w * lf + cos_w * lc + mgd_w * lm + rel_w * lr_
                    loss = l_lat + r * (lam_h * lh + lam_p * lp)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                acc["L_feat"] += float(lf.detach())
                acc["L_cos"] += float(lc.detach())
                acc["L_mgd"] += float(lm.detach())
                acc["L_rel"] += float(lr_.detach())
                acc["L_h"] += float(lh.detach())
                acc["L_peak"] += float(lp.detach())
                acc["L_lat"] += float(l_lat.detach())
                acc["L_total"] += float(loss.detach())
                n_b += 1
                if step == 0:
                    # per-term encoder grad magnitudes (extra forwards)
                    model.zero_grad(set_to_none=True)
                    with autocast:
                        out = model(b["clip"])
                        out["predictor_ref"] = model.predictor
                        lf, lc, lr_, lm = latent_terms(out, b, mu, sigma)
                        lh, lp = task_losses(out["Hhat_1s"].float(), b["gt_1s"].float(), cw_t[peak_class_t(b["gt_1s"])])
                    g_epoch["g_feat"] = gnorm(enc_p, lf.float())
                    g_epoch["g_cos"] = gnorm(enc_p, lc.float())
                    g_epoch["g_mgd"] = gnorm(enc_p, lm.float())
                    g_epoch["g_rel"] = gnorm(enc_p, lr_.float())
                    g_epoch["g_h"] = gnorm(enc_p, lh.float())
                    g_epoch["g_peak"] = gnorm(enc_p, lp.float())
                    g_n = 1
                    opt.zero_grad(set_to_none=True)
            if sch is not None:
                sch.step()
            if decoder_state_hash(model.decoder) != EXPECTED_HASH:
                decoder_hash_ok = False
                raise SystemExit("decoder hash drifted")
            train_m = {k: v / max(n_b, 1) for k, v in acc.items()}
            train_m.update({"ramp": r, "sec": time.perf_counter() - t0, **g_epoch})
            sel_m = eval_sel(model, sel_loader, device, amp)
            rec = {"epoch": ep, "train": train_m, "selection": {k: sel_m[k] for k in sel_m if k != "operating"}, "operating": sel_m.get("operating")}
            (EXP / f"metrics_epoch_{ep:02d}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
            ck = {
                "epoch": ep,
                "encoder": model.encoder.state_dict(),
                "predictor": model.predictor.state_dict(),
                "decoder": model.decoder.state_dict(),
                "from": "epoch38",
                "src_sha256": src_hash,
            }
            torch.save(ck, EXP / f"checkpoint_epoch_{ep:02d}.pt")
            rk = s75.rank_checkpoint(sel_m)
            if best_rank is None or rk > best_rank:
                best_rank, best_ep = rk, ep
                torch.save({**ck, "selection": sel_m}, EXP / "checkpoint_best.pt")
            history.append(
                {
                    "epoch": ep,
                    "ramp": r,
                    "L_total": train_m["L_total"],
                    "L_feat": train_m["L_feat"],
                    "L_h": train_m["L_h"],
                    "L_peak": train_m["L_peak"],
                    "g_feat": train_m.get("g_feat"),
                    "g_h": train_m.get("g_h"),
                    "g_peak": train_m.get("g_peak"),
                    "SAFE": sel_m["safe_recall"],
                    "THREAT": sel_m["threat_recall"],
                    "STOP": sel_m["stop_recall"],
                    "STOP_prec": sel_m["stop_precision"],
                    "pre": sel_m["pre_fork_stop_recall"],
                    "post": sel_m["post_fork_stop_recall"],
                    "cos": sel_m["fdec_cosine"],
                    "stop_peak_mean": sel_m["peaks"]["gt_stop"]["mean"],
                }
            )
            log(
                f"epoch {ep} ramp={r:.2f} L={train_m['L_total']:.4f} feat={train_m['L_feat']:.4f} "
                f"H={train_m['L_h']:.4f} peak={train_m['L_peak']:.4f} "
                f"gF={train_m.get('g_feat'):.3g} gH={train_m.get('g_h'):.3g} gP={train_m.get('g_peak'):.3g} "
                f"SAFE={sel_m['safe_recall']:.3f} THR={sel_m['threat_recall']:.3f} STOP={sel_m['stop_recall']:.3f} "
                f"prec={sel_m['stop_precision']:.3f} pre={sel_m['pre_fork_stop_recall']:.3f} post={sel_m['post_fork_stop_recall']:.3f} "
                f"cos={sel_m['fdec_cosine']:.3f} sec={train_m['sec']:.1f}"
            )
        (EXP / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (EXP / "best.json").write_text(json.dumps({"best_epoch": best_ep, "rank": list(best_rank)}, indent=2), encoding="utf-8")
        assert sha256(SRC38) == src_hash
        log(f"DONE best_epoch={best_ep} decoder_ok={decoder_hash_ok} source38_unmodified=True")
        (EXP / ".success").write_text("ok\n")
    except Exception:
        traceback.print_exc()
        (EXP / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
