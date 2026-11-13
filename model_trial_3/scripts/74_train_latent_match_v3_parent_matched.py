#!/usr/bin/env python3
"""Train LatentMatch V3 on Pair-CF parent-matched 16-frame caches (RAID v2)."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
TEMP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp")
RAID = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_3")
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.preprocessing import uint8_to_nchw01  # noqa: E402
from collision_jepa3.student_research.cache_access import OFFSETS_16, ParentTargetCache, RGBCache  # noqa: E402
from collision_jepa3.student_research.fdec_student import build_parent_decoder, decoder_state_hash  # noqa: E402
from collision_jepa3.student_research.latent_match_v3 import StudentLatentMatchV3  # noqa: E402
from collision_jepa3.student_research.student_final.eval import (  # noqa: E402
    confusion_and_rates,
    forward_collect,
    search_cutoffs,
)
from collision_jepa3.student_research.student_final.losses import weighted_ordinal_loss  # noqa: E402
from collision_jepa3.student_research.student_final.train import adamw_param_groups  # noqa: E402
from collision_jepa3.utils.device import resolve_device_safe  # noqa: E402

EXP = TEMP / "student_latent_match_v3_parent_matched"
RGB_ROOT = RAID / "shared_rgb_cache_v2"
PARENT_ROOT = RAID / "parent_target_cache_v2"
EXPECTED_HASH = "77f78c8714667617"


def schedule_weights(epoch: int, lam_gt: float, lam_f: float, lam_kd: float) -> tuple[float, float, float]:
    # 1-indexed epochs: 1-5 latent, 6-10 ramp, 11-20 full
    if epoch <= 5:
        return 0.0, lam_f, 0.0
    if epoch <= 10:
        a = (epoch - 5) / 5.0
        return a * lam_gt, lam_f, a * lam_kd
    return lam_gt, lam_f, lam_kd


class V3Dataset(Dataset):
    def __init__(self, rows, row_ids, gt, rgb, parent, group_w):
        self.rows = rows
        self.row_ids = list(row_ids)
        self.gt = gt
        self.rgb = rgb
        self.parent = parent
        self.group_w = group_w

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, i):
        rid = int(self.row_ids[i])
        row = self.rows[rid]
        cid, t = row["clip_id"], int(row["frame"])
        frames = self.rgb.get_frames(cid, t, OFFSETS_16)
        if frames is None:
            raise KeyError(f"missing rgb {cid} {t}")
        parent = self.parent.get_parent_targets(cid, t)
        gt = np.asarray(self.gt[rid], dtype=np.float32)
        return {
            "clip": torch.from_numpy(uint8_to_nchw01(frames)),
            "gt_1s": torch.from_numpy(gt),
            "p_gt": float(gt.max()),
            "pre_fork": bool(row.get("pre_fork")),
            "amb_1s": bool(row.get("amb_1s")),
            "sample_id": row.get("sample_id", ""),
            "has_parent": parent is not None,
            "group_weight": float(self.group_w.get(str(row.get("sample_id", "")), 1.0)),
            "F_teacher": torch.from_numpy(parent["F_dec"]) if parent else torch.zeros(128, 16, 16),
            "H_teacher": torch.from_numpy(parent["H_teacher"]) if parent else torch.zeros(5, 5),
        }


def collate(batch):
    out = {
        "clip": torch.stack([b["clip"] for b in batch]),
        "gt_1s": torch.stack([b["gt_1s"] for b in batch]),
        "p_gt": torch.tensor([b["p_gt"] for b in batch], dtype=torch.float32),
        "pre_fork": torch.tensor([b["pre_fork"] for b in batch], dtype=torch.bool),
        "amb_1s": torch.tensor([b["amb_1s"] for b in batch], dtype=torch.bool),
        "sample_id": [b["sample_id"] for b in batch],
        "has_parent": torch.tensor([b["has_parent"] for b in batch], dtype=torch.bool),
        "group_weight": torch.tensor([b["group_weight"] for b in batch], dtype=torch.float32),
        "F_teacher": torch.stack([b["F_teacher"] for b in batch]),
        "H_teacher": torch.stack([b["H_teacher"] for b in batch]),
    }
    gw = out["group_weight"]
    out["group_weight"] = gw * (len(batch) / gw.sum().clamp_min(1e-6))
    return out


def v3_loss(out, batch, *, lam_gt, lam_f, lam_kd, f_mu, f_sigma, w_bar):
    gw = batch["group_weight"]
    l_ord, terms = weighted_ordinal_loss(out["q_warn"], out["q_stop"], batch["p_gt"], gw)
    huber = F.smooth_l1_loss(out["Hhat_1s"], batch["gt_1s"], reduction="none", beta=0.05)
    w_ij = (0.05 + 4.0 * batch["gt_1s"] ** 2) / max(w_bar, 1e-6)
    l_gt = (huber * w_ij * gw.view(-1, 1, 1)).mean()
    l_f = torch.tensor(0.0, device=l_ord.device)
    l_kd = torch.tensor(0.0, device=l_ord.device)
    mask = batch["has_parent"]
    if mask.any():
        fs = out["F_student"][mask]
        ft = batch["F_teacher"][mask]
        l_f = F.smooth_l1_loss((fs - f_mu) / (f_sigma + 1e-6), (ft - f_mu) / (f_sigma + 1e-6))
        l_kd = (
            F.smooth_l1_loss(out["Hhat_1s"][mask], batch["H_teacher"][mask], reduction="none", beta=0.1)
            * gw[mask].view(-1, 1, 1)
        ).mean()
    total = l_ord + lam_gt * l_gt + lam_f * l_f + lam_kd * l_kd
    terms.update({"L_gt": float(l_gt.detach()), "L_fdec": float(l_f.detach()), "L_kd": float(l_kd.detach()), "L_total": float(total.detach())})
    return total, terms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda:6")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = load_cfg(args)
    seed = int(args.seed if args.seed is not None else 1337)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = resolve_device_safe(str(args.device))
    EXP.mkdir(parents=True, exist_ok=True)
    log_path = EXP / "run.log"

    def log(m: str) -> None:
        print(m, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(m + "\n")

    try:
        if not (RGB_ROOT / "COMPLETE").is_file() or not (PARENT_ROOT / "COMPLETE").is_file():
            raise SystemExit("caches not COMPLETE")
        rows = [json.loads(l) for l in (PARENT_ROOT / "index.jsonl").open(encoding="utf-8")]
        gt = np.load(PARENT_ROOT / "gt_1s.npy").astype(np.float32)
        train_ids = [i for i, r in enumerate(rows) if r["split"] == "train"]
        sel_ids = [i for i, r in enumerate(rows) if r["split"] == "selection"]
        test_ids = [i for i, r in enumerate(rows) if r["split"] == "test"]
        log(f"windows train={len(train_ids)} sel={len(sel_ids)} test={len(test_ids)}")
        if (len(train_ids), len(sel_ids), len(test_ids)) != (22680, 5120, 5120):
            raise SystemExit("window counts != parent 22680/5120/5120")

        rgb = RGBCache(RGB_ROOT)
        parent = ParentTargetCache(PARENT_ROOT)
        train_gt = gt[np.array(train_ids)]
        w_bar = float(np.mean(0.05 + 4.0 * train_gt**2))
        counts: dict[str, int] = defaultdict(int)
        for rid in train_ids:
            counts[str(rows[rid]["sample_id"])] += 1
        group_w = {s: 1.0 / max(1, n) for s, n in counts.items()}

        f_list = []
        rng = np.random.default_rng(seed)
        for rid in rng.choice(train_ids, size=min(4000, len(train_ids)), replace=False):
            pt = parent.get_parent_targets(rows[int(rid)]["clip_id"], int(rows[int(rid)]["frame"]))
            if pt:
                f_list.append(pt["F_dec"])
        arr = np.stack(f_list, 0)
        f_mu = torch.tensor(arr.mean(axis=(0, 2, 3), keepdims=True), dtype=torch.float32, device=device)
        f_sigma = torch.tensor(arr.std(axis=(0, 2, 3), keepdims=True) + 1e-6, dtype=torch.float32, device=device)

        decoder = build_parent_decoder(cfg, device="cpu")
        dec_h = decoder_state_hash(decoder)
        log(f"decoder_hash={dec_h}")
        if dec_h != EXPECTED_HASH:
            raise SystemExit(f"decoder hash mismatch {dec_h}")
        torch.manual_seed(seed)
        model = StudentLatentMatchV3(decoder, n_frames=16).to(device)
        audit_p = model.param_audit()
        log(f"params {audit_p}")
        (EXP / "param_audit.json").write_text(json.dumps(audit_p, indent=2), encoding="utf-8")
        torch.save({"model": model.state_dict(), "param_audit": audit_p}, EXP / "checkpoint_initial.pt")

        ds_kw = dict(rows=rows, gt=gt, rgb=rgb, parent=parent, group_w=group_w)
        train_ds = V3Dataset(row_ids=train_ids, **ds_kw)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, collate_fn=collate, pin_memory=True)

        trunk = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder")]

        def gnorm(loss):
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            s = 0.0
            for p in trunk:
                if p.grad is not None:
                    s += float(p.grad.detach().pow(2).sum())
            return s**0.5

        lams = []
        n_cal = 0
        for batch in train_loader:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(b["clip"])
            l_ord, _ = weighted_ordinal_loss(out["q_warn"], out["q_stop"], b["p_gt"], b["group_weight"])
            go = gnorm(l_ord)
            out = model(b["clip"])
            huber = F.smooth_l1_loss(out["Hhat_1s"], b["gt_1s"], reduction="none", beta=0.05)
            w_ij = (0.05 + 4.0 * b["gt_1s"] ** 2) / max(w_bar, 1e-6)
            l_gt = (huber * w_ij * b["group_weight"].view(-1, 1, 1)).mean()
            gg = gnorm(l_gt)
            out = model(b["clip"])
            mask = b["has_parent"]
            gf = gk = 0.0
            if mask.any():
                fs, ft = out["F_student"][mask], b["F_teacher"][mask]
                l_f = F.smooth_l1_loss((fs - f_mu) / (f_sigma + 1e-6), (ft - f_mu) / (f_sigma + 1e-6))
                gf = gnorm(l_f)
                out = model(b["clip"])
                l_kd = (
                    F.smooth_l1_loss(out["Hhat_1s"][mask], b["H_teacher"][mask], reduction="none", beta=0.1)
                    * b["group_weight"][mask].view(-1, 1, 1)
                ).mean()
                gk = gnorm(l_kd)
            lams.append((go / (gg + 1e-8), 0.25 * go / (gf + 1e-8) if gf > 0 else 0.0, 0.10 * go / (gk + 1e-8) if gk > 0 else 0.0))
            n_cal += 1
            if n_cal >= 4:
                break
        lam_gt = float(np.median([x[0] for x in lams]))
        lam_f = float(np.median([x[1] for x in lams]))
        lam_kd = float(np.median([x[2] for x in lams]))
        log(f"calibration lam_gt={lam_gt} lam_f={lam_f} lam_kd={lam_kd} w_bar={w_bar}")
        (EXP / "config.json").write_text(
            json.dumps(
                {
                    "epochs": args.epochs,
                    "batch": args.batch,
                    "lr": args.lr,
                    "n_train": len(train_ids),
                    "n_sel": len(sel_ids),
                    "n_test": len(test_ids),
                    "lambda_gt": lam_gt,
                    "lambda_f": lam_f,
                    "lambda_kd": lam_kd,
                    "w_bar": w_bar,
                    "decoder_hash": dec_h,
                    "rgb": str(RGB_ROOT),
                    "parent": str(PARENT_ROOT),
                    "frames_per_window": 16,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        decay, no_decay = adamw_param_groups(model)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        def eval_ids(ids):
            ds = V3Dataset(row_ids=ids, **ds_kw)
            loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=collate)
            return forward_collect(model, loader, device)["rows"]

        best_ep, best_cut, best_score = None, None, -1.0
        smooth = []
        for ep in range(1, args.epochs + 1):
            sg, sf, sk = schedule_weights(ep, lam_gt, lam_f, lam_kd)
            t0 = time.perf_counter()
            model.train()
            acc = defaultdict(float)
            n_b = 0
            for batch in train_loader:
                b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                out = model(b["clip"])
                loss, terms = v3_loss(out, b, lam_gt=sg, lam_f=sf, lam_kd=sk, f_mu=f_mu, f_sigma=f_sigma, w_bar=w_bar)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                for k, v in terms.items():
                    acc[k] += float(v)
                n_b += 1
            sched.step()
            train_m = {k: v / max(1, n_b) for k, v in acc.items()}
            train_m.update({"sec": time.perf_counter() - t0, "lam_gt": sg, "lam_f": sf, "lam_kd": sk})
            (EXP / f"metrics_train_epoch_{ep:02d}.json").write_text(json.dumps(train_m, indent=2), encoding="utf-8")
            torch.save({"epoch": ep, "model": model.state_dict()}, EXP / f"checkpoint_epoch_{ep:02d}.pt")
            sel_rows = eval_ids(sel_ids)
            cut = search_cutoffs(sel_rows, parent_safe_floor=0.84)
            score = float(cut.get("selection_score", float("nan"))) if cut.get("feasible") else float("nan")
            smooth.append(score)
            ma = float(np.nanmean(smooth[-3:]))
            (EXP / f"metrics_selection_epoch_{ep:02d}.json").write_text(
                json.dumps({"epoch": ep, "selection": cut, "ma": ma}, indent=2), encoding="utf-8"
            )
            log(f"epoch {ep} L={train_m.get('L_total')} lam=({sg:.3g},{sf:.3g},{sk:.3g}) sel={cut.get('feasible')} ma={ma:.4f} sec={train_m['sec']:.1f}")
            if ep >= 5 and cut.get("feasible") and ma > best_score:
                best_score, best_ep, best_cut = ma, ep, {"c_warn": cut["c_warn"], "c_stop": cut["c_stop"]}

        if best_ep is None:
            best_ep = args.epochs
            best_cut = {"c_warn": 0.5, "c_stop": 0.55}
        torch.save(torch.load(EXP / f"checkpoint_epoch_{best_ep:02d}.pt", map_location="cpu"), EXP / "checkpoint_final.pt")
        (EXP / "selection_cutoffs.json").write_text(json.dumps({"epoch": best_ep, "score": best_score, **best_cut}, indent=2), encoding="utf-8")
        blob = torch.load(EXP / "checkpoint_final.pt", map_location=device)
        model.load_state_dict(blob["model"])
        model.eval()
        test_rows = eval_ids(test_ids)
        cos_vals = []
        ds = V3Dataset(row_ids=test_ids, **ds_kw)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate)
        with torch.no_grad():
            for batch in loader:
                b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                out = model(b["clip"])
                m = b["has_parent"]
                if m.any():
                    cos_vals.append(F.cosine_similarity(out["F_student"][m].flatten(1), b["F_teacher"][m].flatten(1)).cpu().numpy())
        mean_cos = float(np.concatenate(cos_vals).mean()) if cos_vals else float("nan")
        test_rep = confusion_and_rates(test_rows, c_warn=best_cut["c_warn"], c_stop=best_cut["c_stop"])
        mse = float(np.mean([np.mean((r["h_pred"] - r["h_gt"]) ** 2) for r in test_rows]))
        final = {
            "selected_epoch": best_ep,
            "cutoffs": best_cut,
            "test": test_rep,
            "heatmap_mse": mse,
            "fdec_cosine_test": mean_cos,
            "disclaimer": "F_dec cosine is a latent match score, not STOP recall. STOP is predicted occupancy ~1s ahead on synthetic data.",
        }
        (EXP / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        log(f"DONE epoch={best_ep} stop={test_rep['metrics'].get('stop_recall')} safe={test_rep['metrics'].get('safe_recall')} cos={mean_cos}")
        (EXP / ".success").write_text("ok\n", encoding="utf-8")
    except Exception:
        traceback.print_exc()
        (EXP / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
