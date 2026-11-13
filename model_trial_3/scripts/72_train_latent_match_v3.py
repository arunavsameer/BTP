#!/usr/bin/env python3
"""Train LatentMatch V3 from scratch (frozen parent decoder + ordinal head)."""

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
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.preprocessing import uint8_to_nchw01  # noqa: E402
from collision_jepa3.student_research.cache_access import OFFSETS_5, ParentTargetCache, RGBCache  # noqa: E402
from collision_jepa3.student_research.fdec_pilot_dataset import SparseClipFrameStore, build_frame_store  # noqa: E402
from collision_jepa3.student_research.fdec_student import build_parent_decoder, decoder_state_hash  # noqa: E402
from collision_jepa3.student_research.frames import indices_at_anchor  # noqa: E402
from collision_jepa3.student_research.latent_match_v3 import StudentLatentMatchV3  # noqa: E402
from collision_jepa3.student_research.student_final.eval import (  # noqa: E402
    confusion_and_rates,
    forward_collect,
    search_cutoffs,
)
from collision_jepa3.student_research.student_final.losses import (  # noqa: E402
    spatial_gt_loss,
    teacher_kd_loss,
    weighted_ordinal_loss,
)
from collision_jepa3.student_research.student_final.splits import build_data_splits  # noqa: E402
from collision_jepa3.student_research.student_final.targets import build_group_mean_gt_array  # noqa: E402
from collision_jepa3.student_research.student_final.train import adamw_param_groups, iter_epoch_batches  # noqa: E402
from collision_jepa3.utils.device import resolve_device_safe  # noqa: E402

EXP = TEMP / "student_latent_match_v3"
CACHE_TEACHER = ROOT / "reports/student_architecture_tests/teacher_cache_train_pilot"
ARCH_REPORT = ROOT / "reports/student_architecture_tests"
PARENT_CKPT = ROOT / "checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt"
EXPECTED_HASH = "77f78c8714667617"


class V3Dataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        row_ids: list[int],
        gt: np.ndarray,
        rgb: RGBCache,
        parent: ParentTargetCache,
        store: SparseClipFrameStore | None,
        *,
        w_bar: float,
        group_w: dict[str, float],
    ):
        self.rows = rows
        self.row_ids = list(row_ids)
        self.gt = gt
        self.rgb = rgb
        self.parent = parent
        self.store = store
        self.w_bar = float(w_bar)
        self.group_w = group_w

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, i: int) -> dict:
        rid = int(self.row_ids[i])
        row = self.rows[rid]
        cid = row["clip_id"]
        t = int(row["frame"])
        frames = self.rgb.get_frames(cid, t, OFFSETS_5)
        if frames is None:
            if self.store is None:
                raise KeyError(cid)
            idxs = indices_at_anchor(t, OFFSETS_5)
            frames = np.stack([np.asarray(self.store.get(cid, fi)) for fi in idxs], axis=0)
        clip = uint8_to_nchw01(frames)
        gt = np.asarray(self.gt[rid], dtype=np.float32)
        parent = self.parent.get_parent_targets(cid, t)
        has_p = parent is not None
        sample = {
            "clip": torch.from_numpy(clip),
            "gt_1s": torch.from_numpy(gt),
            "p_gt": float(gt.max()),
            "pre_fork": bool(row.get("pre_fork")),
            "amb_1s": bool(row.get("amb_1s")),
            "sample_id": row.get("sample_id", ""),
            "row_id": rid,
            "has_parent": has_p,
            "group_weight": float(self.group_w.get(str(row.get("sample_id", "")), 1.0)),
            "F_teacher": torch.from_numpy(parent["F_dec"]) if has_p else torch.zeros(128, 16, 16),
            "H_teacher": torch.from_numpy(parent["H_teacher"]) if has_p else torch.zeros(5, 5),
        }
        return sample


def collate(batch: list[dict]) -> dict:
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
        fsn = (fs - f_mu) / (f_sigma + 1e-6)
        ftn = (ft - f_mu) / (f_sigma + 1e-6)
        l_f = F.smooth_l1_loss(fsn, ftn)
        l_kd = (
            F.smooth_l1_loss(out["Hhat_1s"][mask], batch["H_teacher"][mask], reduction="none", beta=0.1)
            * gw[mask].view(-1, 1, 1)
        ).mean()
    total = l_ord + lam_gt * l_gt + lam_f * l_f + lam_kd * l_kd
    terms.update(
        {
            "L_gt": float(l_gt.detach()),
            "L_fdec": float(l_f.detach()),
            "L_kd": float(l_kd.detach()),
            "L_total": float(total.detach()),
        }
    )
    return total, terms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
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
        rows = [json.loads(l) for l in (CACHE_TEACHER / "index.jsonl").open(encoding="utf-8")]
        with np.load(CACHE_TEACHER / "gt_heatmaps.npz") as z:
            raw_gt = np.asarray(z["gt_1s"])
        gt, gt_meta = build_group_mean_gt_array(cfg, rows, raw_gt=raw_gt)
        log(f"group_mean_gt {gt_meta}")
        audit = build_data_splits(cfg, CACHE_TEACHER, ARCH_REPORT, gt, seed=seed)
        train_ids, sel_ids, dev_ids = audit["train_row_ids"], audit["selection_row_ids"], audit["dev_row_ids"]
        log(f"splits train={len(train_ids)} sel={len(sel_ids)} dev={len(dev_ids)}")

        rgb = RGBCache()
        parent = ParentTargetCache()
        need_store = []
        for rid in train_ids + sel_ids + dev_ids:
            cid = rows[rid]["clip_id"]
            t = int(rows[rid]["frame"])
            if rgb.get_frames(cid, t, OFFSETS_5) is None:
                need_store.append(rid)
        store = None
        if need_store:
            store, prep = build_frame_store(cfg, CACHE_TEACHER, sorted(set(need_store)), OFFSETS_5, prepare_workers=2)
            log(f"rgb fallback decode {prep}")
        else:
            log("all windows served from shared_rgb_cache_v1")

        # drop windows we still cannot load
        def usable(ids):
            keep = []
            for rid in ids:
                cid, t = rows[rid]["clip_id"], int(rows[rid]["frame"])
                if rgb.get_frames(cid, t, OFFSETS_5) is not None:
                    keep.append(rid)
                elif store is not None:
                    try:
                        for fi in indices_at_anchor(t, OFFSETS_5):
                            store.get(cid, fi)
                        keep.append(rid)
                    except Exception:
                        pass
            return keep

        train_ids, sel_ids, dev_ids = usable(train_ids), usable(sel_ids), usable(dev_ids)
        log(f"usable train={len(train_ids)} sel={len(sel_ids)} dev={len(dev_ids)}")

        train_gt = gt[np.array(train_ids)]
        w_bar = float(np.mean(0.05 + 4.0 * train_gt**2))
        counts: dict[str, int] = defaultdict(int)
        for rid in train_ids:
            counts[str(rows[rid]["sample_id"])] += 1
        group_w = {s: 1.0 / max(1, n) for s, n in counts.items()}

        f_list = []
        for rid in train_ids:
            pt = parent.get_parent_targets(rows[rid]["clip_id"], int(rows[rid]["frame"]))
            if pt:
                f_list.append(pt["F_dec"])
        if f_list:
            arr = np.stack(f_list, 0)
            f_mu = torch.tensor(arr.mean(axis=(0, 2, 3), keepdims=True), dtype=torch.float32, device=device)
            f_sigma = torch.tensor(arr.std(axis=(0, 2, 3), keepdims=True) + 1e-6, dtype=torch.float32, device=device)
        else:
            f_mu = torch.zeros(1, 128, 1, 1, device=device)
            f_sigma = torch.ones(1, 128, 1, 1, device=device)

        decoder = build_parent_decoder(cfg, device="cpu")
        dec_h = decoder_state_hash(decoder)
        log(f"decoder_hash={dec_h} expected={EXPECTED_HASH}")
        if dec_h != EXPECTED_HASH:
            raise SystemExit(f"decoder hash mismatch {dec_h}")
        model = StudentLatentMatchV3(decoder).to(device)
        audit_p = model.param_audit()
        log(f"params {audit_p}")
        if not audit_p["under_750k"]:
            raise SystemExit("param budget exceeded")
        (EXP / "param_audit.json").write_text(json.dumps(audit_p, indent=2), encoding="utf-8")
        torch.save({"model": model.state_dict(), "param_audit": audit_p}, EXP / "checkpoint_initial.pt")

        ds_kw = dict(rows=rows, gt=gt, rgb=rgb, parent=parent, store=store, w_bar=w_bar, group_w=group_w)
        train_ds = V3Dataset(row_ids=train_ids, **ds_kw)

        # calibrate on 4 batches
        model.train()
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
        for bi in iter_epoch_batches(train_ids, rows, batch_size=args.batch, seed=seed, epoch=0):
            batch = collate([train_ds[i] for i in bi[: args.batch]])
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(b["clip"])
            l_ord, _ = weighted_ordinal_loss(out["q_warn"], out["q_stop"], b["p_gt"], b["group_weight"])
            go = gnorm(l_ord)
            out = model(b["clip"])
            lg, _ = v3_loss(out, b, lam_gt=1, lam_f=0, lam_kd=0, f_mu=f_mu, f_sigma=f_sigma, w_bar=w_bar)
            # isolate GT
            model.zero_grad(set_to_none=True)
            out = model(b["clip"])
            huber = F.smooth_l1_loss(out["Hhat_1s"], b["gt_1s"], reduction="none", beta=0.05)
            w_ij = (0.05 + 4.0 * b["gt_1s"] ** 2) / max(w_bar, 1e-6)
            l_gt = (huber * w_ij * b["group_weight"].view(-1, 1, 1)).mean()
            gg = gnorm(l_gt)
            out = model(b["clip"])
            mask = b["has_parent"]
            gf = gk = 0.0
            if mask.any():
                fs = out["F_student"][mask]
                ft = b["F_teacher"][mask]
                l_f = F.smooth_l1_loss((fs - f_mu) / (f_sigma + 1e-6), (ft - f_mu) / (f_sigma + 1e-6))
                gf = gnorm(l_f)
                out = model(b["clip"])
                l_kd = (
                    F.smooth_l1_loss(out["Hhat_1s"][mask], b["H_teacher"][mask], reduction="none", beta=0.1)
                    * b["group_weight"][mask].view(-1, 1, 1)
                ).mean()
                gk = gnorm(l_kd)
            lam_gt = go / (gg + 1e-8)
            lam_f = 0.25 * go / (gf + 1e-8) if gf > 0 else 0.0
            lam_kd = 0.10 * go / (gk + 1e-8) if gk > 0 else 0.0
            lams.append((lam_gt, lam_f, lam_kd, go, gg, gf, gk))
            n_cal += 1
            if n_cal >= 4:
                break
        lam_gt = float(np.median([x[0] for x in lams]))
        lam_f = float(np.median([x[1] for x in lams]))
        lam_kd = float(np.median([x[2] for x in lams]))
        calib = {"lambda_gt": lam_gt, "lambda_f": lam_f, "lambda_kd": lam_kd, "raw": lams, "w_bar": w_bar}
        log(f"calibration {calib}")
        (EXP / "config.json").write_text(
            json.dumps(
                {
                    "epochs": args.epochs,
                    "batch": args.batch,
                    "lr": args.lr,
                    "seed": seed,
                    "device": device,
                    "n_train": len(train_ids),
                    "n_sel": len(sel_ids),
                    "n_dev": len(dev_ids),
                    **{k: calib[k] for k in ("lambda_gt", "lambda_f", "lambda_kd", "w_bar")},
                    "decoder_hash": dec_h,
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

        warning = cfg.get("warning") or {}
        # parent SAFE floor on selection via H_teacher peaks when available else skip 0.02 slack on 0.86
        parent_safe_floor = 0.84

        best_ep, best_cut, best_score = None, None, -1.0
        smooth = []

        def eval_ids(ids):
            ds = V3Dataset(row_ids=ids, **ds_kw)
            loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate)
            return forward_collect(model, loader, device)["rows"]

        for ep in range(1, args.epochs + 1):
            t0 = time.perf_counter()
            model.train()
            acc = defaultdict(float)
            n_b = 0
            for bi in iter_epoch_batches(train_ids, rows, batch_size=args.batch, seed=seed, epoch=ep):
                batch = collate([train_ds[i] for i in bi])
                b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                out = model(b["clip"])
                loss, terms = v3_loss(
                    out, b, lam_gt=lam_gt, lam_f=lam_f, lam_kd=lam_kd, f_mu=f_mu, f_sigma=f_sigma, w_bar=w_bar
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                for k, v in terms.items():
                    acc[k] += float(v)
                n_b += 1
            sched.step()
            train_m = {k: v / max(1, n_b) for k, v in acc.items()}
            train_m["sec"] = time.perf_counter() - t0
            (EXP / f"metrics_train_epoch_{ep:02d}.json").write_text(json.dumps(train_m, indent=2), encoding="utf-8")
            torch.save({"epoch": ep, "model": model.state_dict()}, EXP / f"checkpoint_epoch_{ep:02d}.pt")

            sel_rows = eval_ids(sel_ids)
            cut = search_cutoffs(sel_rows, parent_safe_floor=parent_safe_floor)
            score = float(cut.get("selection_score", float("nan"))) if cut.get("feasible") else float("nan")
            smooth.append(score)
            ma = float(np.nanmean(smooth[-3:]))
            (EXP / f"metrics_selection_epoch_{ep:02d}.json").write_text(
                json.dumps({"epoch": ep, "selection": cut, "ma": ma}, indent=2), encoding="utf-8"
            )
            log(f"epoch {ep} train_L={train_m.get('L_total')} sel_feasible={cut.get('feasible')} ma={ma:.4f} sec={train_m['sec']:.1f}")
            if ep >= 5 and cut.get("feasible") and ma > best_score:
                best_score, best_ep, best_cut = ma, ep, {"c_warn": cut["c_warn"], "c_stop": cut["c_stop"]}

        if best_ep is None:
            best_ep = args.epochs
            best_cut = {"c_warn": 0.5, "c_stop": 0.55}
        torch.save(torch.load(EXP / f"checkpoint_epoch_{best_ep:02d}.pt", map_location="cpu"), EXP / "checkpoint_final.pt")
        (EXP / "selection_cutoffs.json").write_text(
            json.dumps({"epoch": best_ep, "score": best_score, **best_cut}, indent=2), encoding="utf-8"
        )
        blob = torch.load(EXP / "checkpoint_final.pt", map_location=device)
        model.load_state_dict(blob["model"])
        model.eval()
        dev_rows = eval_ids(dev_ids)
        # extra: fdec cosine where parent exists
        cos_vals = []
        ds = V3Dataset(row_ids=dev_ids, **ds_kw)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate)
        with torch.no_grad():
            for batch in loader:
                b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                out = model(b["clip"])
                m = b["has_parent"]
                if m.any():
                    a = out["F_student"][m].flatten(1)
                    t = b["F_teacher"][m].flatten(1)
                    cos_vals.append(F.cosine_similarity(a, t).cpu().numpy())
        mean_cos = float(np.concatenate(cos_vals).mean()) if cos_vals else float("nan")
        dev_rep = confusion_and_rates(dev_rows, c_warn=best_cut["c_warn"], c_stop=best_cut["c_stop"])
        mse = float(np.mean([np.mean((r["h_pred"] - r["h_gt"]) ** 2) for r in dev_rows]))
        final = {
            "selected_epoch": best_ep,
            "cutoffs": best_cut,
            "dev": dev_rep,
            "heatmap_mse": mse,
            "fdec_cosine_dev": mean_cos,
            "disclaimer": "STOP is predicted occupancy ~1s ahead on synthetic data, not guaranteed collision.",
        }
        (EXP / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        log(f"DONE epoch={best_ep} post_fork_stop={dev_rep['metrics'].get('post_fork_stop_recall')} stop={dev_rep['metrics'].get('stop_recall')} safe={dev_rep['metrics'].get('safe_recall')} cos={mean_cos}")
        (EXP / ".success").write_text("ok\n", encoding="utf-8")
    except Exception:
        traceback.print_exc()
        (EXP / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
