#!/usr/bin/env python3
"""Train CollisionStudentV5 (V4 gated backbone, V5 loss + decoder unfreeze).

Isolated from V3/V4 jobs. Architecture is CollisionStudentV4. Training recipe:
L_feat cosine 0.50→0.10 from epoch 20, HeatmapDecoder unfreezes at epoch 21
(0.15× backbone LR, 2-epoch ramp), QFL 5×5 + hazard-weighted 16×16.

  python scripts/16_train_student_v5.py --overfit --device cpu
  python scripts/16_train_student_v5.py --epochs 100 --batch 16 --accum 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa4.utils.device import resolve_device, select_amp_dtype  # noqa: E402
from collision_jepa4.config import Config  # noqa: E402
from student.caches import ParentTargetCache, RGBCache256, SpatialHeatmaps  # noqa: E402
from student.dataset import WinDS, collate  # noqa: E402
from student.metrics import collect, rank_checkpoint  # noqa: E402
from student.parent import decoder_state_hash, load_parent_decoder  # noqa: E402
from student.paths import parent_cache, parent_ckpt, rgb_cache, run_dir, z16_cache  # noqa: E402
from student.v2.data import augment_batch, hazard_sample_weights, weighted_sampler  # noqa: E402
from student.v5.losses import get_lambda_feat_v5, student_v5_loss  # noqa: E402
from student.v5.model import (  # noqa: E402
    DECAY_START_EPOCH,
    CollisionStudentV5,
    decoder_lr,
)


def resolve_device_safe(pref: str) -> str:
    device = resolve_device(pref)
    if pref == "cpu":
        return "cpu"
    if not str(device).startswith("cuda"):
        raise SystemExit(f"need cuda, got {device} (pref={pref})")
    last = None
    for attempt in range(12):
        try:
            torch.zeros(1, device=device)
            torch.cuda.synchronize()
            return device
        except Exception as exc:
            last = exc
            print(f"cuda not ready ({attempt + 1}/12): {exc}", flush=True)
            time.sleep(5)
    raise SystemExit(f"cuda unusable ({pref} -> {device}): {last}")


def warmup_cosine(step: int, warmup: int, total: int, base: float, min_lr: float) -> float:
    if step < warmup:
        return (1e-5 + (base - 1e-5) * step / max(warmup, 1)) / base
    progress = (step - warmup) / max(total - warmup, 1)
    lr = min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return lr / base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-name", default="student_v5_unfreeze_v1")
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--feat-w", type=float, default=0.5)
    parser.add_argument("--feat-decay-start", type=int, default=20)
    parser.add_argument("--feat-decay-duration", type=int, default=25)
    parser.add_argument("--feat-w-end", type=float, default=0.10)
    parser.add_argument("--init-gate-bias", type=float, default=-2.0)
    args = parser.parse_args()

    rgb_root = rgb_cache()
    par_root = parent_cache()
    zroot = z16_cache()
    ckpt = parent_ckpt()
    exp = run_dir(args.run_name)
    exp.mkdir(parents=True, exist_ok=True)
    log_path = exp / "run.log"

    def log(m: str) -> None:
        print(m, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(m + "\n")

    try:
        rows = [json.loads(l) for l in (par_root / "index.jsonl").open(encoding="utf-8")]
        gt = np.load(par_root / "gt_1s.npy").astype(np.float32)
        train_ids = [i for i, r in enumerate(rows) if r["split"] == "train"]
        sel_ids = [i for i, r in enumerate(rows) if r["split"] == "selection"]
        test_ids = [i for i, r in enumerate(rows) if r["split"] == "test"]
        log(f"windows {(len(train_ids), len(sel_ids), len(test_ids))}")

        parent = ParentTargetCache(par_root, z16_root=zroot)
        decoder = load_parent_decoder(ckpt)
        dec_h = decoder_state_hash(decoder)
        log(f"decoder_hash={dec_h} z16={'yes' if parent.Z is not None else 'on_the_fly'}")

        cfg = Config.load()
        device = resolve_device_safe(str(args.device))
        amp_dtype = select_amp_dtype(cfg, device)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = CollisionStudentV5(decoder, init_gate_bias=args.init_gate_bias).to(device)
        audit = model.param_audit()
        log(f"device={device} params {audit} arch=v5 gate_bias={args.init_gate_bias}")
        if not (415_000 <= audit["trainable"] <= 430_000):
            raise SystemExit(f"trainable params {audit['trainable']} outside 415k-430k (frozen decoder)")

        rgb = RGBCache256(rgb_root)
        heatmaps = SpatialHeatmaps()
        train_ds = WinDS(rows, train_ids, gt, rgb, parent, heatmaps)
        sel_ds = WinDS(rows, sel_ids, gt, rgb, parent, heatmaps)
        test_ds = WinDS(rows, test_ids, gt, rgb, parent, heatmaps)

        if args.overfit:
            rng = np.random.default_rng(args.seed)
            peaks = []
            for i in train_ids:
                pt = parent.get_parent_targets(rows[i]["clip_id"], int(rows[i]["frame"]))
                peaks.append(float(np.max(pt["H_teacher"])))
            peaks = np.asarray(peaks)
            stop_i = np.where(peaks >= 0.60)[0]
            rest_i = np.where(peaks < 0.60)[0]
            n_stop = min(10, int(len(stop_i)))
            pick_stop = rng.choice(stop_i, size=n_stop, replace=False)
            pick_rest = rng.choice(rest_i, size=32 - n_stop, replace=False)
            subset = [train_ids[int(j)] for j in np.concatenate([pick_stop, pick_rest])]
            train_ds = WinDS(rows, subset, gt, rgb, parent, heatmaps)
            log(f"overfit n={len(subset)} stop={n_stop}")

        peaks = []
        for rid in train_ds.ids:
            row = rows[int(rid)]
            pt = parent.get_parent_targets(row["clip_id"], int(row["frame"]))
            peaks.append(float(np.max(pt["H_teacher"])))
        sampler = None if args.overfit else weighted_sampler(hazard_sample_weights(peaks))
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=str(device).startswith("cuda"),
            drop_last=False,
        )
        sel_loader = DataLoader(
            sel_ds, batch_size=args.batch, shuffle=False, num_workers=min(2, args.num_workers), collate_fn=collate
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch, shuffle=False, num_workers=min(2, args.num_workers), collate_fn=collate
        )

        opt = torch.optim.AdamW(
            [
                {"params": list(model.backbone.parameters()), "lr": args.lr, "weight_decay": args.weight_decay},
                {"params": list(model.decoder.parameters()), "lr": 0.0, "weight_decay": args.weight_decay},
            ],
            betas=(0.9, 0.95),
        )
        steps_per_epoch = max(len(train_loader) // max(args.accum, 1), 1)
        total_steps = args.overfit_steps if args.overfit else args.epochs * steps_per_epoch
        warmup = max(int(0.05 * total_steps), 1)

        # LambdaLR multiplies every group by the same factor from *initial* lr.
        # Decoder initial lr is 0 so it stays 0 until we overwrite group 1 each step.
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: warmup_cosine(s, warmup, total_steps, args.lr, 1e-5)
        )
        start_ep = 1
        global_step = 0
        if args.resume:
            rpath = Path(args.resume)
            blob = torch.load(rpath, map_location=device, weights_only=False)
            model.backbone.load_state_dict(blob["backbone"])
            if "decoder" in blob:
                model.decoder.load_state_dict(blob["decoder"])
            start_ep = int(blob["epoch"]) + 1
            global_step = int(blob.get("global_step", int(blob["epoch"]) * steps_per_epoch))
            if blob.get("optimizer") is not None:
                opt.load_state_dict(blob["optimizer"])
                log(f"resume optimizer from {rpath}")
            else:
                log(f"resume weights only from {rpath}")
            if blob.get("scheduler") is not None:
                sched.load_state_dict(blob["scheduler"])
            else:
                sched.last_epoch = global_step
            log(f"resume epoch={blob['epoch']} next={start_ep} step={global_step}")

        (exp / "config.json").write_text(
            json.dumps(
                {
                    "run": args.run_name,
                    "epochs": args.epochs,
                    "batch": args.batch,
                    "accum": args.accum,
                    "lr": args.lr,
                    "weights": {"L_5x5": 1.0, "L_16x16": 1.0, "L_feat": args.feat_w},
                    "feat_decay": {
                        "start": args.feat_decay_start,
                        "duration": args.feat_decay_duration,
                        "lam_start": args.feat_w,
                        "lam_end": args.feat_w_end,
                    },
                    "decoder_unfreeze": {
                        "start_epoch": DECAY_START_EPOCH + 1,
                        "lr_mult": 0.15,
                        "warmup_epochs": 2,
                    },
                    "init_gate_bias": args.init_gate_bias,
                    "resume": args.resume,
                    "start_epoch": start_ep,
                    "sampler": "hazard_3_1.5_1",
                    "arch": "CollisionStudentV5",
                    "overfit": bool(args.overfit),
                    "decoder_hash": dec_h,
                    "param_audit": audit,
                    "z16": parent.Z is not None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        use_cuda = str(device).startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16 and use_cuda)
        autocast = (
            torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None and use_cuda else nullcontext()
        )
        need_z = parent.Z is None
        history = []
        best_rank = None
        best_ep = None
        epochs = args.overfit_steps if args.overfit else args.epochs

        for ep in range(start_ep, epochs + 1):
            lam_feat = get_lambda_feat_v5(
                ep,
                start_epoch=args.feat_decay_start,
                duration=args.feat_decay_duration,
                lam_start=args.feat_w,
                lam_end=args.feat_w_end,
            )
            if ep == DECAY_START_EPOCH + 1:
                model.set_decoder_trainable(True)
                log(f"unfreeze decoder at epoch {ep}")
            elif ep <= DECAY_START_EPOCH:
                model.set_decoder_trainable(False)
            else:
                model.set_decoder_trainable(True)
            model.train()
            t0 = time.perf_counter()
            acc = defaultdict(float)
            n_b = 0
            opt.zero_grad(set_to_none=True)
            stop_now = False
            for bi, batch in enumerate(train_loader):
                b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                if not args.overfit:
                    b = augment_batch(b)
                if need_z:
                    with torch.no_grad():
                        b["Z_teacher_16"] = model.decoder.forward_fine_logits(b["F_teacher"])
                with autocast:
                    out = model(b["clip"])
                    loss, terms = student_v5_loss(out, b, w_feat=lam_feat)
                    loss = loss / max(args.accum, 1)
                if "G_t" in out:
                    terms["G_mean"] = float(out["G_t"].detach().float().mean())
                    terms["G_std"] = float(out["G_t"].detach().float().std(unbiased=False))
                scaler.scale(loss).backward()
                if (bi + 1) % max(args.accum, 1) == 0:
                    scaler.unscale_(opt)
                    clip_params = [p for p in model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    bb_lr = float(opt.param_groups[0]["lr"])
                    opt.param_groups[1]["lr"] = decoder_lr(ep, bb_lr)
                    global_step += 1
                for k, v in terms.items():
                    if isinstance(v, (int, float)):
                        acc[k] += v
                n_b += 1
                if args.overfit and global_step >= args.overfit_steps:
                    stop_now = True
                    break
                if args.max_batches and n_b >= args.max_batches:
                    break
            train_m = {k: v / max(n_b, 1) for k, v in acc.items()}
            train_m["sec"] = time.perf_counter() - t0
            _, sel_m, _ = collect(model, sel_loader if not args.overfit else train_loader, device, amp_dtype)
            rec = {
                "epoch": ep,
                "train": train_m,
                "selection": sel_m,
                "step": global_step,
                "lam_feat": lam_feat,
                "decoder_unfrozen": model.decoder_unfrozen,
                "decoder_lr": float(opt.param_groups[1]["lr"]),
            }
            (exp / f"metrics_epoch_{ep:02d}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
            ckpt_blob = {
                "epoch": ep,
                "backbone": model.backbone.state_dict(),
                "decoder": model.decoder.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "global_step": global_step,
                "lam_feat": lam_feat,
                "audit": model.param_audit(),
                "decoder_unfrozen": model.decoder_unfrozen,
            }
            torch.save(ckpt_blob, exp / f"checkpoint_epoch_{ep:02d}.pt")
            rk = rank_checkpoint(sel_m)
            if best_rank is None or rk > best_rank:
                best_rank, best_ep = rk, ep
                torch.save({**ckpt_blob, "selection": sel_m}, exp / "checkpoint_best.pt")
            history.append(
                {
                    "epoch": ep,
                    "lam_feat": lam_feat,
                    **{k: train_m.get(k) for k in ("L_total", "L_feat", "L_5x5", "L_16x16", "share_feat", "share_5x5", "share_16x16", "G_mean", "G_std")},
                    "sel_stop": sel_m["stop_recall"],
                    "sel_safe": sel_m["safe_recall"],
                    "decoder_lr": float(opt.param_groups[1]["lr"]),
                }
            )
            log(
                f"epoch {ep} step={global_step} L={train_m['L_total']:.4f} "
                f"5x5={train_m['L_5x5']:.4f} 16={train_m['L_16x16']:.4f} feat={train_m['L_feat']:.4f} "
                f"lam={lam_feat:.3f} share_f={train_m.get('share_feat', 0):.2f} "
                f"G={train_m.get('G_mean', 0):.3f}±{train_m.get('G_std', 0):.3f} "
                f"STOP={sel_m['stop_recall']:.3f} SAFE={sel_m['safe_recall']:.3f} "
                f"lr={opt.param_groups[0]['lr']:.2e} dec_lr={opt.param_groups[1]['lr']:.2e} "
                f"sec={train_m['sec']:.1f}"
            )
            if stop_now:
                break

        blob = torch.load(exp / "checkpoint_best.pt", map_location=device, weights_only=False)
        model.backbone.load_state_dict(blob["backbone"])
        if "decoder" in blob:
            model.decoder.load_state_dict(blob["decoder"])
        if not args.overfit:
            _, test_m, parent_test = collect(model, test_loader, device, amp_dtype)
        else:
            test_m, parent_test = sel_m, {}
        (exp / "final_metrics.json").write_text(
            json.dumps(
                {
                    "selected_epoch": best_ep,
                    "test_student": test_m,
                    "test_parent": parent_test,
                    "history": history,
                    "audit": audit,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        log(f"DONE selected_epoch={best_ep} test_STOP={test_m.get('stop_recall')} test_SAFE={test_m.get('safe_recall')}")
        (exp / ".success").write_text("ok\n", encoding="utf-8")
    except Exception:
        traceback.print_exc()
        (exp / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
