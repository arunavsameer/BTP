#!/usr/bin/env python3
"""Train 3D ConvNeXt-V2 student against cached Pair-CF F_dec.

Reads Trial-3 parent_target_cache_v2 (no teacher forward, no npy copy).
Writes under STUDENT_RUN_DIR or model_trial_5/student_runs/<name>/.

  python scripts/90_train_convnextv2_student.py --variant atto --epochs 30 --batch 32
  python scripts/90_train_convnextv2_student.py --resume
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

from collision_jepa3.utils.device import resolve_device, select_amp_dtype  # noqa: E402
from collision_jepa3.config import Config  # noqa: E402
from student_convnextv2.data import (  # noqa: E402
    ParentTargetCache,
    RGBCache256,
    SpatialHeatmaps,
    WinDS,
    collate,
    compute_fdec_stats,
)
from student_convnextv2.losses import convnext_student_loss  # noqa: E402
from student_convnextv2.metrics import collect, rank_checkpoint  # noqa: E402
from student_convnextv2.model import CollisionStudentConvNeXtV2  # noqa: E402
from student_convnextv2.parent import decoder_state_hash, load_parent_decoder  # noqa: E402
from student_convnextv2.paths import (  # noqa: E402
    parent_cache,
    parent_ckpt,
    rgb_cache,
    run_dir,
    stats_path,
)


def resolve_device_safe(pref: str) -> str:
    device = resolve_device(pref)
    if not str(device).startswith("cuda"):
        return device
    try:
        torch.zeros(1, device=device)
        return device
    except Exception as exc:
        print(f"cuda unusable ({pref} -> {device}): {exc}; falling back to cpu", flush=True)
        return "cpu"


def warmup_cosine(step: int, warmup: int, total: int, base: float, min_lr: float) -> float:
    if step < warmup:
        return (1e-5 + (base - 1e-5) * step / max(warmup, 1)) / base
    progress = (step - warmup) / max(total - warmup, 1)
    lr = min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return lr / base


def load_stats(path: Path, parent, rows, train_ids, log) -> tuple[np.ndarray, np.ndarray]:
    if path.is_file():
        st = np.load(path)
        log(f"loaded F_dec stats {path}")
        return np.asarray(st["mu"], dtype=np.float32), np.asarray(st["sigma"], dtype=np.float32)
    log(f"computing F_dec channel stats on train → {path}")
    return compute_fdec_stats(parent, rows, train_ids, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-name", default="convnextv2_atto_v1")
    parser.add_argument("--variant", default="atto", choices=["atto", "tiny", "base", "large"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--sparse-prob", type=float, default=0.5)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        help="Resume from checkpoint_last.pt in the run dir, or from an explicit path",
    )
    args = parser.parse_args()

    rgb_root = rgb_cache()
    par_root = parent_cache()
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
        log(f"windows {(len(train_ids), len(sel_ids), len(test_ids))} cache={par_root}")
        if (len(train_ids), len(sel_ids), len(test_ids)) != (22680, 5120, 5120):
            log(f"WARN split contract mismatch {(len(train_ids), len(sel_ids), len(test_ids))}")

        parent = ParentTargetCache(par_root)
        decoder = load_parent_decoder(ckpt)
        dec_h = decoder_state_hash(decoder)
        log(f"decoder_hash={dec_h} parent_ckpt={ckpt}")
        log(f"rgb_cache={rgb_root}")

        cfg = Config.load()
        device = resolve_device_safe(str(args.device))
        amp_dtype = select_amp_dtype(cfg, device)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if str(device).startswith("cuda"):
            torch.backends.cudnn.benchmark = True

        model = CollisionStudentConvNeXtV2(
            decoder,
            variant=args.variant,
            mask_ratio=args.mask_ratio,
            sparse_prob=0.0 if args.overfit else args.sparse_prob,
            grad_checkpoint=not args.no_grad_checkpoint,
        ).to(device)
        audit = model.param_audit()
        log(f"device={device} amp={amp_dtype} params {audit}")

        mu_np, sigma_np = load_stats(stats_path(), parent, rows, train_ids, log)
        mu = torch.tensor(mu_np, dtype=torch.float32, device=device)
        sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)

        rgb = RGBCache256(rgb_root)
        heatmaps = SpatialHeatmaps()
        train_ds = WinDS(rows, train_ids, gt, rgb, parent, heatmaps)
        sel_ds = WinDS(rows, sel_ids, gt, rgb, parent, heatmaps)
        test_ds = WinDS(rows, test_ids, gt, rgb, parent, heatmaps)

        if args.overfit:
            rng = np.random.default_rng(args.seed)
            peaks = gt[np.array(train_ids)].reshape(len(train_ids), -1).max(1)
            stop_i = np.where(peaks >= 0.60)[0]
            rest_i = np.where(peaks < 0.60)[0]
            n_stop = min(10, int(len(stop_i)))
            pick_stop = rng.choice(stop_i, size=n_stop, replace=False)
            pick_rest = rng.choice(rest_i, size=32 - n_stop, replace=False)
            subset = [train_ids[int(j)] for j in np.concatenate([pick_stop, pick_rest])]
            train_ds = WinDS(rows, subset, gt, rgb, parent, heatmaps)
            sel_ds = train_ds
            log(f"overfit n={len(subset)} stop={n_stop}")

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=str(device).startswith("cuda"),
            drop_last=False,
        )
        sel_loader = DataLoader(
            sel_ds,
            batch_size=max(args.batch, 1),
            shuffle=False,
            num_workers=min(2, args.num_workers),
            collate_fn=collate,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=max(args.batch, 1),
            shuffle=False,
            num_workers=min(2, args.num_workers),
            collate_fn=collate,
        )

        params = list(model.student_parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        steps_per_epoch = max(math.ceil(len(train_loader) / max(args.accum, 1)), 1)
        total_steps = args.overfit_steps if args.overfit else args.epochs * steps_per_epoch
        warmup = max(int(0.05 * total_steps), 1)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: warmup_cosine(s, warmup, total_steps, args.lr, 1e-6)
        )

        start_ep = 1
        global_step = 0
        if args.resume is not None:
            rpath = Path(args.resume) if args.resume else (exp / "checkpoint_last.pt")
            if not rpath.is_file():
                raise FileNotFoundError(f"--resume checkpoint missing: {rpath}")
            blob = torch.load(rpath, map_location=device, weights_only=False)
            model.backbone.load_state_dict(blob["backbone"])
            if "optimizer" in blob:
                opt.load_state_dict(blob["optimizer"])
            if "scheduler" in blob:
                sched.load_state_dict(blob["scheduler"])
            start_ep = int(blob.get("epoch", 0)) + 1
            global_step = int(blob.get("global_step", 0))
            log(f"resumed {rpath} next_epoch={start_ep} step={global_step}")

        (exp / "config.json").write_text(
            json.dumps(
                {
                    "run": args.run_name,
                    "variant": args.variant,
                    "epochs": args.epochs,
                    "batch": args.batch,
                    "accum": args.accum,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "mask_ratio": args.mask_ratio,
                    "sparse_prob": 0.0 if args.overfit else args.sparse_prob,
                    "overfit": bool(args.overfit),
                    "resume": args.resume,
                    "start_epoch": start_ep,
                    "decoder_hash": dec_h,
                    "param_audit": audit,
                    "parent_cache": str(par_root),
                    "rgb_cache": str(rgb_root),
                    "parent_ckpt": str(ckpt),
                    "loss": "whitened_L1+cosine+gram on F_dec; teacher BCE/blob/hinges/KD on 5x5",
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
        history = []
        best_rank = None
        best_ep = None
        smoke = bool(args.max_batches) and not args.overfit
        epochs = args.overfit_steps if args.overfit else args.epochs

        for ep in range(start_ep, epochs + 1):
            model.train()
            t0 = time.perf_counter()
            acc = defaultdict(float)
            n_b = 0
            opt.zero_grad(set_to_none=True)
            stop_now = False
            for bi, batch in enumerate(train_loader):
                b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                with autocast:
                    out = model(b["clip"])
                    loss, terms = convnext_student_loss(out, b, mu, sigma)
                    loss = loss / max(args.accum, 1)
                scaler.scale(loss).backward()
                if (bi + 1) % max(args.accum, 1) == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    global_step += 1
                with torch.no_grad():
                    acc["cos"] += float(
                        torch.nn.functional.cosine_similarity(
                            out["F"].float().flatten(1), b["F_teacher"].float().flatten(1)
                        ).mean()
                    )
                for k, v in terms.items():
                    acc[k] += v
                n_b += 1
                if args.overfit and global_step >= args.overfit_steps:
                    stop_now = True
                    break
                if args.max_batches and n_b >= args.max_batches:
                    stop_now = True
                    break
            train_m = {k: v / max(n_b, 1) for k, v in acc.items()}
            train_m["sec"] = time.perf_counter() - t0
            train_m["lr"] = float(opt.param_groups[0]["lr"])
            do_eval = (not smoke) and (
                args.overfit or (ep % max(args.eval_every, 1) == 0) or ep == start_ep or ep == epochs
            )
            if do_eval:
                sel_m = collect(model, sel_loader, device, amp_dtype)
            else:
                sel_m = {"skipped": True}
            rec = {"epoch": ep, "train": train_m, "selection": sel_m, "step": global_step}
            if use_cuda:
                rec["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            (exp / f"metrics_epoch_{ep:02d}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
            payload = {
                "epoch": ep,
                "global_step": global_step,
                "backbone": model.backbone.state_dict(),
                "decoder": model.decoder.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "audit": audit,
                "variant": args.variant,
            }
            if not smoke:
                torch.save(payload, exp / f"checkpoint_epoch_{ep:02d}.pt")
            torch.save(payload, exp / "checkpoint_last.pt")
            if do_eval and not sel_m.get("skipped"):
                rk = rank_checkpoint(sel_m)
                if best_rank is None or rk > best_rank:
                    best_rank, best_ep = rk, ep
                    torch.save(
                        {
                            **payload,
                            "selection": sel_m,
                        },
                        exp / "checkpoint_best.pt",
                    )
            history.append(
                {
                    "epoch": ep,
                    "L_total": train_m.get("L_total"),
                    "L_feat": train_m.get("L_feat"),
                    "L_cos": train_m.get("L_cos"),
                    "train_cos": train_m.get("cos"),
                    "sel_stop": sel_m.get("stop_recall"),
                    "sel_safe": sel_m.get("safe_recall"),
                    "sel_fdec_cos": sel_m.get("fdec_cosine"),
                }
            )
            log(
                f"epoch {ep} step={global_step} L={train_m.get('L_total'):.4f} "
                f"feat={train_m.get('L_feat'):.4f} lcos={train_m.get('L_cos'):.4f} "
                f"train_cos={train_m.get('cos'):.3f} "
                f"STOP={sel_m.get('stop_recall')} SAFE={sel_m.get('safe_recall')} "
                f"fdec_cos={sel_m.get('fdec_cosine')} "
                f"lr={train_m['lr']:.2e} sec={train_m['sec']:.1f}"
            )
            if stop_now:
                break

        if smoke:
            test_m = {"skipped": True, "reason": "max-batches smoke"}
            log("smoke: skipped full selection/test eval")
        else:
            best_path = exp / "checkpoint_best.pt"
            if best_path.is_file():
                blob = torch.load(best_path, map_location=device, weights_only=False)
                model.backbone.load_state_dict(blob["backbone"])
            if args.eval_test and not args.overfit:
                test_m = collect(model, test_loader, device, amp_dtype)
            else:
                test_m = collect(model, sel_loader, device, amp_dtype)
                if args.overfit:
                    log("overfit: reporting the 32-clip set instead of held-out test")
                else:
                    log("selection metrics on best ckpt (pass --eval-test for held-out test)")
        (exp / "final_metrics.json").write_text(
            json.dumps(
                {
                    "selected_epoch": best_ep,
                    "held_out": test_m,
                    "history": history,
                    "audit": audit,
                    "variant": args.variant,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        log(
            f"DONE selected_epoch={best_ep} STOP={test_m.get('stop_recall')} "
            f"SAFE={test_m.get('safe_recall')} fdec_cos={test_m.get('fdec_cosine')} "
            f"collapsed={test_m.get('collapsed_stop_frac')}"
        )
        (exp / ".success").write_text("ok\n", encoding="utf-8")
    except Exception:
        traceback.print_exc()
        (exp / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
