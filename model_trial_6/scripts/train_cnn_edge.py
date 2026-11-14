#!/usr/bin/env python3
"""Train CollisionStudentV2 + Canny against cached Pair-CF F_dec. Laptop CUDA only.

Do not launch while Atto owns the 4060. This script is the training entrypoint;
the local wrapper is run_cnn_edge_local.sh.
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
sys.path.insert(0, "/storage/BTP/model_trial_3")
sys.path.insert(0, "/storage/BTP/model_trial_5")

from cnn_edge.data import ParentTargetCache, RGBCache256, WinDS, collate, rgb_clip_exists
from cnn_edge.losses import student_v2_loss
from cnn_edge.metrics import collect, rank_checkpoint
from cnn_edge.model import CollisionStudentV2
from cnn_edge.parent import decoder_state_hash, load_parent_decoder
from cnn_edge.paths import parent_cache, parent_ckpt, rgb_cache, run_dir, stats_path


def warmup_cosine(step: int, warmup: int, total: int, base: float, min_lr: float) -> float:
    if step < warmup:
        return (1e-5 + (base - 1e-5) * step / max(warmup, 1)) / base
    progress = (step - warmup) / max(total - warmup, 1)
    lr = min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return lr / base


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--eval-test", action="store_true")
    p.add_argument("--run-name", default="student_v2_canny")
    p.add_argument("--no-canny", action="store_true")
    p.add_argument("--resume", action="store_true", help="Continue from checkpoint_last.pt in the run dir")
    args = p.parse_args()

    rgb_root, par_root, ckpt = rgb_cache(), parent_cache(), parent_ckpt()
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

        def filt(ids, name):
            kept = [i for i in ids if rgb_clip_exists(rgb_root, rows[i]["clip_id"])]
            dropped = len(ids) - len(kept)
            if dropped:
                log(f"skip {dropped} {name} windows missing local RGB")
            return kept

        train_ids, sel_ids, test_ids = filt(train_ids, "train"), filt(sel_ids, "selection"), filt(test_ids, "test")
        log(f"windows {(len(train_ids), len(sel_ids), len(test_ids))} rgb={rgb_root} parent={par_root}")

        decoder = load_parent_decoder(ckpt)
        log(f"decoder_hash={decoder_state_hash(decoder)} ckpt={ckpt}")
        device = str(args.device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("cuda requested but not available")
        model = CollisionStudentV2(decoder).to(device)
        log(f"arch=CollisionStudentV2+Canny device={device} params {model.param_audit()}")

        st_path = stats_path()
        if st_path.is_file():
            st = np.load(st_path)
            mu = torch.tensor(st["mu"], dtype=torch.float32, device=device)
            sigma = torch.tensor(st["sigma"], dtype=torch.float32, device=device)
            log(f"whitened L1 stats {st_path}")
        else:
            mu, sigma = None, None
            log(f"no fdec stats at {st_path}; whitened L1 off")

        rgb = RGBCache256(rgb_root, canny=not args.no_canny)
        train_ds = WinDS(rows, train_ids, gt, rgb, ParentTargetCache(par_root))
        sel_ds = WinDS(rows, sel_ids, gt, rgb, ParentTargetCache(par_root))
        test_ds = WinDS(rows, test_ids, gt, rgb, ParentTargetCache(par_root))
        pin = device.startswith("cuda")
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=pin,
        )
        sel_loader = DataLoader(
            sel_ds, batch_size=args.batch, shuffle=False, num_workers=min(2, args.num_workers), collate_fn=collate
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch, shuffle=False, num_workers=min(2, args.num_workers), collate_fn=collate
        )

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        steps_per_epoch = max(math.ceil(len(train_loader) / max(args.accum, 1)), 1)
        total_steps = args.epochs * steps_per_epoch
        warmup = max(int(0.05 * total_steps), 1)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: warmup_cosine(s, warmup, total_steps, args.lr, 1e-6)
        )
        amp_dtype = torch.bfloat16 if device.startswith("cuda") else None
        autocast = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else nullcontext()
        (exp / "config.json").write_text(
            json.dumps(
                {
                    "arch": "CollisionStudentV2+Canny",
                    "batch": args.batch,
                    "accum": args.accum,
                    "canny": not args.no_canny,
                    "canny_cfg": {"blur": 5, "low": 40, "high": 120, "color": [0, 255, 255]},
                    "rgb_cache": str(rgb_root),
                    "parent_cache": str(par_root),
                    "parent_ckpt": str(ckpt),
                    "fdec_stats": str(st_path) if st_path.is_file() else None,
                    "params": model.param_audit(),
                },
                indent=2,
            )
        )

        best_rank, best_ep, global_step, history = None, None, 0, []
        start_ep = 1
        last_ckpt = exp / "checkpoint_last.pt"
        if args.resume and last_ckpt.is_file():
            blob = torch.load(last_ckpt, map_location=device, weights_only=False)
            model.backbone.load_state_dict(blob["backbone"], strict=True)
            start_ep = int(blob.get("epoch", 0)) + 1
            log(f"resume {last_ckpt} completed_epoch={blob.get('epoch')} -> from epoch {start_ep}")
            best_ckpt = exp / "checkpoint_best.pt"
            if best_ckpt.is_file():
                bb = torch.load(best_ckpt, map_location="cpu", weights_only=False)
                best_ep = int(bb.get("epoch", 0))
                if isinstance(bb.get("selection"), dict):
                    best_rank = rank_checkpoint(bb["selection"])
                    log(f"restored best epoch {best_ep} rank={best_rank}")
        for ep in range(start_ep, args.epochs + 1):
            model.train()
            t0 = time.perf_counter()
            acc, n_b = defaultdict(float), 0
            opt.zero_grad(set_to_none=True)
            for bi, batch in enumerate(train_loader):
                b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                with autocast:
                    out = model(b["clip"])
                    loss, terms = student_v2_loss(out, b, mu, sigma)
                    loss = loss / max(args.accum, 1)
                loss.backward()
                if n_b < 3 or (n_b + 1) % 50 == 0:
                    mem = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else 0.0
                    log(
                        f"epoch {ep} batch {n_b + 1} L={float(loss.detach()) * max(args.accum, 1):.4f} "
                        f"feat={terms.get('L_feat', float('nan')):.4f} "
                        f"focal={terms.get('L_focal', float('nan')):.4f} gpu_gb={mem:.2f}"
                    )
                if (bi + 1) % max(args.accum, 1) == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    global_step += 1
                for k, v in terms.items():
                    acc[k] += v
                n_b += 1
            train_m = {k: v / max(n_b, 1) for k, v in acc.items()}
            train_m["sec"] = time.perf_counter() - t0
            do_eval = ep % max(args.eval_every, 1) == 0 or ep == 1 or ep == args.epochs
            sel_m = collect(model, sel_loader, device, amp_dtype) if do_eval else {"skipped": True}
            payload = {
                "epoch": ep,
                "backbone": model.backbone.state_dict(),
                "decoder": model.decoder.state_dict(),
            }
            torch.save(payload, exp / "checkpoint_last.pt")
            torch.save(payload, exp / f"checkpoint_epoch_{ep:02d}.pt")
            if do_eval and not sel_m.get("skipped"):
                rk = rank_checkpoint(sel_m)
                if best_rank is None or rk > best_rank:
                    best_rank, best_ep = rk, ep
                    torch.save({**payload, "selection": sel_m}, exp / "checkpoint_best.pt")
            history.append(
                {
                    "epoch": ep,
                    "L_total": train_m.get("L_total"),
                    "sel_stop": sel_m.get("stop_recall"),
                    "sel_safe": sel_m.get("safe_recall"),
                }
            )
            log(
                f"epoch {ep} L={train_m.get('L_total'):.4f} feat={train_m.get('L_feat'):.4f} "
                f"focal={train_m.get('L_focal'):.4f} "
                f"STOP={sel_m.get('stop_recall')} SAFE={sel_m.get('safe_recall')} "
                f"fdec_cos={sel_m.get('fdec_cosine')} sec={train_m['sec']:.1f}"
            )

        best_path = exp / "checkpoint_best.pt"
        if best_path.is_file():
            blob = torch.load(best_path, map_location=device, weights_only=False)
            model.backbone.load_state_dict(blob["backbone"])
        test_m = collect(model, test_loader if args.eval_test else sel_loader, device, amp_dtype)
        (exp / "final_metrics.json").write_text(
            json.dumps({"selected_epoch": best_ep, "held_out": test_m, "history": history}, indent=2, default=str)
        )
        log(
            f"DONE selected_epoch={best_ep} STOP={test_m.get('stop_recall')} "
            f"SAFE={test_m.get('safe_recall')} fdec_cos={test_m.get('fdec_cosine')} "
            f"collapsed={test_m.get('collapsed_stop_frac')}"
        )
        (exp / ".success").write_text("ok\n")
    except Exception:
        traceback.print_exc()
        (exp / ".failed").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
