#!/usr/bin/env python3
"""Train a Trial-4 student against the Trial-3 Pair-CF parent F_dec cache.

Loss is two components with **fixed** weights:
  1) L_feat: channel-normalized L1 vs cached parent F_dec
  2) Trial-3 G heatmap recipe (appear / small-blob / STOP+SAFE hinge / blob / FP / dir)

Component weights are constants (default feat_weight=1, heatmap_weight=1).

Reads (never writes):
  - 256 RGB cache (shared_rgb_cache_256_v1)
  - parent_target_cache_v2 (F_dec / H_teacher / gt_1s)
  - Pair-CF checkpoint teacher_phase_a_best.pt (frozen decoder only)
  - spatial_annotations.json for h_now (G appear weights)

Writes only under model_trial_4/student_runs/<name>/.
Does not import Trial-3 student code or touch Trial-3 checkpoints/logs.

  python scripts/10_smoke_student.py
  python scripts/10_train_student.py --integrity-only
  python scripts/10_train_student.py --epochs 100 --device auto
"""

from __future__ import annotations

import argparse
import json
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

from collision_jepa4.training.optimizer import build_scheduler  # noqa: E402
from collision_jepa4.utils.device import resolve_device, select_amp_dtype  # noqa: E402
from collision_jepa4.config import Config  # noqa: E402
from collision_jepa4.losses import TeacherLoss  # noqa: E402
from student.caches import ParentTargetCache, RGBCache256, SpatialHeatmaps  # noqa: E402
from student.dataset import WinDS, collate  # noqa: E402
from student.losses import student_loss  # noqa: E402
from student.metrics import collect, rank_checkpoint  # noqa: E402
from student.model import StudentLatPairCF  # noqa: E402
from student.parent import decoder_state_hash, load_parent_decoder  # noqa: E402
from student.paths import (  # noqa: E402
    EXPECTED_DECODER_HASH,
    OFFSETS_16,
    parent_cache,
    parent_ckpt,
    rgb_cache,
    run_dir,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--integrity-only", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0, help="If >0, cap train batches per epoch (local test).")
    parser.add_argument("--run-name", default="feat_gheatmap_v2")
    parser.add_argument("--feat-weight", type=float, default=1.0)
    parser.add_argument("--heatmap-weight", type=float, default=1.0)
    parser.add_argument("--rgb-cache", default=None)
    parser.add_argument("--parent-cache", default=None)
    parser.add_argument("--parent-ckpt", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    rgb_root = Path(args.rgb_cache) if args.rgb_cache else rgb_cache()
    par_root = Path(args.parent_cache) if args.parent_cache else parent_cache()
    ckpt = Path(args.parent_ckpt) if args.parent_ckpt else parent_ckpt()
    exp = run_dir(args.run_name)
    exp.mkdir(parents=True, exist_ok=True)
    log_path = exp / "run.log"

    def log(m: str) -> None:
        print(m, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(m + "\n")

    try:
        rgb_ok = (rgb_root / "COMPLETE").is_file()
        par_ok = (par_root / "COMPLETE").is_file()
        if not rgb_ok or not par_ok:
            raise SystemExit(f"caches incomplete rgb={rgb_ok} ({rgb_root}) parent={par_ok} ({par_root})")
        rows = [json.loads(l) for l in (par_root / "index.jsonl").open(encoding="utf-8")]
        gt = np.load(par_root / "gt_1s.npy").astype(np.float32)
        train_ids = [i for i, r in enumerate(rows) if r["split"] == "train"]
        sel_ids = [i for i, r in enumerate(rows) if r["split"] == "selection"]
        test_ids = [i for i, r in enumerate(rows) if r["split"] == "test"]
        counts = (len(train_ids), len(sel_ids), len(test_ids))
        log(f"windows {counts}")
        if counts != (22680, 5120, 5120):
            raise SystemExit(f"sampling contract mismatch {counts}")

        parent = ParentTargetCache(par_root)
        decoder = load_parent_decoder(ckpt)
        dec_h = decoder_state_hash(decoder)
        log(f"decoder_hash={dec_h} ckpt={ckpt}")
        replay_abs = []
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(train_ids, size=128, replace=False)
        decoder.eval()
        with torch.no_grad():
            for rid in pick:
                pt = parent.get_parent_targets(rows[int(rid)]["clip_id"], int(rows[int(rid)]["frame"]))
                f = torch.from_numpy(pt["F_dec"][None])
                h = decoder.forward_with_logits(f)[0][0].detach().cpu().numpy()
                replay_abs.append(float(np.max(np.abs(h - pt["H_teacher"]))))
        replay = {
            "n": len(replay_abs),
            "max_abs": max(replay_abs),
            "mean_abs": float(np.mean(replay_abs)),
            "ok": max(replay_abs) < 2e-2,
        }
        log(f"F_dec→decoder vs H_teacher {replay}")
        if not replay["ok"]:
            raise SystemExit(f"decoder replay failed {replay}")

        f_list = []
        for rid in rng.choice(train_ids, size=min(4000, len(train_ids)), replace=False):
            pt = parent.get_parent_targets(rows[int(rid)]["clip_id"], int(rows[int(rid)]["frame"]))
            f_list.append(pt["F_dec"])
        arr = np.stack(f_list, 0)
        ch_mu = arr.mean(axis=(0, 2, 3), keepdims=True)
        ch_sigma = arr.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        np.savez(exp / "fdec_channel_stats.npz", mu=ch_mu, sigma=ch_sigma)

        cfg = Config.load()
        device = resolve_device_safe(str(args.device))
        amp_dtype = select_amp_dtype(cfg, device)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = StudentLatPairCF(decoder).to(device)
        audit = model.param_audit()
        log(f"device={device} params {audit}")
        integrity = {
            "rgb_complete": rgb_ok,
            "parent_complete": par_ok,
            "rgb_root": str(rgb_root),
            "parent_root": str(par_root),
            "parent_ckpt": str(ckpt),
            "windows": {"train": 22680, "selection": 5120, "test": 5120},
            "offsets": OFFSETS_16,
            "rgb_size": 256,
            "decoder_hash": dec_h,
            "expected_hash": EXPECTED_DECODER_HASH,
            "replay": replay,
            "param_audit": audit,
            "writes_under": str(exp),
            "trial3_read_only": True,
        }
        (exp / "integrity.json").write_text(json.dumps(integrity, indent=2, default=str), encoding="utf-8")
        (exp / "param_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        if args.integrity_only:
            log("integrity-only OK")
            return

        rgb = RGBCache256(rgb_root)
        heatmaps = SpatialHeatmaps()
        mu = torch.tensor(ch_mu, dtype=torch.float32, device=device)
        sigma = torch.tensor(ch_sigma, dtype=torch.float32, device=device)
        train_ds = WinDS(rows, train_ids, gt, rgb, parent, heatmaps)
        sel_ds = WinDS(rows, sel_ids, gt, rgb, parent, heatmaps)
        test_ds = WinDS(rows, test_ids, gt, rgb, parent, heatmaps)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=str(device).startswith("cuda"),
        )
        sel_loader = DataLoader(sel_ds, batch_size=args.batch, shuffle=False, num_workers=min(2, args.num_workers), collate_fn=collate)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=min(2, args.num_workers), collate_fn=collate)

        enc_p = [p for p in model.encoder.parameters() if p.requires_grad]
        opt = torch.optim.AdamW([{"params": enc_p, "lr": 0.0003, "weight_decay": 1e-4}])
        sch = build_scheduler(opt, args.epochs, 1e-6, warmup_epochs=5)
        heatmap_loss = TeacherLoss(cfg, device=device)
        feat_w, heatmap_w = float(args.feat_weight), float(args.heatmap_weight)
        log(f"fixed_weights feat_w={feat_w} heatmap_w={heatmap_w}")
        use_cuda = str(device).startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16 and use_cuda)
        autocast = (
            torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None and use_cuda else nullcontext()
        )
        (exp / "config.json").write_text(
            json.dumps(
                {
                    "epochs": args.epochs,
                    "batch": args.batch,
                    "device": device,
                    "seed": args.seed,
                    "latent": "L_feat_only",
                    "heatmap": "trial3_G",
                    "weights_mode": "constant",
                    "feat_weight": feat_w,
                    "heatmap_weight": heatmap_w,
                    "g_terms": {
                        "appear_weight": cfg.get("loss.bce_appear_weight"),
                        "small_blob_weight": cfg.get("loss.small_blob_weight"),
                        "stop_hinge_weight": cfg.get("loss.stop_hinge_weight"),
                        "safe_hinge_weight": cfg.get("loss.safe_hinge_weight"),
                        "blob_bce_weight": cfg.get("loss.blob_bce_weight"),
                        "close_weight": cfg.get("loss.close_weight"),
                        "close_active": False,
                        "false_positive_weight": cfg.get("loss.false_positive_weight"),
                        "direction_weight": cfg.get("loss.direction_weight"),
                    },
                    "decoder_hash": dec_h,
                    "rgb": str(rgb_root),
                    "parent": str(par_root),
                    "parent_ckpt": str(ckpt),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        torch.save({"model": model.state_dict(), "epoch": 0, "audit": audit}, exp / "checkpoint_initial.pt")

        history = []
        best_rank = None
        best_ep = None
        params = enc_p
        for ep in range(1, args.epochs + 1):
            model.train()
            t0 = time.perf_counter()
            acc = defaultdict(float)
            n_b = 0
            opt.zero_grad(set_to_none=True)
            for batch in train_loader:
                b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                with autocast:
                    out = model(b["clip"])
                    loss, terms = student_loss(
                        out,
                        b,
                        mu=mu,
                        sigma=sigma,
                        heatmap_loss=heatmap_loss,
                        feat_weight=feat_w,
                        heatmap_weight=heatmap_w,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                for k, v in terms.items():
                    acc[k] += v
                n_b += 1
                if args.max_batches and n_b >= args.max_batches:
                    break
            if sch is not None:
                sch.step()
            train_m = {k: v / max(n_b, 1) for k, v in acc.items()}
            train_m["sec"] = time.perf_counter() - t0
            _, sel_m, _ = collect(model, sel_loader, device, amp_dtype)
            rec = {"epoch": ep, "train": train_m, "selection": sel_m}
            (exp / f"metrics_epoch_{ep:02d}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
            torch.save(
                {
                    "epoch": ep,
                    "encoder": model.encoder.state_dict(),
                    "predictor": model.predictor.state_dict(),
                    "decoder": model.decoder.state_dict(),
                    "audit": audit,
                },
                exp / f"checkpoint_epoch_{ep:02d}.pt",
            )
            rk = rank_checkpoint(sel_m)
            if best_rank is None or rk > best_rank:
                best_rank, best_ep = rk, ep
                torch.save(
                    {
                        "epoch": ep,
                        "encoder": model.encoder.state_dict(),
                        "predictor": model.predictor.state_dict(),
                        "decoder": model.decoder.state_dict(),
                        "selection": sel_m,
                        "audit": audit,
                    },
                    exp / "checkpoint_best.pt",
                )
            history.append(
                {
                    "epoch": ep,
                    "L_total": train_m["L_total"],
                    "L_feat": train_m["L_feat"],
                    "L_heatmap": train_m["L_heatmap"],
                    "feat_weight": feat_w,
                    "heatmap_weight": heatmap_w,
                    "share_feat": train_m["share_feat"],
                    "share_heatmap": train_m["share_heatmap"],
                    "sel_stop": sel_m["stop_recall"],
                    "sel_safe": sel_m["safe_recall"],
                    "sel_post_stop": sel_m["post_fork_stop_recall"],
                    "sel_cos": sel_m["fdec_cosine"],
                }
            )
            log(
                f"epoch {ep} L={train_m['L_total']:.4f} feat={train_m['L_feat']:.4f} "
                f"hm={train_m['L_heatmap']:.4f} share_f={100 * train_m['share_feat']:.1f}% "
                f"share_h={100 * train_m['share_heatmap']:.1f}% "
                f"STOP={sel_m['stop_recall']:.3f} SAFE={sel_m['safe_recall']:.3f} "
                f"post={sel_m['post_fork_stop_recall']:.3f} fcos={sel_m['fdec_cosine']:.3f} sec={train_m['sec']:.1f}"
            )

        if best_ep is None:
            best_ep = args.epochs
        blob = torch.load(exp / "checkpoint_best.pt", map_location=device, weights_only=False)
        model.encoder.load_state_dict(blob["encoder"])
        model.predictor.load_state_dict(blob["predictor"])
        torch.save(blob, exp / "checkpoint_final.pt")
        _, test_m, parent_test = collect(model, test_loader, device, amp_dtype)
        final = {
            "selected_epoch": best_ep,
            "test_student": test_m,
            "test_parent": parent_test,
            "history": history,
        }
        (exp / "final_metrics.json").write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")
        log(f"DONE selected_epoch={best_ep} test_STOP={test_m['stop_recall']} test_SAFE={test_m['safe_recall']}")
        (exp / ".success").write_text("ok\n", encoding="utf-8")
    except Exception:
        traceback.print_exc()
        (exp / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
