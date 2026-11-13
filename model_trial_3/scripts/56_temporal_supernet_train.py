#!/usr/bin/env python3
"""Parts 3–8: train temporal supernet (20–30 epochs) + multi-pattern eval."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.cnn import MAX_TOTAL_PARAMS, assert_param_budget  # noqa: E402
from collision_jepa3.student_research.dev_manifest import PATTERN_IDS  # noqa: E402
from collision_jepa3.student_research.experiment_runner import gpu_mem_mb, write_epoch_metrics  # noqa: E402
from collision_jepa3.student_research.frames import FRAME_CANDIDATES  # noqa: E402
from collision_jepa3.student_research.rgb_supernet_cache import RGBSupernetCache  # noqa: E402
from collision_jepa3.student_research.supernet_ablate import run_temporal_ablations  # noqa: E402
from collision_jepa3.student_research.supernet_dataset import SupernetAnchorDataset, make_loader  # noqa: E402
from collision_jepa3.student_research.supernet_metrics import eval_pattern  # noqa: E402
from collision_jepa3.student_research.temporal_supernet import build_supernet, pack_pattern_clip  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _loss(batch, out, alpha: float) -> tuple[torch.Tensor, dict[str, float]]:
    gt = batch["gt_1s"]
    h = out["Hhat_1s"]
    l_gt = F.mse_loss(h, gt)
    l_kd = F.mse_loss(h, batch["H_teacher"])
    total = l_gt + alpha * l_kd
    return total, {"L_gt": float(l_gt.detach()), "L_kd": float(l_kd.detach()), "L_total": float(total.detach())}


def eval_all_patterns(model, dev_loader, device, warning, epoch: int) -> dict[str, dict]:
    out = {}
    for pid in PATTERN_IDS:
        out[pid] = eval_pattern(model, dev_loader, device, pid, warning)
        out[pid]["epoch"] = epoch
    # aggregate screening score
    stops = [out[p]["stop_recall"] for p in PATTERN_IDS if out[p]["stop_recall"] == out[p]["stop_recall"]]
    safes = [out[p]["safe_recall"] for p in PATTERN_IDS if out[p]["safe_recall"] == out[p]["safe_recall"]]
    out["_aggregate"] = {
        "mean_stop_recall": float(np.mean(stops)) if stops else float("nan"),
        "mean_safe_recall": float(np.mean(safes)) if safes else float("nan"),
        "epoch": epoch,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "reports/student_architecture_tests"))
    parser.add_argument("--exp-dir", default="experiments/temporal_supernet")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha-kd", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    add_config_args(parser)
    args = parser.parse_args()
    epochs = min(max(args.epochs, args.min_epochs), args.max_epochs)
    seed = int(args.seed if args.seed is not None else 1337)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out = Path(args.out)
    exp_d = out / args.exp_dir
    exp_d.mkdir(parents=True, exist_ok=True)
    log_path = exp_d / "run.log"

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)
        print(msg, flush=True)

    cfg = load_cfg(argparse.Namespace(config=args.config, dataset_root=args.dataset_root, seed=seed))
    device = resolve_device(args.device)
    warning = cfg.get("warning") or {}
    meta_rgb = json.loads((out / "shared_rgb_supernet_cache.json").read_text())
    rgb = RGBSupernetCache(out / "rgb_supernet_cache", meta_rgb)
    teacher = out / "teacher_cache_train_pilot"
    train_m = json.loads((out / "part1_train_manifest.json").read_text())
    dev_m = json.loads((out / "part1_dev_manifest.json").read_text())

    model = build_supernet()
    nparams = assert_param_budget(model, MAX_TOTAL_PARAMS)
    audit = model.parameter_audit()
    arch_path = out / "part3_supernet_architecture.json"
    atomic_write_json(
        arch_path,
        {
            **audit,
            "max_params_budget": MAX_TOTAL_PARAMS,
            "frame_patterns": FRAME_CANDIDATES,
            "patterns_tested": PATTERN_IDS,
        },
    )

    start_epoch = 0
    if args.resume and (exp_d / "checkpoint_last.pt").exists():
        model.load_state_dict(torch.load(exp_d / "checkpoint_last.pt", map_location="cpu", weights_only=True))
        for p in sorted(exp_d.glob("metrics_epoch_*.json")):
            start_epoch = max(start_epoch, int(p.stem.split("_")[-1]))
        log(f"resume from epoch {start_epoch}")

    config = {
        "epochs": epochs,
        "batch": args.batch,
        "lr": args.lr,
        "alpha_kd": args.alpha_kd,
        "seed": seed,
        "params": nparams,
        "n_train": len(train_m["train_row_ids"]),
        "n_dev": len(dev_m["dev_row_ids"]),
    }
    atomic_write_json(exp_d / "config.json", config)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    eval_epochs = {0, 5, 10, 15, 20, 25, 30}

    tr_ds = SupernetAnchorDataset(teacher, rgb, train_m["train_row_ids"])
    va_ds = SupernetAnchorDataset(teacher, rgb, dev_m["dev_row_ids"], epoch=0)
    va_loader = make_loader(va_ds, args.batch, False)

    pattern_eval_history: dict = {"_snapshots": []}
    for pid in PATTERN_IDS:
        pattern_eval_history[pid] = []

    if start_epoch == 0:
        log("epoch0 eval (init)")
        e0 = eval_all_patterns(model, va_loader, device, warning, 0)
        pattern_eval_history["_snapshots"] = [{"epoch": 0, "patterns": e0}]
        atomic_write_json(exp_d / "metrics_eval_epoch_00.json", e0)

    for ep in range(start_epoch, epochs):
        tr_ds.set_epoch(ep)
        tr_loader = make_loader(tr_ds, args.batch, True)
        model.train()
        t0 = time.perf_counter()
        running = {"L_total": 0.0, "L_gt": 0.0, "L_kd": 0.0}
        steps = 0
        pattern_counts = {p: 0 for p in PATTERN_IDS}
        for batch in tr_loader:
            batch_t = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            patterns = batch_t["train_pattern"] if isinstance(batch_t["train_pattern"], list) else batch["train_pattern"]
            # one pattern per batch (pair-balanced assignment is per-row; pack per-sample pattern)
            # pack each sample with its pattern (batch may mix patterns)
            clips, masks, offs = [], [], []
            for i in range(batch_t["flat_frames"].shape[0]):
                pat = patterns[i]
                pattern_counts[pat] = pattern_counts.get(pat, 0) + 1
                c, m, o = pack_pattern_clip(batch_t["flat_frames"][i : i + 1], pat)
                clips.append(c)
                masks.append(m)
                offs.append(o)
            clip = torch.cat(clips, dim=0)
            mask = torch.cat(masks, dim=0)
            off = torch.cat(offs, dim=0)
            opt.zero_grad(set_to_none=True)
            out = model(clip, mask=mask, offsets_norm=off)
            loss, terms = _loss(batch_t, out, args.alpha_kd)
            loss.backward()
            opt.step()
            for k, v in terms.items():
                running[k] += v
            steps += 1
        train_means = {k: v / max(steps, 1) for k, v in running.items()}
        dur = time.perf_counter() - t0
        epoch_num = ep + 1
        rec = {
            "epoch": epoch_num,
            "train": train_means,
            "pattern_exposure": pattern_counts,
            "steps": steps,
            "seconds": dur,
            "lr": args.lr,
            "gpu_mem": gpu_mem_mb(),
        }
        if epoch_num in eval_epochs or epoch_num >= epochs:
            log(f"eval epoch {epoch_num}")
            va_ds.set_epoch(0)
            ev = eval_all_patterns(model, va_loader, device, warning, epoch_num)
            rec["dev_patterns"] = ev
            pattern_eval_history["_snapshots"] = pattern_eval_history.get("_snapshots", []) + [
                {"epoch": epoch_num, "patterns": ev}
            ]
            atomic_write_json(exp_d / f"metrics_eval_epoch_{epoch_num:02d}.json", ev)
            for pid in PATTERN_IDS:
                pattern_eval_history[pid].append({"epoch": epoch_num, **ev[pid]})
        write_epoch_metrics(exp_d, epoch_num, rec)
        ckpt = exp_d / f"checkpoint_epoch_{epoch_num:02d}.pt"
        torch.save(model.state_dict(), ckpt)
        torch.save(model.state_dict(), exp_d / "checkpoint_last.pt")
        log(f"epoch {epoch_num} loss={train_means['L_total']:.4f} steps={steps} sec={dur:.1f}")

    log("part7 ablations")
    va_ds.set_epoch(0)
    ablations = {pid: run_temporal_ablations(model, va_loader, device, pid, warning) for pid in PATTERN_IDS}
    atomic_write_json(out / "part7_temporal_utilization.json", ablations)

    summary = {
        "epochs_trained": epochs,
        "steps_per_epoch_last": steps,
        "optimizer_updates": steps * epochs,
        "params": nparams,
        "pattern_eval_snapshots": pattern_eval_history.get("_snapshots", []),
        "exp_dir": str(exp_d),
    }
    atomic_write_json(exp_d / "result.json", summary)
    (exp_d / ".success").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")

    # part10 stub report
    report = out / "part10_temporal_supernet_report.md"
    report.write_text(
        "\n".join(
            [
                "# Temporal supernet screening",
                "",
                f"- Params: {nparams}",
                f"- Train windows: {len(train_m['train_row_ids'])}",
                f"- Dev windows: {len(dev_m['dev_row_ids'])}",
                f"- Epochs: {epochs}",
                f"- Experiment dir: `{exp_d.relative_to(ROOT)}`",
                "",
                "See `metrics_eval_epoch_*.json`, `part7_temporal_utilization.json`, `part3_supernet_architecture.json`.",
                "",
                "**No final frame or architecture selection in this report.**",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
