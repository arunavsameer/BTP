#!/usr/bin/env python3
"""Write comparison table after the six overnight runs (selection only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.student_research.overnight6.train import NAMES, exp_dir  # noqa: E402

OUT = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp/overnight6")


def last_metrics(mid: int):
    d = exp_dir(mid)
    files = sorted(d.glob("metrics_epoch_*.json"))
    if not files:
        return None, None
    rec = json.loads(files[-1].read_text())
    return rec, files[-1]


def best_rec(mid: int):
    d = exp_dir(mid)
    p = d / "checkpoint_best.pt"
    mets = sorted(d.glob("metrics_epoch_*.json"))
    best = None
    for f in mets:
        rec = json.loads(f.read_text())
        sel = rec.get("selection") or {}
        sa = float(sel.get("safe_recall") or 0)
        st = float(sel.get("stop_recall") or 0)
        key = (1 if sa >= 0.85 else 0, st, sa)
        if best is None or key > best[0]:
            best = (key, rec)
    return best[1] if best else None, p if p.is_file() else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Overnight Pair-CF six-model comparison (selection split only)",
        "",
        "| Model | Best epoch | SAFE | THREAT | STOP | STOP prec | Pre-fork STOP | Post-fork STOP | Parent recovery | Cosine | Params | CPU ms | SAFE≥85 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    summary = []
    for mid in range(1, 7):
        rec, ck = best_rec(mid)
        cfgp = exp_dir(mid) / "config.json"
        cfg = json.loads(cfgp.read_text()) if cfgp.is_file() else {}
        ok = (exp_dir(mid) / ".success").is_file()
        fail = (exp_dir(mid) / ".failed").is_file()
        if rec is None:
            lines.append(f"| {NAMES[mid]} | — | — | — | — | — | — | — | — | — | {cfg.get('params', {}).get('total')} | {cfg.get('cpu_latency_ms')} | {'FAIL' if fail else 'missing'} |")
            summary.append({"model": NAMES[mid], "status": "failed" if fail else "missing"})
            continue
        sel = rec["selection"]
        pk = sel.get("peaks") or {}
        sa = sel.get("safe_recall")
        row = {
            "model": NAMES[mid],
            "epoch": rec["epoch"],
            "safe": sa,
            "threat": sel.get("threat_recall"),
            "stop": sel.get("stop_recall"),
            "stop_prec": sel.get("stop_precision"),
            "pre": sel.get("pre_fork_stop_recall"),
            "post": sel.get("post_fork_stop_recall"),
            "recovery": pk.get("parent_correct_stop_recovery"),
            "cos": sel.get("fdec_cosine"),
            "params": cfg.get("params", {}).get("total"),
            "cpu_ms": cfg.get("cpu_latency_ms"),
            "safe85": bool(sa is not None and sa >= 0.85),
            "best_ckpt": str(ck) if ck else None,
            "last_ckpt": str(exp_dir(mid) / "checkpoint_last.pt"),
            "success": ok,
            "failed": fail,
            "drift": rec.get("train", {}).get("decoder_drift_rms"),
        }
        summary.append(row)
        lines.append(
            f"| {NAMES[mid]} | {row['epoch']} | {row['safe']} | {row['threat']} | {row['stop']} | {row['stop_prec']} | {row['pre']} | {row['post']} | {row['recovery']} | {row['cos']} | {row['params']} | {row['cpu_ms']} | {row['safe85']} |"
        )
    lines += ["", "## Status", json.dumps(summary, indent=2, default=str)]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print((OUT / "FINAL_REPORT.md").read_text())


if __name__ == "__main__":
    main()
