#!/usr/bin/env python3
"""CLI: score and render student vs GT clips.

    /storage/BTP/impl/venv/bin/python run.py --model m1_stop_floor --checkpoint /path.pt

Outputs go to output/<model>/ so a second student does not overwrite the first.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    BATCH_SIZE,
    DEFAULT_CHECKPOINT,
    DEFAULT_MODEL,
    FRAME_OFFSETS,
    N_CLIPS,
    ROOT as CHECK_ROOT,
    T_MIN,
)
from data import clip_stem, input_indices, load_clip_frames, load_gt_heatmaps, pick_clips, uint8_to_nchw01  # noqa: E402
from infer_student import load_model  # noqa: E402
from metrics import fmt_score, score_clip  # noqa: E402
from render import render_clip_frames, write_mp4  # noqa: E402


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("cuda unavailable; using cpu", flush=True)
        return "cpu"
    return requested


@torch.no_grad()
def predict_clip(student, letterboxed: np.ndarray, *, batch_size: int) -> dict[int, np.ndarray]:
    n = int(letterboxed.shape[0])
    anchors = list(range(T_MIN, n))
    out: dict[int, np.ndarray] = {}
    for start in range(0, len(anchors), batch_size):
        chunk = anchors[start : start + batch_size]
        clips = []
        for t in chunk:
            idx = input_indices(t, FRAME_OFFSETS)
            window = np.stack([np.asarray(letterboxed[i]) for i in idx], axis=0)
            clips.append(uint8_to_nchw01(window))
        x = np.stack(clips, axis=0)
        h = student.predict(x)
        if h.ndim == 2:
            h = h[None]
        for i, t in enumerate(chunk):
            out[int(t)] = np.asarray(h[i], dtype=np.float32)
        done = start + len(chunk)
        if done % 80 == 0 or done == len(anchors):
            print(f"  {done}/{len(anchors)} anchors  peak {float(h.max()):.3f}", flush=True)
    return out


def _na(v) -> str:
    return "" if v is None else f"{float(v):.6f}"


def write_tables(out_dir: Path, clip_rows: list[dict], extra: dict) -> None:
    csv_path = out_dir / "scores.csv"
    fields = [
        "clip_id", "n_frames", "n_safe_frames", "n_stop_frames",
        "safe_score", "stop_score", "mean_spatial", "mean_intent",
        "video",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in clip_rows:
            w.writerow({
                "clip_id": r["clip_id"],
                "n_frames": r["n_frames"],
                "n_safe_frames": r["n_safe_frames"],
                "n_stop_frames": r["n_stop_frames"],
                "safe_score": _na(r["safe_score"]),
                "stop_score": _na(r["stop_score"]),
                "mean_spatial": _na(r["mean_spatial"]),
                "mean_intent": _na(r["mean_intent"]),
                "video": r["video"],
            })
    json_path = out_dir / "scores.json"
    json_path.write_text(json.dumps({"extra": extra, "clips": clip_rows}, indent=2, default=str), encoding="utf-8")

    lines = [
        "# SAFE / STOP check",
        "",
        f"- model: `{extra.get('model')}`",
        f"- checkpoint: `{extra.get('checkpoint')}`",
        f"- device: `{extra.get('device')}`",
        "",
        "| clip | SAFE | STOP | n_safe | n_stop | spatial | intent | video |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in clip_rows:
        rel = Path(r["video"]).name
        lines.append(
            f"| `{r['clip_id']}` | {fmt_score(r['safe_score'])} | {fmt_score(r['stop_score'])} | "
            f"{r['n_safe_frames']} | {r['n_stop_frames']} | {fmt_score(r['mean_spatial'])} | "
            f"{fmt_score(r['mean_intent'])} | [{rel}]({rel}) |"
        )
    lines.append("")
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {out_dir / 'index.md'}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL, help="adapter name in models/registry.py")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--out-dir", type=Path, default=None, help="default: output/<model>/")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--n-clips", type=int, default=N_CLIPS)
    p.add_argument("--smoke", action="store_true", help="render only the first clip")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    out_dir = (args.out_dir or (CHECK_ROOT / "output" / args.model)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint)
    print(f"model={args.model} checkpoint={ckpt} device={device} out={out_dir}", flush=True)

    try:
        student = load_model(args.model, ckpt, device)
    except Exception as exc:
        print(f"LOAD ERROR: {exc}", flush=True)
        raise SystemExit(f"checkpoint failed to load; not inventing heatmaps. {exc}") from exc

    n_want = 1 if args.smoke else int(args.n_clips)
    clips = pick_clips(n=n_want)
    print(f"picked {len(clips)} clips", flush=True)
    for i, rec in enumerate(clips, 1):
        print(f"  {i:02d} {rec['clip_id']}  {rec.get('outcome_type')}  {rec.get('scenario_type')}", flush=True)

    table = []
    extra = {
        "model": args.model,
        "checkpoint": str(ckpt),
        "device": device,
        "adapter_meta": student.extra() if hasattr(student, "extra") else {},
        "smoke": bool(args.smoke),
    }
    for i, rec in enumerate(clips, 1):
        stem = clip_stem(i, rec)
        print(f"\n[{i}/{len(clips)}] {rec['clip_id']}", flush=True)
        letterboxed, meta = load_clip_frames(rec)
        gt = load_gt_heatmaps(rec)
        preds = predict_clip(student, letterboxed, batch_size=int(args.batch_size))
        if not preds:
            raise RuntimeError(f"no predictions for {rec['clip_id']}")
        rows, clip = score_clip(gt, preds)
        rows_by_t = {r["t"]: r for r in rows}
        frames = render_clip_frames(
            rec,
            letterboxed,
            preds,
            gt,
            rows_by_t,
            clip,
            src_w=int(meta["source_width"]),
            src_h=int(meta["source_height"]),
        )
        video = out_dir / f"{stem}.mp4"
        write_mp4(video, frames)
        row = {
            "clip_id": rec["clip_id"],
            "sample_id": rec.get("sample_id"),
            "outcome_type": rec.get("outcome_type"),
            "scenario_type": rec.get("scenario_type"),
            "video": str(video),
            "n_frames": clip["n_frames"],
            "n_safe_frames": clip["n_safe_frames"],
            "n_stop_frames": clip["n_stop_frames"],
            "n_threat_frames": clip["n_threat_frames"],
            "safe_score": clip["safe_score"],
            "stop_score": clip["stop_score"],
            "mean_spatial": clip["mean_spatial"],
            "mean_intent": clip["mean_intent"],
            "frame_source": meta.get("source"),
        }
        table.append(row)
        print(
            f"  wrote {video}  SAFE={fmt_score(clip['safe_score'])}  "
            f"STOP={fmt_score(clip['stop_score'])}  "
            f"n_safe={clip['n_safe_frames']} n_stop={clip['n_stop_frames']}",
            flush=True,
        )
        write_tables(out_dir, table, extra)

    print("\ndone", flush=True)
    for r in table:
        print(
            f"  {r['clip_id']:55s}  SAFE={fmt_score(r['safe_score']):>6s}  "
            f"STOP={fmt_score(r['stop_score']):>6s}",
            flush=True,
        )


if __name__ == "__main__":
    main()
