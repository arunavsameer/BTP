#!/usr/bin/env python3
"""Student H+1s street-vlog overlays (nine windows, same layout as teacher 16_render).

ConvNeXt-V2 (trial 5) or Canny CNN (trial 6): vlog + heatmap + 5×5 grid via ffmpeg decode.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

TRIAL3 = Path("/storage/BTP/model_trial_3")
TRIAL5 = Path(__file__).resolve().parent.parent
TRIAL6 = Path("/storage/BTP/model_trial_6")
SOURCE = Path("/storage/BTP/impl/input/street_walking_480p.mp4")
OFFSETS = [-90, -84, -78, -72, -66, -60, -54, -48, -42, -36, -30, -24, -18, -12, -6, 0]


def _load_teacher_render_module():
    sys.path.insert(0, str(TRIAL3))
    path = TRIAL3 / "scripts" / "16_render_street_vlog.py"
    spec = importlib.util.spec_from_file_location("street_vlog_teacher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_convnext(checkpoint: Path, device: str):
    sys.path.insert(0, str(TRIAL5))
    from collision_jepa3.data.preprocessing import uint8_to_nchw01  # noqa: WPS433
    from collision_jepa3.data.sampling import input_indices  # noqa: WPS433
    from collision_jepa3.evaluation.visualizations import (  # noqa: WPS433
        heatmap_to_image,
        overlay_heatmap,
    )
    from student_convnextv2.model import CollisionStudentConvNeXtV2  # noqa: WPS433
    from student_convnextv2.parent import load_parent_decoder  # noqa: WPS433

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    variant = str(blob.get("variant", "atto"))
    decoder = load_parent_decoder()
    model = CollisionStudentConvNeXtV2(
        decoder,
        variant=variant,
        mask_ratio=0.0,
        sparse_prob=0.0,
        grad_checkpoint=False,
    )
    model.backbone.load_state_dict(blob["backbone"], strict=True)
    model.to(device).eval()
    print(
        f"ConvNeXt-V2 {variant} from {checkpoint} epoch={blob.get('epoch')}",
        flush=True,
    )
    return model, uint8_to_nchw01, input_indices


def _load_canny(checkpoint: Path, device: str):
    sys.path.insert(0, str(TRIAL5))
    sys.path.insert(0, str(TRIAL6))
    from collision_jepa3.data.preprocessing import uint8_to_nchw01  # noqa: WPS433
    from collision_jepa3.data.sampling import input_indices  # noqa: WPS433
    from cnn_edge.canny import highlight_clip  # noqa: WPS433
    from cnn_edge.model import CollisionStudentV2  # noqa: WPS433
    from cnn_edge.parent import load_parent_decoder  # noqa: WPS433

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    decoder = load_parent_decoder()
    model = CollisionStudentV2(decoder)
    model.backbone.load_state_dict(blob["backbone"], strict=True)
    model.to(device).eval()
    print(f"Canny CNN from {checkpoint} epoch={blob.get('epoch')}", flush=True)
    return model, uint8_to_nchw01, input_indices, highlight_clip


@torch.no_grad()
def predict_student(
    model,
    letterboxed: np.ndarray,
    *,
    device: str,
    batch_size: int,
    uint8_to_nchw01,
    input_indices,
    highlight_clip=None,
    dense_mask=None,
) -> dict[int, np.ndarray]:
    n = letterboxed.shape[0]
    t0 = -min(OFFSETS)
    anchors = list(range(t0, n))
    amp = torch.bfloat16 if device.startswith("cuda") else None
    out: dict[int, np.ndarray] = {}
    for start in range(0, len(anchors), batch_size):
        chunk = anchors[start : start + batch_size]
        clips = []
        for t in chunk:
            idx = input_indices(t, OFFSETS)
            window = letterboxed[idx]
            if highlight_clip is not None:
                window = highlight_clip(window)
            clips.append(uint8_to_nchw01(window))
        x = torch.from_numpy(np.stack(clips, axis=0)).to(device)
        if amp is not None:
            with torch.autocast(device_type="cuda", dtype=amp):
                if dense_mask is not None:
                    pred = model(clip=x, mask=dense_mask)
                else:
                    pred = model(clip=x)
        else:
            pred = model(clip=x, mask=dense_mask) if dense_mask is not None else model(clip=x)
        h_1s = pred["Hhat_1s"].float().cpu().numpy()
        for i, t in enumerate(chunk):
            out[t] = h_1s[i]
        done = start + len(chunk)
        if done % 80 == 0 or done == len(anchors):
            print(f"  {done}/{len(anchors)} anchors  peak {float(h_1s.max()):.3f}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", choices=("convnext", "canny"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output folder for mp4s (default: trial visualizations/street_vlog)",
    )
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("cuda unavailable; using cpu", flush=True)
        device = "cpu"

    render = _load_teacher_render_module()
    from collision_jepa3.data.preprocessing import letterbox_rgb  # noqa: WPS433

    clips = list(render.ALL_CLIPS)
    size = int(args.size)

    if args.student == "convnext":
        model, u8, idx_fn = _load_convnext(args.checkpoint.resolve(), device)
        highlight = None
        mask = None
        label_tag = "ConvNeXt student H+1s"
        default_out = TRIAL5 / "visualizations" / "street_vlog"
    else:
        model, u8, idx_fn, highlight = _load_canny(args.checkpoint.resolve(), device)
        mask = None
        label_tag = "Canny-CNN student H+1s"
        default_out = TRIAL6 / "visualizations" / "street_vlog"

    out_dir = (args.out_dir or default_out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing nine clips to {out_dir}", flush=True)

    written = []
    for name, start, end in clips:
        print(f"decoding {name} {start:.1f}s–{end:.1f}s", flush=True)
        rgb = render.decode_window(args.source, start, end)
        letterboxed = np.stack([letterbox_rgb(rgb[i], size, 0) for i in range(rgb.shape[0])], axis=0)
        preds = predict_student(
            model,
            letterboxed,
            device=device,
            batch_size=args.batch_size,
            uint8_to_nchw01=u8,
            input_indices=idx_fn,
            highlight_clip=highlight,
            dense_mask=mask,
        )
        peak = max(float(hm.max()) for hm in preds.values())
        frames = []
        for t in range(rgb.shape[0]):
            hm = preds.get(t)
            if hm is None:
                label = f"{name}  t={t}  waiting for 3s history"
            else:
                label = f"{name}  t={t}  {label_tag}  peak {float(hm.max()):.2f}"
            frames.append(render.compose(rgb[t], hm, size, label))
        path = out_dir / name
        render.write_mp4(path, frames, render.FPS)
        written.append((path, len(frames), peak))
        print(f"  wrote {path} ({len(frames)} frames, pred peak {peak:.3f})", flush=True)

    print("summary", flush=True)
    for path, nframes, peak in written:
        print(f"  {path.name}  frames={nframes}  peak={peak:.3f}", flush=True)


if __name__ == "__main__":
    main()