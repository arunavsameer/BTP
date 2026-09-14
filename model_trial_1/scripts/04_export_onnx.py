"""Export the student deployment path (no shuffle) to ONNX."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rs_jepa.config import Config  # noqa: E402
from rs_jepa.models.student import RSJEPA  # noqa: E402
from rs_jepa.models.tiny_cnn import count_params  # noqa: E402


class DeployStudent(torch.nn.Module):
    def __init__(self, student: RSJEPA):
        super().__init__()
        self.student = student

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.student.predict_future_heatmap(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    cfg = Config.load(args.config)
    ckpt = Path(cfg.get("paths.ckpt_dir")) / "student.pt"
    model = RSJEPA.from_config(cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    model.eval()
    wrap = DeployStudent(model)
    wrap.eval()
    size = int(cfg.get("student.img_size"))
    n_clip = len(list(cfg.get("student.frame_offsets")))
    dummy = torch.zeros(1, n_clip, 3, size, size)
    out_dir = Path(cfg.get("paths.export_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "student_fp32.onnx"
    torch.onnx.export(
        wrap,
        dummy,
        str(onnx_path),
        input_names=["frames"],
        output_names=["heatmap"],
        opset_version=17,
        dynamo=False,
    )
    print(f"[export] wrote {onnx_path}  params={count_params(model):,}")

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        x = dummy.numpy()
        for _ in range(10):
            sess.run(None, {"frames": x})
        t0 = time.perf_counter()
        n = 100
        for _ in range(n):
            sess.run(None, {"frames": x})
        ms = (time.perf_counter() - t0) * 1000 / n
        print(f"[export] CPU latency {ms:.2f} ms/frame")
    except Exception as exc:
        print(f"[export] onnxruntime bench skipped: {exc}")


if __name__ == "__main__":
    main()
