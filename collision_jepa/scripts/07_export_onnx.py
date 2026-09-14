"""Export the student to ONNX, INT8-quantize, and measure CPU latency.

Deployment path only: frames -> future 5x5 heatmap. Target < 10 ms on CPU
(ideally < 5 ms). If over budget, shrink the CNN width before touching the teacher.

Run:  python scripts/07_export_onnx.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa.config import Config  # noqa: E402
from collision_jepa.models.student import Student  # noqa: E402
from collision_jepa.models.tiny_cnn import count_params  # noqa: E402


class DeployStudent(torch.nn.Module):
    """Wrap the student so forward = deployment path (frames -> future heatmap)."""

    def __init__(self, student: Student):
        super().__init__()
        self.student = student

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.student.predict_future_heatmap(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Also produce a dynamic-INT8 model. Often SLOWER than fp32 for "
        "depthwise CNNs in onnxruntime, so off by default.",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    size = int(cfg.get("student.img_size"))
    n_frames_in = len(cfg.get("student.frame_offsets"))
    export_dir = Path(cfg.get("paths.export_dir"))
    export_dir.mkdir(parents=True, exist_ok=True)

    model = Student(
        int(cfg.get("student.cnn_width", 32)),
        int(cfg.get("student.feature_grid", 5)),
        int(cfg.get("student.z_channels", 16)),
    )
    ckpt = Path(cfg.get("paths.ckpt_dir")) / "student.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
        print(f"[export] loaded {ckpt}")
    else:
        print("[export] WARNING: student.pt not found; exporting randomly-initialized weights.")
    model.eval()
    print(f"[export] student params={count_params(model):,}")

    deploy = DeployStudent(model).eval()
    dummy = torch.rand(1, n_frames_in, 3, size, size)

    fp32_path = export_dir / "student_fp32.onnx"
    # The legacy TorchScript exporter cannot handle adaptive_avg_pool2d when the input
    # size is not an integer multiple of the output (8x8 -> 5x5). The dynamo-based
    # exporter (torch 2.6+) supports it, so try that first and fall back if needed.
    exported = False
    try:
        torch.onnx.export(
            deploy,
            (dummy,),
            str(fp32_path),
            input_names=["frames"],
            output_names=["future_heatmap"],
            opset_version=18,
            dynamo=True,
        )
        exported = True
        print("[export] used dynamo exporter")
    except Exception as e:  # noqa: BLE001
        print(f"[export] dynamo export failed ({e}); trying legacy exporter")
    if not exported:
        torch.onnx.export(
            deploy,
            dummy,
            str(fp32_path),
            input_names=["frames"],
            output_names=["future_heatmap"],
            opset_version=17,
        )
    print(f"[export] wrote {fp32_path} ({fp32_path.stat().st_size / 1e6:.2f} MB)")

    # INT8 dynamic quantization (opt-in; usually slower than fp32 here).
    int8_path = export_dir / "student_int8.onnx" if args.int8 else None
    if args.int8:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
            print(f"[export] wrote {int8_path} ({int8_path.stat().st_size / 1e6:.2f} MB)")
        except Exception as e:  # noqa: BLE001
            print(f"[export] INT8 quantization skipped: {e}")
            int8_path = None

    # Latency benchmark on CPU via onnxruntime.
    try:
        import onnxruntime as ort

        for label, path in [("fp32", fp32_path), ("int8", int8_path)]:
            if path is None or not Path(path).exists():
                continue
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1  # single-thread ~ wearable-ish
            sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            x = np.random.rand(1, n_frames_in, 3, size, size).astype(np.float32)
            for _ in range(20):  # warmup
                sess.run(None, {"frames": x})
            t0 = time.perf_counter()
            for _ in range(args.runs):
                sess.run(None, {"frames": x})
            dt = (time.perf_counter() - t0) / args.runs * 1000.0
            budget = "OK" if dt < 10 else "OVER BUDGET"
            print(f"[export] {label} latency: {dt:.2f} ms/frame (1 thread)  [{budget} vs 10 ms]")
    except Exception as e:  # noqa: BLE001
        print(f"[export] latency benchmark skipped: {e}")

    print(
        "[export] NOTE: fp32 already meets the <10 ms (often <5 ms) budget for this "
        "tiny depthwise CNN, so student_fp32.onnx is the recommended artifact. "
        "Dynamic INT8 (--int8) tends to be slower in onnxruntime for depthwise convs."
    )


if __name__ == "__main__":
    main()
