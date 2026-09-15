"""Evaluate student and/or teacher on val/test: heatmap + wearable nuisance/miss."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvjepa.config import Config, resolve_device  # noqa: E402
from cvjepa.data import load_json  # noqa: E402
from cvjepa.data.dataset import StudentDataset  # noqa: E402
from cvjepa.engine import evaluate_future_heatmap, evaluate_wearable  # noqa: E402
from cvjepa.metrics import format_summary, format_wearable  # noqa: E402
from cvjepa.models.student import CollisionStudent  # noqa: E402
from cvjepa.models.teacher import CollisionTeacher  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--split", default="val", choices=["val", "test", "both"])
    parser.add_argument("--who", default="student", choices=["student", "teacher", "both"])
    parser.add_argument("--student-ckpt", default="")
    parser.add_argument("--teacher-ckpt", default="")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    device = resolve_device(cfg.get("train.device", "auto"))
    split = load_json(cfg.get("data.split_file"))
    has_rare = split.get("has_rare", {})
    names = []
    if args.split in ("val", "both"):
        names.append(("val", split["val"]))
    if args.split in ("test", "both"):
        names.append(("test", split.get("test") or []))

    if args.who in ("student", "both"):
        offsets = list(cfg.get("student.frame_offsets"))
        model = CollisionStudent(
            width=int(cfg.get("student.cnn_width", 32)),
            feature_grid=int(cfg.get("student.feature_grid", 5)),
            z_channels=int(cfg.get("student.z_channels", 32)),
            n_frames=len(offsets),
            use_loom=bool(cfg.get("student.use_loom", True)),
            copy_residual=bool(cfg.get("student.copy_residual", True)),
        ).to(device)
        ckpt = Path(args.student_ckpt) if args.student_ckpt else Path(cfg.get("paths.ckpt_dir")) / "student_best.pt"
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()
        print(f"[eval] student {ckpt}")
        for split_name, eps in names:
            if not eps:
                continue
            ds = StudentDataset(
                cfg.get("data.cache_dir"),
                eps,
                int(cfg.get("student.img_size")),
                offsets,
                int(cfg.get("horizon.tau_frames")),
                load_teacher_z=False,
            )
            loader = DataLoader(ds, batch_size=int(cfg.get("train.batch_size")), shuffle=False)
            hm = evaluate_future_heatmap(
                lambda b: model.predict_future_heatmap(b["frames"]),
                loader,
                cfg,
                device,
                ds.episodes,
                has_rare,
            )
            wear = evaluate_wearable(cfg, eps, has_rare, device, model=model)
            print(f"[eval] student {split_name}  {format_summary('hm', hm['overall'])}  {format_wearable('wear', wear['overall'])}")

    if args.who in ("teacher", "both"):
        import os

        hf_cache = Path(cfg.get("teacher.hf_cache_dir", ROOT / "hf_cache"))
        os.environ.setdefault("HF_HOME", str(hf_cache))
        teacher = CollisionTeacher(
            hf_model_id=cfg.get("teacher.hf_model_id"),
            z_channels=int(cfg.get("teacher.z_channels", 32)),
            feature_grid=int(cfg.get("student.feature_grid", 5)),
            pool_hidden=int(cfg.get("teacher.pool_hidden", 128)),
            cache_dir=hf_cache,
            torch_dtype=str(cfg.get("teacher.torch_dtype", "float16")),
        ).to(device)
        ckpt = Path(args.teacher_ckpt) if args.teacher_ckpt else Path(cfg.get("paths.ckpt_dir")) / "teacher_best.pt"
        state = torch.load(ckpt, map_location=device, weights_only=False)
        if state.get("lora"):
            teacher.backbone.enable_lora(
                last_n=int(cfg.get("teacher.lora_last_blocks", 6)),
                rank=int(cfg.get("teacher.lora_rank", 16)),
                alpha=float(cfg.get("teacher.lora_alpha", 16.0)),
            )
            teacher.backbone.set_backbone_grad(False)
        teacher.load_state_dict(state["model"], strict=False)
        teacher.eval()
        print(f"[eval] teacher {ckpt}")
        for split_name, eps in names:
            if not eps:
                continue
            wear = evaluate_wearable(cfg, eps, has_rare, device, model=teacher, teacher=True)
            print(f"[eval] teacher {split_name}  {format_wearable('wear', wear['overall'])}")


if __name__ == "__main__":
    main()
