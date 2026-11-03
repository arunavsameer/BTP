"""Frozen-encoder trainer. The trunk is the only optimized module."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..evaluation.evaluator import evaluate_loader
from ..losses.composite import TeacherLoss
from ..models.teacher import CollisionTeacher4
from ..reproducibility import software_versions
from ..utils.device import describe_hardware, select_amp_dtype
from ..utils.logging import JsonlLogger
from .checkpoint import extract_teacher_state, load_checkpoint, load_teacher_state, save_checkpoint
from .early_stopping import EarlyStopping
from .engine import assert_finite, current_lrs, horizon_mse, move_batch, selection_better, teacher_forward
from .optimizer import build_optimizer, build_scheduler

logger = logging.getLogger("collision_jepa4.trainer")


class TeacherTrainer:
    def __init__(
        self,
        cfg,
        model: CollisionTeacher4,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: str,
        split_fingerprint: str,
        manifest_fingerprint: str,
        preprocess_signature: dict,
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.split_fingerprint = split_fingerprint
        self.manifest_fingerprint = manifest_fingerprint
        self.preprocess_signature = preprocess_signature
        self.loss_fn = TeacherLoss(cfg, device=device)
        self.amp_dtype = select_amp_dtype(cfg, device)
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp_dtype is torch.float16)
            if device.startswith("cuda")
            else torch.amp.GradScaler("cuda", enabled=False)
        )
        self.ckpt_root = cfg.path("paths.checkpoint_root")
        self.log = JsonlLogger(cfg.path("paths.logs_root") / "train.jsonl")
        self.global_step = 0
        self.best_sel: dict[str, Any] | None = None

    def _autocast(self):
        if self.amp_dtype is None or not self.device.startswith("cuda"):
            from contextlib import nullcontext

            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _forward_loss(self, batch) -> tuple[torch.Tensor, dict[str, float]]:
        batch = move_batch(batch, self.device)
        with self._autocast():
            out = teacher_forward(self.model, batch)
            packed = self.loss_fn(out, batch)
        assert_finite(packed.total, "L_total")
        return packed.total, packed.as_floats()

    def _make_optimizer(self, epochs: int):
        opt = build_optimizer(
            self.model.trainable_parameters(),
            float(self.cfg.get("optimizer.adapter_lr", 0.0003)),
            float(self.cfg.get("optimizer.weight_decay", 0.0001)),
            name=str(self.cfg.get("optimizer.name", "AdamW")),
        )
        sch = build_scheduler(
            opt,
            epochs,
            float(self.cfg.get("optimizer.min_lr", 1e-6)),
            warmup_epochs=int(self.cfg.get("optimizer.warmup_epochs", 1)),
        )
        return opt, sch

    def _state(self, epoch: int, optimizer, scheduler, extra: dict) -> dict[str, Any]:
        blob = extract_teacher_state(self.model)
        blob.update(
            {
                "phase": "frozen",
                "epoch": epoch,
                "global_step": self.global_step,
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": self.scaler.state_dict(),
                "best_selection": self.best_sel,
                "config": self.cfg.to_dict(),
                "backbone_model_id": getattr(self.model.backbone, "model_id", None),
                "backbone_revision": getattr(self.model.backbone, "revision", None),
                "preprocess_signature": self.preprocess_signature,
                "seed": int(self.cfg.get("project.seed", 1337)),
                "split_fingerprint": self.split_fingerprint,
                "manifest_fingerprint": self.manifest_fingerprint,
                "software": software_versions(),
                "parameter_audit": self.model.parameter_audit(),
                **extra,
            }
        )
        return blob

    def _validate(self) -> dict[str, Any]:
        if self.val_loader is None:
            return {"selection": {"score": float("nan"), "eligible": False}, "horizons": {}}
        return evaluate_loader(self.model, self.val_loader, self.cfg, self.device, loss_fn=self.loss_fn)

    def _save(self, name: str, blob: dict) -> Path:
        path = self.ckpt_root / name
        save_checkpoint(blob, path)
        logger.info("wrote %s", path)
        return path

    def run_phase(self, epochs: int, resume: dict | None = None) -> dict[str, Any]:
        logger.info("=== frozen trunk for %d epochs ===", epochs)
        logger.info("parameter audit: %s", self.model.parameter_audit())
        logger.info("hardware: %s", describe_hardware(self.device, getattr(self.model.backbone, "model_id", None)))
        opt, sch = self._make_optimizer(epochs)
        start_epoch = 0
        if resume is not None:
            load_teacher_state(self.model, resume, strict=True)
            if resume.get("optimizer"):
                opt.load_state_dict(resume["optimizer"])
            if sch is not None and resume.get("scheduler"):
                sch.load_state_dict(resume["scheduler"])
            if resume.get("scaler"):
                self.scaler.load_state_dict(resume["scaler"])
            start_epoch = int(resume.get("epoch", 0)) + 1
            self.global_step = int(resume.get("global_step", 0))
            self.best_sel = resume.get("best_selection")
            logger.info("resumed frozen trunk from epoch %d step %d", start_epoch, self.global_step)
        stopper = EarlyStopping(int(self.cfg.get("training.patience", 4)))
        stop_metric = str(self.cfg.get("training.early_stop_metric", "selection"))
        if stop_metric not in {"selection", "val_mse"}:
            raise ValueError(f"unsupported training.early_stop_metric {stop_metric}")
        patience_start = int(self.cfg.get("training.patience_start_epoch", 0))
        prev_val_mse: float | None = None
        accum = max(1, int(self.cfg.get("training.gradient_accumulation_steps", 1)))
        clip_norm = float(self.cfg.get("optimizer.grad_clip_norm", 1.0))
        last_name = "teacher_phase_a_last.pt"
        best_name = "teacher_phase_a_best.pt"
        history = []
        for epoch in range(start_epoch, epochs):
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            self.model.train()
            t0 = time.perf_counter()
            running: dict[str, float] = {}
            n_steps = 0
            opt.zero_grad(set_to_none=True)
            max_steps = int(self.cfg.get("training.max_steps", 0) or 0)
            log_every = int(self.cfg.get("training.log_every_steps", 0) or 0)
            for step, batch in enumerate(self.train_loader):
                loss, terms = self._forward_loss(batch)
                self.scaler.scale(loss / accum).backward()
                if (step + 1) % accum == 0:
                    self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), clip_norm)
                    self.scaler.step(opt)
                    self.scaler.update()
                    opt.zero_grad(set_to_none=True)
                    self.global_step += 1
                for k, v in terms.items():
                    running[k] = running.get(k, 0.0) + v
                n_steps += 1
                if log_every > 0 and n_steps % log_every == 0:
                    logger.info(
                        "epoch=%d step=%d/%s L_total=%.4f L_1s=%.4f",
                        epoch,
                        n_steps,
                        len(self.train_loader),
                        terms.get("L_total", float("nan")),
                        terms.get("L_1s", float("nan")),
                    )
                if max_steps > 0 and n_steps >= max_steps:
                    logger.info("stopping epoch early after max_steps=%d", max_steps)
                    break
            if n_steps and (n_steps % accum) != 0:
                self.scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), clip_norm)
                self.scaler.step(opt)
                self.scaler.update()
                opt.zero_grad(set_to_none=True)
            if sch is not None:
                sch.step()
            train_means = {k: v / max(n_steps, 1) for k, v in running.items()}
            val = self._validate()
            sel = val.get("selection", {})
            mse = horizon_mse(val, self.cfg)
            h1s = (val.get("horizons") or {}).get("1s") or {}
            if stop_metric == "val_mse":
                better = prev_val_mse is None or (mse == mse and mse < prev_val_mse)
            else:
                better = selection_better(sel, self.best_sel)
            blob = self._state(epoch, opt, sch, {"val": val, "train": train_means})
            self._save(last_name, blob)
            if bool(self.cfg.get("training.save_every_epoch", False)):
                self._save(f"teacher_phase_a_epoch_{epoch:02d}.pt", blob)
            if better:
                tagged = dict(sel)
                tagged["val_mse"] = mse
                tagged["best_epoch"] = epoch
                tagged["stop_f1"] = h1s.get("stop_f1")
                tagged["nuisance"] = h1s.get("nuisance")
                tagged["miss"] = h1s.get("miss")
                self.best_sel = tagged
                self._save(best_name, blob)
            dt = time.perf_counter() - t0
            rec = {
                "phase": "frozen",
                "epoch": epoch,
                "seconds": dt,
                "lrs": current_lrs(opt),
                "train": train_means,
                "val_selection": sel,
                "val_1s": h1s,
                "val_mse": mse,
                "gpu": describe_hardware(self.device),
            }
            self.log.write(rec)
            history.append(rec)
            logger.info(
                "epoch=%d L_total=%.4f L_1s=%.4f val_mse=%.5f score=%s eligible=%s time=%.1fs",
                epoch,
                train_means.get("L_total", float("nan")),
                train_means.get("L_1s", float("nan")),
                mse,
                sel.get("score"),
                sel.get("eligible"),
                dt,
            )
            logger.info(
                "epoch=%d val 1s: ht_p=%.3f ht_r=%.3f ht_f1=%.3f dir_acc=%.3f "
                "stop_p=%.3f stop_r=%.3f stop_f1=%.3f safe_r=%.3f nuisance=%.3f miss=%.3f "
                "pre_stop_r=%.3f post_stop_r=%.3f",
                epoch,
                float(h1s.get("high_threat_precision", float("nan"))),
                float(h1s.get("high_threat_recall", float("nan"))),
                float(h1s.get("high_threat_f1", float("nan"))),
                float(h1s.get("direction_accuracy", float("nan"))),
                float(h1s.get("stop_precision", float("nan"))),
                float(h1s.get("stop_recall", float("nan"))),
                float(h1s.get("stop_f1", float("nan"))),
                float(h1s.get("safe_recall", float("nan"))),
                float(h1s.get("nuisance", float("nan"))),
                float(h1s.get("miss", float("nan"))),
                float(h1s.get("pre_fork_stop_recall", float("nan"))),
                float(h1s.get("post_fork_stop_recall", float("nan"))),
            )
            if stop_metric == "val_mse":
                prev_val_mse = mse
            if epoch < patience_start:
                stopper.step(True)
            elif stopper.step(better):
                keep = None if self.best_sel is None else self.best_sel.get("best_epoch")
                why = "val_mse" if stop_metric == "val_mse" else "val_score"
                logger.info(
                    "early stopping at epoch %d (%s did not improve for %d epochs); keeping epoch %s",
                    epoch,
                    why,
                    stopper.patience,
                    keep,
                )
                break
        return {"phase": "frozen", "history": history, "best": self.best_sel}

    def run(self, resume_path: Path | None = None) -> dict[str, Any]:
        resume = load_checkpoint(resume_path, map_location=self.device) if resume_path else None
        epochs = int(self.cfg.get("training.frozen_epochs", 8))
        return {"frozen": self.run_phase(epochs, resume if resume and resume.get("phase") == "frozen" else None)}
