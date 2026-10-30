"""Train the RGB student against a frozen teacher G (adapter on token cache)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from collision_jepa3.reproducibility import software_versions
from collision_jepa3.training.checkpoint import load_checkpoint, save_checkpoint
from collision_jepa3.training.early_stopping import EarlyStopping
from collision_jepa3.training.engine import (
    assert_finite,
    balanced_warning_better,
    current_lrs,
    horizon_mse,
    move_batch,
    prefork_stop_better,
    recall5_better,
    selection_better,
)
from collision_jepa3.training.optimizer import build_scheduler
from collision_jepa3.utils.device import describe_hardware, select_amp_dtype
from collision_jepa3.utils.logging import JsonlLogger

from .evaluate import evaluate_student, student_forward
from .loss import StudentLoss
from .student import CollisionStudent

logger = logging.getLogger("student_model.trainer")


@torch.no_grad()
def teacher_features(teacher, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    z, fine = teacher.encode_pair(tokens=tokens)
    out = teacher._heads(z, fine)
    out["F"] = fine
    return out


class StudentTrainer:
    def __init__(
        self,
        cfg,
        student: CollisionStudent,
        teacher,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: str,
        split_fingerprint: str,
        manifest_fingerprint: str,
        preprocess_signature: dict,
        teacher_ckpt: str,
    ):
        self.cfg = cfg
        self.student = student.to(device)
        self.teacher = teacher.to(device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.split_fingerprint = split_fingerprint
        self.manifest_fingerprint = manifest_fingerprint
        self.preprocess_signature = preprocess_signature
        self.teacher_ckpt = str(teacher_ckpt)
        self.loss_fn = StudentLoss(cfg, device=device)
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

    def _forward_loss(self, batch, epoch: int):
        batch = move_batch(batch, self.device)
        with self._autocast():
            t_out = None
            if "tokens" in batch:
                t_out = teacher_features(self.teacher, batch["tokens"])
            s_out = student_forward(self.student, batch)
            packed = self.loss_fn(
                s_out,
                batch,
                t_out,
                epoch=epoch,
                predictor=self.student.predictor,
            )
        assert_finite(packed.total, "L_total")
        return packed.total, packed.as_floats()

    def _state(self, epoch: int, optimizer, scheduler, extra: dict) -> dict[str, Any]:
        return {
            "encoder": self.student.encoder.state_dict(),
            "predictor": self.student.predictor.state_dict(),
            "hint": self.student.predictor.state_dict(),
            "bottleneck": self.student.bottleneck.state_dict(),
            "norm": self.student.norm.state_dict(),
            "decoder": self.student.decoder.state_dict(),
            "close_head": self.student.close_head.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": self.scaler.state_dict(),
            "best_selection": self.best_sel,
            "config": self.cfg.to_dict(),
            "teacher_checkpoint": self.teacher_ckpt,
            "preprocess_signature": self.preprocess_signature,
            "seed": int(self.cfg.get("project.seed", 1337)),
            "split_fingerprint": self.split_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "software": software_versions(),
            "parameter_audit": self.student.parameter_audit(),
            **extra,
        }

    def _validate(self) -> dict[str, Any]:
        if self.val_loader is None:
            return {"selection": {"score": float("nan"), "eligible": False}}
        return evaluate_student(
            self.student,
            self.val_loader,
            self.cfg,
            self.device,
            loss_fn=self.loss_fn,
            teacher=self.teacher,
        )

    def run(self, epochs: int) -> dict[str, Any]:
        logger.info("parameter audit: %s", self.student.parameter_audit())
        logger.info("hardware: %s", describe_hardware(self.device, "student_g"))
        enc = self.student.encoder_parameters()
        pred = self.student.predictor_parameters()
        params = enc + pred
        wd = float(self.cfg.get("optimizer.weight_decay", 0.0001))
        groups = [{"params": enc, "lr": float(self.cfg.get("optimizer.adapter_lr", 0.0003)), "weight_decay": wd}]
        if pred:
            groups.append(
                {
                    "params": pred,
                    "lr": float(self.cfg.get("student.predictor_lr", 0.0003)),
                    "weight_decay": wd,
                }
            )
        opt = torch.optim.AdamW(groups)
        sch = build_scheduler(
            opt,
            epochs,
            float(self.cfg.get("optimizer.min_lr", 1e-6)),
            warmup_epochs=int(self.cfg.get("optimizer.warmup_epochs", 1)),
        )
        stopper = EarlyStopping(int(self.cfg.get("training.patience", 4)))
        stop_metric = str(self.cfg.get("training.early_stop_metric", "latent_l1"))
        patience_start = int(self.cfg.get("training.patience_start_epoch", 0))
        post_floor = float(self.cfg.get("training.post_fork_stop_floor", 0.90))
        safe_floor = float(self.cfg.get("training.safe_recall_floor", 0.82))
        stop_floor = float(self.cfg.get("training.stop_recall_floor", 0.75))
        max_safe_stop = float(self.cfg.get("training.max_safe_stop_rate", 0.08))
        accum = max(1, int(self.cfg.get("training.gradient_accumulation_steps", 1)))
        clip_norm = float(self.cfg.get("optimizer.grad_clip_norm", 1.0))
        history = []
        for epoch in range(epochs):
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            self.student.train()
            self.teacher.eval()
            t0 = time.perf_counter()
            running: dict[str, float] = {}
            n_steps = 0
            opt.zero_grad(set_to_none=True)
            max_steps = int(self.cfg.get("training.max_steps", 0) or 0)
            for step, batch in enumerate(self.train_loader):
                loss, terms = self._forward_loss(batch, epoch)
                self.scaler.scale(loss / accum).backward()
                if (step + 1) % accum == 0:
                    self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, clip_norm)
                    self.scaler.step(opt)
                    self.scaler.update()
                    opt.zero_grad(set_to_none=True)
                    self.global_step += 1
                for k, v in terms.items():
                    running[k] = running.get(k, 0.0) + v
                n_steps += 1
                if max_steps > 0 and n_steps >= max_steps:
                    logger.info("stopping epoch early after max_steps=%d", max_steps)
                    break
            if n_steps and (n_steps % accum) != 0:
                self.scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, clip_norm)
                self.scaler.step(opt)
                self.scaler.update()
                opt.zero_grad(set_to_none=True)
            if sch is not None:
                sch.step()
            train_means = {k: v / max(n_steps, 1) for k, v in running.items()}
            val = self._validate()
            sel = val.get("selection", {})
            mse = horizon_mse(val, self.cfg)
            operating = val.get("operating") or {}
            h1s = (val.get("horizons") or {}).get("1s") or {}
            latent = val.get("latent") or {}
            if stop_metric == "blob_l1":
                cur = latent.get("blob_l1")
                prev = None if self.best_sel is None else self.best_sel.get("latent_blob_l1")
                better = cur is not None and cur == cur and (prev is None or float(cur) < float(prev) - 1e-8)
            elif stop_metric == "small_stop_recall":
                cur = latent.get("small_stop_recall")
                prev = None if self.best_sel is None else self.best_sel.get("small_stop_recall")
                better = cur is not None and cur == cur and (prev is None or float(cur) > float(prev) + 1e-8)
            elif stop_metric == "latent_l1":
                cur = latent.get("l1")
                prev = None if self.best_sel is None else self.best_sel.get("latent_l1")
                better = cur is not None and cur == cur and (prev is None or float(cur) < float(prev) - 1e-8)
            elif stop_metric == "recall_at_5fa":
                better = recall5_better(operating, self.best_sel)
            elif stop_metric == "prefork_stop":
                better = prefork_stop_better(h1s, self.best_sel, post_floor=post_floor)
            elif stop_metric == "balanced_warning":
                better = balanced_warning_better(
                    h1s,
                    self.best_sel,
                    post_floor=post_floor,
                    safe_floor=safe_floor,
                    stop_floor=stop_floor,
                    max_safe_stop=max_safe_stop,
                )
            else:
                better = selection_better(sel, self.best_sel)
            blob = self._state(epoch, opt, sch, {"val": val, "train": train_means})
            save_checkpoint(blob, self.ckpt_root / "student_last.pt")
            if bool(self.cfg.get("training.save_every_epoch", False)):
                save_checkpoint(blob, self.ckpt_root / f"student_epoch_{epoch:02d}.pt")
            if better:
                tagged = dict(sel)
                tagged["val_mse"] = mse
                tagged["best_epoch"] = epoch
                tagged["latent_l1"] = latent.get("l1")
                tagged["latent_cos"] = latent.get("cos")
                tagged["latent_blob_l1"] = latent.get("blob_l1")
                tagged["small_stop_recall"] = latent.get("small_stop_recall")
                tagged.update(
                    {
                        k: operating.get(k)
                        for k in (
                            "recall_at_5fa",
                            "ap_high",
                            "mean_pred_gt80",
                            "mse",
                            "threshold_5fa",
                            "fa_achieved_5fa",
                            "fa_at_0.5",
                            "recall_at_0.5",
                        )
                    }
                )
                tagged.update(
                    {
                        k: h1s.get(k)
                        for k in (
                            "pre_fork_stop_recall",
                            "post_fork_stop_recall",
                            "pre_fork_stop_n",
                            "post_fork_stop_n",
                            "post_fork_safe_stop_rate",
                            "safe_recall",
                            "stop_recall",
                            "safe_stop_rate",
                        )
                    }
                )
                self.best_sel = tagged
                save_checkpoint(blob, self.ckpt_root / "student_best.pt")
            dt = time.perf_counter() - t0
            rec = {
                "epoch": epoch,
                "seconds": dt,
                "lrs": current_lrs(opt),
                "train": train_means,
                "val_selection": sel,
                "val_mse": mse,
                "val_latent": latent,
                "val_operating": operating,
                "val_fork_stop": {
                    "pre_fork_stop_recall": h1s.get("pre_fork_stop_recall"),
                    "post_fork_stop_recall": h1s.get("post_fork_stop_recall"),
                    "post_fork_safe_stop_rate": h1s.get("post_fork_safe_stop_rate"),
                    "safe_recall": h1s.get("safe_recall"),
                    "stop_recall": h1s.get("stop_recall"),
                    "safe_stop_rate": h1s.get("safe_stop_rate"),
                },
                "gpu": describe_hardware(self.device),
            }
            self.log.write(rec)
            history.append(rec)
            logger.info(
                "epoch=%d L_feat=%.4f L_h=%s L_stop=%s L_safe=%s L_total=%.4f val_L1=%s blob_L1=%s val_cos=%s rec@5FA=%s SAFE=%s STOP=%s smallSTOP=%s time=%.1fs",
                epoch,
                train_means.get("L_feat", float("nan")),
                train_means.get("L_h"),
                train_means.get("L_stop"),
                train_means.get("L_safe"),
                train_means.get("L_total", float("nan")),
                latent.get("l1"),
                latent.get("blob_l1"),
                latent.get("cos"),
                operating.get("recall_at_5fa"),
                h1s.get("safe_recall"),
                h1s.get("stop_recall"),
                latent.get("small_stop_recall"),
                dt,
            )
            if epoch < patience_start:
                stopper.step(True)
            elif stopper.step(better):
                logger.info(
                    "early stopping at epoch %d; keeping epoch %s",
                    epoch,
                    None if self.best_sel is None else self.best_sel.get("best_epoch"),
                )
                break
        return {"history": history, "best": self.best_sel}
