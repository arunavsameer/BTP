"""Two-phase teacher trainer: frozen encoder, then optional LoRA."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..evaluation.evaluator import evaluate_loader
from ..losses.composite import TeacherLoss
from ..models.teacher import CollisionTeacher3
from ..reproducibility import software_versions
from ..utils.device import describe_hardware, select_amp_dtype
from ..utils.logging import JsonlLogger
from .checkpoint import extract_teacher_state, load_checkpoint, load_teacher_state, save_checkpoint
from .early_stopping import EarlyStopping
from .engine import (
    assert_finite,
    current_lrs,
    horizon_mse,
    move_batch,
    balanced_warning_better,
    prefork_stop_better,
    recall5_better,
    selection_better,
    teacher_forward,
)
from .optimizer import build_param_groups, build_scheduler

logger = logging.getLogger("collision_jepa3.trainer")


class PairBuffer:
    """Stop-grad store of recent Z_t keyed by (background_pair_id, anchor)."""

    def __init__(self, max_items: int = 64):
        self.max_items = int(max_items)
        self._store: OrderedDict[tuple[str, int], tuple[str, torch.Tensor]] = OrderedDict()

    def pop_mate(self, pair_id: str, anchor: int, clip_id: str) -> torch.Tensor | None:
        key = (pair_id, int(anchor))
        if key not in self._store:
            return None
        mate_id, z = self._store[key]
        if mate_id == clip_id:
            return None
        return z

    def put(self, pair_id: str, anchor: int, clip_id: str, z: torch.Tensor) -> None:
        key = (pair_id, int(anchor))
        self._store[key] = (clip_id, z.detach())
        self._store.move_to_end(key)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)


class TeacherTrainer:
    def __init__(
        self,
        cfg,
        model: CollisionTeacher3,
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
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype is torch.float16) if device.startswith("cuda") else torch.amp.GradScaler("cuda", enabled=False)
        self.pair_buffer = PairBuffer(int(cfg.get("training.max_pair_buffer", 64)))
        self.ckpt_root = cfg.path("paths.checkpoint_root")
        self.log = JsonlLogger(cfg.path("paths.logs_root") / "train.jsonl")
        self.global_step = 0
        self.best_sel: dict[str, Any] | None = None

    def _autocast(self):
        if self.amp_dtype is None or not self.device.startswith("cuda"):
            from contextlib import nullcontext

            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _bg_pair_tensor(self, out, batch) -> torch.Tensor | None:
        z = out["Z_t"]
        mates = []
        valid = batch["bg_pair_valid"]
        valid_list = valid.detach().cpu().tolist() if torch.is_tensor(valid) else list(valid)
        anchors = batch["anchor"]
        anchor_list = anchors.detach().cpu().tolist() if torch.is_tensor(anchors) else list(anchors)
        for i in range(z.shape[0]):
            if not bool(valid_list[i]):
                continue
            pair_id = batch["background_pair_id"][i]
            anchor = int(anchor_list[i])
            clip_id = batch["clip_id"][i]
            mate = self.pair_buffer.pop_mate(pair_id, anchor, clip_id)
            if mate is not None:
                mates.append((i, mate.to(device=z.device, dtype=z.dtype)))
            self.pair_buffer.put(pair_id, anchor, clip_id, z[i])
        if not mates:
            return None
        # Align stored mates to the current batch slots via gathering one pair at a time.
        # Average cosine over available pairs by stacking them as a mini comparison.
        z_cur = torch.stack([z[i] for i, _ in mates], dim=0)
        z_mate = torch.stack([m for _, m in mates], dim=0)
        from ..losses.background_consistency import background_consistency_loss

        return background_consistency_loss(z_cur, z_mate)

    def _forward_loss(self, batch) -> tuple[torch.Tensor, dict[str, float]]:
        batch = move_batch(batch, self.device)
        with self._autocast():
            out = teacher_forward(self.model, batch)
            packed = self.loss_fn(out, batch)
            bg_term = self._bg_pair_tensor(out, batch)
            if bg_term is not None:
                packed.terms["L_bg"] = bg_term
                packed.total = packed.total - self.loss_fn.bg_w * packed.terms["L_bg"].detach() + self.loss_fn.bg_w * bg_term
        assert_finite(packed.total, "L_total")
        return packed.total, packed.as_floats()

    def _make_optimizer(self, phase: str, epochs: int):
        include_lora = phase == "lora"
        adapter_params = self.model.adapter_parameters()
        lora_params = self.model.lora_parameters() if include_lora else []
        if include_lora:
            groups = build_param_groups(
                adapter_params,
                lora_params,
                float(self.cfg.get("optimizer.lora_adapter_lr", 0.0002)),
                float(self.cfg.get("optimizer.lora_lr", 0.0001)),
                float(self.cfg.get("optimizer.weight_decay", 0.0001)),
            )
        else:
            groups = build_param_groups(
                adapter_params,
                [],
                float(self.cfg.get("optimizer.adapter_lr", 0.001)),
                float(self.cfg.get("optimizer.lora_lr", 0.0001)),
                float(self.cfg.get("optimizer.weight_decay", 0.0001)),
            )
        opt = torch.optim.AdamW(groups)
        sch = build_scheduler(
            opt,
            epochs,
            float(self.cfg.get("optimizer.min_lr", 1e-6)),
            warmup_epochs=int(self.cfg.get("optimizer.warmup_epochs", 1)),
        )
        return opt, sch

    def _state(self, phase: str, epoch: int, optimizer, scheduler, extra: dict) -> dict[str, Any]:
        blob = extract_teacher_state(self.model)
        blob.update(
            {
                "phase": phase,
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

    def _disable_token_cache_for_lora(self) -> None:
        """LoRA changes encoder tokens; rebuild pixel loaders if we were on the cache."""
        from ..data.dataset import build_dataloader

        ds_t = getattr(self.train_loader, "dataset", None)
        if ds_t is None or not getattr(ds_t, "use_token_cache", False):
            return
        logger.info("LoRA phase: dropping encoder-token cache and running the backbone")
        ds_t.use_token_cache = False
        self.train_loader = build_dataloader(ds_t, self.cfg, shuffle=True, drop_last=False)
        if self.val_loader is not None:
            ds_v = self.val_loader.dataset
            if hasattr(ds_v, "use_token_cache"):
                ds_v.use_token_cache = False
            self.val_loader = build_dataloader(ds_v, self.cfg, shuffle=False)

    def _validate(self) -> dict[str, Any]:
        if self.val_loader is None:
            return {"selection": {"score": float("nan"), "eligible": False}}
        return evaluate_loader(self.model, self.val_loader, self.cfg, self.device, loss_fn=self.loss_fn)

    def _save(self, name: str, blob: dict) -> Path:
        path = self.ckpt_root / name
        save_checkpoint(blob, path)
        logger.info("wrote %s", path)
        return path

    def run_phase(self, phase: str, epochs: int, resume: dict | None = None) -> dict[str, Any]:
        logger.info("=== phase %s for %d epochs ===", phase, epochs)
        logger.info("parameter audit: %s", self.model.parameter_audit())
        logger.info("hardware: %s", describe_hardware(self.device, getattr(self.model.backbone, "model_id", None)))
        opt, sch = self._make_optimizer(phase, epochs)
        start_epoch = 0
        if resume is not None:
            opt.load_state_dict(resume["optimizer"])
            if sch is not None and resume.get("scheduler"):
                sch.load_state_dict(resume["scheduler"])
            if resume.get("scaler"):
                self.scaler.load_state_dict(resume["scaler"])
            start_epoch = int(resume.get("epoch", 0)) + 1
            self.global_step = int(resume.get("global_step", 0))
            self.best_sel = resume.get("best_selection")
            logger.info("resumed %s from epoch %d step %d", phase, start_epoch, self.global_step)
        stopper = EarlyStopping(int(self.cfg.get("training.patience", 4)))
        stop_metric = str(self.cfg.get("training.early_stop_metric", "selection"))
        if stop_metric not in {"selection", "val_mse", "recall_at_5fa", "prefork_stop", "balanced_warning"}:
            raise ValueError(f"unsupported training.early_stop_metric {stop_metric}")
        patience_start = int(self.cfg.get("training.patience_start_epoch", 0))
        post_floor = float(self.cfg.get("training.post_fork_stop_floor", 0.90))
        safe_floor = float(self.cfg.get("training.safe_recall_floor", 0.82))
        stop_floor = float(self.cfg.get("training.stop_recall_floor", 0.75))
        max_safe_stop = float(self.cfg.get("training.max_safe_stop_rate", 0.08))
        prev_val_mse: float | None = None
        accum = max(1, int(self.cfg.get("training.gradient_accumulation_steps", 1)))
        clip_norm = float(self.cfg.get("optimizer.grad_clip_norm", 1.0))
        last_name = f"teacher_phase_{'a' if phase == 'frozen' else 'b'}_last.pt"
        best_name = f"teacher_phase_{'a' if phase == 'frozen' else 'b'}_best.pt"
        history = []
        for epoch in range(start_epoch, epochs):
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            self.model.train()
            self.model.backbone.eval()
            t0 = time.perf_counter()
            running: dict[str, float] = {}
            n_steps = 0
            opt.zero_grad(set_to_none=True)
            max_steps = int(self.cfg.get("training.max_steps", 0) or 0)
            for step, batch in enumerate(self.train_loader):
                loss, terms = self._forward_loss(batch)
                self.scaler.scale(loss / accum).backward()
                if (step + 1) % accum == 0:
                    self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(include_lora=(phase == "lora")), clip_norm)
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
                torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(include_lora=(phase == "lora")), clip_norm)
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
            rec5 = float(operating.get("recall_at_5fa", float("nan")))
            if stop_metric == "val_mse":
                better = prev_val_mse is None or (mse == mse and mse < prev_val_mse)
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
            blob = self._state(phase, epoch, opt, sch, {"val": val, "train": train_means})
            self._save(last_name, blob)
            if bool(self.cfg.get("training.save_every_epoch", False)):
                self._save(f"teacher_phase_{'a' if phase == 'frozen' else 'b'}_epoch_{epoch:02d}.pt", blob)
            if better:
                tagged = dict(sel)
                tagged["val_mse"] = mse
                tagged["best_epoch"] = epoch
                tagged.update({k: operating.get(k) for k in (
                    "recall_at_5fa", "ap_high", "mean_pred_gt80", "mse",
                    "threshold_5fa", "fa_achieved_5fa", "fa_at_0.5", "recall_at_0.5",
                )})
                tagged.update({k: h1s.get(k) for k in (
                    "pre_fork_stop_recall",
                    "post_fork_stop_recall",
                    "pre_fork_stop_n",
                    "post_fork_stop_n",
                    "post_fork_safe_stop_rate",
                    "safe_recall",
                    "stop_recall",
                    "safe_stop_rate",
                )})
                self.best_sel = tagged
                self._save(best_name, blob)
            dt = time.perf_counter() - t0
            rec = {
                "phase": phase,
                "epoch": epoch,
                "seconds": dt,
                "lrs": current_lrs(opt),
                "train": train_means,
                "val_selection": sel,
                "val_mse": mse,
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
                "phase=%s epoch=%d L_total=%.4f val_mse=%.5f rec@5FA=%s AP=%s SAFE=%s STOP=%s FAstop=%s preSTOP=%s postSTOP=%s time=%.1fs",
                phase,
                epoch,
                train_means.get("L_total", float("nan")),
                mse,
                operating.get("recall_at_5fa"),
                operating.get("ap_high"),
                h1s.get("safe_recall"),
                h1s.get("stop_recall"),
                h1s.get("safe_stop_rate"),
                h1s.get("pre_fork_stop_recall"),
                h1s.get("post_fork_stop_recall"),
                dt,
            )
            if stop_metric == "val_mse":
                prev_val_mse = mse
            if epoch < patience_start:
                stopper.step(True)
            elif stopper.step(better):
                keep = None if self.best_sel is None else self.best_sel.get("best_epoch")
                why = {
                    "val_mse": "val_mse",
                    "recall_at_5fa": "recall_at_5fa",
                    "prefork_stop": "pre_fork_stop_recall",
                    "balanced_warning": "min(SAFE,STOP) recall",
                }.get(stop_metric, "val_score")
                logger.info(
                    "early stopping at epoch %d (%s did not improve for %d epochs after warmup); keeping epoch %s",
                    epoch,
                    why,
                    stopper.patience,
                    keep,
                )
                break
        return {"phase": phase, "history": history, "best": self.best_sel}

    def run(self, phase: str = "frozen", resume_path: Path | None = None) -> dict[str, Any]:
        resume = load_checkpoint(resume_path, map_location=self.device) if resume_path else None
        results = {}
        if phase in {"frozen", "both"}:
            results["frozen"] = self.run_phase("frozen", int(self.cfg.get("training.frozen_epochs", 8)), resume if resume and resume.get("phase") == "frozen" else None)
            resume = None
        if phase in {"lora", "both"}:
            self._disable_token_cache_for_lora()
            if not getattr(self.model.backbone, "lora_enabled", False):
                n = self.model.enable_lora()
                logger.info("enabled LoRA on %d linear layers", n)
            lora_resume = resume if resume and resume.get("phase") == "lora" else None
            if lora_resume is None:
                best_a = self.ckpt_root / "teacher_phase_a_best.pt"
                if best_a.exists() and phase in {"lora", "both"}:
                    blob = load_checkpoint(best_a, map_location=self.device)
                    # LoRA just enabled: load adapter/heads from Phase A, ignore missing LoRA.
                    load_teacher_state(
                        self.model,
                        {**blob, "lora": {}},
                        strict=True,
                    )
                    self.best_sel = blob.get("best_selection")
                    logger.info("loaded Phase A best adapter/heads from %s", best_a)
            else:
                load_teacher_state(self.model, lora_resume, strict=True)
            results["lora"] = self.run_phase("lora", int(self.cfg.get("training.lora_epochs", 10)), lora_resume)
        return results
