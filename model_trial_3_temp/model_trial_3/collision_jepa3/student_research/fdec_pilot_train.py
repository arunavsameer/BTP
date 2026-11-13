"""Training loop for F_dec pilot S0/S1."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from collision_jepa3.student_research.cnn import MAX_TOTAL_PARAMS
from collision_jepa3.student_research.fdec_pilot_eval import evaluate_fdec_pilot
from collision_jepa3.student_research.fdec_student import (
    FDecNormStats,
    StudentS0,
    StudentS1,
    StudentS2,
    decoder_state_hash,
    extract_backbone_state,
    init_shared_backbone,
    load_backbone_state,
)


def _huber_kd(pred: torch.Tensor, teacher: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    return F.smooth_l1_loss(pred, teacher, beta=delta)


def _loss_s0(out, batch, alpha: float) -> tuple[torch.Tensor, dict[str, float]]:
    l_gt = F.mse_loss(out["Hhat_1s"], batch["gt_1s"])
    l_kd = F.mse_loss(out["Hhat_1s"], batch["H_teacher"])
    total = l_gt + alpha * l_kd
    return total, {"L_gt": float(l_gt.detach()), "L_kd": float(l_kd.detach()), "L_total": float(total.detach())}


def _loss_s1(
    out,
    batch,
    *,
    alpha: float,
    eta: float,
    norm: FDecNormStats,
    use_huber_kd: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    l_gt = F.mse_loss(out["Hhat_1s"], batch["gt_1s"])
    if use_huber_kd:
        l_kd = _huber_kd(out["Hhat_1s"], batch["H_teacher"])
    else:
        l_kd = F.mse_loss(out["Hhat_1s"], batch["H_teacher"])
    f_s = out["F_student"]
    f_t = batch["F_teacher"]
    f_sn = (f_s - norm.mu) / (norm.sigma + 1e-6)
    f_tn = (f_t - norm.mu) / (norm.sigma + 1e-6)
    l_f = F.smooth_l1_loss(f_sn, f_tn)
    total = l_gt + alpha * l_kd + eta * l_f
    with torch.no_grad():
        cos = F.cosine_similarity(f_s.flatten(1), f_t.flatten(1), dim=1).mean()
        mse_f = F.mse_loss(f_s, f_t)
    terms = {
        "L_gt": float(l_gt.detach()),
        "L_kd": float(l_kd.detach()),
        "L_fdec": float(l_f.detach()),
        "L_total": float(total.detach()),
        "latent_mse": float(mse_f.detach()),
        "latent_cosine": float(cos.detach()),
    }
    return total, terms


def _grad_norm(params: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total**0.5


def calibrate_eta_s1(
    model: StudentS1,
    batch: dict,
    norm: FDecNormStats,
    device: str,
    alpha: float = 0.5,
    target_frac: float = 0.25,
) -> dict[str, Any]:
    model.train()
    trainable = [p for p in list(model.backbone.parameters()) + list(model.proj.parameters()) if p.requires_grad]
    b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    opt_params = trainable
    model.zero_grad(set_to_none=True)
    out = model(b["clip"])
    loss_gt = F.mse_loss(out["Hhat_1s"], b["gt_1s"])
    loss_gt.backward(retain_graph=True)
    gn_gt = _grad_norm(opt_params)
    model.zero_grad(set_to_none=True)
    out = model(b["clip"])
    f_s = out["F_student"]
    f_t = b["F_teacher"]
    f_sn = (f_s - norm.mu) / (norm.sigma + 1e-6)
    f_tn = (f_t - norm.mu) / (norm.sigma + 1e-6)
    l_f = F.smooth_l1_loss(f_sn, f_tn)
    l_f.backward()
    gn_f = _grad_norm(opt_params)
    eta = float(target_frac * gn_gt / max(gn_f, 1e-8))
    return {
        "grad_norm_gt": gn_gt,
        "grad_norm_fdec": gn_f,
        "eta_selected": eta,
        "target_latent_fraction": target_frac,
    }


def _backbone_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state.keys()):
        h.update(k.encode())
        h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def run_fdec_pilot(
    variant: str,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    *,
    device: str,
    epochs: int,
    lr: float,
    alpha_kd: float,
    eta_fdec: float,
    norm: FDecNormStats | None,
    cfg_eval,
    exp_dir: Path,
    report_root: Path,
    verified_pair_keys: list[tuple],
    start_epoch: int = 0,
    decoder_freeze_epochs: int = 3,
    eval_epochs: tuple[int, ...] = (1, 3, 6, 9, 12, 15),
) -> dict[str, Any]:
    assert variant in ("S0", "S1", "S2")
    if sum(p.numel() for p in model.parameters()) > MAX_TOTAL_PARAMS:
        raise ValueError("parameter budget exceeded")
    model.to(device)
    if norm is not None:
        norm = norm.to(device)

    def build_optimizer(decoder_unfrozen: bool):
        if variant == "S0":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        assert isinstance(model, (StudentS1, StudentS2))
        enc_p = list(model.backbone.stem.parameters()) + list(model.backbone.encoder.parameters())  # type: ignore[arg-type]
        enc_p += list(model.backbone.head.parameters())  # type: ignore[union-attr]
        enc_p += list(model.proj.parameters())
        if decoder_unfrozen and variant == "S1":
            return torch.optim.AdamW(
                [
                    {"params": enc_p, "lr": lr},
                    {"params": model.decoder.parameters(), "lr": lr * 0.1},
                ],
                weight_decay=1e-4,
            )
        return torch.optim.AdamW(enc_p, lr=lr, weight_decay=1e-4)

    decoder_unfrozen = start_epoch >= decoder_freeze_epochs and variant == "S1"
    if variant in ("S1", "S2"):
        model.set_decoder_trainable(decoder_unfrozen if variant == "S1" else False)
    dec_hash0 = None
    if variant in ("S1", "S2"):
        dec_hash0 = decoder_state_hash(model.decoder)
        (exp_dir / "decoder_frozen_hash.json").write_text(
            json.dumps({"initial_hash": dec_hash0, "variant": variant}, indent=2), encoding="utf-8"
        )
    opt = build_optimizer(decoder_unfrozen)
    history: list[dict] = []
    eval_epochs_set = set(eval_epochs)

    for ep in range(start_epoch, epochs):
        if variant == "S1" and ep == decoder_freeze_epochs and not decoder_unfrozen:
            torch.save(model.decoder.state_dict(), exp_dir / "decoder_after_freeze.pt")
            model.set_decoder_trainable(True)
            decoder_unfrozen = True
            opt = build_optimizer(True)

        model.train()
        use_cuda = device.startswith("cuda")
        running: dict[str, float] = {}
        n_steps = 0
        t_wait = t_h2d = t_fwd = t_bwd = t_opt = 0.0
        t_epoch0 = time.perf_counter()
        t_iter_end = t_epoch0
        for batch in train_loader:
            t_iter_start = time.perf_counter()
            t_wait += t_iter_start - t_iter_end
            t1 = time.perf_counter()
            batch = {
                k: v.to(device, non_blocking=use_cuda) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            if use_cuda:
                torch.cuda.synchronize()
            t_h2d += time.perf_counter() - t1
            opt.zero_grad(set_to_none=True)
            t2 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
                out = model(batch["clip"])
                if variant == "S0":
                    loss, terms = _loss_s0(out, batch, alpha_kd)
                else:
                    assert norm is not None
                    loss, terms = _loss_s1(
                        out, batch, alpha=alpha_kd, eta=eta_fdec, norm=norm, use_huber_kd=True
                    )
            if use_cuda:
                torch.cuda.synchronize()
            t_fwd += time.perf_counter() - t2
            t3 = time.perf_counter()
            loss.backward()
            if use_cuda:
                torch.cuda.synchronize()
            t_bwd += time.perf_counter() - t3
            t4 = time.perf_counter()
            opt.step()
            if use_cuda:
                torch.cuda.synchronize()
            t_opt += time.perf_counter() - t4
            t_iter_end = time.perf_counter()
            for k, v in terms.items():
                running[k] = running.get(k, 0.0) + v
            n_steps += 1

        train_sec = time.perf_counter() - t_epoch0
        train_means = {k: v / max(n_steps, 1) for k, v in running.items()}
        epoch_idx = ep + 1
        train_rec: dict[str, Any] = {
            "epoch": epoch_idx,
            "variant": variant,
            "train": train_means,
            "n_steps": n_steps,
            "timing": {
                "dataloader_wait_sec": t_wait,
                "h2d_sec": t_h2d,
                "forward_sec": t_fwd,
                "backward_sec": t_bwd,
                "optimizer_sec": t_opt,
                "train_total_sec": train_sec,
            },
            "decoder_trainable": decoder_unfrozen if variant == "S1" else (variant == "S0"),
            "decoder_hash": decoder_state_hash(model.decoder) if variant in ("S1", "S2") else None,
            "decoder_hash_unchanged": (
                decoder_state_hash(model.decoder) == dec_hash0 if dec_hash0 is not None else None
            ),
        }
        if dec_hash0 is not None and train_rec.get("decoder_hash_unchanged") is False:
            raise RuntimeError(f"decoder hash changed at epoch {epoch_idx}")
        if hasattr(train_loader.dataset, "stats"):
            train_rec["rgb_cache"] = train_loader.dataset.stats()  # type: ignore[attr-defined]
        (exp_dir / f"metrics_train_epoch_{epoch_idx:02d}.json").write_text(json.dumps(train_rec, indent=2), encoding="utf-8")
        torch.save(model.state_dict(), exp_dir / "checkpoint_last.pt")
        torch.save(model.state_dict(), exp_dir / f"checkpoint_epoch_{epoch_idx:02d}.pt")

        if epoch_idx in eval_epochs_set:
            t_dev0 = time.perf_counter()
            warning = cfg_eval.get("warning") or {}
            val = evaluate_fdec_pilot(
                model,
                val_loader,
                device,
                warning,
                verified_pair_keys=verified_pair_keys,
                variant=variant,
            )
            dev_rec = {"epoch": epoch_idx, "variant": variant, "val": val, "dev_eval_sec": time.perf_counter() - t_dev0}
            (exp_dir / f"metrics_dev_epoch_{epoch_idx:02d}.json").write_text(json.dumps(dev_rec, indent=2), encoding="utf-8")
            combined = {**train_rec, **dev_rec, "seconds": train_sec + dev_rec["dev_eval_sec"]}
        else:
            combined = {**train_rec, "seconds": train_sec}
        (exp_dir / f"metrics_epoch_{epoch_idx:02d}.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
        history.append(combined)

    return {"history": history, "n_steps_last_epoch": n_steps}


def prepare_shared_backbone(
    exp_root: Path,
    seed: int = 1337,
    *,
    baseline_init: Path | None = None,
) -> dict[str, torch.Tensor]:
    for path in (baseline_init, exp_root / "shared_backbone_init.pt"):
        if path is not None and path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
    path = exp_root / "shared_backbone_init.pt"
    state = init_shared_backbone(seed=seed, n_frames=5)
    meta = {"seed": seed, "hash": _backbone_hash(state)}
    torch.save(state, path)
    (exp_root / "shared_backbone_init.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return state


def apply_backbone(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    if isinstance(model, StudentS0):
        load_backbone_state(model.net, state)
    elif isinstance(model, (StudentS1, StudentS2)):
        load_backbone_state(model.backbone, state)
