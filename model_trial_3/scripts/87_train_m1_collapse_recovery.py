#!/usr/bin/env python3
"""Finetune M1 ep10: frozen decoder, STOP collapse recovery. Selection only. No test."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.student_research.overnight6.gpu_bind import bind_cuda  # noqa: E402

_p = argparse.ArgumentParser()
_p.add_argument("--epochs", type=int, default=15)
_p.add_argument("--batch", type=int, default=8)
_p.add_argument("--lr", type=float, default=1e-4)
_p.add_argument("--device", default="auto")
_p.add_argument("--num-workers", type=int, default=4)
_p.add_argument("--seed", type=int, default=1001)
_p.add_argument("--smoke", action="store_true")
_args = _p.parse_args()
bind_cuda(str(_args.device))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402

from collision_jepa3.config import Config  # noqa: E402
from collision_jepa3.evaluation.metrics import CAUTION, SAFE, STOP  # noqa: E402
from collision_jepa3.student_research.fdec_student import build_parent_decoder, decoder_state_hash  # noqa: E402
from collision_jepa3.student_research.overnight6.losses import collapse_recovery_losses  # noqa: E402
from collision_jepa3.student_research.overnight6.models import build_model  # noqa: E402
from collision_jepa3.student_research.overnight6.train import (  # noqa: E402
    EXPECTED_HASH,
    STATS,
    amp_ctx,
    decoder_drift,
    eval_selection,
    grad_norm,
    pick_cuda,
    prepare_data,
    save_ckpt,
    set_decoder_trainable,
)
from collision_jepa3.training.optimizer import build_scheduler  # noqa: E402

EXP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp/m1_collapse_recovery_v1")
M1_CKPT = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp/overnight6/m1_stop_floor/checkpoint_epoch_10.pt")
COS_ABORT = 0.72


def load_m1_student(model, path: Path) -> dict:
    """Load M1 student towers; keep the freshly built Pair-CF decoder (ep10 drift ~8e-5)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(ck["encoder"])
    model.predictor.load_state_dict(ck["predictor"])
    model.proj.load_state_dict(ck["proj"])
    return ck


def class_balanced_prestop_sampler(rows, train_ids, gt, seed: int) -> WeightedRandomSampler:
    peaks = gt[np.array(train_ids)].reshape(len(train_ids), -1).max(1)
    cls = np.zeros(len(train_ids), np.int64)
    cls[peaks >= 0.25] = CAUTION
    cls[peaks >= 0.60] = STOP
    counts = np.bincount(cls, minlength=3).astype(np.float64)
    inv = (len(train_ids) / np.maximum(counts, 1.0)) / 3.0
    w = inv[cls]
    pref = np.array([bool(rows[i].get("pre_fork")) for i in train_ids])
    w = w * (1.0 + 2.0 * ((cls == STOP) & pref).astype(np.float64))
    gen = torch.Generator()
    gen.manual_seed(seed)
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(train_ids), replacement=True, generator=gen)


def run() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    log_path = EXP / "run.log"

    def log(m: str) -> None:
        print(m, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(m + "\n")

    if not M1_CKPT.is_file():
        raise SystemExit(f"missing M1 ckpt {M1_CKPT}")
    device = pick_cuda(str(_args.device))
    s75, rows, gt, train_ids, sel_ids, train_ds, sel_ds = prepare_data(forbid_test=True)
    if any(str(rows[i]["split"]) == "test" for i in list(train_ids) + list(sel_ids)):
        raise SystemExit("test ids leaked")
    torch.manual_seed(_args.seed)
    np.random.seed(_args.seed)
    cfg = Config.load()
    decoder = build_parent_decoder(cfg, device="cpu")
    if decoder_state_hash(decoder) != EXPECTED_HASH:
        raise SystemExit("parent decoder hash")
    model = build_model(1, decoder).to(device)
    ck = load_m1_student(model, M1_CKPT)
    set_decoder_trainable(model.decoder, False)
    if any(p.requires_grad for p in model.decoder.parameters()):
        raise SystemExit("decoder still requires grad")
    dec_h0 = decoder_state_hash(model.decoder)
    if dec_h0 != EXPECTED_HASH:
        raise SystemExit(f"loaded decoder hash {dec_h0}")
    dec_init = {k: v.detach().clone() for k, v in model.decoder.named_parameters()}
    torch.save({k: v.cpu() for k, v in dec_init.items()}, EXP / "decoder_init.pt")
    st = np.load(STATS)
    mu = torch.tensor(st["mu"], dtype=torch.float32, device=device)
    sigma = torch.tensor(st["sigma"], dtype=torch.float32, device=device)
    sampler = class_balanced_prestop_sampler(rows, train_ids, gt, _args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=_args.batch,
        sampler=sampler,
        shuffle=False,
        num_workers=_args.num_workers,
        collate_fn=s75.collate,
        pin_memory=True,
    )
    sel_loader = DataLoader(
        sel_ds, batch_size=_args.batch, shuffle=False, num_workers=min(2, _args.num_workers), collate_fn=s75.collate
    )
    b0 = next(iter(DataLoader(train_ds, batch_size=_args.batch, shuffle=False, num_workers=0, collate_fn=s75.collate)))
    b0 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b0.items()}
    model.train()
    model.decoder.eval()
    out0 = model(b0["clip"])
    parts0, tot0, _ = collapse_recovery_losses(out0, b0, mu, sigma, model.predictor, model.decoder, dec_init)
    tot0.backward()
    gdec = grad_norm(list(model.decoder.parameters()))
    gstu = grad_norm(list(model.student_parameters()))
    if gdec > 1e-8:
        raise SystemExit(f"decoder grad {gdec}")
    if gstu <= 0:
        raise SystemExit("student grad zero")
    model.zero_grad(set_to_none=True)
    log(f"preflight_ok init=M1_ep{ck.get('epoch')} hash={dec_h0} L={float(tot0.detach())} g_dec={gdec} vis={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')}")

    opt_s = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": _args.lr, "weight_decay": 1e-4},
            {"params": list(model.predictor.parameters()), "lr": _args.lr, "weight_decay": 1e-4},
            {"params": list(model.proj.parameters()), "lr": 1e-4, "weight_decay": 1e-4},
        ]
    )
    sch = build_scheduler(opt_s, _args.epochs, 1e-6, warmup_epochs=2)
    ac = amp_ctx(device)
    (EXP / "config.json").write_text(
        json.dumps(
            {
                "init": str(M1_CKPT),
                "decoder_frozen": True,
                "epochs": _args.epochs,
                "lr": _args.lr,
                "batch": _args.batch,
                "seed": _args.seed,
                "no_test": True,
                "loss": "collapse_recovery",
                "floor_target": 0.75,
                "floor_w": 1.8,
                "kd": "parent_correct_STOP_only_w1.0",
                "safe_hinge_w": 0.4,
                "stop_boost": 2.0,
                "sample_w": "1+3*GTSTOP+2*preforkSTOP",
                "sampler": "class_balanced_plus_prefork_STOP",
                "select": "SAFE>=0.80 and STOP>=0.70 else max STOP | SAFE>=0.80",
                "decoder_hash": dec_h0,
                "params": model.param_audit(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_ckpt(EXP / "checkpoint_initial.pt", model, 10, {"from_m1_ep10": True})
    log(f"start collapse_recovery device={device} lr={_args.lr} epochs={_args.epochs}")

    best_gate = best_safe80 = None
    n_clip = 0
    max_ep = 1 if _args.smoke else _args.epochs
    max_b = 3 if _args.smoke else 10**9
    low_cos = 0
    for ep in range(1, max_ep + 1):
        set_decoder_trainable(model.decoder, False)
        model.train()
        model.decoder.eval()
        t0 = time.perf_counter()
        acc = defaultdict(float)
        n_b = 0
        g_dec_first = None
        g_stu = 0.0
        for step, batch in enumerate(train_loader):
            if step >= max_b:
                break
            b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            with ac:
                out = model(b["clip"])
                parts, total, _y = collapse_recovery_losses(out, b, mu, sigma, model.predictor, model.decoder, dec_init)
            opt_s.zero_grad(set_to_none=True)
            total.backward()
            gs = torch.nn.utils.clip_grad_norm_(list(model.student_parameters()), 1.0)
            if float(gs) >= 1.0:
                n_clip += 1
            if g_dec_first is None:
                g_dec_first = grad_norm(list(model.decoder.parameters()))
            g_stu = grad_norm(list(model.student_parameters()))
            opt_s.step()
            for k, v in parts.items():
                if k.startswith("L_") and torch.is_tensor(v):
                    acc[k] += float(v.detach())
            acc["L_total"] += float(total.detach())
            acc["cos"] += float(F.cosine_similarity(out["F"].float().flatten(1), b["F_teacher"].float().flatten(1)).mean().detach())
            n_b += 1
        if sch is not None and not _args.smoke:
            sch.step()
        drift = decoder_drift(model.decoder, {k: v.to(device) for k, v in dec_init.items()})
        hnow = decoder_state_hash(model.decoder)
        if hnow != dec_h0 or drift > 1e-8:
            raise SystemExit(f"decoder moved ep={ep} hash={hnow} drift={drift}")
        if g_dec_first not in (None, 0.0) and g_dec_first > 1e-8:
            raise SystemExit(f"decoder grad ep={ep} g={g_dec_first}")
        train_m = {k: v / max(n_b, 1) for k, v in acc.items()}
        train_m.update(
            {
                "decoder_grad_norm": g_dec_first,
                "student_grad_norm": g_stu,
                "decoder_drift_rms": drift,
                "decoder_hash": hnow,
                "sec": time.perf_counter() - t0,
                "clip_events": n_clip,
            }
        )
        if _args.smoke:
            log(f"SMOKE ep={ep} g_dec={g_dec_first} total={train_m['L_total']}")
            (EXP / "smoke_epoch_1.json").write_text(json.dumps(train_m, indent=2, default=str), encoding="utf-8")
            continue
        sel_m = eval_selection(model, sel_loader, device, s75)
        rec = {"epoch": ep, "train": train_m, "selection": sel_m}
        if torch.cuda.is_available():
            rec["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        (EXP / f"metrics_epoch_{ep:02d}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        extra = {"selection": sel_m, "from_m1_ep10": True}
        save_ckpt(EXP / f"checkpoint_epoch_{ep:02d}.pt", model, ep, extra)
        save_ckpt(EXP / "checkpoint_last.pt", model, ep, extra)
        sa = float(sel_m["safe_recall"] or 0)
        st = float(sel_m["stop_recall"] or 0)
        pr = float(sel_m.get("stop_precision") or 0)
        cos = float(sel_m.get("fdec_cosine") or 0)
        recov = (sel_m.get("peaks") or {}).get("parent_correct_stop_recovery")
        coll = (sel_m.get("peaks") or {}).get("collapsed_stop_frac")
        p10 = ((sel_m.get("peaks") or {}).get("gt_stop") or {}).get("p10")
        if sa >= 0.80 and st >= 0.70:
            key = (st, pr, sa)
            if best_gate is None or key > best_gate[0]:
                best_gate = (key, ep)
                save_ckpt(EXP / "checkpoint_best_gate.pt", model, ep, extra)
                save_ckpt(EXP / "checkpoint_best.pt", model, ep, extra)
        if sa >= 0.80:
            key80 = (st, pr, sa)
            if best_safe80 is None or key80 > best_safe80[0]:
                best_safe80 = (key80, ep)
                save_ckpt(EXP / "checkpoint_best_safe80.pt", model, ep, extra)
                if best_gate is None:
                    save_ckpt(EXP / "checkpoint_best.pt", model, ep, extra)
        if cos < COS_ABORT:
            low_cos += 1
            log(f"WARN cosine {cos:.3f} < {COS_ABORT}")
            if low_cos >= 2:
                raise SystemExit(f"abort cosine {cos:.3f} for 2 epochs")
        else:
            low_cos = 0
        log(
            f"epoch {ep} SAFE={sa:.3f} THR={sel_m.get('threat_recall')} STOP={st:.3f} prec={sel_m.get('stop_precision')} "
            f"pre={sel_m.get('pre_fork_stop_recall')} post={sel_m.get('post_fork_stop_recall')} "
            f"coll={coll} p10={p10} recov={recov} cos={cos:.3f} "
            f"g_dec={g_dec_first} drift={drift:.5g} sec={train_m['sec']:.1f}"
        )
    if not _args.smoke:
        if decoder_state_hash(model.decoder) != dec_h0:
            raise SystemExit("decoder hash changed")
        (EXP / ".success").write_text(
            json.dumps({"best_gate_ep": None if best_gate is None else best_gate[1], "best_safe80_ep": None if best_safe80 is None else best_safe80[1]})
            + "\n",
            encoding="utf-8",
        )
        log(f"DONE gate_ep={None if best_gate is None else best_gate[1]} safe80_ep={None if best_safe80 is None else best_safe80[1]}")


def main():
    try:
        run()
    except Exception:
        EXP.mkdir(parents=True, exist_ok=True)
        (EXP / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
