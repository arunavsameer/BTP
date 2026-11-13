#!/usr/bin/env python3
"""M1 heatmap-dominant student. Random init. Selection only. No test."""

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
_p.add_argument("--epochs", type=int, default=30)
_p.add_argument("--batch", type=int, default=8)
_p.add_argument("--lr", type=float, default=3e-4)
_p.add_argument("--device", default="auto")
_p.add_argument("--num-workers", type=int, default=4)
_p.add_argument("--smoke", action="store_true")
_p.add_argument("--seed", type=int, default=2001)
_args = _p.parse_args()
bind_cuda(str(_args.device))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from collision_jepa3.config import Config  # noqa: E402
from collision_jepa3.student_research.fdec_student import build_parent_decoder, decoder_state_hash  # noqa: E402
from collision_jepa3.student_research.overnight6.heatmap_dominant import (  # noqa: E402
    EPS,
    GroupEMA,
    m1_parts,
    task_ratio,
)
from collision_jepa3.student_research.overnight6.models import build_model  # noqa: E402
from collision_jepa3.student_research.overnight6.schedule import LAMBDA_DECODER_REG, decoder_lr_multiplier  # noqa: E402
from collision_jepa3.student_research.overnight6.train import (  # noqa: E402
    EXPECTED_HASH,
    STATS,
    amp_ctx,
    cpu_latency_ms,
    decoder_drift,
    eval_selection,
    grad_norm,
    pick_cuda,
    prepare_data,
    save_ckpt,
    set_decoder_trainable,
)
from collision_jepa3.training.optimizer import build_scheduler  # noqa: E402

EXP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp/m1_heatmap_dominant_v1")
SEED = int(_args.seed)


def gnorm(params, loss):
    if float(loss.detach()) == 0.0:
        return 0.0
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    s = 0.0
    for g in gs:
        if g is not None:
            s += float(g.detach().pow(2).sum())
    return s ** 0.5


def smoke_unit() -> None:
    # schedule
    assert abs(task_ratio(1) - 0.05) < 1e-9
    assert abs(task_ratio(2) - 0.10) < 1e-9
    assert abs(task_ratio(3) - 0.20) < 1e-9
    assert abs(task_ratio(4) - 0.30) < 1e-9
    assert abs(task_ratio(5) - 0.40) < 1e-9
    assert abs(task_ratio(6) - 0.50) < 1e-9
    assert abs(task_ratio(30) - 0.50) < 1e-9
    # floor weights
    from collision_jepa3.student_research.overnight6.heatmap_dominant import adaptive_stop_floor, floor_weight_from_peak

    p = torch.tensor([0.05, 0.30, 0.50, 0.65, 0.80])
    w = floor_weight_from_peak(p)
    assert torch.allclose(w, torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]))
    h = torch.zeros(5, 5, 5)
    gt = torch.zeros(5, 5, 5)
    gt[:, 2, 2] = 0.9
    h[0, 2, 2] = 0.05
    h[1, 2, 2] = 0.30
    h[2, 2, 2] = 0.50
    h[3, 2, 2] = 0.65
    h[4, 2, 2] = 0.80
    loss, bins = adaptive_stop_floor(h, gt)
    assert float(loss) > 0
    assert bins["n_0.0_0.25"] == 1 and bins["n_0.7_10.0"] == 1
    # EMA does not require grad on ema buffers
    ema = GroupEMA()
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    tot, pt, pl = ema.balance(a, b, 0.5)
    tot.backward()
    assert a.grad is not None
    print("UNIT_OK", {"loss_floor": float(loss), "pct_task": pt, "bins": bins}, flush=True)


def extra_peaks(sel_m, rows_unused=None):
    return sel_m


def run() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    log_path = EXP / "run.log"

    def log(m: str) -> None:
        print(m, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(m + "\n")

    smoke_unit()
    device = pick_cuda(str(_args.device))
    s75, rows, gt, train_ids, sel_ids, train_ds, sel_ds = prepare_data(forbid_test=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = Config.load()
    decoder = build_parent_decoder(cfg, device="cpu")
    dec_h = decoder_state_hash(decoder)
    if dec_h != EXPECTED_HASH:
        raise SystemExit(f"decoder hash {dec_h}")
    model = build_model(1, decoder).to(device)
    set_decoder_trainable(model.decoder, False)
    dec_init = {k: v.detach().clone() for k, v in model.decoder.named_parameters()}
    torch.save({k: v.cpu() for k, v in dec_init.items()}, EXP / "decoder_init.pt")
    st = np.load(STATS)
    mu = torch.tensor(st["mu"], dtype=torch.float32, device=device)
    sigma = torch.tensor(st["sigma"], dtype=torch.float32, device=device)
    train_loader = DataLoader(
        train_ds, batch_size=_args.batch, shuffle=True, num_workers=_args.num_workers, collate_fn=s75.collate, pin_memory=True
    )
    sel_loader = DataLoader(sel_ds, batch_size=_args.batch, shuffle=False, num_workers=min(2, _args.num_workers), collate_fn=s75.collate)
    opt_s = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": _args.lr, "weight_decay": 1e-4},
            {"params": list(model.predictor.parameters()), "lr": _args.lr, "weight_decay": 1e-4},
            {"params": list(model.proj.parameters()), "lr": 1e-4, "weight_decay": 1e-4},
        ]
    )
    opt_d = torch.optim.AdamW(model.decoder.parameters(), lr=1e-8, weight_decay=0.0)
    sch = build_scheduler(opt_s, _args.epochs, 1e-6, warmup_epochs=5)
    ac = amp_ctx(device)
    audit = model.param_audit()
    lat = None if _args.smoke else None
    if not _args.smoke:
        try:
            lat = cpu_latency_ms(model)
        except Exception as exc:
            log(f"cpu_latency_failed {exc}")
    (EXP / "config.json").write_text(
        json.dumps(
            {
                "from": "random_student",
                "not_from_m1_ep10": True,
                "seed": SEED,
                "decoder_hash_init": dec_h,
                "epochs": _args.epochs,
                "batch": _args.batch,
                "lr": _args.lr,
                "schedule": "task 5/10/20/30/40 then 50%",
                "adaptive_floor": True,
                "params": audit,
                "cpu_latency_ms": lat,
                "no_test": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"start heatmap_dominant seed={SEED} device={device} hash={dec_h} params={audit} cpu_ms={lat}")
    save_ckpt(EXP / "checkpoint_initial.pt", model, 0, {"random_init": True})
    enc_p = list(model.encoder.parameters()) + list(model.proj.parameters())
    ema = GroupEMA()
    # init EMA on a few train batches
    model.train()
    n_init = 0
    for batch in train_loader:
        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with ac:
            out = model(b["clip"])
            parts, Llat, Ltask, lreg, _ = m1_parts(out, b, mu, sigma, model.predictor, model.decoder, dec_init)
        ema.update(Llat, Ltask)
        n_init += 1
        if n_init >= 4:
            break
    log(f"ema_init lat={ema.ema_lat} task={ema.ema_task}")
    best_prim = best82 = best_bal = None
    n_clip = 0
    max_ep = 1 if _args.smoke else _args.epochs
    max_b = 3 if _args.smoke else 10**9
    for ep in range(1, max_ep + 1):
        r = task_ratio(ep)
        mult = decoder_lr_multiplier(ep)
        unfreeze = mult > 0
        set_decoder_trainable(model.decoder, unfreeze)
        for g in opt_d.param_groups:
            g["lr"] = float(_args.lr) * float(mult)
        model.train()
        model.decoder.eval()
        t0 = time.perf_counter()
        acc = defaultdict(float)
        n_b = 0
        g_dec_first = None
        gdiag = {}
        bin_acc = defaultdict(float)
        for step, batch in enumerate(train_loader):
            if step >= max_b:
                break
            b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            with ac:
                out = model(b["clip"])
                parts, Llat, Ltask, lreg, bins = m1_parts(out, b, mu, sigma, model.predictor, model.decoder, dec_init)
                ema.update(Llat, Ltask)
                Lbal, pct_t, pct_l = ema.balance(Llat, Ltask, r)
                loss = Lbal + LAMBDA_DECODER_REG * lreg
            opt_s.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)
            loss.backward()
            gs = torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            if float(gs) >= 1.0:
                n_clip += 1
            if g_dec_first is None:
                g_dec_first = grad_norm(list(model.decoder.parameters()))
            opt_s.step()
            if unfreeze:
                opt_d.step()
            for k, v in parts.items():
                if torch.is_tensor(v):
                    acc[k] += float(v.detach())
            acc["L_balanced"] += float(Lbal.detach())
            acc["L_total"] += float(loss.detach())
            acc["pct_task"] += pct_t
            acc["pct_lat"] += pct_l
            acc["cos"] += float(F.cosine_similarity(out["F"].float().flatten(1), b["F_teacher"].float().flatten(1)).mean().detach())
            for k, v in bins.items():
                bin_acc[k] += float(v)
            n_b += 1
            if step == 0:
                with ac:
                    out = model(b["clip"])
                    parts, Llat, Ltask, lreg, _ = m1_parts(out, b, mu, sigma, model.predictor, model.decoder, dec_init)
                gdiag = {
                    "g_latent": gnorm(enc_p, Llat.float()),
                    "g_task": gnorm(enc_p, Ltask.float()),
                    "g_h": gnorm(enc_p, parts["L_h"].float()),
                    "g_floor": gnorm(enc_p, parts["L_floor"].float()) if float(parts["L_floor"].detach()) > 0 else 0.0,
                    "g_kd": gnorm(enc_p, parts["L_kd"].float()),
                }
                gl, gt_ = [], []
                gs1 = torch.autograd.grad(Llat.float(), enc_p, retain_graph=True, allow_unused=True)
                gs2 = torch.autograd.grad(Ltask.float(), enc_p, retain_graph=True, allow_unused=True)
                num = den1 = den2 = 0.0
                for a, c in zip(gs1, gs2):
                    if a is None or c is None:
                        continue
                    num += float((a * c).sum().detach())
                    den1 += float(a.pow(2).sum().detach())
                    den2 += float(c.pow(2).sum().detach())
                gdiag["cos_lat_task"] = num / ((den1**0.5) * (den2**0.5) + 1e-12)
                opt_s.zero_grad(set_to_none=True)
                opt_d.zero_grad(set_to_none=True)
        if sch is not None and not _args.smoke:
            sch.step()
        drift = decoder_drift(model.decoder, {k: v.to(device) for k, v in dec_init.items()})
        if not unfreeze and g_dec_first not in (None, 0.0) and g_dec_first > 1e-8:
            log(f"WARN decoder grad frozen ep={ep} g={g_dec_first}")
        if unfreeze and (g_dec_first is None or g_dec_first < 1e-12):
            raise SystemExit(f"decoder grads still zero after unfreeze epoch {ep}")
        train_m = {k: v / max(n_b, 1) for k, v in acc.items()}
        train_m.update(
            {
                "r_target": r,
                "ema_lat": ema.ema_lat,
                "ema_task": ema.ema_task,
                "decoder_lr_mult": mult,
                "decoder_drift_rms": drift,
                "decoder_grad_norm": g_dec_first,
                "clip_events": n_clip,
                "sec": time.perf_counter() - t0,
                "floor_bins": {k: v / max(n_b, 1) for k, v in bin_acc.items()},
                **gdiag,
            }
        )
        if abs(train_m["pct_task"] - r) > 0.15 and n_b > 2:
            log(f"WARN contribution drift ep={ep} pct_task={train_m['pct_task']:.3f} target={r}")
        if _args.smoke:
            log(f"SMOKE ep={ep} unfreeze={unfreeze} g_dec={g_dec_first} pct_task={train_m['pct_task']:.3f} r={r} total={train_m['L_total']}")
            (EXP / f"smoke_epoch_{ep}.json").write_text(json.dumps(train_m, indent=2, default=str), encoding="utf-8")
            continue
        sel_m = eval_selection(model, sel_loader, device, s75)
        rec = {"epoch": ep, "train": train_m, "selection": sel_m}
        if torch.cuda.is_available():
            rec["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        (EXP / f"metrics_epoch_{ep:02d}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        extra = {"selection": sel_m, "decoder_drift_rms": drift}
        save_ckpt(EXP / f"checkpoint_epoch_{ep:02d}.pt", model, ep, extra)
        save_ckpt(EXP / "checkpoint_last.pt", model, ep, extra)
        sa, st, bal = float(sel_m["safe_recall"] or 0), float(sel_m["stop_recall"] or 0), float(sel_m.get("balanced_stop_safe_recall") or 0)
        prim = (1 if sa >= 0.85 else 0, st, sa)
        if best_prim is None or prim > best_prim[0]:
            best_prim = (prim, ep)
            save_ckpt(EXP / "checkpoint_best.pt", model, ep, extra)
        if sa >= 0.82 and st >= 0.70:
            if best82 is None or st > best82[0]:
                best82 = (st, ep)
                save_ckpt(EXP / "checkpoint_best_safe82_stop70.pt", model, ep, extra)
        if best_bal is None or bal > best_bal[0]:
            best_bal = (bal, ep)
            save_ckpt(EXP / "checkpoint_best_balanced.pt", model, ep, extra)
        pk = sel_m.get("peaks") or {}
        gst = pk.get("gt_stop") or {}
        log(
            f"epoch {ep} r={r:.2f} pct_task={train_m['pct_task']:.3f} pct_lat={train_m['pct_lat']:.3f} "
            f"SAFE={sa:.3f} THR={sel_m.get('threat_recall')} STOP={st:.3f} prec={sel_m.get('stop_precision')} "
            f"pre={sel_m.get('pre_fork_stop_recall')} post={sel_m.get('post_fork_stop_recall')} "
            f"cos={sel_m.get('fdec_cosine')} coll={pk.get('collapsed_stop_frac')} "
            f"stop_p10={gst.get('p10')} stop_p50={gst.get('p50')} "
            f"gcos={gdiag.get('cos_lat_task')} dec_mult={mult} drift={drift:.5g} sec={train_m['sec']:.1f}"
        )
    if not _args.smoke:
        (EXP / ".success").write_text("ok\n")
        log("DONE")


def main() -> None:
    try:
        run()
    except Exception:
        EXP.mkdir(parents=True, exist_ok=True)
        (EXP / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
