"""Shared training loop. Selection only. No test tensors. Random student init."""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collision_jepa3.evaluation.metrics import CAUTION, SAFE, STOP
from collision_jepa3.student_research.cache_access import ParentTargetCache
from collision_jepa3.student_research.fdec_student import build_parent_decoder, decoder_state_hash
from collision_jepa3.student_research.overnight6.losses import compute_losses, peak_class_t
from collision_jepa3.student_research.overnight6.models import build_model
from collision_jepa3.student_research.overnight6.schedule import LAMBDA_DECODER_REG, decoder_lr_multiplier
from collision_jepa3.training.optimizer import build_scheduler
def pick_cuda(pref: str = "auto") -> str:
    """Use the GPU bound via CUDA_VISIBLE_DEVICES (always cuda:0 in-process)."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit("CUDA is required; bind_cuda() must run before importing torch")
    try:
        torch.zeros(1, device="cuda:0")
        torch.cuda.synchronize()
    except Exception as exc:
        raise SystemExit(f"bound GPU is not allocatable: {exc}") from exc
    return "cuda:0"


def amp_ctx(device: str):
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

EXPECTED_HASH = "77f78c8714667617"
TEMP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp")
RAID = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_3")
RGB_ROOT = TEMP / "shared_rgb_cache_256_v1"
PARENT_ROOT = TEMP / "parent_target_cache_v2"
if not PARENT_ROOT.is_dir():
    PARENT_ROOT = RAID / "parent_target_cache_v2"
STATS = TEMP / "student_lat_paircf" / "fdec_channel_stats.npz"
SEEDS = {1: 1001, 2: 1002, 3: 1003, 4: 1004, 5: 1005, 6: 1006}
NAMES = {
    1: "m1_stop_floor",
    2: "m2_ordinal",
    3: "m3_asymmetric",
    4: "m4_parent_recovery",
    5: "m5_multiscale",
    6: "m6_taskaware",
}


def exp_dir(mid: int) -> Path:
    return TEMP / "overnight6" / NAMES[mid]


def set_decoder_trainable(decoder: nn.Module, on: bool) -> None:
    for p in decoder.parameters():
        p.requires_grad_(bool(on))


def decoder_drift(decoder: nn.Module, init: dict[str, torch.Tensor]) -> float:
    s = 0.0
    n = 0
    with torch.no_grad():
        for name, p in decoder.named_parameters():
            s += float((p - init[name]).pow(2).sum())
            n += p.numel()
    return (s / max(n, 1)) ** 0.5


def grad_norm(params) -> float:
    s = 0.0
    for p in params:
        if p.grad is not None:
            s += float(p.grad.detach().pow(2).sum())
    return s ** 0.5


def lerp_m6(phase: str, t: float) -> dict:
    A = dict(feat=1.0, cos=0.20, mgd=0.50, rel=0.20, h=0.10, peak=0.10, focal=0.03, rank=0.08, floor=0.40)
    C = dict(feat=0.40, cos=0.10, mgd=0.20, rel=0.10, h=0.70, peak=0.50, focal=0.08, rank=0.20, floor=0.80)
    if phase == "A":
        return A
    if phase == "C":
        return C
    return {k: (1 - t) * A[k] + t * C[k] for k in A}


def peak_stats(rows):
    from importlib.machinery import SourceFileLoader
    import sys

    root = Path(__file__).resolve().parents[3]
    # model_trial_3/collision_jepa3/... -> parents[3] is model_trial_3
    s75_path = root / "scripts/75_train_student_lat_paircf.py"
    spec_name = "s75_ovn"
    if spec_name not in sys.modules:
        mod = SourceFileLoader(spec_name, str(s75_path)).load_module()
        sys.modules[spec_name] = mod
    s75 = sys.modules[spec_name]
    peaks = np.array([float(np.max(r["H_student"])) for r in rows])
    gt_c = np.array([s75.peak_class(r["gt_1s"]) for r in rows])
    pr_c = np.array([s75.peak_class(r["H_student"]) for r in rows])
    pt_c = np.array([s75.peak_class(r["H_teacher"]) for r in rows])
    stop = gt_c == STOP
    recov_den = (stop & (pt_c == STOP)).sum()
    recov_hit = (stop & (pt_c == STOP) & (pr_c == STOP)).sum()
    out = {}
    for name, m in (("gt_safe", gt_c == SAFE), ("gt_threat", gt_c == CAUTION), ("gt_stop", stop)):
        x = peaks[m]
        if x.size == 0:
            out[name] = {}
            continue
        out[name] = {
            "n": int(x.size),
            "mean": float(x.mean()),
            "p50": float(np.median(x)),
            "p10": float(np.quantile(x, 0.1)),
            "p25": float(np.quantile(x, 0.25)),
            "p90": float(np.quantile(x, 0.9)),
            "lt_025": float(np.mean(x < 0.25)),
            "lt_060": float(np.mean(x < 0.60)),
        }
    out["collapsed_stop_frac"] = float(np.mean(peaks[stop] < 0.25)) if stop.any() else None
    out["parent_correct_stop_recovery"] = float(recov_hit / recov_den) if recov_den else None
    out["n_parent_correct_stop"] = int(recov_den)
    return out


def cpu_latency_ms(model, n_warm=3, n_run=8) -> float:
    m = deepcopy(model).cpu().eval()
    x = torch.rand(1, 16, 3, 256, 256)
    with torch.no_grad():
        for _ in range(n_warm):
            m(x)
        t0 = time.perf_counter()
        for _ in range(n_run):
            m(x)
        dt = (time.perf_counter() - t0) / n_run
    del m
    return dt * 1000.0


def load_s75():
    import sys
    from importlib.machinery import SourceFileLoader

    root = Path(__file__).resolve().parents[3]
    name = "s75_ovn"
    if name not in sys.modules:
        sys.modules[name] = SourceFileLoader(name, str(root / "scripts/75_train_student_lat_paircf.py")).load_module()
    return sys.modules[name]


def prepare_data(forbid_test: bool = True):
    s75 = load_s75()
    rows = [json.loads(l) for l in (PARENT_ROOT / "index.jsonl").open(encoding="utf-8")]
    gt = np.load(PARENT_ROOT / "gt_1s.npy").astype(np.float32)
    train_ids = [i for i, r in enumerate(rows) if r["split"] == "train"]
    sel_ids = [i for i, r in enumerate(rows) if r["split"] == "selection"]
    n_test = sum(1 for r in rows if r["split"] == "test")
    if (len(train_ids), len(sel_ids), n_test) != (22680, 5120, 5120):
        raise SystemExit(f"split contract {(len(train_ids), len(sel_ids), n_test)}")
    if forbid_test:
        # never materialize test window tensors
        pass
    rgb = s75.RGBCache256(RGB_ROOT)
    parent = ParentTargetCache(PARENT_ROOT)
    train_ds = s75.WinDS(rows, train_ids, gt, rgb, parent)
    sel_ds = s75.WinDS(rows, sel_ids, gt, rgb, parent)
    return s75, rows, gt, train_ids, sel_ids, train_ds, sel_ds


def save_ckpt(path: Path, model, epoch, extra: dict):
    payload = {
        "epoch": epoch,
        "encoder": model.encoder.state_dict(),
        "predictor": model.predictor.state_dict(),
        "proj": model.proj.state_dict(),
        "decoder": model.decoder.state_dict(),
        "variant": model.variant,
        **extra,
    }
    if model.aux is not None:
        payload["aux"] = model.aux.state_dict()
    if model.recent is not None:
        payload["recent"] = model.recent.state_dict()
        payload["gate"] = model.gate.state_dict()
    torch.save(payload, path)


def run_train(mid: int, *, epochs: int, batch: int, lr: float, device: str, num_workers: int, smoke: bool = False):
    EXP = exp_dir(mid)
    EXP.mkdir(parents=True, exist_ok=True)
    try:
        return _run_train(mid, epochs=epochs, batch=batch, lr=lr, device=device, num_workers=num_workers, smoke=smoke)
    except Exception:
        if not smoke:
            (EXP / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _run_train(mid: int, *, epochs: int, batch: int, lr: float, device: str, num_workers: int, smoke: bool = False):
    s75, rows, gt, train_ids, sel_ids, train_ds, sel_ds = prepare_data()
    EXP = exp_dir(mid)
    EXP.mkdir(parents=True, exist_ok=True)
    log_path = EXP / "run.log"
    seed = SEEDS[mid]
    torch.manual_seed(seed)
    np.random.seed(seed)

    def log(m: str) -> None:
        print(m, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(m + "\n")

    device = pick_cuda(device)
    from collision_jepa3.config import Config

    cfg = Config.load()
    decoder = build_parent_decoder(cfg, device="cpu")
    dec_h = decoder_state_hash(decoder)
    if dec_h != EXPECTED_HASH:
        raise SystemExit(f"decoder hash {dec_h}")
    model = build_model(mid, decoder).to(device)
    set_decoder_trainable(model.decoder, False)
    dec_init = {k: v.detach().clone() for k, v in model.decoder.state_dict().items()}
    torch.save(dec_init, EXP / "decoder_init.pt")
    st = np.load(STATS)
    mu = torch.tensor(st["mu"], dtype=torch.float32, device=device)
    sigma = torch.tensor(st["sigma"], dtype=torch.float32, device=device)
    peaks = gt[np.array(train_ids)].reshape(len(train_ids), -1).max(1)
    cls = np.zeros(len(train_ids), np.int64)
    cls[peaks >= 0.25] = CAUTION
    cls[peaks >= 0.60] = STOP
    counts = np.bincount(cls, minlength=3).astype(np.float64)
    inv = (len(train_ids) / np.maximum(counts, 1)) / 3.0
    class_w = torch.tensor(inv / inv.mean(), dtype=torch.float32, device=device)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=num_workers, collate_fn=s75.collate, pin_memory=True)
    sel_loader = DataLoader(sel_ds, batch_size=batch, shuffle=False, num_workers=min(2, num_workers), collate_fn=s75.collate)
    stud = list(model.student_parameters())
    opt_s = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": lr, "weight_decay": 1e-4},
            {"params": list(model.predictor.parameters()), "lr": lr, "weight_decay": 1e-4},
            {"params": list(model.proj.parameters()), "lr": 1e-4, "weight_decay": 1e-4},
        ]
        + ([{"params": list(model.aux.parameters()), "lr": lr, "weight_decay": 1e-4}] if model.aux is not None else [])
        + (
            [
                {"params": list(model.recent.parameters()), "lr": lr, "weight_decay": 1e-4},
                {"params": list(model.gate.parameters()), "lr": lr, "weight_decay": 1e-4},
            ]
            if model.recent is not None
            else []
        )
    )
    opt_d = torch.optim.AdamW(model.decoder.parameters(), lr=1e-8, weight_decay=0.0)
    sch = build_scheduler(opt_s, epochs, 1e-6, warmup_epochs=5)
    amp = str(device).startswith("cuda")
    ac = amp_ctx(device)
    audit = model.param_audit()
    lat = None
    if not smoke:
        try:
            lat = cpu_latency_ms(model)
        except Exception as exc:
            lat = None
            log(f"cpu_latency_failed {exc}")
    (EXP / "config.json").write_text(
        json.dumps(
            {
                "model": NAMES[mid],
                "seed": seed,
                "epochs": epochs,
                "batch": batch,
                "lr_student": lr,
                "decoder_hash_init": dec_h,
                "from": "random_student",
                "no_test": True,
                "params": audit,
                "cpu_latency_ms": lat,
                "lambda_decoder_reg": LAMBDA_DECODER_REG,
                "grad_clip": 1.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"start {NAMES[mid]} seed={seed} device={device} hash={dec_h} params={audit} cpu_ms={lat}")
    save_ckpt(EXP / "checkpoint_initial.pt", model, 0, {"random_init": True})

    phase, trans_t, trans_left = "A", 0.0, 0
    best_prim = best_stop = best_bal = None
    n_clip = 0
    n_steps = 0
    max_epochs = 1 if smoke else epochs
    max_batches = 2 if smoke else 10**9

    for ep in range(1, max_epochs + 1):
        mult = decoder_lr_multiplier(ep)
        unfreeze = mult > 0
        set_decoder_trainable(model.decoder, unfreeze)
        for g in opt_d.param_groups:
            g["lr"] = float(lr) * float(mult)
        model.train()
        # decoder has no dropout/BN running-stat layers that must train; keep eval() for BN-less towers
        model.decoder.eval()
        t0 = time.perf_counter()
        acc = defaultdict(float)
        n_b = 0
        g_stud = g_dec = 0.0
        g_dec_first = None
        for step, batch in enumerate(train_loader):
            if step >= max_batches:
                break
            b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            with ac:
                out = model(b["clip"])
                parts, total, y = compute_losses(mid, out, b, mu, sigma, model.predictor, model.decoder, dec_init, class_w)
                if mid == 6:
                    wts = lerp_m6(phase, trans_t)
                    total = (
                        wts["feat"] * parts["L_feat"]
                        + wts["cos"] * parts["L_cos"]
                        + wts["mgd"] * parts["L_mgd"]
                        + wts["rel"] * parts["L_rel"]
                        + wts["h"] * parts["L_h"]
                        + wts["peak"] * parts["L_peak"]
                        + wts["focal"] * parts["L_focal"]
                        + wts["rank"] * parts["L_rank"]
                        + wts["floor"] * parts["L_floor"]
                        + LAMBDA_DECODER_REG * parts["L_dec_reg"]
                    )
            opt_s.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)
            total.backward()
            gs = torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            if float(gs) >= 1.0:
                n_clip += 1
            g_stud = grad_norm(stud)
            g_dec = grad_norm(list(model.decoder.parameters()))
            if g_dec_first is None:
                g_dec_first = g_dec
            opt_s.step()
            if unfreeze:
                opt_d.step()
            for k, v in parts.items():
                if k.startswith("L_") and torch.is_tensor(v):
                    acc[k] += float(v.detach())
            acc["L_total"] += float(total.detach())
            acc["cos"] += float(F.cosine_similarity(out["F"].float().flatten(1), b["F_teacher"].float().flatten(1)).mean().detach())
            n_b += 1
            n_steps += 1
        if sch is not None and not smoke:
            sch.step()
        drift = decoder_drift(model.decoder, {k: v.to(device) for k, v in dec_init.items()})
        if not unfreeze and g_dec_first not in (None, 0.0) and g_dec_first > 1e-8:
            log(f"WARN decoder grad while frozen ep={ep} g={g_dec_first}")
        if unfreeze and (g_dec_first is None or g_dec_first < 1e-12):
            raise SystemExit(f"decoder grads still zero after unfreeze epoch {ep}")
        train_m = {k: v / max(n_b, 1) for k, v in acc.items()}
        train_m.update(
            {
                "decoder_lr_mult": mult,
                "decoder_lr": float(lr) * mult,
                "decoder_grad_norm": g_dec_first,
                "student_grad_norm": g_stud,
                "decoder_drift_rms": drift,
                "sec": time.perf_counter() - t0,
                "phase": phase if mid == 6 else "fixed",
            }
        )
        sel_m = eval_selection(model, sel_loader, device, s75) if not smoke else {}
        if smoke:
            model.eval()
            sb = next(iter(sel_loader))
            sb = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sb.items()}
            with amp_ctx(device):
                sout = model(sb["clip"])
            assert torch.isfinite(sout["Hhat_1s"]).all()
            log(
                f"smoke ep={ep} device={device} unfreeze={unfreeze} g_dec={g_dec_first} "
                f"drift={drift} train={train_m.get('L_total')}"
            )
            (EXP / f"smoke_epoch_{ep}.json").write_text(json.dumps(train_m, indent=2, default=str), encoding="utf-8")
            continue
        rec = {"epoch": ep, "train": train_m, "selection": sel_m, "clip_events": n_clip, "steps": n_steps}
        (EXP / f"metrics_epoch_{ep:02d}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        extra = {"selection": sel_m, "decoder_drift_rms": drift, "phase": phase}
        save_ckpt(EXP / f"checkpoint_epoch_{ep:02d}.pt", model, ep, extra)
        save_ckpt(EXP / "checkpoint_last.pt", model, ep, extra)
        sa, st, bal = float(sel_m["safe_recall"] or 0), float(sel_m["stop_recall"] or 0), float(sel_m.get("balanced_stop_safe_recall") or 0)
        prim = (1 if sa >= 0.85 else 0, st, sa)
        if best_prim is None or prim > best_prim[0]:
            best_prim = (prim, ep)
            save_ckpt(EXP / "checkpoint_best.pt", model, ep, extra)
        if sa >= 0.85 and (best_stop is None or st > best_stop[0]):
            best_stop = (st, ep)
            save_ckpt(EXP / "checkpoint_best_stop_safe85.pt", model, ep, extra)
        if best_bal is None or bal > best_bal[0]:
            best_bal = (bal, ep)
            save_ckpt(EXP / "checkpoint_best_balanced.pt", model, ep, extra)
        if mid == 6:
            sel_cos = float(sel_m.get("fdec_cosine") or 0)
            if phase == "A":
                ready = ep >= 16 and sel_cos >= 0.82 and sa >= 0.40
                if ready or ep >= 20:
                    phase, trans_left, trans_t = "B", 6, 0.0
            elif phase == "B":
                trans_left -= 1
                trans_t = 1.0 - max(trans_left, 0) / 6.0
                if trans_left <= 0:
                    phase, trans_t = "C", 1.0
        log(
            f"epoch {ep} SAFE={sa:.3f} THR={sel_m.get('threat_recall')} STOP={st:.3f} "
            f"prec={sel_m.get('stop_precision')} pre={sel_m.get('pre_fork_stop_recall')} "
            f"post={sel_m.get('post_fork_stop_recall')} cos={sel_m.get('fdec_cosine')} "
            f"dec_mult={mult} drift={drift:.5g} g_dec={g_dec_first} collapse={sel_m.get('peaks', {}).get('collapsed_stop_frac')} "
            f"sec={train_m['sec']:.1f}"
        )
        if torch.cuda.is_available():
            rec["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            (EXP / f"metrics_epoch_{ep:02d}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    if not smoke:
        (EXP / ".success").write_text("ok\n")
        log("DONE")
    return EXP


@torch.no_grad()
def eval_selection(model, loader, device, s75):
    model.eval()
    rows = []
    cos, l1 = [], []
    ctx = amp_ctx(device)
    aux_ok = aux_n = 0
    agree = 0
    for batch in loader:
        b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with ctx:
            out = model(b["clip"])
        h = out["Hhat_1s"].float().cpu().numpy()
        ht = b["H_teacher"].float().cpu().numpy()
        gt = b["gt_1s"].float().cpu().numpy()
        pref = b["pre_fork"].cpu().numpy()
        fs, ft = out["F"].float(), b["F_teacher"].float()
        c = F.cosine_similarity(fs.flatten(1), ft.flatten(1)).cpu().numpy()
        cos.append(c)
        l1.append((fs - ft).abs().mean(dim=(1, 2, 3)).cpu().numpy())
        y = peak_class_t(b["gt_1s"]).cpu().numpy()
        hcls = np.array([s75.peak_class(hh) for hh in h])
        if "aux_logits" in out:
            ax = out["aux_logits"].float().argmax(1).cpu().numpy()
            aux_ok += int((ax == y).sum())
            aux_n += len(y)
            agree += int((ax == hcls).sum())
        for i in range(h.shape[0]):
            rows.append(
                {
                    "H_student": h[i],
                    "H_teacher": ht[i],
                    "gt_1s": gt[i],
                    "pre_fork": bool(pref[i]),
                    "sample_id": b["sample_id"][i],
                    "frame": int(b["frame"][i]),
                    "background_id": b["background_id"][i],
                    "clip_id": b["clip_id"][i],
                }
            )
    m = s75.metrics_from_rows(rows, "H_student")
    m["fdec_cosine"] = float(np.concatenate(cos).mean())
    m["fdec_l1"] = float(np.concatenate(l1).mean())
    m["peaks"] = peak_stats(rows)
    if aux_n:
        m["aux_acc"] = aux_ok / aux_n
        m["aux_heatmap_agree"] = agree / aux_n
    return m


def run_smoke_all(device: str, batch: int = 2):
    results = {}
    for mid in range(1, 7):
        EXP = exp_dir(mid)
        EXP.mkdir(parents=True, exist_ok=True)
        try:
            s75, rows, gt, train_ids, sel_ids, train_ds, sel_ds = prepare_data()
            from collision_jepa3.config import Config

            cfg = Config.load()
            torch.manual_seed(SEEDS[mid])
            device_r = pick_cuda(device)
            decoder = build_parent_decoder(cfg, device="cpu")
            h0 = decoder_state_hash(decoder)
            assert h0 == EXPECTED_HASH, h0
            model = build_model(mid, decoder).to(device_r)
            set_decoder_trainable(model.decoder, False)
            dec_init = {k: v.detach().clone() for k, v in model.decoder.named_parameters()}
            x = torch.rand(batch, 16, 3, 256, 256, device=device_r)
            out = model(x)
            assert tuple(out["F"].shape) == (batch, 128, 16, 16), out["F"].shape
            assert tuple(out["Hhat_1s"].shape) == (batch, 5, 5), out["Hhat_1s"].shape
            assert torch.isfinite(out["F"]).all() and torch.isfinite(out["Hhat_1s"]).all()
            st = np.load(STATS)
            mu = torch.tensor(st["mu"], dtype=torch.float32, device=device_r)
            sigma = torch.tensor(st["sigma"], dtype=torch.float32, device=device_r)
            # one real train batch, no test
            loader = DataLoader(train_ds, batch_size=batch, shuffle=False, num_workers=0, collate_fn=s75.collate)
            batch0 = next(iter(loader))
            b = {k: v.to(device_r) if torch.is_tensor(v) else v for k, v in batch0.items()}
            class_w = torch.ones(3, device=device_r)
            parts, total, y = compute_losses(mid, model(b["clip"]), b, mu, sigma, model.predictor, model.decoder, dec_init, class_w)
            if mid == 6:
                total = parts["L_feat"] + 0.01 * parts["L_dec_reg"] + 0.1 * parts["L_h"] + 0.03 * parts["L_focal"]
            assert torch.isfinite(total)
            total.backward()
            g_dec = grad_norm(list(model.decoder.parameters()))
            g_stu = grad_norm(list(model.student_parameters()))
            assert g_dec == 0.0 or g_dec < 1e-12, g_dec
            assert g_stu > 0
            model.zero_grad(set_to_none=True)
            set_decoder_trainable(model.decoder, True)
            parts, total, y = compute_losses(mid, model(b["clip"]), b, mu, sigma, model.predictor, model.decoder, dec_init, class_w)
            if mid == 6:
                total = parts["L_feat"] + parts["L_h"] + parts["L_floor"] + 0.01 * parts["L_dec_reg"]
            before = {k: p.detach().clone() for k, p in model.decoder.named_parameters()}
            total.backward()
            g_dec2 = grad_norm(list(model.decoder.parameters()))
            if g_dec2 < 1e-12:
                raise RuntimeError("decoder grad still zero after unfreeze")
            opt = torch.optim.SGD(model.decoder.parameters(), lr=1e-2)
            opt.step()
            changed = any(float((p - before[n]).detach().abs().max()) > 0 for n, p in model.decoder.named_parameters())
            if not changed:
                raise RuntimeError("decoder params did not change")
            del model, decoder
            torch.cuda.empty_cache()
            run_train(mid, epochs=1, batch=batch, lr=3e-4, device=device_r, num_workers=0, smoke=True)
            results[mid] = {
                "ok": True,
                "hash": h0,
                "g_dec_frozen": g_dec,
                "g_dec_unfrozen": g_dec2,
                "g_student": g_stu,
                "device": device_r,
                "train_loop_smoke": True,
            }
            print(f"SMOKE M{mid} OK {results[mid]}", flush=True)
        except Exception:
            results[mid] = {"ok": False, "error": traceback.format_exc()}
            print(f"SMOKE M{mid} FAIL\n{results[mid]['error']}", flush=True)
    return results
