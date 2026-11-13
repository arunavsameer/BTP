#!/usr/bin/env python3
"""M9: frozen Pair-CF decoder, parent cell losses, no STOP/SAFE hinges. Selection only."""

from __future__ import annotations

import argparse
import hashlib
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
_p.add_argument("--seed", type=int, default=1001)
_p.add_argument("--smoke", action="store_true")
_args = _p.parse_args()
bind_cuda(str(_args.device))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from collision_jepa3.config import Config  # noqa: E402
from collision_jepa3.evaluation.metrics import direction_scores_np  # noqa: E402
from collision_jepa3.student_research.fdec_student import build_parent_decoder, decoder_state_hash  # noqa: E402
from collision_jepa3.student_research.overnight6.models import build_model  # noqa: E402
from collision_jepa3.student_research.overnight6.parent_task import (  # noqa: E402
    M9_WEIGHTS,
    build_teacher_loss_m9,
    load_h_now_table,
    m9_losses,
)
from collision_jepa3.student_research.overnight6.train import (  # noqa: E402
    EXPECTED_HASH,
    amp_ctx,
    decoder_drift,
    grad_norm,
    peak_stats,
    pick_cuda,
    prepare_data,
    save_ckpt,
    set_decoder_trainable,
)
from collision_jepa3.training.optimizer import build_scheduler  # noqa: E402

EXP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp/m9_frozen_decoder_no_hinges_v1")
TEMP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp")
SEED = int(_args.seed)
SAFE, THREAT, STOP = 0, 1, 2


class TaskDS(Dataset):
    def __init__(self, inner, hnow):
        self.inner = inner
        self.hnow = hnow

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        x = self.inner[i]
        x["h_now"] = torch.from_numpy(np.ascontiguousarray(self.hnow[(str(x["clip_id"]), int(x["frame"]))]))
        return x


def collate_fn(batch):
    return {
        "clip": torch.stack([b["clip"] for b in batch]),
        "F_teacher": torch.stack([b["F_teacher"] for b in batch]),
        "H_teacher": torch.stack([b["H_teacher"] for b in batch]),
        "gt_1s": torch.stack([b["gt_1s"] for b in batch]),
        "h_now": torch.stack([b["h_now"] for b in batch]),
        "pre_fork": torch.tensor([b["pre_fork"] for b in batch], dtype=torch.bool),
        "amb_1s": torch.tensor([b["amb_1s"] for b in batch], dtype=torch.bool),
        "sample_id": [b["sample_id"] for b in batch],
        "clip_id": [b["clip_id"] for b in batch],
        "frame": torch.tensor([b["frame"] for b in batch], dtype=torch.int64),
        "background_id": [b["background_id"] for b in batch],
        "outcome": [b["outcome"] for b in batch],
    }


def student_hash(model) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        if str(k).startswith("decoder."):
            continue
        h.update(k.encode())
        h.update(v.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def participation_ratio(x: np.ndarray) -> float:
    xc = x - x.mean(0, keepdims=True)
    if xc.shape[0] < 2:
        return float("nan")
    cov = (xc.T @ xc) / max(xc.shape[0] - 1, 1)
    w = np.clip(np.linalg.eigvalsh(cov), 0, None)
    s = float(w.sum())
    if s <= 0:
        return float("nan")
    return float((s * s) / max(float((w * w).sum()), 1e-12))


def roc_pr_auc(scores, labels):
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, np.int64)
    if labels.min() == labels.max():
        return float("nan"), float("nan")
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    npos, nneg = int(y.sum()), int(len(y) - y.sum())
    tpr = tp / max(npos, 1)
    fpr = fp / max(nneg, 1)
    rec = tp / max(npos, 1)
    prec = tp / np.maximum(tp + fp, 1)
    trap = getattr(np, "trapezoid", np.trapz)
    return float(trap(tpr, fpr)), float(trap(prec, rec))


@torch.no_grad()
def eval_sel(model, loader, device, s75):
    model.eval()
    rows = []
    Fs, Ft, peaks_s, gt_c = [], [], [], []
    ctx = amp_ctx(device)
    for batch in loader:
        b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with ctx:
            out = model(b["clip"])
        h = out["Hhat_1s"].float().cpu().numpy()
        ht = b["H_teacher"].float().cpu().numpy()
        gt = b["gt_1s"].float().cpu().numpy()
        pref = b["pre_fork"].cpu().numpy()
        fs = out["F"].float().cpu().numpy()
        ft = b["F_teacher"].float().cpu().numpy()
        Fs.append(fs)
        Ft.append(ft)
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
            peaks_s.append(float(h[i].max()))
            gc = s75.peak_class(gt[i])
            gt_c.append(gc)
    m = s75.metrics_from_rows(rows, "H_student")
    m["peaks"] = peak_stats(rows)
    warn = s75.warn_cfg()
    buckets = {k: [0, 0] for k in ("LEFT", "CENTER", "RIGHT")}
    n_false_stop = 0
    n_safe = 0
    for r in rows:
        hg = np.asarray(r["gt_1s"], np.float32)
        hp = np.asarray(r["H_student"], np.float32)
        gc, pc = s75.peak_class(hg), s75.peak_class(hp)
        if gc == SAFE:
            n_safe += 1
            n_false_stop += int(pc == STOP)
        if gc != STOP:
            continue
        sc = direction_scores_np(hg, [1, 1, 1, 1, 1], warn["left_cols"], warn["center_cols"], warn["right_cols"])
        d = max(sc, key=sc.get)
        buckets[d][0] += int(pc == STOP)
        buckets[d][1] += 1
    for k, (h, n) in buckets.items():
        m[f"stop_recall_dir_{k}"] = float(h / n) if n else float("nan")
    m["gt_safe_false_stop"] = float(n_false_stop / n_safe) if n_safe else float("nan")
    fs = np.concatenate(Fs)
    ft = np.concatenate(Ft)
    n = fs.shape[0]
    cos = []
    for i in range(n):
        a, b = fs[i].reshape(-1), ft[i].reshape(-1)
        cos.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    ch = fs.mean(axis=(2, 3))
    m["fdec_cosine"] = float(np.mean(cos))
    m["feat_norm_mean"] = float(np.sqrt((fs**2).mean(axis=(1, 2, 3))).mean())
    m["feat_var"] = float(fs.var())
    m["eff_dim_pr"] = participation_ratio(ch)
    m["ch_mean_abs"] = float(np.abs(fs.mean(axis=(0, 2, 3))).mean())
    m["ch_std_mean"] = float(fs.std(axis=(0, 2, 3)).mean())
    peaks = np.array(peaks_s)
    y = np.array(gt_c)
    mask = y != THREAT
    roc, pr = roc_pr_auc(peaks[mask], (y[mask] == STOP).astype(np.int64))
    m["stop_vs_safe_roc_auc"] = roc
    m["stop_vs_safe_pr_auc"] = pr
    return m


def pull(folder: Path, ep: int):
    p = folder / f"metrics_epoch_{ep:02d}.json"
    if not p.is_file():
        return None
    sel = json.loads(p.read_text()).get("selection") or {}
    pk = (sel.get("peaks") or {}).get("gt_stop") or {}
    return {
        "SAFE": sel.get("safe_recall"),
        "THREAT": sel.get("threat_recall"),
        "STOP": sel.get("stop_recall"),
        "STOP_prec": sel.get("stop_precision"),
        "pre": sel.get("pre_fork_stop_recall"),
        "post": sel.get("post_fork_stop_recall"),
        "p10": pk.get("p10"),
        "collapse": (sel.get("peaks") or {}).get("collapsed_stop_frac"),
        "roc": sel.get("stop_vs_safe_roc_auc"),
        "RIGHT": sel.get("stop_recall_dir_RIGHT"),
        "eff_dim": sel.get("eff_dim_pr"),
    }


def write_final(best_off_ep, last_ep, sel_off):
    rows = []
    for name, ep, folder in (
        ("M1", 10, TEMP / "overnight6" / "m1_stop_floor"),
        ("M5", 17, TEMP / "overnight6" / "m5_multiscale"),
        ("heatmap_dominant", 14, TEMP / "m1_heatmap_dominant_v1"),
        ("M7", 23, TEMP / "m7_parent_task_latent_v1"),
        ("M8", 20, TEMP / "m8_parent_task_only_v1"),
        ("M9_official", best_off_ep, EXP),
        ("M9_last", last_ep, EXP),
    ):
        if ep is None:
            continue
        b = pull(folder, int(ep))
        if b:
            b["name"] = f"{name} ep{ep}"
            rows.append(b)
    lines = [
        "# M9 frozen decoder, no STOP/SAFE hinges\n\n",
        "| model | SAFE | THR | STOP | prec | pre | post | p10 | collapse | ROC | RIGHT |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r.get('SAFE')} | {r.get('THREAT')} | {r.get('STOP')} | {r.get('STOP_prec')} | {r.get('pre')} | {r.get('post')} | {r.get('p10')} | {r.get('collapse')} | {r.get('roc')} | {r.get('RIGHT')} |\n"
        )
    m1 = pull(TEMP / "overnight6" / "m1_stop_floor", 10) or {}
    s = sel_off or {}
    pk = (s.get("peaks") or {})
    gst = pk.get("gt_stop") or {}
    sa, st = float(s.get("safe_recall") or 0), float(s.get("stop_recall") or 0)
    col = float(pk.get("collapsed_stop_frac") or 1)
    col1 = float(m1.get("collapse") or 1)
    roc = s.get("stop_vs_safe_roc_auc")
    ans = {
        "1_hinge_removal_prevented_amplitude_collapse": bool(col < col1 - 1e-6) if sel_off else None,
        "M9_collapse": col if sel_off else None,
        "M1_collapse": col1,
        "M9_gt_stop_p10": gst.get("p10"),
        "2_STOP_improved_vs_M1": bool(st > float(m1.get("STOP") or 0) + 1e-6) if sel_off else None,
        "STOP_M9": st if sel_off else None,
        "STOP_M1": m1.get("STOP"),
        "3_SAFE_acceptable": {"ge_082": sa >= 0.82, "ge_085": sa >= 0.85, "SAFE": sa} if sel_off else None,
        "4_ROC": roc,
        "5_prefork_improved": bool(float(s.get("pre_fork_stop_recall") or 0) > float(m1.get("pre") or 0) + 1e-6) if sel_off else None,
        "6_RIGHT_improved": bool(float(s.get("stop_recall_dir_RIGHT") or 0) > 0.413) if sel_off else None,
        "7_eff_dim": s.get("eff_dim_pr") if sel_off else None,
        "8_better_tradeoff_than_M1": bool(sa >= 0.85 and st >= float(m1.get("STOP") or 0)) if sel_off else None,
        "gates": {
            "SAFE082_STOP070": sa >= 0.82 and st >= 0.70,
            "SAFE085_STOP060": sa >= 0.85 and st >= 0.60,
        }
        if sel_off
        else None,
        "official_epoch": best_off_ep,
        "note": "ROC-AUC is ranking; STOP recall is calibration at 0.60.",
    }
    (EXP / "report.md").write_text("".join(lines) + "\n## Answers\n\n" + json.dumps(ans, indent=2) + "\n", encoding="utf-8")
    (EXP / "answers.json").write_text(json.dumps(ans, indent=2), encoding="utf-8")


def run() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    log_path = EXP / "run.log"

    def log(m: str) -> None:
        print(m, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(m + "\n")

    device = pick_cuda(str(_args.device))
    s75, rows, gt, train_ids, sel_ids, train_ds, sel_ds = prepare_data(forbid_test=True)
    if any(str(rows[i]["split"]) == "test" for i in list(train_ids) + list(sel_ids)):
        raise SystemExit("test ids leaked")
    hnow_tr = load_h_now_table(rows, train_ids)
    hnow_sel = load_h_now_table(rows, sel_ids)
    train_ds = TaskDS(train_ds, hnow_tr)
    sel_ds = TaskDS(sel_ds, hnow_sel)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = Config.load()
    decoder = build_parent_decoder(cfg, device="cpu")
    dec_h0 = decoder_state_hash(decoder)
    if dec_h0 != EXPECTED_HASH:
        raise SystemExit(f"decoder hash {dec_h0}")
    model = build_model(1, decoder).to(device)
    set_decoder_trainable(model.decoder, False)
    if any(p.requires_grad for p in model.decoder.parameters()):
        raise SystemExit("decoder still requires grad")
    dec_init = {k: v.detach().clone() for k, v in model.decoder.named_parameters()}
    torch.save({k: v.cpu() for k, v in dec_init.items()}, EXP / "decoder_init.pt")
    sh = student_hash(model)
    teacher = build_teacher_loss_m9(device)
    gen = torch.Generator()
    gen.manual_seed(SEED)
    train_loader = DataLoader(
        train_ds, batch_size=_args.batch, shuffle=True, num_workers=_args.num_workers, collate_fn=collate_fn, pin_memory=True, generator=gen
    )
    sel_loader = DataLoader(sel_ds, batch_size=_args.batch, shuffle=False, num_workers=min(2, _args.num_workers), collate_fn=collate_fn)
    b0 = next(iter(DataLoader(train_ds, batch_size=_args.batch, shuffle=False, num_workers=0, collate_fn=collate_fn)))
    b0 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b0.items()}
    model.train()
    model.decoder.eval()
    out0 = model(b0["clip"])
    if tuple(out0["F"].shape[1:]) != (128, 16, 16) or tuple(out0["Hhat_1s"].shape[1:]) != (5, 5):
        raise SystemExit("shape")
    if not torch.isfinite(out0["Hhat_1s"]).all():
        raise SystemExit("nonfinite heatmap")
    parts0 = m9_losses(teacher, out0, b0)
    if set(k for k in parts0 if k.startswith("L_") and k != "L_task") != set(M9_WEIGHTS):
        raise SystemExit(f"unexpected loss keys {parts0.keys()}")
    parts0["L_task"].backward()
    gdec = grad_norm(list(model.decoder.parameters()))
    gstu = grad_norm(list(model.student_parameters()))
    if gdec > 1e-12:
        raise SystemExit(f"decoder grad {gdec}")
    if gstu <= 0 or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise SystemExit("bad student grads")
    model.zero_grad(set_to_none=True)
    log(f"preflight_ok hash={dec_h0} student={sh} L_task={float(parts0['L_task'].detach())} g_dec={gdec}")

    opt_s = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": _args.lr, "weight_decay": 1e-4},
            {"params": list(model.predictor.parameters()), "lr": _args.lr, "weight_decay": 1e-4},
            {"params": list(model.proj.parameters()), "lr": 1e-4, "weight_decay": 1e-4},
        ]
    )
    sch = build_scheduler(opt_s, _args.epochs, 1e-6, warmup_epochs=5)
    ac = amp_ctx(device)
    (EXP / "config.json").write_text(
        json.dumps(
            {
                "model": 9,
                "seed": SEED,
                "decoder_frozen": True,
                "decoder_optimizer": False,
                "hinges": False,
                "latent": False,
                "decoder_hash_init": dec_h0,
                "student_init_hash": sh,
                "weights": M9_WEIGHTS,
                "no_test": True,
                "params": model.param_audit(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_ckpt(EXP / "checkpoint_initial.pt", model, 0, {"random_init": True})
    log(f"start M9 seed={SEED} device={device} vis={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')}")

    best_off = best_unr = None
    n_clip = 0
    max_ep = 1 if _args.smoke else _args.epochs
    max_b = 3 if _args.smoke else 10**9
    last_sel = None
    for ep in range(1, max_ep + 1):
        set_decoder_trainable(model.decoder, False)
        model.train()
        model.decoder.eval()
        t0 = time.perf_counter()
        acc = defaultdict(float)
        n_b = 0
        g_dec_first = None
        for step, batch in enumerate(train_loader):
            if step >= max_b:
                break
            b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            with ac:
                out = model(b["clip"])
                parts = m9_losses(teacher, out, b)
                loss = parts["L_task"]
            opt_s.zero_grad(set_to_none=True)
            loss.backward()
            gs = torch.nn.utils.clip_grad_norm_(list(model.student_parameters()), 1.0)
            if float(gs) >= 1.0:
                n_clip += 1
            if g_dec_first is None:
                g_dec_first = grad_norm(list(model.decoder.parameters()))
            opt_s.step()
            for k, v in parts.items():
                if torch.is_tensor(v):
                    acc[k] += float(v.detach())
            acc["L_total"] += float(loss.detach())
            for name, w in M9_WEIGHTS.items():
                acc[f"{name}_weighted"] += w * float(parts[name].detach())
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
        train_m.update({"decoder_grad_norm": g_dec_first, "decoder_drift_rms": drift, "decoder_hash": hnow, "sec": time.perf_counter() - t0, "clip_events": n_clip})
        if _args.smoke:
            log(f"SMOKE ep={ep} g_dec={g_dec_first} drift={drift} total={train_m['L_total']}")
            (EXP / "smoke_epoch_1.json").write_text(json.dumps(train_m, indent=2, default=str), encoding="utf-8")
            continue
        sel_m = eval_sel(model, sel_loader, device, s75)
        last_sel = sel_m
        rec = {"epoch": ep, "train": train_m, "selection": sel_m}
        if torch.cuda.is_available():
            rec["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        (EXP / f"metrics_epoch_{ep:02d}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        extra = {"selection": sel_m, "decoder_hash": hnow}
        save_ckpt(EXP / f"checkpoint_epoch_{ep:02d}.pt", model, ep, extra)
        save_ckpt(EXP / "checkpoint_last.pt", model, ep, extra)
        sa, st, pr = float(sel_m["safe_recall"] or 0), float(sel_m["stop_recall"] or 0), float(sel_m.get("stop_precision") or 0)
        off = (1 if sa >= 0.85 else 0, st, pr)
        unr = (st, sa, pr)
        if best_off is None or off > best_off[0]:
            best_off = (off, ep)
            save_ckpt(EXP / "checkpoint_best.pt", model, ep, extra)
        if sa >= 0.85:
            save_ckpt(EXP / "checkpoint_best_qualifying.pt", model, ep, extra)
        if best_unr is None or unr > best_unr[0]:
            best_unr = (unr, ep)
            save_ckpt(EXP / "checkpoint_best_unrestricted.pt", model, ep, extra)
        pk = sel_m.get("peaks") or {}
        gst = pk.get("gt_stop") or {}
        log(
            f"epoch {ep} SAFE={sa:.3f} THR={sel_m.get('threat_recall')} STOP={st:.3f} prec={sel_m.get('stop_precision')} "
            f"pre={sel_m.get('pre_fork_stop_recall')} post={sel_m.get('post_fork_stop_recall')} "
            f"coll={pk.get('collapsed_stop_frac')} p10={gst.get('p10')} p25={gst.get('p25')} med={gst.get('p50')} "
            f"RIGHT={sel_m.get('stop_recall_dir_RIGHT')} roc={sel_m.get('stop_vs_safe_roc_auc')} "
            f"cos={sel_m.get('fdec_cosine')} dim={sel_m.get('eff_dim_pr')} var={sel_m.get('feat_var')} "
            f"g_dec={g_dec_first} drift={drift:.5g} sec={train_m['sec']:.1f}"
        )
    if not _args.smoke:
        if decoder_state_hash(model.decoder) != dec_h0:
            raise SystemExit("decoder hash changed at end")
        (EXP / ".success").write_text("ok\n")
        log("DONE")
        off_ep = best_off[1] if best_off else None
        sel_off = None
        if off_ep and (EXP / f"metrics_epoch_{off_ep:02d}.json").is_file():
            sel_off = json.loads((EXP / f"metrics_epoch_{off_ep:02d}.json").read_text())["selection"]
        write_final(off_ep, max_ep, sel_off)


def main():
    try:
        run()
    except Exception:
        EXP.mkdir(parents=True, exist_ok=True)
        (EXP / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
