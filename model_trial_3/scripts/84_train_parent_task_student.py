#!/usr/bin/env python3
"""M7 parent-task+latent and M8 parent-task-only RGB students. Selection only. No test."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch.utils.data

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.student_research.overnight6.gpu_bind import bind_cuda  # noqa: E402

_p = argparse.ArgumentParser()
_p.add_argument("--model", type=int, required=True, choices=(7, 8))
_p.add_argument("--epochs", type=int, default=30)
_p.add_argument("--batch", type=int, default=8)
_p.add_argument("--lr", type=float, default=3e-4)
_p.add_argument("--device", default="auto")
_p.add_argument("--num-workers", type=int, default=4)
_p.add_argument("--seed", type=int, default=1001)
_p.add_argument("--calibrate-batches", type=int, default=200)
_p.add_argument("--smoke", action="store_true")
_args = _p.parse_args()
bind_cuda(str(_args.device))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from collision_jepa3.config import Config  # noqa: E402
from collision_jepa3.evaluation.metrics import CAUTION, SAFE, STOP, direction_scores_np  # noqa: E402
from collision_jepa3.student_research.fdec_student import build_parent_decoder, decoder_state_hash  # noqa: E402
from collision_jepa3.student_research.overnight6.losses import decoder_anchor  # noqa: E402
from collision_jepa3.student_research.overnight6.models import build_model  # noqa: E402
from collision_jepa3.student_research.overnight6.parent_task import (  # noqa: E402
    TASK_WEIGHTS,
    build_teacher_loss,
    load_h_now_table,
    m7_losses,
    m8_losses,
)
from collision_jepa3.student_research.overnight6.schedule import LAMBDA_DECODER_REG, decoder_lr_multiplier  # noqa: E402
from collision_jepa3.student_research.overnight6.train import (  # noqa: E402
    EXPECTED_HASH,
    STATS,
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

TEMP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp")
DIRS = {
    7: TEMP / "m7_parent_task_latent_v1",
    8: TEMP / "m8_parent_task_only_v1",
}
SEED = int(_args.seed)
CAL_MIN = 1e-6


class TaskDS(Dataset):
    def __init__(self, inner, hnow, drop_fdec: bool):
        self.inner = inner
        self.hnow = hnow
        self.drop_fdec = bool(drop_fdec)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        x = self.inner[i]
        key = (str(x["clip_id"]), int(x["frame"]))
        x["h_now"] = torch.from_numpy(np.ascontiguousarray(self.hnow[key]))
        if self.drop_fdec:
            x.pop("F_teacher", None)
        return x


def collate_fn(batch):
    out = {
        "clip": torch.stack([b["clip"] for b in batch]),
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
    if "F_teacher" in batch[0]:
        out["F_teacher"] = torch.stack([b["F_teacher"] for b in batch])
    return out


def student_hash(model) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        if str(k).startswith("decoder."):
            continue
        h.update(k.encode())
        h.update(v.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def gvec(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = []
    n = 0.0
    for g in gs:
        if g is None:
            continue
        x = g.detach().reshape(-1)
        flat.append(x)
        n += float(x.pow(2).sum())
    if not flat:
        return None, 0.0
    return torch.cat(flat), n ** 0.5


def extra_region_metrics(rows, s75):
    warn = s75.warn_cfg()
    left, center, right = warn["left_cols"], warn["center_cols"], warn["right_cols"]
    buckets = {
        "LEFT": [0, 0],
        "CENTER": [0, 0],
        "RIGHT": [0, 0],
        "spatial_LEFT": [0, 0],
        "spatial_CENTER": [0, 0],
        "spatial_RIGHT": [0, 0],
    }
    for r in rows:
        hg = np.asarray(r["gt_1s"], dtype=np.float32)
        hp = np.asarray(r["H_student"], dtype=np.float32)
        if s75.peak_class(hg) != STOP:
            continue
        hit = int(s75.peak_class(hp) == STOP)
        scores = direction_scores_np(hg, [1, 1, 1, 1, 1], left, center, right)
        d = max(scores, key=scores.get)
        buckets[d][0] += hit
        buckets[d][1] += 1
        col = int(np.unravel_index(int(np.argmax(hg)), hg.shape)[1])
        if col in left:
            sp = "spatial_LEFT"
        elif col in center:
            sp = "spatial_CENTER"
        else:
            sp = "spatial_RIGHT"
        buckets[sp][0] += hit
        buckets[sp][1] += 1

    def rec(k):
        h, n = buckets[k]
        return float(h / n) if n else float("nan")

    return {
        "stop_recall_dir_LEFT": rec("LEFT"),
        "stop_recall_dir_CENTER": rec("CENTER"),
        "stop_recall_dir_RIGHT": rec("RIGHT"),
        "stop_recall_spatial_LEFT": rec("spatial_LEFT"),
        "stop_recall_spatial_CENTER": rec("spatial_CENTER"),
        "stop_recall_spatial_RIGHT": rec("spatial_RIGHT"),
        "n_stop_dir": {k: buckets[k][1] for k in ("LEFT", "CENTER", "RIGHT")},
    }


@torch.no_grad()
def eval_sel(model, loader, device, s75, use_fdec: bool):
    model.eval()
    rows = []
    cos, l1 = [], []
    ctx = amp_ctx(device)
    for batch in loader:
        b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with ctx:
            out = model(b["clip"])
        h = out["Hhat_1s"].float().cpu().numpy()
        ht = b["H_teacher"].float().cpu().numpy()
        gt = b["gt_1s"].float().cpu().numpy()
        pref = b["pre_fork"].cpu().numpy()
        if use_fdec:
            fs, ft = out["F"].float(), b["F_teacher"].float()
            cos.append(F.cosine_similarity(fs.flatten(1), ft.flatten(1)).cpu().numpy())
            l1.append((fs - ft).abs().mean(dim=(1, 2, 3)).cpu().numpy())
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
    m["fdec_cosine"] = float(np.concatenate(cos).mean()) if cos else None
    m["fdec_l1"] = float(np.concatenate(l1).mean()) if l1 else None
    m["peaks"] = peak_stats(rows)
    m.update(extra_region_metrics(rows, s75))
    return m


def official_key(sel):
    sa = float(sel["safe_recall"] or 0)
    st = float(sel["stop_recall"] or 0)
    pr = float(sel.get("stop_precision") or 0)
    return (1 if sa >= 0.85 else 0, st, pr)


def pull_baseline(path: Path, epoch: int) -> dict | None:
    p = path / f"metrics_epoch_{epoch:02d}.json"
    if not p.is_file():
        return None
    rec = json.loads(p.read_text())
    sel = rec.get("selection") or {}
    pk = (sel.get("peaks") or {}).get("gt_stop") or {}
    return {
        "tag": f"{path.name} ep{epoch}",
        "SAFE": sel.get("safe_recall"),
        "THREAT": sel.get("threat_recall"),
        "STOP": sel.get("stop_recall"),
        "STOP_prec": sel.get("stop_precision"),
        "pre_fork_STOP": sel.get("pre_fork_stop_recall"),
        "post_fork_STOP": sel.get("post_fork_stop_recall"),
        "gt_stop_p10": pk.get("p10"),
        "gt_stop_p25": pk.get("p25"),
        "collapse_lt_025": (sel.get("peaks") or {}).get("collapsed_stop_frac"),
    }


def write_report(exp: Path, mid: int, best_ep: int | None) -> None:
    rows = []
    for name, ep, folder in (
        ("M1", 10, TEMP / "overnight6" / "m1_stop_floor"),
        ("M5", 17, TEMP / "overnight6" / "m5_multiscale"),
        ("heatmap_dominant", 14, TEMP / "m1_heatmap_dominant_v1"),
    ):
        b = pull_baseline(folder, ep)
        if b:
            b["name"] = name
            rows.append(b)
    for tag, folder in (("M7", DIRS[7]), ("M8", DIRS[8])):
        bestp = folder / "checkpoint_best.pt"
        ep = None
        if bestp.is_file():
            try:
                ep = int(torch.load(bestp, map_location="cpu", weights_only=False).get("epoch", 0))
            except Exception:
                ep = None
        if ep:
            b = pull_baseline(folder, ep)
            if b:
                b["name"] = f"{tag}_official_ep{ep}"
                rows.append(b)
    lines = ["# M7/M8 parent-task students\n", "| model | SAFE | THREAT | STOP | STOP prec | pre-fork STOP | post-fork STOP | GT-STOP p10 | collapse |\n", "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"]
    for r in rows:
        lines.append(
            f"| {r.get('name') or r.get('tag')} | {r.get('SAFE')} | {r.get('THREAT')} | {r.get('STOP')} | {r.get('STOP_prec')} | {r.get('pre_fork_STOP')} | {r.get('post_fork_STOP')} | {r.get('gt_stop_p10')} | {r.get('collapse_lt_025')} |\n"
        )
    lines.append("\nQuestions are answered in `answers.json` after both runs finish.\n")
    (exp / "comparison_partial.md").write_text("".join(lines), encoding="utf-8")
    (TEMP / "m7_m8_comparison.md").write_text("".join(lines), encoding="utf-8")


def answers_if_ready() -> None:
    a7, a8 = DIRS[7] / ".success", DIRS[8] / ".success"
    if not (a7.is_file() and a8.is_file()):
        return
    def load_off(folder: Path):
        ck = torch.load(folder / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        ep = int(ck["epoch"])
        rec = json.loads((folder / f"metrics_epoch_{ep:02d}.json").read_text())
        return ep, rec["selection"]
    e7, s7 = load_off(DIRS[7])
    e8, s8 = load_off(DIRS[8])
    m1 = pull_baseline(TEMP / "overnight6" / "m1_stop_floor", 10) or {}
    def f(sel, k, default=None):
        if isinstance(sel, dict) and "SAFE" in sel and k in ("SAFE", "STOP"):
            return sel.get(k)
        return (sel or {}).get(k, default)
    stop7 = float(s7.get("stop_recall") or 0)
    stop8 = float(s8.get("stop_recall") or 0)
    stop1 = float(m1.get("STOP") or 0)
    col7 = float((s7.get("peaks") or {}).get("collapsed_stop_frac") or 1)
    col1 = float(m1.get("collapse_lt_025") or 1)
    sa7 = float(s7.get("safe_recall") or 0)
    sa8 = float(s8.get("safe_recall") or 0)
    pre7 = float(s7.get("pre_fork_stop_recall") or 0)
    pre8 = float(s8.get("pre_fork_stop_recall") or 0)
    pre1 = float(m1.get("pre_fork_STOP") or 0)
    ans = {
        "1_parent_task_improved_STOP": bool(max(stop7, stop8) > stop1 + 1e-6),
        "STOP_M1": stop1,
        "STOP_M7": stop7,
        "STOP_M8": stop8,
        "2_hinge_reduced_low_peak_GT_STOP": bool(col7 < col1 - 1e-6 or float((s8.get("peaks") or {}).get("collapsed_stop_frac") or 1) < col1 - 1e-6),
        "collapse_M1": col1,
        "collapse_M7": col7,
        "collapse_M8": (s8.get("peaks") or {}).get("collapsed_stop_frac"),
        "3_SAFE_acceptable": {"M7_ge_082": sa7 >= 0.82, "M8_ge_082": sa8 >= 0.82, "M7": sa7, "M8": sa8},
        "4_prefork_improved": bool(max(pre7, pre8) > pre1 + 1e-6),
        "pre_M1": pre1,
        "pre_M7": pre7,
        "pre_M8": pre8,
        "5_latent_helped_M7_vs_M8": {
            "M7_STOP": stop7,
            "M8_STOP": stop8,
            "M7_SAFE": sa7,
            "M8_SAFE": sa8,
            "helped_STOP": bool(stop7 > stop8 + 1e-6),
        },
        "gates": {
            "any_SAFE082_STOP070": (sa7 >= 0.82 and stop7 >= 0.70) or (sa8 >= 0.82 and stop8 >= 0.70),
            "any_SAFE085_STOP060": (sa7 >= 0.85 and stop7 >= 0.60) or (sa8 >= 0.85 and stop8 >= 0.60),
        },
        "official_epochs": {"M7": e7, "M8": e8},
    }
    (TEMP / "m7_m8_answers.json").write_text(json.dumps(ans, indent=2), encoding="utf-8")


def run() -> None:
    mid = int(_args.model)
    EXP = DIRS[mid]
    EXP.mkdir(parents=True, exist_ok=True)
    log_path = EXP / "run.log"

    def log(m: str) -> None:
        print(m, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(m + "\n")

    device = pick_cuda(str(_args.device))
    s75, rows, gt, train_ids, sel_ids, train_ds, sel_ds = prepare_data(forbid_test=True)
    if any(str(rows[i]["split"]) == "test" for i in list(train_ids) + list(sel_ids)):
        raise SystemExit("test ids in train/selection")
    hnow_train = load_h_now_table(rows, train_ids)
    hnow_sel = load_h_now_table(rows, sel_ids)
    drop_f = mid == 8
    train_ds = TaskDS(train_ds, hnow_train, drop_f)
    sel_ds = TaskDS(sel_ds, hnow_sel, drop_f)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = Config.load()
    decoder = build_parent_decoder(cfg, device="cpu")
    dec_h = decoder_state_hash(decoder)
    if dec_h != EXPECTED_HASH:
        raise SystemExit(f"decoder hash {dec_h} != {EXPECTED_HASH}")
    model = build_model(1, decoder).to(device)
    sh = student_hash(model)
    (EXP / "student_init_hash.txt").write_text(sh + "\n", encoding="utf-8")
    set_decoder_trainable(model.decoder, False)
    dec_init = {k: v.detach().clone() for k, v in model.decoder.named_parameters()}
    torch.save({k: v.cpu() for k, v in dec_init.items()}, EXP / "decoder_init.pt")
    teacher = build_teacher_loss(device)
    st = np.load(STATS)
    mu = torch.tensor(st["mu"], dtype=torch.float32, device=device)
    sigma = torch.tensor(st["sigma"], dtype=torch.float32, device=device)

    cf = collate_fn

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=_args.batch,
        shuffle=True,
        num_workers=_args.num_workers,
        collate_fn=cf,
        pin_memory=True,
        generator=g,
    )
    cal_loader = DataLoader(
        train_ds,
        batch_size=_args.batch,
        shuffle=False,
        num_workers=0,
        collate_fn=cf,
    )
    sel_loader = DataLoader(
        sel_ds,
        batch_size=_args.batch,
        shuffle=False,
        num_workers=min(2, _args.num_workers),
        collate_fn=cf,
    )

    # safety: task match + logits + finite on one batch
    b0 = next(iter(cal_loader))
    b0 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b0.items()}
    if mid == 8 and "F_teacher" in b0:
        raise SystemExit("M8 loaded F_teacher")
    if mid == 7 and "F_teacher" not in b0:
        raise SystemExit("M7 missing F_teacher")
    model.train()
    out0 = model(b0["clip"])
    if out0["Hhat_1s_logits"].shape != out0["Hhat_1s"].shape:
        raise SystemExit("logits/prob shape mismatch")
    if tuple(out0["F"].shape[1:]) != (128, 16, 16) or tuple(out0["Hhat_1s"].shape[1:]) != (5, 5):
        raise SystemExit("output rank mismatch")
    if mid == 7:
        parts0 = m7_losses(teacher, out0, b0, mu, sigma, model.predictor, model.decoder, dec_init)
    else:
        parts0 = m8_losses(teacher, out0, b0, model.decoder, dec_init)
    for k, v in parts0.items():
        if torch.is_tensor(v) and (not torch.isfinite(v).all()):
            raise SystemExit(f"non-finite {k}")
    parts0["L_task"].backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise SystemExit("non-finite grads")
    gdec0 = grad_norm(list(model.decoder.parameters()))
    if gdec0 > 1e-8:
        raise SystemExit(f"decoder grads while frozen {gdec0}")
    model.zero_grad(set_to_none=True)
    log(f"preflight_ok hash={dec_h} student={sh} L_task={float(parts0['L_task'].detach())}")

    s_task = s_lat = 1.0
    if mid == 7:
        n_cal = 2 if _args.smoke else int(_args.calibrate_batches)
        acc_t = acc_l = 0.0
        n = 0
        model.train()
        model.decoder.eval()
        with torch.no_grad():
            for step, batch in enumerate(cal_loader):
                if step >= n_cal:
                    break
                b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                out = model(b["clip"])
                parts = m7_losses(teacher, out, b, mu, sigma, model.predictor, model.decoder, dec_init)
                if not torch.isfinite(parts["L_task"]) or not torch.isfinite(parts["L_latent"]):
                    raise SystemExit("calibration non-finite")
                acc_t += float(parts["L_task"])
                acc_l += float(parts["L_latent"])
                n += 1
        if n < n_cal:
            raise SystemExit(f"calibration saw {n} < {n_cal} batches")
        s_task, s_lat = acc_t / n, acc_l / n
        if s_task < CAL_MIN or s_lat < CAL_MIN or not np.isfinite([s_task, s_lat]).all():
            raise SystemExit(f"calibration failed S_task={s_task} S_latent={s_lat}")
        (EXP / "calibration.json").write_text(
            json.dumps({"S_task": s_task, "S_latent": s_lat, "n": n, "frozen": True}, indent=2),
            encoding="utf-8",
        )
        log(f"calibrated S_task={s_task:.6g} S_latent={s_lat:.6g}")
        model.train()
        model.decoder.eval()

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
    enc_p = list(model.encoder.parameters()) + list(model.proj.parameters())
    (EXP / "config.json").write_text(
        json.dumps(
            {
                "model": mid,
                "seed": SEED,
                "decoder_hash_init": dec_h,
                "student_init_hash": sh,
                "from": "random_student",
                "no_test": True,
                "S_task": s_task if mid == 7 else None,
                "S_latent": s_lat if mid == 7 else None,
                "task_weights": TASK_WEIGHTS,
                "params": model.param_audit(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_ckpt(EXP / "checkpoint_initial.pt", model, 0, {"random_init": True, "student_init_hash": sh})
    log(f"start M{mid} seed={SEED} device={device} vis={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')}")

    best_key = None
    n_clip = 0
    max_ep = 1 if _args.smoke else _args.epochs
    max_b = 3 if _args.smoke else 10**9
    for ep in range(1, max_ep + 1):
        mult = decoder_lr_multiplier(ep)
        unfreeze = mult > 0
        set_decoder_trainable(model.decoder, unfreeze)
        for gpg in opt_d.param_groups:
            gpg["lr"] = float(_args.lr) * float(mult)
        model.train()
        model.decoder.eval()
        t0 = time.perf_counter()
        acc = defaultdict(float)
        n_b = 0
        g_dec_first = None
        gdiag = {}
        for step, batch in enumerate(train_loader):
            if step >= max_b:
                break
            b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            with ac:
                out = model(b["clip"])
                if mid == 7:
                    parts = m7_losses(teacher, out, b, mu, sigma, model.predictor, model.decoder, dec_init)
                    loss = parts["L_task"] / s_task + 0.10 * parts["L_latent"] / s_lat + LAMBDA_DECODER_REG * parts["L_dec_reg"]
                else:
                    parts = m8_losses(teacher, out, b, model.decoder, dec_init)
                    loss = parts["L_task"] + LAMBDA_DECODER_REG * parts["L_dec_reg"]
            opt_s.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)
            loss.backward()
            gs = torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            if float(gs) >= 1.0:
                n_clip += 1
            if g_dec_first is None:
                g_dec_first = grad_norm(list(model.decoder.parameters()))
            g_stud = grad_norm(list(model.student_parameters()))
            opt_s.step()
            if unfreeze:
                opt_d.step()
            for k, v in parts.items():
                if torch.is_tensor(v):
                    acc[k] += float(v.detach())
            acc["L_total"] += float(loss.detach())
            for name, w in TASK_WEIGHTS.items():
                acc[f"{name}_weighted"] += w * float(parts[name].detach())
            if mid == 7:
                acc["L_task_norm"] += float((parts["L_task"] / s_task).detach())
                acc["L_latent_norm"] += float((0.10 * parts["L_latent"] / s_lat).detach())
            n_b += 1
            if step == 0:
                with ac:
                    out = model(b["clip"])
                    if mid == 7:
                        parts = m7_losses(teacher, out, b, mu, sigma, model.predictor, model.decoder, dec_init)
                        vt, nn_t = gvec(parts["L_task"].float(), enc_p)
                        vl, nn_l = gvec(parts["L_latent"].float(), enc_p)
                        if vt is not None and vl is not None:
                            gdiag["g_task"] = nn_t
                            gdiag["g_latent"] = nn_l
                            gdiag["cos_task_latent"] = float(
                                (vt * vl).sum() / ((vt.norm() * vl.norm()).clamp_min(1e-12))
                            )
                    else:
                        parts = m8_losses(teacher, out, b, model.decoder, dec_init)
                        _, nn_t = gvec(parts["L_task"].float(), enc_p)
                        gdiag["g_task"] = nn_t
                gdiag["g_student"] = g_stud
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
                "decoder_lr_mult": mult,
                "decoder_grad_norm": g_dec_first,
                "decoder_drift_rms": drift,
                "clip_events": n_clip,
                "sec": time.perf_counter() - t0,
                **gdiag,
            }
        )
        if _args.smoke:
            log(f"SMOKE M{mid} ep={ep} unfreeze={unfreeze} g_dec={g_dec_first} total={train_m['L_total']}")
            (EXP / f"smoke_epoch_{ep}.json").write_text(json.dumps(train_m, indent=2, default=str), encoding="utf-8")
            continue
        sel_m = eval_sel(model, sel_loader, device, s75, use_fdec=(mid == 7))
        rec = {"epoch": ep, "train": train_m, "selection": sel_m}
        if torch.cuda.is_available():
            rec["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        (EXP / f"metrics_epoch_{ep:02d}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        extra = {"selection": sel_m, "decoder_drift_rms": drift}
        save_ckpt(EXP / f"checkpoint_epoch_{ep:02d}.pt", model, ep, extra)
        save_ckpt(EXP / "checkpoint_last.pt", model, ep, extra)
        key = official_key(sel_m)
        if best_key is None or key > best_key[0]:
            best_key = (key, ep)
            save_ckpt(EXP / "checkpoint_best.pt", model, ep, extra)
        sa, st = float(sel_m["safe_recall"] or 0), float(sel_m["stop_recall"] or 0)
        pk = sel_m.get("peaks") or {}
        gst = pk.get("gt_stop") or {}
        log(
            f"epoch {ep} SAFE={sa:.3f} THR={sel_m.get('threat_recall')} STOP={st:.3f} "
            f"prec={sel_m.get('stop_precision')} pre={sel_m.get('pre_fork_stop_recall')} "
            f"post={sel_m.get('post_fork_stop_recall')} coll={pk.get('collapsed_stop_frac')} "
            f"p10={gst.get('p10')} p25={gst.get('p25')} med={gst.get('p50')} "
            f"dirR={sel_m.get('stop_recall_dir_RIGHT')} spC={sel_m.get('stop_recall_spatial_CENTER')} "
            f"dec_mult={mult} drift={drift:.5g} g_dec={g_dec_first} "
            f"gcos={gdiag.get('cos_task_latent')} sec={train_m['sec']:.1f}"
        )
        write_report(EXP, mid, best_key[1] if best_key else None)
    if not _args.smoke:
        (EXP / ".success").write_text("ok\n")
        log("DONE")
        write_report(EXP, mid, best_key[1] if best_key else None)
        answers_if_ready()


def main() -> None:
    EXP = DIRS[int(_args.model)]
    try:
        run()
    except Exception:
        EXP.mkdir(parents=True, exist_ok=True)
        (EXP / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
