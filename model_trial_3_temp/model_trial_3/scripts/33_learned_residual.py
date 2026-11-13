#!/usr/bin/env python3
"""Train tiny residual splitters on frozen F + frozen G decoder. Does not overwrite G.

Each variant sees one clip at a time (deployable). Pair losses use the mate in the batch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.evaluation.counterfactual import warning_report  # noqa: E402
from collision_jepa3.losses.heatmap import soft_bce_logits  # noqa: E402
from collision_jepa3.u1_runtime import load_frozen_g  # noqa: E402
from collision_jepa3.utils.device import describe_hardware, resolve_device  # noqa: E402


def _load24():
    p = Path(__file__).with_name("24_cadet_oracles.py")
    spec = importlib.util.spec_from_file_location("cadet24", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Add1x1(nn.Module):
    def __init__(self, k=1):
        super().__init__()
        self.r = nn.Conv2d(128, 128, k, padding=k // 2)
        nn.init.zeros_(self.r.weight)
        nn.init.zeros_(self.r.bias)

    def forward(self, f):
        fs = self.r(f)
        return f - fs, fs


class BottleneckAdd(nn.Module):
    def __init__(self, hid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(128, hid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, 128, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f):
        fs = self.net(f)
        return f - fs, fs


class Gate1x1(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Conv2d(128, 128, 1)
        nn.init.zeros_(self.g.weight)
        nn.init.constant_(self.g.bias, 4.0)

    def forward(self, f):
        g = torch.sigmoid(self.g(f))
        return g * f, (1.0 - g) * f


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Conv2d(128, 1, 1)
        nn.init.zeros_(self.g.weight)
        nn.init.constant_(self.g.bias, 4.0)

    def forward(self, f):
        g = torch.sigmoid(self.g(f))
        return g * f, (1.0 - g) * f


class SEGate(nn.Module):
    def __init__(self, hid=16):
        super().__init__()
        self.down = nn.Conv2d(128, hid, 1)
        self.up = nn.Conv2d(hid, 128, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.constant_(self.up.bias, 4.0)

    def forward(self, f):
        h = F.relu(self.down(f.mean(dim=(2, 3), keepdim=True)), inplace=True)
        g = torch.sigmoid(self.up(h))
        return g * f, (1.0 - g) * f


class FiLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Conv2d(128, 128, 1)
        self.beta = nn.Conv2d(128, 128, 1)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, f):
        p = f.mean(dim=(2, 3), keepdim=True)
        fc = f * (1.0 + torch.tanh(self.gamma(p))) + self.beta(p)
        return fc, f - fc


def pair_idx(mate: np.ndarray):
    i = np.arange(len(mate))
    ok = (mate >= 0) & (i < mate)
    return i[ok], mate[ok]


@torch.no_grad()
def std_mix_teacher(fa, fb):
    sa = fa.std(dim=1, keepdim=True)
    sb = fb.std(dim=1, keepdim=True)
    w = torch.softmax(torch.stack([sa, sb], 1) / 0.3, dim=1)
    return w[:, 0] * fa + w[:, 1] * fb


def score_split(name, pred, pack, g_rep, c24):
    row = c24.score_plan(name, pred, pack, g_rep if name != "G" else None)
    h = row["headline"]
    feas = row.get("feasible_vs_g") or {}
    mark = ""
    if feas:
        mark = " FEASIBLE" if feas.get("feasible") else " no"
    print(
        f"  STOP={h['stop_recall']:.3f} SAFE={h['safe_recall']:.3f} "
        f"pre={h['pre_fork_stop_recall']:.3f} rec5={h['recall_at_5fa']:.3f} "
        f"post={h['post_fork_stop_recall']:.3f} small={h['small_blob_stop_recall']:.3f} "
        f"flip={row['flips']['warn_flip']:.3f}{mark}",
        flush=True,
    )
    return {
        "name": name,
        "stop_recall": h["stop_recall"],
        "safe_recall": h["safe_recall"],
        "pre_fork_stop_recall": h["pre_fork_stop_recall"],
        "post_fork_stop_recall": h["post_fork_stop_recall"],
        "recall_at_5fa": h["recall_at_5fa"],
        "small_blob_stop_recall": h["small_blob_stop_recall"],
        "warn_flip": row["flips"]["warn_flip"],
        "feasible": feas.get("feasible"),
    }


@torch.no_grad()
def eval_split(split, decoder, pack, device, c24, g_rep, name):
    split.eval()
    decoder.eval()
    fine = pack["fine"]
    hs = []
    for i in range(0, len(fine), 64):
        f = fine[i : i + 64].to(device)
        fc, _ = split(f)
        h, _ = decoder.forward_with_logits(fc)
        hs.append(h.float().cpu())
    pred = torch.cat(hs, 0).numpy()
    return score_split(name, pred, pack, g_rep, c24)


def train_one(split, decoder, train, val_pack, device, c24, g_rep, name, *, epochs, lam_h, lam_inv, lam_var, distill, lr):
    opt = torch.optim.AdamW(split.parameters(), lr=lr, weight_decay=1e-4)
    feat, y, mate = train["fine"], train["gt"], train["mate"]
    lo, hi = pair_idx(mate)
    n_p = len(lo)
    best = None
    best_key = (-1.0, -1.0)
    for ep in range(epochs):
        split.train()
        decoder.eval()
        perm = np.random.permutation(n_p)
        total = 0.0
        nb = 0
        bs = 32
        for s in range(0, n_p, bs):
            sl = perm[s : s + bs]
            ia, ib = lo[sl], hi[sl]
            fa = feat[ia].to(device)
            fb = feat[ib].to(device)
            ya = y[ia].to(device)
            yb = y[ib].to(device)
            f = torch.cat([fa, fb], 0)
            tgt = torch.cat([ya, yb], 0)
            fc, fs = split(f)
            _, logits = decoder.forward_with_logits(fc)
            lh = soft_bce_logits(logits, tgt)
            fca, fcb = fc.chunk(2)
            linv = ((fca - fcb) ** 2).mean()
            lvar = f.new_zeros(())
            if lam_var > 0:
                std = fc.flatten(2).std(dim=2)
                lvar = F.relu(1.0 - std).mean()
            ld = f.new_zeros(())
            if distill == "mean":
                cstar = 0.5 * (fa + fb)
                ld = ((fca - cstar.detach()) ** 2).mean() + ((fcb - cstar.detach()) ** 2).mean()
            elif distill == "stdmix":
                cstar = std_mix_teacher(fa, fb).detach()
                ld = ((fca - cstar) ** 2).mean() + ((fcb - cstar) ** 2).mean()
            loss = lam_h * lh + lam_inv * linv + lam_var * lvar + (0.1 if distill else 0.0) * ld
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
        row = eval_split(split, decoder, val_pack, device, c24, g_rep, f"{name} ep{ep}")
        key = (row["stop_recall"] if row["safe_recall"] >= 0.90 else -1.0, row["recall_at_5fa"])
        if key > best_key:
            best_key = key
            best = {**row, "name": name, "epoch": ep, "train_loss": total / max(nb, 1)}
        print(f"    ep{ep} loss={total / max(nb, 1):.4f}", flush=True)
    return best


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--teacher-ckpt",
        default="checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt",
    )
    parser.add_argument("--out", default="reports/cadet_diag/learned_residual.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    c24 = _load24()
    cfg = load_cfg(args)
    cfg.set("heads.mode", "direct_h1s")
    cfg.set("adapter.variance_head", False)
    cfg.set("training.num_workers", 0)
    device = resolve_device(args.device)
    print(json.dumps(describe_hardware(device), indent=2), flush=True)
    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    recs = {r.clip_id: r for r in records}
    model = load_frozen_g(cfg, Path(args.teacher_ckpt), device, variance_head=False)
    for p in model.parameters():
        p.requires_grad = False
    decoder = model.decoder
    decoder.eval()

    caches = {}
    for split_name in ("val", "train"):
        print("collect", split_name, flush=True)
        ds = TeacherDataset(cfg, records, split_name, splits, dummy_backbone=False, use_token_cache=True)
        caches[split_name] = c24.collect(model, ds, cfg, device, recs)

    val = caches["val"]
    pred_g = c24.decode(model, val["fine"], val["z"], device)
    g_rep = warning_report(
        pred_g, val["gt"].numpy(), pre_fork=val["pre"], collision=val["coll"],
        blob_n=val["blob"], paired=val["paired"],
    )
    results = [score_split("G", pred_g, val, g_rep, c24)]
    # pair-mean oracle (no train)
    mate = val["mate"]
    idx = torch.from_numpy(np.where(mate >= 0, mate, np.arange(len(mate))))
    mix = 0.5 * (val["fine"] + val["fine"][idx])
    pred_m = c24.decode(model, mix, val["z"], device)
    results.append(score_split("pair-mean oracle (no train)", pred_m, val, g_rep, c24))

    variants = [
        # name, ctor, lam_h, lam_inv, lam_var, distill
        ("add1x1 L_H+L_inv", lambda: Add1x1(1), 1.0, 0.1, 0.0, None),
        ("add1x1 L_H only", lambda: Add1x1(1), 1.0, 0.0, 0.0, None),
        ("add1x1 L_inv only", lambda: Add1x1(1), 0.0, 1.0, 0.0, None),
        ("add1x1 L_H+L_inv+var", lambda: Add1x1(1), 1.0, 0.1, 0.05, None),
        ("add1x1 L_H+distill mean", lambda: Add1x1(1), 1.0, 0.05, 0.0, "mean"),
        ("add1x1 L_H+distill cell-std mix", lambda: Add1x1(1), 1.0, 0.05, 0.0, "stdmix"),
        ("add3x3 L_H+L_inv", lambda: Add1x1(3), 1.0, 0.1, 0.0, None),
        ("bottleneck add L_H+L_inv", lambda: BottleneckAdd(32), 1.0, 0.1, 0.0, None),
        ("gate1x1 L_H+L_inv", lambda: Gate1x1(), 1.0, 0.1, 0.0, None),
        ("spatial gate L_H+L_inv", lambda: SpatialGate(), 1.0, 0.1, 0.0, None),
        ("SE gate L_H+L_inv", lambda: SEGate(), 1.0, 0.1, 0.0, None),
        ("FiLM L_H+L_inv", lambda: FiLM(), 1.0, 0.1, 0.0, None),
    ]
    for name, ctor, lam_h, lam_inv, lam_var, distill in variants:
        print("train", name, flush=True)
        split = ctor().to(device)
        best = train_one(
            split, decoder, caches["train"], val, device, c24, g_rep, name,
            epochs=int(args.epochs), lam_h=lam_h, lam_inv=lam_inv, lam_var=lam_var,
            distill=distill, lr=float(args.lr),
        )
        results.append(best)
        print("  BEST", best["name"], f"STOP={best['stop_recall']:.3f} SAFE={best['safe_recall']:.3f} rec5={best['recall_at_5fa']:.3f}", flush=True)

    slim = [r for r in results if r]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"epochs": args.epochs, "results": slim}, indent=2), encoding="utf-8")
    print("wrote", path, flush=True)
    print("\nRANK by STOP then rec5 (SAFE>=0.90 preferred in training best)", flush=True)
    for r in sorted(slim, key=lambda x: (x["stop_recall"], x["recall_at_5fa"]), reverse=True):
        print(
            f"  {r['stop_recall']:.3f} rec5={r['recall_at_5fa']:.3f} SAFE={r['safe_recall']:.3f} "
            f"pre={r.get('pre_fork_stop_recall', 0):.3f}  {r['name']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
