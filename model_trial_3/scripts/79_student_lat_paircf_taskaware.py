#!/usr/bin/env python3
"""Task-aware Pair-CF student_lat from random init. Selection only. No test. No OLS."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
TEMP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp")
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402
from collision_jepa3.evaluation.metrics import CAUTION, SAFE, STOP  # noqa: E402
from collision_jepa3.models.vjepa_backbone import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from collision_jepa3.student_research.cache_access import ParentTargetCache  # noqa: E402
from collision_jepa3.student_research.fdec_student import build_parent_decoder, decoder_state_hash  # noqa: E402
from collision_jepa3.training.optimizer import build_scheduler  # noqa: E402
from collision_jepa3.utils.device import resolve_device_safe  # noqa: E402
from student_model.encoder import build_student_encoder  # noqa: E402
from student_model.loss import blob_weights, relation_loss, spatial_cosine, weighted_l1  # noqa: E402
from student_model.student import JEPAPredictor  # noqa: E402

spec = importlib.util.spec_from_file_location("s75", ROOT / "scripts/75_train_student_lat_paircf.py")
s75 = importlib.util.module_from_spec(spec)
sys.modules["s75"] = s75
spec.loader.exec_module(s75)

EXP = TEMP / "student_lat_paircf_taskaware"
RGB_ROOT = TEMP / "shared_rgb_cache_256_v1"
PARENT_ROOT = TEMP / "parent_target_cache_v2"
EXPECTED_HASH = "77f78c8714667617"
STATS = TEMP / "student_lat_paircf" / "fdec_channel_stats.npz"
PREV = TEMP / "student_lat_paircf"


class ChannelProj(nn.Module):
    """Identity-init 1×1. Decoder still sees [B,128,16,16]."""

    def __init__(self, c: int = 128):
        super().__init__()
        self.conv = nn.Conv2d(c, c, kernel_size=1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, :, 0, 0] = torch.eye(c)
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AuxTierHead(nn.Module):
    def __init__(self, c: int = 128, hidden: int = 64, n_class: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_class),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class StudentTaskAware(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.encoder = build_student_encoder("lat", out_channels=128)
        self.predictor = JEPAPredictor(128, grid=16)
        self.proj = ChannelProj(128)
        self.aux = AuxTierHead(128)
        self.decoder = decoder
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.eval()
        return self

    def encode_context(self, clip: torch.Tensor) -> torch.Tensor:
        x = (clip - self._mean.to(dtype=clip.dtype, device=clip.device)) / self._std.to(
            dtype=clip.dtype, device=clip.device
        )
        return self.encoder(x.permute(0, 2, 1, 3, 4).contiguous())

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        ctx = self.encode_context(clip)
        f_dec_in = self.proj(ctx)
        pred = self.predictor(ctx)
        h, logits = self.decoder.forward_with_logits(f_dec_in)
        return {
            "F_ctx": ctx,
            "F": f_dec_in,
            "F_pred": pred,
            "Hhat_1s": h,
            "Hhat_1s_logits": logits,
            "aux_logits": self.aux(ctx),
        }

    def param_audit(self) -> dict:
        def n(m):
            return int(sum(p.numel() for p in m.parameters()))

        total = int(sum(p.numel() for p in self.parameters()))
        tr = int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        return {
            "total": total,
            "trainable": tr,
            "frozen": total - tr,
            "encoder": n(self.encoder),
            "predictor": n(self.predictor),
            "proj_1x1": n(self.proj),
            "aux_head": n(self.aux),
            "decoder": n(self.decoder),
        }


def peak_class_t(gt: torch.Tensor) -> torch.Tensor:
    p = gt.flatten(1).max(dim=1).values
    y = torch.zeros(gt.size(0), dtype=torch.long, device=gt.device)
    y = torch.where(p >= 0.25, torch.full_like(y, CAUTION), y)
    y = torch.where(p >= 0.60, torch.full_like(y, STOP), y)
    return y


def focal_ce(logits, y, alpha, gamma=2.0):
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    pt = p.gather(1, y.view(-1, 1)).squeeze(1)
    logpt = logp.gather(1, y.view(-1, 1)).squeeze(1)
    w = alpha.gather(0, y)
    return (-w * (1.0 - pt).clamp(min=0, max=1).pow(gamma) * logpt).mean()


def ranking_margin(peaks: torch.Tensor, y: torch.Tensor, margin: float = 0.20):
    stop = y == STOP
    non = y != STOP
    if not bool(stop.any()) or not bool(non.any()):
        return peaks.new_zeros(())
    ps = peaks[stop]
    pn = peaks[non]
    # pairwise: STOP peaks should exceed non-STOP by margin
    return F.relu(margin - (ps.unsqueeze(1) - pn.unsqueeze(0))).mean()


def task_heatmap(h, gt, class_w):
    w_cell = 0.10 + 4.0 * gt.clamp(0, 1).pow(2) + 2.0 * (gt >= 0.50).to(gt.dtype)
    sl1 = F.smooth_l1_loss(h, gt, reduction="none", beta=0.05)
    l_h = (sl1 * w_cell * class_w.view(-1, 1, 1)).mean()
    l_peak = (
        F.smooth_l1_loss(h.flatten(1).max(1).values, gt.flatten(1).max(1).values, reduction="none", beta=0.05) * class_w
    ).mean()
    return l_h, l_peak


def latent_terms(out, batch, mu, sigma, proj):
    fs, ft = out["F"], batch["F_teacher"]
    w = blob_weights(ft, None, batch["gt_1s"], energy_alpha=0.0, appear_boost=0.0, stop_boost=0.0)
    fs_n = (fs - mu) / (sigma + 1e-6)
    ft_n = (ft - mu) / (sigma + 1e-6)
    l_feat = weighted_l1(fs_n, ft_n, w)
    l_cos = spatial_cosine(fs, ft, w)
    l_rel = relation_loss(fs, ft)
    mask = s75.random_spatial_mask(out["F_ctx"], 0.5)
    f_mgd = proj(out["predictor_ref"](out["F_ctx"], mask=mask))
    l_mgd = weighted_l1((f_mgd - mu) / (sigma + 1e-6), ft_n, w)
    return l_feat, l_cos, l_rel, l_mgd


def gnorm(params, loss):
    if float(loss.detach()) == 0.0:
        return 0.0
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    s = 0.0
    for g in gs:
        if g is not None:
            s += float(g.detach().pow(2).sum())
    return s ** 0.5


def peak_dist(hs, hg):
    peaks = np.max(hs, axis=(1, 2))
    gc = np.array([s75.peak_class(h) for h in hg])
    stop = peaks[gc == STOP]

    def s(x):
        x = np.asarray(x, np.float64)
        if x.size == 0:
            return {}
        return {"n": int(x.size), "mean": float(x.mean()), "p50": float(np.median(x)), "p10": float(np.quantile(x, 0.1)), "p90": float(np.quantile(x, 0.9))}

    return {
        "gt_safe": s(peaks[gc == SAFE]),
        "gt_threat": s(peaks[gc == CAUTION]),
        "gt_stop": s(stop),
        "collapsed_stop_frac": float(np.mean(stop < 0.25)) if stop.size else None,
    }


@torch.no_grad()
def eval_sel(model, loader, device, amp):
    model.eval()
    rows, hs, hg, cos = [], [], [], []
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.nullcontext()
    aux_ok = 0
    n = 0
    hmse = []
    for batch in loader:
        b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with ctx:
            out = model(b["clip"])
        h = out["Hhat_1s"].float().cpu().numpy()
        gt = b["gt_1s"].float().cpu().numpy()
        ht = b["H_teacher"].float().cpu().numpy()
        pref = b["pre_fork"].cpu().numpy()
        y = peak_class_t(b["gt_1s"]).cpu().numpy()
        pred_aux = out["aux_logits"].float().argmax(1).cpu().numpy()
        aux_ok += int((pred_aux == y).sum())
        n += len(y)
        fs, ft = out["F"].float(), b["F_teacher"].float()
        cos.append(F.cosine_similarity(fs.flatten(1), ft.flatten(1)).cpu().numpy())
        hmse.append(((h - gt) ** 2).mean(axis=(1, 2)))
        hs.append(h)
        hg.append(gt)
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
    m["gt_heatmap_mse"] = float(np.concatenate(hmse).mean())
    m["aux_acc"] = aux_ok / max(n, 1)
    m["peaks"] = peak_dist(np.concatenate(hs), np.concatenate(hg))
    m["threat_precision"] = (
        int(np.logical_and(np.array([s75.peak_class(r["gt_1s"]) for r in rows]) == CAUTION, np.array([s75.peak_class(r["H_student"]) for r in rows]) == CAUTION).sum())
        / max(1, int(sum(s75.peak_class(r["H_student"]) == CAUTION for r in rows)))
    )
    return m


def lerp_w(phase: str, t: float) -> dict:
    """t in [0,1] during B. Cosine never 0."""
    A = dict(feat=1.0, cos=0.20, mgd=0.50, rel=0.20, h=0.08, peak=0.08, focal=0.08, rank=0.08)
    C = dict(feat=0.20, cos=0.06, mgd=0.12, rel=0.08, h=1.00, peak=1.00, focal=1.00, rank=1.00)
    if phase == "A":
        return A
    if phase == "C":
        return C
    return {k: (1 - t) * A[k] + t * C[k] for k in A}


def cpu_latency_ms(model, n_warm=5, n_run=15) -> float:
    from copy import deepcopy

    dec = deepcopy(model.decoder).cpu().eval()
    m = StudentTaskAware(dec).cpu().eval()
    m.encoder.load_state_dict({k: v.cpu() for k, v in model.encoder.state_dict().items()})
    m.predictor.load_state_dict({k: v.cpu() for k, v in model.predictor.state_dict().items()})
    m.proj.load_state_dict({k: v.cpu() for k, v in model.proj.state_dict().items()})
    x = torch.rand(1, 16, 3, 256, 256)
    with torch.no_grad():
        for _ in range(n_warm):
            m(x)
        t0 = time.perf_counter()
        for _ in range(n_run):
            m(x)
        dt = (time.perf_counter() - t0) / n_run
    del m, dec
    return dt * 1000.0


def write_audit(path: Path, audit: dict, incompat: list[str]) -> None:
    lines = [
        "# Task-aware student_lat design audit",
        "",
        "## Decoder interface (unchanged)",
        "- Pair-CF H1s consumes `F` **[B,128,16,16]** then pools to **[B,5,5]** occupancy.",
        "- No bottleneck, LayerNorm, task_whiten, or Z_t on this path.",
        "- Identity 1×1 is shape-preserving; at init `F_proj == F_ctx`.",
        "",
        "## Tensors",
        "- RGB cache uint8 256 letterbox → [0,1] → ImageNet in the model.",
        "- `F_ctx` encoder [B,128,16,16]; `F` = proj(`F_ctx`) matched to cached `F_dec`.",
        "- Heatmap GT is group-mean `gt_1s` [B,5,5], not `H_teacher`.",
        "- Aux logits [B,3] from GAP(`F_ctx`); not decoder outputs.",
        "",
        "## Gradient path",
        "- Decoder params frozen (`requires_grad=False`) but **not** under `no_grad`.",
        "- Heatmap / peak / ranking losses backprop through decoder into 1×1 and encoder.",
        "- Latent L1/cos/rel/MGD on `F` / MGD=`proj(predictor(mask·F_ctx))`.",
        "- Focal only through aux head → `F_ctx`.",
        "",
        "## Incompatibilities flagged",
    ]
    if not incompat:
        lines.append("- None vs Pair-CF decoder contract.")
    else:
        lines += [f"- {x}" for x in incompat]
    lines.append("")
    lines.append(json.dumps(audit, indent=2))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_prev_sel(ep: int) -> dict:
    p = PREV / f"metrics_epoch_{ep:02d}.json"
    if not p.is_file():
        return {}
    s = json.loads(p.read_text())["selection"]
    return {k: s.get(k) for k in (
        "safe_recall", "threat_recall", "stop_recall", "stop_precision",
        "balanced_stop_safe_recall", "pre_fork_stop_recall", "post_fork_stop_recall", "fdec_cosine",
    )}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--ema-decay", type=float, default=0.98)
    parser.add_argument("--cos-trigger", type=float, default=0.80)
    parser.add_argument("--confirm-epochs", type=int, default=2)
    parser.add_argument("--transition-epochs", type=int, default=6)
    parser.add_argument("--max-phase-a-epochs", type=int, default=12)
    parser.add_argument("--rank-margin", type=float, default=0.20)
    parser.add_argument("--num-workers", type=int, default=4)
    add_config_args(parser)
    args = parser.parse_args()
    cfg = load_cfg(args)
    EXP.mkdir(parents=True, exist_ok=True)
    log_path = EXP / "run.log"

    def log(m: str) -> None:
        print(m, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(m + "\n")

    incompat = []
    try:
        rows = [json.loads(l) for l in (PARENT_ROOT / "index.jsonl").open()]
        gt = np.load(PARENT_ROOT / "gt_1s.npy").astype(np.float32)
        train_ids = [i for i, r in enumerate(rows) if r["split"] == "train"]
        sel_ids = [i for i, r in enumerate(rows) if r["split"] == "selection"]
        assert len(train_ids) == 22680 and len(sel_ids) == 5120
        peaks = gt[np.array(train_ids)].reshape(len(train_ids), -1).max(1)
        cls = np.zeros(len(train_ids), np.int64)
        cls[peaks >= 0.25] = CAUTION
        cls[peaks >= 0.60] = STOP
        counts = np.bincount(cls, minlength=3).astype(np.float64)
        inv = (len(train_ids) / np.maximum(counts, 1)) / 3.0
        sample_w = inv[cls]
        class_w_vec = inv / inv.mean()
        rgb = s75.RGBCache256(RGB_ROOT)
        parent = ParentTargetCache(PARENT_ROOT)
        train_ds = s75.WinDS(rows, train_ids, gt, rgb, parent)
        sel_ds = s75.WinDS(rows, sel_ids, gt, rgb, parent)
        sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double), num_samples=len(train_ids), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=args.num_workers, collate_fn=s75.collate, pin_memory=True)
        sel_loader = DataLoader(sel_ds, batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=s75.collate)

        device = resolve_device_safe(str(args.device))
        torch.manual_seed(1337)
        decoder = build_parent_decoder(cfg, device="cpu")
        dec_h = decoder_state_hash(decoder)
        if dec_h != EXPECTED_HASH:
            raise SystemExit(f"decoder hash {dec_h}")
        model = StudentTaskAware(decoder).to(device)
        # verify identity 1x1
        with torch.no_grad():
            z = torch.randn(2, 128, 16, 16, device=device)
            id_err = float((model.proj(z) - z).abs().max())
            # 1×1 identity is exact on CPU; CUDA TF32/cuDNN can leave ~1e-3 residual.
            if id_err > 5e-3:
                incompat.append(f"identity 1x1 max abs {id_err}")
                raise SystemExit(f"identity 1x1 broken err={id_err}")
            log(f"identity_1x1_max_abs={id_err}")
        audit = model.param_audit()
        lat_ms = None
        try:
            lat_ms = cpu_latency_ms(model)
        except Exception as exc:
            lat_ms = None
            log(f"cpu_latency_failed {exc}")
        log(f"params {audit} cpu_ms={lat_ms}")
        write_audit(EXP / "DESIGN_AUDIT.md", {"params": audit, "cpu_latency_ms": lat_ms, "decoder_hash": dec_h, "incompatibilities": incompat}, incompat)
        (TEMP / "reports").mkdir(exist_ok=True)
        shutil = __import__("shutil")
        shutil.copy2(EXP / "DESIGN_AUDIT.md", TEMP / "reports/student_lat_paircf_taskaware_audit.md")

        st = np.load(STATS)
        mu = torch.tensor(st["mu"], dtype=torch.float32, device=device)
        sigma = torch.tensor(st["sigma"], dtype=torch.float32, device=device)
        enc_p = list(model.encoder.parameters())
        pred_p = list(model.predictor.parameters())
        proj_p = list(model.proj.parameters())
        aux_p = list(model.aux.parameters())
        opt = torch.optim.AdamW(
            [
                {"params": enc_p, "lr": args.lr, "weight_decay": 1e-4},
                {"params": pred_p, "lr": args.lr, "weight_decay": 1e-4},
                {"params": proj_p, "lr": 1e-4, "weight_decay": 1e-4},
                {"params": aux_p, "lr": args.lr, "weight_decay": 1e-4},
            ]
        )
        sch = build_scheduler(opt, args.epochs, 1e-6, warmup_epochs=5)
        amp = str(device).startswith("cuda")
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.nullcontext()
        cw_t = torch.tensor(class_w_vec, dtype=torch.float32, device=device)
        alpha = torch.tensor(class_w_vec, dtype=torch.float32, device=device)
        trunk = enc_p + proj_p

        # scale task losses to L_feat grads (train batches only)
        model.train()
        gcal = defaultdict(list)
        ncal = 0
        for batch in train_loader:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with autocast:
                out = model(b["clip"])
                out["predictor_ref"] = model.predictor
                lf, lc, lr, lm = latent_terms(out, b, mu, sigma, model.proj)
                y = peak_class_t(b["gt_1s"])
                gw = cw_t[y]
                lh, lp = task_heatmap(out["Hhat_1s"].float(), b["gt_1s"].float(), gw)
                lfo = focal_ce(out["aux_logits"].float(), y, alpha)
                lrk = ranking_margin(out["Hhat_1s"].float().flatten(1).max(1).values, y, args.rank_margin)
            gcal["feat"].append(gnorm(trunk, lf.float()))
            gcal["h"].append(gnorm(trunk, lh.float()))
            gcal["peak"].append(gnorm(trunk, lp.float()))
            gcal["focal"].append(gnorm(enc_p, lfo.float()))
            gcal["rank"].append(gnorm(trunk, lrk.float()) if float(lrk.detach()) > 0 else 0.0)
            ncal += 1
            if ncal >= 4:
                break
        gf = float(np.median(gcal["feat"]) + 1e-8)
        def sc(name, target):
            g = float(np.median([x for x in gcal[name] if x > 0] or [1e-8]) + 1e-8)
            return target * gf / g
        scales = {"h": sc("h", 0.35), "peak": sc("peak", 0.20), "focal": sc("focal", 0.15), "rank": sc("rank", 0.20)}
        log(f"grad_scales {scales} g_feat={gf} raw={ {k: float(np.median(v)) for k,v in gcal.items()} }")
        (EXP / "loss_calibration.json").write_text(json.dumps({"scales": scales, "gcal": {k: v for k, v in gcal.items()}}, indent=2), encoding="utf-8")
        cfgj = {
            "from": "random_init_seed_1337",
            "not_from_epoch38": True,
            "ols": False,
            "parent_H_as_gt": False,
            "decoder_hash": dec_h,
            "optimizer": "AdamW wd=1e-4 warmup=5 cosine; encoder/predictor/aux lr=3e-4 (historical); proj 1e-4 (new)",
            "latent_A": "feat=1 cos=0.2 mgd=0.5 rel=0.2 (historical student_lat)",
            "cosine_floor": 0.06,
            "ema_decay": args.ema_decay,
            "cos_trigger": args.cos_trigger,
            "confirm_epochs": args.confirm_epochs,
            "params": audit,
            "cpu_latency_ms": lat_ms,
            "class_balanced_sampler": True,
            "no_test": True,
        }
        (EXP / "config.json").write_text(json.dumps(cfgj, indent=2), encoding="utf-8")
        torch.save({"encoder": model.encoder.state_dict(), "predictor": model.predictor.state_dict(), "proj": model.proj.state_dict(), "aux": model.aux.state_dict()}, EXP / "checkpoint_initial.pt")

        ema_cos = None
        confirm = 0
        phase = "A"
        trans_left = 0
        trans_t = 0.0
        best_rank = None
        best_ep = None
        baselines = {"parent_sel_from_ep38_file": load_prev_sel(38) and json.loads((PREV / "metrics_epoch_38.json").read_text())["parent_selection"],
                     "student_ep12": load_prev_sel(12), "student_ep38": load_prev_sel(38), "student_ep62": load_prev_sel(62)}
        # parent keys
        if (PREV / "metrics_epoch_38.json").is_file():
            ps = json.loads((PREV / "metrics_epoch_38.json").read_text()).get("parent_selection") or {}
            baselines["parent"] = {k: ps.get(k) for k in ("safe_recall", "threat_recall", "stop_recall", "stop_precision", "pre_fork_stop_recall", "post_fork_stop_recall", "balanced_stop_safe_recall")}
        (EXP / "baselines_selection.json").write_text(json.dumps(baselines, indent=2), encoding="utf-8")

        for ep in range(1, args.epochs + 1):
            model.train()
            model.decoder.eval()
            t0 = time.perf_counter()
            acc = defaultdict(float)
            n_b = 0
            cos_sum = 0.0
            wts = lerp_w(phase, trans_t)
            g_epoch = {}
            for step, batch in enumerate(train_loader):
                b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                with autocast:
                    out = model(b["clip"])
                    out["predictor_ref"] = model.predictor
                    lf, lc, lr, lm = latent_terms(out, b, mu, sigma, model.proj)
                    y = peak_class_t(b["gt_1s"])
                    gw = cw_t[y]
                    lh, lp = task_heatmap(out["Hhat_1s"].float(), b["gt_1s"].float(), gw)
                    lfo = focal_ce(out["aux_logits"].float(), y, alpha)
                    lrk = ranking_margin(out["Hhat_1s"].float().flatten(1).max(1).values, y, args.rank_margin)
                    loss = (
                        wts["feat"] * lf + wts["cos"] * lc + wts["mgd"] * lm + wts["rel"] * lr
                        + wts["h"] * scales["h"] * lh
                        + wts["peak"] * scales["peak"] * lp
                        + wts["focal"] * scales["focal"] * lfo
                        + wts["rank"] * scales["rank"] * lrk
                    )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
                opt.step()
                batch_cos = float(F.cosine_similarity(out["F"].float().flatten(1), b["F_teacher"].float().flatten(1)).mean().detach())
                ema_cos = batch_cos if ema_cos is None else args.ema_decay * ema_cos + (1 - args.ema_decay) * batch_cos
                cos_sum += batch_cos
                for k, v in (("L_feat", lf), ("L_cos", lc), ("L_mgd", lm), ("L_rel", lr), ("L_h", lh), ("L_peak", lp), ("L_focal", lfo), ("L_rank", lrk), ("L_total", loss)):
                    acc[k] += float(v.detach())
                n_b += 1
                if step == 0:
                    with autocast:
                        out = model(b["clip"])
                        out["predictor_ref"] = model.predictor
                        lf, lc, lr, lm = latent_terms(out, b, mu, sigma, model.proj)
                        y = peak_class_t(b["gt_1s"])
                        lh, lp = task_heatmap(out["Hhat_1s"].float(), b["gt_1s"].float(), cw_t[y])
                        lfo = focal_ce(out["aux_logits"].float(), y, alpha)
                        lrk = ranking_margin(out["Hhat_1s"].float().flatten(1).max(1).values, y, args.rank_margin)
                    g_epoch = {
                        "g_feat": gnorm(trunk, lf.float()),
                        "g_h": gnorm(trunk, lh.float()),
                        "g_peak": gnorm(trunk, lp.float()),
                        "g_focal": gnorm(enc_p, lfo.float()),
                        "g_rank": gnorm(trunk, lrk.float()) if float(lrk.detach()) > 0 else 0.0,
                    }
                    opt.zero_grad(set_to_none=True)
            if sch is not None:
                sch.step()
            if decoder_state_hash(model.decoder) != EXPECTED_HASH:
                raise SystemExit("decoder mutated")
            mean_cos = cos_sum / max(n_b, 1)
            # phase machine at epoch boundary (not a single batch)
            triggered = False
            if phase == "A":
                if ema_cos is not None and ema_cos >= args.cos_trigger:
                    confirm += 1
                else:
                    confirm = 0
                if confirm >= args.confirm_epochs or ep >= args.max_phase_a_epochs:
                    phase = "B"
                    trans_left = args.transition_epochs
                    trans_t = 0.0
                    triggered = True
                    confirm = 0
            elif phase == "B":
                trans_left -= 1
                trans_t = 1.0 - max(trans_left, 0) / max(args.transition_epochs, 1)
                if trans_left <= 0:
                    phase = "C"
                    trans_t = 1.0
            train_m = {k: v / max(n_b, 1) for k, v in acc.items()}
            train_m.update({"ema_cos": ema_cos, "mean_cos": mean_cos, "phase": phase, "wts": wts, **g_epoch, "sec": time.perf_counter() - t0})
            sel_m = eval_sel(model, sel_loader, device, amp)
            rec = {"epoch": ep, "train": train_m, "selection": sel_m, "phase": phase, "triggered": triggered}
            (EXP / f"metrics_epoch_{ep:02d}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
            ck = {
                "epoch": ep,
                "encoder": model.encoder.state_dict(),
                "predictor": model.predictor.state_dict(),
                "proj": model.proj.state_dict(),
                "aux": model.aux.state_dict(),
                "decoder": model.decoder.state_dict(),
                "phase": phase,
            }
            torch.save(ck, EXP / f"checkpoint_epoch_{ep:02d}.pt")
            rk = s75.rank_checkpoint(sel_m)
            if best_rank is None or rk > best_rank:
                best_rank, best_ep = rk, ep
                torch.save({**ck, "selection": sel_m}, EXP / "checkpoint_best.pt")
            log(
                f"epoch {ep} phase={phase} ema_cos={ema_cos:.3f} sel_cos={sel_m['fdec_cosine']:.3f} "
                f"SAFE={sel_m['safe_recall']:.3f} THR={sel_m['threat_recall']:.3f} STOP={sel_m['stop_recall']:.3f} "
                f"prec={sel_m['stop_precision']:.3f} bal={sel_m['balanced_stop_safe_recall']:.3f} "
                f"pre={sel_m['pre_fork_stop_recall']:.3f} post={sel_m['post_fork_stop_recall']:.3f} "
                f"collapse={sel_m['peaks']['collapsed_stop_frac']} Hmse={sel_m['gt_heatmap_mse']:.4f} "
                f"aux={sel_m['aux_acc']:.3f} sec={train_m['sec']:.1f}"
            )
        log(f"DONE best_epoch={best_ep}")
        (EXP / ".success").write_text("ok\n")
    except Exception:
        traceback.print_exc()
        (EXP / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
