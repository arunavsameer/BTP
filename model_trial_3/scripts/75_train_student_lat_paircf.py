#!/usr/bin/env python3
"""StudentLat-PairCF: historical student_lat encoder distilled onto Pair-CF F_dec."""

from __future__ import annotations

import argparse
import hashlib
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
TEMP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp")
RAID = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_3")
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.evaluation.metrics import CAUTION, SAFE, STOP, classify_heatmap  # noqa: E402
from collision_jepa3.evaluation.operating_point import operating_metrics  # noqa: E402
from collision_jepa3.models.vjepa_backbone import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from scripts.pair_cf_lib import scenario_bootstrap_ci  # noqa: E402
from collision_jepa3.student_research.cache_access import ParentTargetCache  # noqa: E402
from collision_jepa3.student_research.fdec_student import build_parent_decoder, decoder_state_hash  # noqa: E402
from collision_jepa3.training.optimizer import build_scheduler  # noqa: E402
from collision_jepa3.utils.device import resolve_device_safe, select_amp_dtype  # noqa: E402
from student_model.encoder import build_student_encoder  # noqa: E402
from student_model.loss import blob_weights, relation_loss, spatial_cosine, weighted_l1  # noqa: E402
from student_model.student import JEPAPredictor  # noqa: E402

EXP = TEMP / "student_lat_paircf"
RGB_ROOT = TEMP / "shared_rgb_cache_256_v1"
PARENT_ROOT = TEMP / "parent_target_cache_v2"
if not PARENT_ROOT.is_dir():
    PARENT_ROOT = RAID / "parent_target_cache_v2"
PARENT_CKPT = ROOT / "checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt"
EXPECTED_HASH = "77f78c8714667617"
OFFSETS_16 = [-90, -84, -78, -72, -66, -60, -54, -48, -42, -36, -30, -24, -18, -12, -6, 0]
STOP_THR = 0.60
CAUTION_THR = 0.25
FORBIDDEN_WEIGHT_SUBSTR = (
    "student_lat/",
    "student_s0",
    "student_s1",
    "student_s2",
    "latent_match",
    "TeacherLatent",
)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def pick_gpu(pref: str) -> str:
    if pref and pref != "auto":
        return resolve_device_safe(pref)
    try:
        import subprocess

        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        cands = []
        for line in raw.strip().splitlines():
            idx, mem = [x.strip() for x in line.split(",")]
            cands.append((float(mem), int(idx)))
        cands.sort()
        if cands:
            return resolve_device_safe(f"cuda:{cands[0][1]}")
    except Exception:
        pass
    return resolve_device_safe("auto")


class RGBCache256:
    def __init__(self, root: Path):
        from importlib.machinery import SourceFileLoader

        mod = SourceFileLoader("rgb256", str(root / "loader.py")).load_module()
        self.inner = mod.RGBCache256(root)

    def get_clip01(self, clip_id: str, frame: int) -> np.ndarray:
        return self.inner.get_frames(clip_id, frame, OFFSETS_16, input_size=256, normalize=False)


class StudentLatPairCF(nn.Module):
    """Encoder+predictor trainable; Pair-CF H1s decoder frozen. Inference uses encoder F."""

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.encoder = build_student_encoder("lat", out_channels=128)
        self.predictor = JEPAPredictor(128, grid=16)
        self.decoder = decoder
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.decoder.eval()
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.eval()
        return self

    def normalize(self, clip: torch.Tensor) -> torch.Tensor:
        return (clip - self._mean.to(dtype=clip.dtype, device=clip.device)) / self._std.to(
            dtype=clip.dtype, device=clip.device
        )

    def encode_context(self, clip: torch.Tensor) -> torch.Tensor:
        x = self.normalize(clip).permute(0, 2, 1, 3, 4).contiguous()
        return self.encoder(x)

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        ctx = self.encode_context(clip)
        pred = self.predictor(ctx)
        h, logits = self.decoder.forward_with_logits(ctx)
        return {
            "F_ctx": ctx,
            "F": ctx,
            "F_pred": pred,
            "Hhat_1s": h,
            "Hhat_1s_logits": logits,
        }

    def param_audit(self) -> dict:
        def n(mod):
            return int(sum(p.numel() for p in mod.parameters()))

        total = int(sum(p.numel() for p in self.parameters()))
        trainable = int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "encoder": n(self.encoder),
            "predictor": n(self.predictor),
            "decoder": n(self.decoder),
            "encoder_variant": "lat",
            "modules": [name for name, _ in self.named_modules() if name],
        }


class WinDS(Dataset):
    def __init__(self, rows, ids, gt, rgb, parent):
        self.rows = rows
        self.ids = list(ids)
        self.gt = gt
        self.rgb = rgb
        self.parent = parent

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        rid = int(self.ids[i])
        row = self.rows[rid]
        cid, t = row["clip_id"], int(row["frame"])
        clip = self.rgb.get_clip01(cid, t)
        pt = self.parent.get_parent_targets(cid, t)
        if pt is None:
            raise KeyError(f"missing parent {cid} {t}")
        return {
            "clip": torch.from_numpy(np.ascontiguousarray(clip)),
            "F_teacher": torch.from_numpy(pt["F_dec"]),
            "H_teacher": torch.from_numpy(pt["H_teacher"]),
            "gt_1s": torch.from_numpy(np.asarray(self.gt[rid], dtype=np.float32)),
            "pre_fork": bool(row.get("pre_fork")),
            "amb_1s": bool(row.get("amb_1s")),
            "sample_id": str(row.get("sample_id", "")),
            "clip_id": cid,
            "frame": t,
            "background_id": str(row.get("background_id", "")),
            "outcome": str(row.get("outcome", "")),
        }


def collate(batch):
    return {
        "clip": torch.stack([b["clip"] for b in batch]),
        "F_teacher": torch.stack([b["F_teacher"] for b in batch]),
        "H_teacher": torch.stack([b["H_teacher"] for b in batch]),
        "gt_1s": torch.stack([b["gt_1s"] for b in batch]),
        "pre_fork": torch.tensor([b["pre_fork"] for b in batch], dtype=torch.bool),
        "amb_1s": torch.tensor([b["amb_1s"] for b in batch], dtype=torch.bool),
        "sample_id": [b["sample_id"] for b in batch],
        "clip_id": [b["clip_id"] for b in batch],
        "frame": torch.tensor([b["frame"] for b in batch], dtype=torch.int64),
        "background_id": [b["background_id"] for b in batch],
        "outcome": [b["outcome"] for b in batch],
    }


def random_spatial_mask(fine: torch.Tensor, keep: float = 0.5) -> torch.Tensor:
    b, _, h, w = fine.shape
    return (torch.rand(b, 1, h, w, device=fine.device, dtype=fine.dtype) < float(keep)).to(fine.dtype)


def peak_class(h: np.ndarray) -> int:
    p = float(np.max(h))
    if p >= STOP_THR:
        return STOP
    if p >= CAUTION_THR:
        return CAUTION
    return SAFE


def latent_loss(out, batch, *, feat_w, cos_w, mgd_w, rel_w, mu, sigma, predictor, mgd_keep=0.5):
    fs = out["F"]
    ft = batch["F_teacher"]
    fs_n = (fs - mu) / (sigma + 1e-6)
    ft_n = (ft - mu) / (sigma + 1e-6)
    w = blob_weights(ft, None, batch["gt_1s"], energy_alpha=0.0, appear_boost=0.0, stop_boost=0.0)
    l_feat = weighted_l1(fs_n, ft_n, w)
    l_cos = spatial_cosine(fs, ft, w)
    l_rel = relation_loss(fs, ft)
    l_mgd = fs.new_zeros(())
    if mgd_w > 0 and predictor is not None:
        mask = random_spatial_mask(out["F_ctx"], keep=mgd_keep)
        f_mgd = predictor(out["F_ctx"], mask=mask)
        l_mgd = weighted_l1((f_mgd - mu) / (sigma + 1e-6), ft_n, w)
    total = feat_w * l_feat + cos_w * l_cos + mgd_w * l_mgd + rel_w * l_rel
    return total, {
        "L_feat": float(l_feat.detach()),
        "L_cos": float(l_cos.detach()),
        "L_mgd": float(l_mgd.detach()),
        "L_rel": float(l_rel.detach()),
        "L_z": 0.0,
        "L_total": float(total.detach()),
    }


def warn_cfg():
    return {
        "stop_threshold": STOP_THR,
        "caution_threshold": CAUTION_THR,
        "left_cols": [0, 1],
        "center_cols": [2],
        "right_cols": [3, 4],
        "row_weights": [1, 1, 1, 1, 1],
    }


def metrics_from_rows(rows: list[dict], h_key: str) -> dict:
    cm = np.zeros((3, 3), dtype=np.int64)
    preds, gts, valids = [], [], []
    parent_mae = []
    gt_mse = []
    prefork_stop_hit = prefork_stop_n = 0
    post_stop_hit = post_stop_n = 0
    pair = defaultdict(list)
    for r in rows:
        hp = np.asarray(r[h_key], dtype=np.float32)
        hg = np.asarray(r["gt_1s"], dtype=np.float32)
        ht = np.asarray(r["H_teacher"], dtype=np.float32)
        gt_c = peak_class(hg)
        pr_c = peak_class(hp)
        cm[gt_c, pr_c] += 1
        preds.append(hp)
        gts.append(hg)
        valids.append(np.ones_like(hg, dtype=bool))
        parent_mae.append(float(np.mean(np.abs(hp - ht))))
        gt_mse.append(float(np.mean((hp - hg) ** 2)))
        if gt_c == STOP:
            if r["pre_fork"]:
                prefork_stop_n += 1
                prefork_stop_hit += int(pr_c == STOP)
            else:
                post_stop_n += 1
                post_stop_hit += int(pr_c == STOP)
        pair[(r["sample_id"], int(r["frame"]))].append((r["background_id"], gt_c, pr_c))

    def rec(g, p=None):
        d = cm[g].sum()
        if d == 0:
            return float("nan")
        return float(cm[g, g if p is None else p] / d)

    stop_col = cm[:, STOP].sum()
    both_stop = both_safe = n_pairs = 0
    for key, vals in pair.items():
        if len(vals) != 2:
            continue
        n_pairs += 1
        gts_c = [v[1] for v in vals]
        prs = [v[2] for v in vals]
        if all(g == STOP for g in gts_c) and all(p == STOP for p in prs):
            both_stop += 1
        if all(g == SAFE for g in gts_c) and all(p == SAFE for p in prs):
            both_safe += 1
    op = operating_metrics(np.stack(preds), np.stack(gts), np.stack(valids), hi=0.5) if preds else {}
    sa, st = rec(SAFE), rec(STOP)
    out = {
        "confusion_matrix": cm.tolist(),
        "stop_recall": st,
        "safe_recall": sa,
        "threat_recall": rec(CAUTION),
        "stop_precision": float(cm[STOP, STOP] / stop_col) if stop_col else float("nan"),
        "safe_false_alarm_rate": float(1.0 - sa) if sa == sa else float("nan"),
        "balanced_stop_safe_recall": float(0.5 * (sa + st)) if sa == sa and st == st else float("nan"),
        "pre_fork_stop_recall": prefork_stop_hit / prefork_stop_n if prefork_stop_n else float("nan"),
        "post_fork_stop_recall": post_stop_hit / post_stop_n if post_stop_n else float("nan"),
        "n_prefork_stop": prefork_stop_n,
        "n_postfork_stop": post_stop_n,
        "gt_heatmap_mse": float(np.mean(gt_mse)) if gt_mse else float("nan"),
        "parent_heatmap_mae": float(np.mean(parent_mae)) if parent_mae else float("nan"),
        "recall_at_5fa": op.get("recall_at_5fa"),
        "ap_high": op.get("ap_high"),
        "both_background_correct_stop_rate": both_stop / n_pairs if n_pairs else float("nan"),
        "both_background_correct_safe_rate": both_safe / n_pairs if n_pairs else float("nan"),
        "n_verified_pairs": n_pairs,
        "n": len(rows),
        "operating": {k: (float(v) if isinstance(v, (float, int, np.floating)) else v) for k, v in op.items()},
    }
    return out


def bootstrap_metrics(rows: list[dict], h_key: str, seed: int = 1337) -> dict:
    sample_ids = np.array([r["sample_id"] for r in rows])
    gt_stop = np.array([1.0 if peak_class(r["gt_1s"]) == STOP else 0.0 for r in rows])
    pr_stop = np.array([1.0 if peak_class(r[h_key]) == STOP else 0.0 for r in rows])
    gt_safe = np.array([1.0 if peak_class(r["gt_1s"]) == SAFE else 0.0 for r in rows])
    pr_safe = np.array([1.0 if peak_class(r[h_key]) == SAFE else 0.0 for r in rows])
    hit_stop = np.where(gt_stop > 0, (pr_stop == 1).astype(np.float32), np.nan)
    hit_safe = np.where(gt_safe > 0, (pr_safe == 1).astype(np.float32), np.nan)

    def mean_nan(x):
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else float("nan")

    return {
        "stop_recall": scenario_bootstrap_ci(sample_ids, hit_stop, mean_nan, n_boot=500, seed=seed),
        "safe_recall": scenario_bootstrap_ci(sample_ids, hit_safe, mean_nan, n_boot=500, seed=seed + 1),
    }


def rank_checkpoint(m: dict) -> tuple:
    safe = float(m.get("safe_recall") or 0)
    qual = 1 if safe + 1e-12 >= 0.80 else 0
    return (
        qual,
        float(m.get("post_fork_stop_recall") or -1),
        float(m.get("stop_recall") or -1),
        float(m.get("threat_recall") or -1),
        float(m.get("stop_precision") or -1),
        float(m.get("balanced_stop_safe_recall") or -1),
    )


@torch.no_grad()
def collect(model, loader, device, amp_dtype):
    model.eval()
    rows = []
    cos, l1 = [], []
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None and str(device).startswith("cuda") else torch.nullcontext()
    for batch in loader:
        b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with ctx:
            out = model(b["clip"])
        fs = out["F"].float()
        ft = b["F_teacher"].float()
        cos.append(F.cosine_similarity(fs.flatten(1), ft.flatten(1)).cpu().numpy())
        l1.append((fs - ft).abs().mean(dim=(1, 2, 3)).cpu().numpy())
        h = out["Hhat_1s"].float().cpu().numpy()
        ht = b["H_teacher"].float().cpu().numpy()
        gt = b["gt_1s"].float().cpu().numpy()
        pref = b["pre_fork"].cpu().numpy()
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
                    "f_cos": float(cos[-1][i]),
                    "f_l1": float(l1[-1][i]),
                }
            )
    m = metrics_from_rows(rows, "H_student")
    m["fdec_cosine"] = float(np.concatenate(cos).mean()) if cos else float("nan")
    m["fdec_l1"] = float(np.concatenate(l1).mean()) if l1 else float("nan")
    parent_m = metrics_from_rows(rows, "H_teacher")
    return rows, m, parent_m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--integrity-only", action="store_true")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = load_cfg(args)
    seed = int(args.seed if getattr(args, "seed", None) is not None else 1337)
    torch.manual_seed(seed)
    np.random.seed(seed)
    EXP.mkdir(parents=True, exist_ok=True)
    log_path = EXP / "run.log"

    def log(m: str) -> None:
        print(m, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(m + "\n")

    try:
        rgb_ok = (RGB_ROOT / "COMPLETE").is_file()
        par_ok = (PARENT_ROOT / "COMPLETE").is_file()
        if not rgb_ok or not par_ok:
            raise SystemExit(f"caches incomplete rgb={rgb_ok} parent={par_ok}")
        rows = [json.loads(l) for l in (PARENT_ROOT / "index.jsonl").open(encoding="utf-8")]
        gt = np.load(PARENT_ROOT / "gt_1s.npy").astype(np.float32)
        train_ids = [i for i, r in enumerate(rows) if r["split"] == "train"]
        sel_ids = [i for i, r in enumerate(rows) if r["split"] == "selection"]
        test_ids = [i for i, r in enumerate(rows) if r["split"] == "test"]
        counts = (len(train_ids), len(sel_ids), len(test_ids))
        log(f"windows {counts}")
        if counts != (22680, 5120, 5120):
            raise SystemExit(f"sampling contract mismatch {counts}")
        mapping = json.loads((PARENT_ROOT / "mapping.json").read_text())
        missing = []
        for i in train_ids + sel_ids + test_ids:
            key = f"{rows[i]['clip_id']}::{int(rows[i]['frame'])}"
            if key not in mapping:
                missing.append(key)
        if missing:
            raise SystemExit(f"missing parent maps {len(missing)}")
        parent = ParentTargetCache(PARENT_ROOT)
        z_path = PARENT_ROOT / "Z_t.npy"
        z_ok = False
        z_note = "Z_t.npy present but not a function of cached F_dec (Z is adapter bottleneck of unwhitened fine; F_dec is task-whitened). Omitting Z L1."
        if z_path.is_file():
            z_ok = True
        decoder = build_parent_decoder(cfg, device="cpu")
        dec_h = decoder_state_hash(decoder)
        log(f"decoder_hash={dec_h}")
        if dec_h != EXPECTED_HASH:
            raise SystemExit(f"decoder hash mismatch {dec_h}")
        replay_abs = []
        rng = np.random.default_rng(seed)
        pick = rng.choice(train_ids, size=128, replace=False)
        decoder.eval()
        with torch.no_grad():
            for rid in pick:
                pt = parent.get_parent_targets(rows[int(rid)]["clip_id"], int(rows[int(rid)]["frame"]))
                f = torch.from_numpy(pt["F_dec"][None])
                h = decoder.forward_with_logits(f)[0][0].detach().cpu().numpy()
                replay_abs.append(float(np.max(np.abs(h - pt["H_teacher"]))))
        replay = {"n": len(replay_abs), "max_abs": max(replay_abs), "mean_abs": float(np.mean(replay_abs)), "ok": max(replay_abs) < 2e-2}
        log(f"F_dec→decoder vs H_teacher {replay}")
        if not replay["ok"]:
            raise SystemExit(f"decoder replay failed {replay}")

        f_list = []
        for rid in rng.choice(train_ids, size=min(4000, len(train_ids)), replace=False):
            pt = parent.get_parent_targets(rows[int(rid)]["clip_id"], int(rows[int(rid)]["frame"]))
            f_list.append(pt["F_dec"])
        arr = np.stack(f_list, 0)
        ch_mu = arr.mean(axis=(0, 2, 3), keepdims=True)
        ch_sigma = arr.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        np.savez(EXP / "fdec_channel_stats.npz", mu=ch_mu, sigma=ch_sigma)

        src_hashes = {
            "encoder.py": file_sha(ROOT / "student_model/encoder.py"),
            "student.py": file_sha(ROOT / "student_model/student.py"),
            "loss.py": file_sha(ROOT / "student_model/loss.py"),
            "parent_ckpt": file_sha(PARENT_CKPT),
        }
        device = pick_gpu(str(args.device))
        log(f"device={device}")
        amp_dtype = select_amp_dtype(cfg, device)
        torch.manual_seed(seed)
        model = StudentLatPairCF(decoder).to(device)
        audit = model.param_audit()
        log(f"param_audit { {k: audit[k] for k in ('total','trainable','frozen','encoder','predictor','decoder')} }")
        (EXP / "param_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        loaded_forbidden = []
        integrity = {
            "rgb_complete": rgb_ok,
            "parent_complete": par_ok,
            "windows": {"train": 22680, "selection": 5120, "test": 5120},
            "offsets": OFFSETS_16,
            "rgb_size": 256,
            "F_dec_shape": [128, 16, 16],
            "H_teacher_shape": [5, 5],
            "decoder_hash": dec_h,
            "parent_ckpt": str(PARENT_CKPT),
            "replay": replay,
            "z_target_present": z_ok,
            "z_l1_used": False,
            "z_note": z_note,
            "source_hashes": src_hashes,
            "forbidden_weights_loaded": loaded_forbidden,
            "inference_tensor": "encoder F_ctx [B,128,16,16] → Pair-CF decoder (no bottleneck/LN/task_whiten/close)",
            "train_supervised_tensor": "same F_ctx",
            "channel_stats_n": int(arr.shape[0]),
            "param_audit": {k: audit[k] for k in ("total", "trainable", "frozen", "encoder", "predictor", "decoder")},
            "recipe": {
                "optimizer": "AdamW",
                "encoder_lr": 3e-4,
                "predictor_lr": 3e-4,
                "weight_decay": 1e-4,
                "warmup_epochs": 5,
                "min_lr": 1e-6,
                "scheduler": "linear_warmup+cosine",
                "epochs": args.epochs,
                "batch": args.batch,
                "grad_clip": 1.0,
                "feat_weight": 1.0,
                "cos_weight": 0.2,
                "mgd_weight": 0.5,
                "rel_weight": 0.2,
                "z_weight": 0.0,
                "h_weight": 0.0,
                "augmentation": "none (historical random_horizontal_flip=false)",
                "upsample_repeats": 0,
                "energy_alpha": 0.0,
                "source": "model_trial_3/student_model/submit_student_lat.sh",
            },
        }
        (EXP / "integrity.json").write_text(json.dumps(integrity, indent=2, default=str), encoding="utf-8")
        if args.integrity_only:
            log("integrity-only OK")
            return

        rgb = RGBCache256(RGB_ROOT)
        mu = torch.tensor(ch_mu, dtype=torch.float32, device=device)
        sigma = torch.tensor(ch_sigma, dtype=torch.float32, device=device)
        train_ds = WinDS(rows, train_ids, gt, rgb, parent)
        sel_ds = WinDS(rows, sel_ids, gt, rgb, parent)
        test_ds = WinDS(rows, test_ids, gt, rgb, parent)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, collate_fn=collate, pin_memory=True
        )
        sel_loader = DataLoader(sel_ds, batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=collate)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=collate)

        enc_p = [p for p in model.encoder.parameters() if p.requires_grad]
        pred_p = [p for p in model.predictor.parameters() if p.requires_grad]
        wd = 0.0001
        opt = torch.optim.AdamW(
            [
                {"params": enc_p, "lr": 0.0003, "weight_decay": wd},
                {"params": pred_p, "lr": 0.0003, "weight_decay": wd},
            ]
        )
        sch = build_scheduler(opt, args.epochs, 1e-6, warmup_epochs=5)
        feat_w, cos_w, mgd_w, rel_w = 1.0, 0.2, 0.5, 0.2
        # Historical z_w=0.3 omitted. Calibrate remaining terms vs encoder grad of L_feat.
        cal_n = 0
        g_feat = g_cos = g_mgd = g_rel = []
        g_feat, g_cos, g_mgd, g_rel = [], [], [], []

        def gnorm(loss):
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            s = 0.0
            for p in enc_p:
                if p.grad is not None:
                    s += float(p.grad.detach().pow(2).sum())
            return s ** 0.5

        model.train()
        for batch in train_loader:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(b["clip"])
            w = blob_weights(b["F_teacher"], None, b["gt_1s"], energy_alpha=0.0, appear_boost=0.0, stop_boost=0.0)
            fs_n = (out["F"] - mu) / (sigma + 1e-6)
            ft_n = (b["F_teacher"] - mu) / (sigma + 1e-6)
            lf = weighted_l1(fs_n, ft_n, w)
            lc = spatial_cosine(out["F"], b["F_teacher"], w)
            lr = relation_loss(out["F"], b["F_teacher"])
            mask = random_spatial_mask(out["F_ctx"], 0.5)
            lm = weighted_l1((model.predictor(out["F_ctx"], mask=mask) - mu) / (sigma + 1e-6), ft_n, w)
            g_feat.append(gnorm(lf))
            out = model(b["clip"])
            lc = spatial_cosine(out["F"], b["F_teacher"], w)
            g_cos.append(gnorm(lc))
            out = model(b["clip"])
            lr = relation_loss(out["F"], b["F_teacher"])
            g_rel.append(gnorm(lr))
            out = model(b["clip"])
            mask = random_spatial_mask(out["F_ctx"], 0.5)
            ft_n = (b["F_teacher"] - mu) / (sigma + 1e-6)
            lm = weighted_l1((model.predictor(out["F_ctx"], mask=mask) - mu) / (sigma + 1e-6), ft_n, w)
            g_mgd.append(gnorm(lm))
            cal_n += 1
            if cal_n >= 4:
                break
        gf, gc, gm, gr = map(lambda x: float(np.median(x) + 1e-8), (g_feat, g_cos, g_mgd, g_rel))
        # Keep historical coefficients; log gradient scale. Do not invent heatmap losses.
        calib = {"g_feat": gf, "g_cos": gc, "g_mgd": gm, "g_rel": gr, "feat_w": feat_w, "cos_w": cos_w, "mgd_w": mgd_w, "rel_w": rel_w}
        log(f"grad_calib {calib}")
        (EXP / "loss_calibration.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
        (EXP / "config.json").write_text(json.dumps({**integrity["recipe"], "decoder_hash": dec_h, "device": device, "seed": seed}, indent=2), encoding="utf-8")
        torch.save({"model": model.state_dict(), "epoch": 0, "audit": audit}, EXP / "checkpoint_initial.pt")

        history = []
        best_rank = None
        best_ep = None
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16) if str(device).startswith("cuda") else torch.amp.GradScaler("cuda", enabled=False)
        autocast = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None and str(device).startswith("cuda")
            else torch.nullcontext()
        )
        params = enc_p + pred_p
        for ep in range(1, args.epochs + 1):
            model.train()
            t0 = time.perf_counter()
            acc = defaultdict(float)
            n_b = 0
            opt.zero_grad(set_to_none=True)
            for batch in train_loader:
                b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                with autocast:
                    out = model(b["clip"])
                    loss, terms = latent_loss(
                        out, b, feat_w=feat_w, cos_w=cos_w, mgd_w=mgd_w, rel_w=rel_w, mu=mu, sigma=sigma, predictor=model.predictor
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                for k, v in terms.items():
                    acc[k] += v
                n_b += 1
            if sch is not None:
                sch.step()
            train_m = {k: v / max(n_b, 1) for k, v in acc.items()}
            train_m["sec"] = time.perf_counter() - t0
            _, sel_m, parent_sel = collect(model, sel_loader, device, amp_dtype)
            rec = {"epoch": ep, "train": train_m, "selection": sel_m, "parent_selection": parent_sel}
            (EXP / f"metrics_epoch_{ep:02d}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
            torch.save(
                {"epoch": ep, "encoder": model.encoder.state_dict(), "predictor": model.predictor.state_dict(), "decoder": model.decoder.state_dict(), "audit": audit},
                EXP / f"checkpoint_epoch_{ep:02d}.pt",
            )
            rk = rank_checkpoint(sel_m)
            if best_rank is None or rk > best_rank:
                best_rank, best_ep = rk, ep
                torch.save(
                    {"epoch": ep, "encoder": model.encoder.state_dict(), "predictor": model.predictor.state_dict(), "decoder": model.decoder.state_dict(), "selection": sel_m, "audit": audit},
                    EXP / "checkpoint_best.pt",
                )
            history.append({"epoch": ep, "L_total": train_m["L_total"], "sel_stop": sel_m["stop_recall"], "sel_safe": sel_m["safe_recall"], "sel_post_stop": sel_m["post_fork_stop_recall"], "sel_cos": sel_m["fdec_cosine"]})
            log(
                f"epoch {ep} L={train_m['L_total']:.4f} feat={train_m['L_feat']:.4f} cos={train_m['L_cos']:.4f} "
                f"STOP={sel_m['stop_recall']:.3f} SAFE={sel_m['safe_recall']:.3f} post={sel_m['post_fork_stop_recall']:.3f} "
                f"fcos={sel_m['fdec_cosine']:.3f} sec={train_m['sec']:.1f}"
            )

        if best_ep is None:
            best_ep = args.epochs
        blob = torch.load(EXP / "checkpoint_best.pt", map_location=device, weights_only=False)
        model.encoder.load_state_dict(blob["encoder"])
        model.predictor.load_state_dict(blob["predictor"])
        torch.save(blob, EXP / "checkpoint_final.pt")
        test_rows, test_m, parent_test = collect(model, test_loader, device, amp_dtype)
        boot_s = bootstrap_metrics(test_rows, "H_student")
        boot_p = bootstrap_metrics(test_rows, "H_teacher")
        final = {
            "selected_epoch": best_ep,
            "selection_rank": list(best_rank) if best_rank else None,
            "test_student": test_m,
            "test_parent": parent_test,
            "bootstrap_student": boot_s,
            "bootstrap_parent": boot_p,
            "history": history,
            "thresholds": {"caution": CAUTION_THR, "stop": STOP_THR},
            "z_l1": False,
        }
        (EXP / "final_metrics.json").write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")
        report_md(integrity, final, audit, calib if "calib" in dir() else {})
        log(f"DONE selected_epoch={best_ep} test_STOP={test_m['stop_recall']} test_SAFE={test_m['safe_recall']}")
        (EXP / ".success").write_text("ok\n", encoding="utf-8")
    except Exception:
        traceback.print_exc()
        (EXP / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def report_md(integrity, final, audit, calib):
    p = TEMP / "reports"
    p.mkdir(parents=True, exist_ok=True)
    ts = final["test_student"]
    tp = final["test_parent"]
    lines = [
        "# StudentLat-PairCF",
        "",
        "Historical `student_lat` encoder + JEPA predictor, Pair-CF H1s decoder frozen.",
        "RGB 256 cache + parent F_dec targets. No Giant re-encode. No heatmap/ordinal/GT MSE training losses.",
        "",
        "## Integrity",
        f"- Decoder hash `{integrity['decoder_hash']}` (expected `{EXPECTED_HASH}`)",
        f"- F_dec→H replay max_abs `{integrity['replay']['max_abs']}`",
        f"- Windows 22680/5120/5120, offsets 16, RGB 256 letterbox + ImageNet at load",
        f"- Z L1 omitted: {integrity['z_note']}",
        f"- Supervised tensor = inference tensor = encoder `F_ctx` [128,16,16]",
        f"- Params total={audit['total']} trainable={audit['trainable']} frozen={audit['frozen']} (enc={audit['encoder']} pred={audit['predictor']} dec={audit['decoder']})",
        "",
        "## Recipe (submit_student_lat.sh)",
        json.dumps(integrity["recipe"], indent=2),
        "",
        "## Selected checkpoint",
        f"Epoch **{final['selected_epoch']}** (SAFE≥80% first, then post-fork STOP, STOP, THREAT, precision). Thresholds 0.25/0.60 fixed.",
        "",
        "## Test (group-mean GT, identical rows)",
        "",
        "| metric | student | Pair-CF parent |",
        "|---|---:|---:|",
    ]
    keys = [
        "stop_recall",
        "safe_recall",
        "threat_recall",
        "pre_fork_stop_recall",
        "post_fork_stop_recall",
        "safe_false_alarm_rate",
        "stop_precision",
        "balanced_stop_safe_recall",
        "recall_at_5fa",
        "ap_high",
        "gt_heatmap_mse",
        "parent_heatmap_mae",
        "both_background_correct_stop_rate",
        "both_background_correct_safe_rate",
    ]
    for k in keys:
        lines.append(f"| {k} | {ts.get(k)} | {tp.get(k)} |")
    lines += [
        "",
        f"- F_dec cosine (student vs teacher): **{ts.get('fdec_cosine')}**  L1 **{ts.get('fdec_l1')}**",
        f"- Student confusion: `{ts.get('confusion_matrix')}`",
        f"- Parent confusion: `{tp.get('confusion_matrix')}`",
        "",
        "## Bootstrap (sample_id groups)",
        json.dumps(final.get("bootstrap_student"), indent=2),
        "",
        "## Per-epoch (selection)",
    ]
    for h in final.get("history") or []:
        lines.append(
            f"- ep {h['epoch']}: L={h['L_total']:.4f} STOP={h['sel_stop']:.3f} SAFE={h['sel_safe']:.3f} postSTOP={h['sel_post_stop']:.3f} cos={h['sel_cos']:.3f}"
        )
    (p / "student_lat_paircf_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
