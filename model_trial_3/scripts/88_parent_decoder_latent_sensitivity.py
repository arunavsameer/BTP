#!/usr/bin/env python3
"""Parent decoder stability: small F_dec noise vs heatmap change. Selection only. No train/test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from collision_jepa3.config import Config
from collision_jepa3.student_research.fdec_student import build_parent_decoder, decoder_state_hash

EXPECTED_HASH = "77f78c8714667617"
PARENT = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_3/parent_target_cache_v2")
OUT = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_3_temp/parent_decoder_latent_sensitivity_v1")
RELS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
SEED = 1337
BATCH = 64


def peak_class(h: np.ndarray) -> np.ndarray:
    p = h.reshape(h.shape[0], -1).max(1)
    y = np.zeros(h.shape[0], np.int64)
    y[p >= 0.25] = 1
    y[p >= 0.60] = 2
    return y


@torch.no_grad()
def decode(dec, F: torch.Tensor) -> torch.Tensor:
    h, _ = dec.forward_with_logits(F)
    return h.float()


def summarize(H0, H1, F0, F1, y0) -> dict:
    dH = (H1 - H0).reshape(H0.shape[0], -1)
    dF = (F1 - F0).reshape(F0.shape[0], -1)
    nH = np.linalg.norm(dH, axis=1)
    nF = np.linalg.norm(dF, axis=1)
    p0 = H0.reshape(H0.shape[0], -1).max(1)
    p1 = H1.reshape(H1.shape[0], -1).max(1)
    y1 = peak_class(H1)
    lip = nH / np.maximum(nF, 1e-12)
    relF = nF / np.maximum(np.linalg.norm(F0.reshape(F0.shape[0], -1), axis=1), 1e-12)
    gain = nH / np.maximum(relF, 1e-12)
    return {
        "mean_abs_dH": float(np.abs(dH).mean()),
        "max_abs_dH": float(np.abs(dH).max()),
        "p90_abs_dH": float(np.quantile(np.abs(dH), 0.90)),
        "mean_abs_dpeak": float(np.abs(p1 - p0).mean()),
        "max_abs_dpeak": float(np.abs(p1 - p0).max()),
        "p90_abs_dpeak": float(np.quantile(np.abs(p1 - p0), 0.90)),
        "class_flip_rate": float(np.mean(y1 != y0)),
        "safe_to_not_safe": float(np.mean((y0 == 0) & (y1 != 0))) if (y0 == 0).any() else None,
        "stop_to_not_stop": float(np.mean((y0 == 2) & (y1 != 2))) if (y0 == 2).any() else None,
        "mean_lipschitz_dH_over_dF": float(np.mean(lip)),
        "p90_lipschitz": float(np.quantile(lip, 0.90)),
        "mean_dH_norm_over_rel_dF": float(np.mean(gain)),
        "mean_rel_dF": float(np.mean(relF)),
        "mean_H_l1": float(np.abs(H0).mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (PARENT / "index.jsonl").open(encoding="utf-8")]
    sel = [i for i, r in enumerate(rows) if r["split"] == "selection"]
    n_test = sum(1 for r in rows if r["split"] == "test")
    if (len(sel), n_test) != (5120, 5120):
        raise SystemExit(f"split contract sel={len(sel)} test={n_test}")
    meta = json.loads((PARENT / "metadata.json").read_text())
    n = int(meta["num_training_anchors"])
    Fmm = np.memmap(PARENT / "F_dec.npy", dtype=np.float16, mode="r", shape=(n, 128, 16, 16))
    Hmm = np.memmap(PARENT / "H_teacher.npy", dtype=np.float16, mode="r", shape=(n, 5, 5))
    F = np.asarray(Fmm[np.array(sel)], dtype=np.float32)
    Ht = np.asarray(Hmm[np.array(sel)], dtype=np.float32)
    cfg = Config.load()
    dec = build_parent_decoder(cfg, device="cpu").eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    hsh = decoder_state_hash(dec)
    if hsh != EXPECTED_HASH:
        raise SystemExit(f"decoder hash {hsh}")
    rms = float(np.sqrt((F**2).mean()))
    H0 = []
    with torch.no_grad():
        tF = torch.from_numpy(F)
        for i in range(0, len(F), BATCH):
            H0.append(decode(dec, tF[i : i + BATCH]).numpy())
    H0 = np.concatenate(H0)
    replay_mae = float(np.abs(H0 - Ht).mean())
    replay_max = float(np.abs(H0 - Ht).max())
    y0 = peak_class(H0)
    rng = np.random.default_rng(SEED)
    results = {
        "n_selection": len(sel),
        "decoder_hash": hsh,
        "F_rms": rms,
        "F_mean_abs": float(np.abs(F).mean()),
        "cache_replay_mae": replay_mae,
        "cache_replay_max_abs": replay_max,
        "baseline_peaks": {
            "mean": float(H0.reshape(len(H0), -1).max(1).mean()),
            "safe_frac": float(np.mean(y0 == 0)),
            "stop_frac": float(np.mean(y0 == 2)),
        },
        "additive_isotropic": [],
        "multiplicative": [],
        "global_bias": [],
    }

    def run_pert(name, make_F1, rels=RELS):
        rows_out = []
        for rel in rels:
            F1 = make_F1(rel)
            Hs = []
            with torch.no_grad():
                t1 = torch.from_numpy(F1)
                for i in range(0, len(F), BATCH):
                    Hs.append(decode(dec, t1[i : i + BATCH]).numpy())
            H1 = np.concatenate(Hs)
            row = {"rel": rel, **summarize(H0, H1, F, F1, y0)}
            rows_out.append(row)
            print(
                f"{name} rel={rel:g} mean|dH|={row['mean_abs_dH']:.5g} max|dH|={row['max_abs_dH']:.5g} "
                f"mean|dpeak|={row['mean_abs_dpeak']:.5g} flip={row['class_flip_rate']:.4f} "
                f"lip={row['mean_lipschitz_dH_over_dF']:.5g}",
                flush=True,
            )
        return rows_out

    results["additive_isotropic"] = run_pert(
        "add",
        lambda rel: F + (rel * rms * rng.standard_normal(F.shape).astype(np.float32)),
    )
    results["multiplicative"] = run_pert(
        "mul",
        lambda rel: F * np.float32(1.0 + rel),
        rels=(1e-4, 1e-3, 1e-2, 3e-2, 1e-1),
    )
    results["global_bias"] = run_pert(
        "bias",
        lambda rel: F + np.float32(rel * rms),
        rels=(1e-4, 1e-3, 1e-2, 3e-2, 1e-1),
    )

    # How large a relative F noise is needed for mean |dpeak| >= 0.05 or flip>=5%
    def first_cross(rows, key, thr):
        for r in rows:
            if r[key] >= thr:
                return r["rel"]
        return None

    results["thresholds"] = {
        "add_rel_for_mean_dpeak_0.05": first_cross(results["additive_isotropic"], "mean_abs_dpeak", 0.05),
        "add_rel_for_flip_5pct": first_cross(results["additive_isotropic"], "class_flip_rate", 0.05),
        "add_rel_for_mean_dH_0.02": first_cross(results["additive_isotropic"], "mean_abs_dH", 0.02),
    }
    (OUT / "report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [
        "# Parent decoder latent sensitivity (selection, frozen Pair-CF)\n\n",
        f"Decoder hash `{hsh}`. F RMS={rms:.5g}. Cache replay MAE={replay_mae:.5g} (decode vs stored H_teacher).\n\n",
        "| kind | rel | mean\\|dH\\| | max\\|dH\\| | mean\\|dpeak\\| | class flip | lip \\|dH\\|/\\|dF\\| |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for kind in ("additive_isotropic", "multiplicative", "global_bias"):
        for r in results[kind]:
            lines.append(
                f"| {kind} | {r['rel']:g} | {r['mean_abs_dH']:.5g} | {r['max_abs_dH']:.5g} | "
                f"{r['mean_abs_dpeak']:.5g} | {r['class_flip_rate']:.4f} | {r['mean_lipschitz_dH_over_dF']:.5g} |\n"
            )
    (OUT / "report.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", OUT / "report.md", flush=True)


if __name__ == "__main__":
    main()
