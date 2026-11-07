"""Token-local pair-conditioned counterfactual utilities (TPCA)."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.pair_cf_lib import (
    PairRow,
    add_channel_delta,
    broadcast_channel_delta,
    pair_shrink_d,
    pooled_F,
    proj_subspace,
    random_orthonormal_U,
)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


class StreamingCovariance:
    """Welford-style sum for mu and sum of xx^T; rows are token difference vectors."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n = 0
        self.sum = np.zeros(self.dim, dtype=np.float64)
        self.xxt = np.zeros((self.dim, self.dim), dtype=np.float64)

    def update(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"expected [*, {self.dim}], got {X.shape}")
        self.n += int(X.shape[0])
        self.sum += X.sum(axis=0)
        self.xxt += X.T @ X

    def mean(self) -> np.ndarray:
        if self.n == 0:
            return np.zeros(self.dim, dtype=np.float64)
        return self.sum / self.n

    def covariance(self, centered: bool = True) -> np.ndarray:
        if self.n < 2:
            return np.zeros((self.dim, self.dim), dtype=np.float64)
        mu = self.mean()
        C = self.xxt / self.n
        if centered:
            C = C - np.outer(mu, mu)
        return C

    def second_moment(self) -> np.ndarray:
        return self.xxt / max(self.n, 1)


def eigh_top_k(C: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns mu unused, eigenvalues descending full, U_top [k,C] rows orthonormal."""
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx].astype(np.float64)
    evecs = evecs[:, idx].astype(np.float64)
    U = evecs[:, :k].T.copy()
    return evals, U, (evals[:k] / evals.sum()) if evals.sum() > 0 else evals[:k]


def effective_ranks(evals: np.ndarray) -> dict[str, float]:
    lam = np.clip(np.asarray(evals, dtype=np.float64), 0.0, None)
    s = lam.sum()
    if s <= 0:
        return {"participation_ratio": 0.0, "entropy_rank": 0.0}
    pr = float((s * s) / np.square(lam).sum())
    p = lam / s
    p = p[p > 1e-30]
    ent = float(np.exp(-(p * np.log(p)).sum()))
    return {"participation_ratio": pr, "entropy_rank": ent}


def token_project_rows(D: np.ndarray, U: np.ndarray, lam: float) -> np.ndarray:
    """D [M,C], U [k,C] -> lam * P_k D row-wise."""
    D = np.asarray(D, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    coeff = D @ U.T
    return (float(lam) * (coeff @ U)).astype(np.float32)


def token_pair_shrink(Fa: np.ndarray, Fb: np.ndarray, U: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Token-local symmetric shrinkage; pair average preserved per token."""
    Fa = np.asarray(Fa, dtype=np.float32)
    Fb = np.asarray(Fb, dtype=np.float32)
    D = 0.5 * (Fa.astype(np.float64) - Fb.astype(np.float64))
    d = token_project_rows(D, U, lam)
    return Fa - d, Fb + d


def pooled_broadcast_shrink(Fa: np.ndarray, Fb: np.ndarray, U: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    d = pair_shrink_d(0.5 * (pooled_F(Fa) - pooled_F(Fb)), U, lam)
    return broadcast_channel_delta(Fa, d), add_channel_delta(Fb, d)


def verify_token_shrink_identity(Fa: np.ndarray, Fb: np.ndarray, U: np.ndarray, lam: float, rtol: float = 1e-4) -> dict[str, float]:
    t0, u0 = token_pair_shrink(Fa, Fb, U, 0.0)
    err0 = float(np.max(np.abs(t0 - Fa)) + np.max(np.abs(u0 - Fb)))
    D = 0.5 * (Fa - Fb)
    ta, tb = token_pair_shrink(Fa, Fb, U, lam)
    Dp = 0.5 * (ta - tb)
    Pd = token_project_rows(D, U, lam)
    if lam >= 1.0 - 1e-6:
        err_p = float(np.max(np.abs(Dp - (D - Pd))))
    else:
        err_p = float(np.max(np.abs(Dp - (D - lam * (D @ U.T) @ U))))
    savg = 0.5 * (Fa + Fb)
    ta_s = 0.5 * (ta + tb)
    err_avg = float(np.max(np.abs(ta_s - savg)))
    return {"lambda0_max_err": err0, "projected_diff_err": err_p, "pair_avg_max_err": err_avg}


def token_index_layout(n_tokens: int, t_prime: int, spatial: int) -> dict[str, Any]:
    """Map linear token index -> (t, row, col) matching adapter tokens_to_grid."""
    s2 = spatial * spatial
    if t_prime * s2 != n_tokens:
        raise ValueError(f"n_tokens {n_tokens} != {t_prime}*{spatial}^2")
    return {
        "n_tokens": n_tokens,
        "temporal_groups": t_prime,
        "spatial": spatial,
        "order": "reshape (B, T', S, S, C) row-major; linear index = t*S*S + row*S + col",
        "index_formula": {"t": "idx // (S*S)", "row": "(idx // S) % S", "col": "idx % S"},
    }


def pair_correction_norms(Fa: np.ndarray, Fb: np.ndarray, U: np.ndarray, lam: float) -> np.ndarray:
    D = 0.5 * (Fa.astype(np.float64) - Fb.astype(np.float64))
    d = token_project_rows(D, U, lam)
    return np.linalg.norm(d, axis=1).astype(np.float32)


def subspace_principal_angles(Ua: np.ndarray, Ub: np.ndarray) -> float:
    M = Ua @ Ub.T
    s = np.linalg.svd(M, compute_uv=False)
    return float(np.mean(np.arccos(np.clip(s, -1.0, 1.0))))


def projection_overlap(Ua: np.ndarray, Ub: np.ndarray) -> float:
    Pa = Ua.T @ Ua
    Pb = Ub.T @ Ub
    fro = float(np.linalg.norm(Pa @ Pb - Pa, "fro"))
    return fro


def pair_robustness_ext(pairs: list[PairRow], pred_by_key: dict) -> dict[str, float]:
    from collision_jepa3.evaluation.metrics import classify_heatmap

    gaps, peak_d, flips = [], [], []
    warn = {
        "caution_threshold": 0.25,
        "stop_threshold": 0.60,
        "row_weights": [1.0] * 5,
        "left_cols": [0, 1],
        "center_cols": [2],
        "right_cols": [3, 4],
    }
    for p in pairs:
        ha = pred_by_key[(p.clip_a, p.anchor)]
        hb = pred_by_key[(p.clip_b, p.anchor)]
        gaps.append(float(np.abs(ha - hb).mean()))
        peak_d.append(abs(float(ha.max()) - float(hb.max())))
        _, sa, _ = classify_heatmap(ha, warn)
        _, sb, _ = classify_heatmap(hb, warn)
        flips.append(int(sa != sb))
    g = np.asarray(gaps)
    return {
        "mean_pair_gap": float(g.mean()),
        "median_pair_gap": float(np.median(g)),
        "p90_pair_gap": float(np.percentile(g, 90)),
        "p95_pair_gap": float(np.percentile(g, 95)),
        "warning_flip_rate": float(np.mean(flips)),
        "mean_peak_diff": float(np.mean(peak_d)),
    }


def cell_pair_breakdown(pairs: list[PairRow], pred_a: dict, pred_b: dict, ds, occ_thr: float = 0.05) -> dict[str, float]:
    occ_g, emp_g, threat_g, n = 0.0, 0.0, 0.0, 0
    for p in pairs:
        ha = pred_a[(p.clip_a, p.anchor)]
        hb = pred_b[(p.clip_b, p.anchor)]
        gt = ds.h1s_at_anchor(p.clip_a, p.anchor)
        diff = np.abs(ha - hb)
        occ = gt >= occ_thr
        emp = ~occ
        occ_g += float(diff[occ].mean()) if occ.any() else 0.0
        emp_g += float(diff[emp].mean()) if emp.any() else 0.0
        if p.gt_occ_rows:
            tr = int(p.gt_occ_rows[min(1, len(p.gt_occ_rows) - 1)])
            threat_g += float(diff[tr, :].mean())
        n += 1
    if n == 0:
        return {}
    return {
        "mean_gap_gt_occupied": occ_g / n,
        "mean_gap_gt_empty": emp_g / n,
        "mean_gap_threat_row": threat_g / n,
    }


def run_unit_tests() -> None:
    rng = np.random.default_rng(0)
    C = 32
    M = 64
    k = 4
    Fa = rng.standard_normal((M, C)).astype(np.float32)
    Fb = Fa + rng.standard_normal((M, C)).astype(np.float32) * 0.1
    U, _ = np.linalg.qr(rng.standard_normal((C, k)), mode="reduced")
    U = U.T.astype(np.float64)
    chk = verify_token_shrink_identity(Fa, Fb, U, 1.0)
    assert chk["lambda0_max_err"] < 1e-5, chk
    assert chk["pair_avg_max_err"] < 1e-4, chk
    assert chk["projected_diff_err"] < 1e-3, chk
    stream = StreamingCovariance(C)
    X = 0.5 * (Fa - Fb)
    stream.update(X.astype(np.float64))
    Cemp = np.cov(X.T, bias=True)
    mu_s, C_s = stream.mean(), stream.covariance()
    assert np.allclose(mu_s, X.mean(0), atol=1e-6)
    assert np.allclose(C_s, Cemp, atol=1e-4)
    print("tpca_lib unit tests OK")


if __name__ == "__main__":
    run_unit_tests()
