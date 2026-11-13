#!/usr/bin/env python3
"""Parent Pair-CF decoder channel sensitivity. Diagnostic only. No train, no test."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.student_research.overnight6.gpu_bind import bind_cuda  # noqa: E402

_p = argparse.ArgumentParser()
_p.add_argument("--device", default="auto")
_p.add_argument("--smoke", action="store_true")
_p.add_argument("--batch", type=int, default=32)
_p.add_argument("--pack", type=int, default=16)
_args = _p.parse_args()
bind_cuda(str(_args.device))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from collision_jepa3.config import Config  # noqa: E402
from collision_jepa3.student_research.fdec_student import build_parent_decoder, decoder_state_hash  # noqa: E402
from collision_jepa3.student_research.overnight6.train import pick_cuda  # noqa: E402

EXPECTED_HASH = "77f78c8714667617"
CKPT = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3/checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt")
PARENT = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_3/parent_target_cache_v2")
OUT = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_3_temp/parent_channel_sensitivity_v1")
SEED = 1337
SAFE_T, STOP_T = 0.25, 0.60
C = 128
SUBSETS = ("all", "gt_safe", "gt_threat", "gt_stop", "prefork_stop", "postfork_stop", "parent_correct_stop")


def peak_class(peaks: np.ndarray) -> np.ndarray:
    y = np.zeros(peaks.shape[0], np.int64)
    y[peaks >= SAFE_T] = 1
    y[peaks >= STOP_T] = 2
    return y


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra.astype(np.float64)
    rb = rb.astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


@torch.no_grad()
def decode(dec, F: torch.Tensor) -> torch.Tensor:
    h, _ = dec.forward_with_logits(F)
    return h.float()


def load_meta():
    rows = [json.loads(l) for l in (PARENT / "index.jsonl").open(encoding="utf-8")]
    train = [i for i, r in enumerate(rows) if r["split"] == "train"]
    sel = [i for i, r in enumerate(rows) if r["split"] == "selection"]
    n_test = sum(1 for r in rows if r["split"] == "test")
    if (len(train), len(sel), n_test) != (22680, 5120, 5120):
        raise SystemExit(f"split contract {(len(train), len(sel), n_test)}")
    return rows, np.array(train, np.int64), np.array(sel, np.int64)


def subset_masks(gt_c, par_c, prefork) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(gt_c.shape[0], bool),
        "gt_safe": gt_c == 0,
        "gt_threat": gt_c == 1,
        "gt_stop": gt_c == 2,
        "prefork_stop": (gt_c == 2) & prefork,
        "postfork_stop": (gt_c == 2) & (~prefork),
        "parent_correct_stop": (gt_c == 2) & (par_c == 2),
    }


def aggregate(abs_dh, abs_dp, sgn_dp, max_dh, y0, y1, masks) -> dict:
    out = {}
    for name, m in masks.items():
        n = int(m.sum())
        if n == 0:
            out[name] = {"n": 0}
            continue
        out[name] = {
            "n": n,
            "I": abs_dh[m].mean(0).astype(np.float64),
            "I_max": max_dh[m].mean(0).astype(np.float64),
            "I_peak": abs_dp[m].mean(0).astype(np.float64),
            "S_peak": sgn_dp[m].mean(0).astype(np.float64),
        }
        stop_leave = (y0[m] == 2)[:, None] & (y1[m] != 2)
        safe_leave = (y0[m] == 0)[:, None] & (y1[m] != 0)
        out[name]["stop_leave_rate"] = stop_leave.mean(0).astype(np.float64) if (y0[m] == 2).any() else np.full(C, np.nan)
        out[name]["safe_leave_rate"] = safe_leave.mean(0).astype(np.float64) if (y0[m] == 0).any() else np.full(C, np.nan)
    return out


def barplot(path: Path, scores: np.ndarray, title: str, k: int = 32):
    order = np.argsort(-scores)
    top = order[:k]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(k), scores[top], color="#3b6ea5")
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels([str(int(i)) for i in top], rotation=90, fontsize=7)
    ax.set_ylabel("sensitivity")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def heatmap_grid(path: Path, maps: list[np.ndarray], titles: list[str]):
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0))
    if n == 1:
        axes = [axes]
    for ax, m, t in zip(axes, maps, titles):
        im = ax.imshow(m, vmin=0, vmax=1, cmap="magma")
        ax.set_title(t, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


class Run:
    def __init__(self, device: str, batch: int, pack: int):
        self.device = device
        self.batch = batch
        self.pack = pack
        self.t0 = time.perf_counter()
        OUT.mkdir(parents=True, exist_ok=True)
        self.logp = OUT / "run.log"

    def log(self, m: str) -> None:
        print(m, flush=True)
        with self.logp.open("a", encoding="utf-8") as f:
            f.write(m + "\n")

    def setup(self):
        self.rows, self.train_idx, self.sel_idx = load_meta()
        meta = json.loads((PARENT / "metadata.json").read_text())
        n = int(meta["num_training_anchors"])
        self.Fmm = np.memmap(PARENT / "F_dec.npy", dtype=np.float16, mode="r", shape=(n, 128, 16, 16))
        self.Hmm = np.memmap(PARENT / "H_teacher.npy", dtype=np.float16, mode="r", shape=(n, 5, 5))
        self.gt = np.load(PARENT / "gt_1s.npy").astype(np.float32)
        cfg = Config.load()
        dec = build_parent_decoder(cfg, device="cpu")
        h = decoder_state_hash(dec)
        if h != EXPECTED_HASH:
            raise SystemExit(f"decoder hash {h}")
        dec.eval()
        for p in dec.parameters():
            p.requires_grad_(False)
        self.dec = dec.to(self.device)
        self.hash = h
        x = torch.zeros(2, 128, 16, 16, device=self.device)
        with torch.no_grad():
            y, logits = self.dec.forward_with_logits(x)
        if tuple(x.shape) != (2, 128, 16, 16) or tuple(y.shape) != (2, 5, 5):
            raise SystemExit(f"shapes F={tuple(x.shape)} H={tuple(y.shape)}")
        self.log(f"integrity ckpt={CKPT} hash={h} F={tuple(x.shape)} H={tuple(y.shape)} test_never_indexed=True")

    def windows(self, idx: np.ndarray):
        F = np.asarray(self.Fmm[idx], dtype=np.float32)
        Ht = np.asarray(self.Hmm[idx], dtype=np.float32)
        gt = self.gt[idx]
        pref = np.array([bool(self.rows[int(i)]["pre_fork"]) for i in idx])
        sid = np.array([self.rows[int(i)]["sample_id"] for i in idx])
        cid = np.array([self.rows[int(i)]["clip_id"] for i in idx])
        amb = np.array([bool(self.rows[int(i)]["amb_1s"]) for i in idx])
        return F, Ht, gt, pref, sid, cid, amb

    def decode_np(self, F: np.ndarray) -> np.ndarray:
        hs = []
        t = torch.from_numpy(F)
        with torch.inference_mode():
            for i in range(0, len(F), self.batch):
                b = t[i : i + self.batch].to(self.device, non_blocking=True)
                hs.append(decode(self.dec, b).cpu().numpy())
        return np.concatenate(hs)

    def perturb_all(self, F: np.ndarray, H0: np.ndarray, mode: str, mu: np.ndarray | None):
        n = F.shape[0]
        abs_dh = np.zeros((n, C), np.float32)
        abs_dp = np.zeros((n, C), np.float32)
        sgn_dp = np.zeros((n, C), np.float32)
        max_dh = np.zeros((n, C), np.float32)
        y1 = np.zeros((n, C), np.int64)
        p0 = H0.reshape(n, -1).max(1)
        chs_all = list(range(C))
        for s in range(0, n, self.batch):
            fb = F[s : s + self.batch]
            hb = H0[s : s + self.batch]
            pb = p0[s : s + self.batch]
            B = fb.shape[0]
            base = torch.from_numpy(fb).to(self.device)
            for p0i in range(0, C, self.pack):
                chs = chs_all[p0i : p0i + self.pack]
                K = len(chs)
                g = base.unsqueeze(1).expand(B, K, C, 16, 16).contiguous()
                for j, c in enumerate(chs):
                    if mode == "zero":
                        g[:, j, c].zero_()
                    else:
                        g[:, j, c] = float(mu[c])
                with torch.inference_mode():
                    h = decode(self.dec, g.reshape(B * K, C, 16, 16)).reshape(B, K, 5, 5).cpu().numpy()
                d = hb[:, None] - h
                ad = np.abs(d).reshape(B, K, 25)
                abs_dh[s : s + B, chs] = ad.mean(-1)
                max_dh[s : s + B, chs] = ad.max(-1)
                pk = h.reshape(B, K, 25).max(-1)
                abs_dp[s : s + B, chs] = np.abs(pb[:, None] - pk)
                sgn_dp[s : s + B, chs] = pb[:, None] - pk
                y1[s : s + B, chs] = peak_class(pk.reshape(B * K)).reshape(B, K)
            if (s // self.batch) % 20 == 0:
                self.log(f"  {mode} {s}/{n}")
        return abs_dh, abs_dp, sgn_dp, max_dh, y1

    def grads(self, F: np.ndarray, H0: np.ndarray, masks):
        n = F.shape[0]
        acc = {k: np.zeros(C, np.float64) for k in masks}
        acc_p = {k: np.zeros(C, np.float64) for k in masks}
        cnt = {k: 0 for k in masks}
        for s in range(0, n, self.batch):
            fb = torch.from_numpy(F[s : s + self.batch]).to(self.device)
            fb.requires_grad_(True)
            h, _ = self.dec.forward_with_logits(fb)
            h = h.float()
            B = fb.shape[0]
            gsum = torch.zeros_like(fb)
            flat = h.reshape(B, 25)
            for k in range(25):
                g = torch.autograd.grad(flat[:, k].sum(), fb, retain_graph=True)[0]
                gsum = gsum + g.abs()
            gmean = (gsum / 25.0).detach()
            pk = h.reshape(B, 25).max(1).values
            gp = torch.autograd.grad(pk.sum(), fb, retain_graph=False)[0].abs().detach()
            gch = gmean.mean(dim=(2, 3)).cpu().numpy()
            pch = gp.mean(dim=(2, 3)).cpu().numpy()
            sl = slice(s, s + B)
            local = {name: m[sl] for name, m in masks.items()}
            for name, m in local.items():
                if not m.any():
                    continue
                acc[name] += gch[m].sum(0)
                acc_p[name] += pch[m].sum(0)
                cnt[name] += int(m.sum())
            fb.grad = None
            if (s // self.batch) % 20 == 0:
                self.log(f"  grad {s}/{n}")
        out = {}
        for name in masks:
            d = max(cnt[name], 1)
            out[name] = {"n": cnt[name], "I_grad": (acc[name] / d), "I_grad_peak": (acc_p[name] / d)}
        return out

    def baseline_metrics(self, H, Ht, gt, prefork):
        peaks = H.reshape(len(H), -1).max(1)
        gt_p = gt.reshape(len(gt), -1).max(1)
        yc, yp = peak_class(gt_p), peak_class(peaks)
        cm = np.zeros((3, 3), np.int64)
        for a, b in zip(yc, yp):
            cm[a, b] += 1
        rec = lambda i: float(cm[i, i] / max(cm[i].sum(), 1))
        prec = lambda i: float(cm[i, i] / max(cm[:, i].sum(), 1))
        stop = yc == 2
        return {
            "n": int(len(H)),
            "replay_mae": float(np.abs(H - Ht).mean()),
            "confusion": cm.tolist(),
            "safe_recall": rec(0),
            "threat_recall": rec(1),
            "stop_recall": rec(2),
            "safe_precision": prec(0),
            "threat_precision": prec(1),
            "stop_precision": prec(2),
            "peak_mean": float(peaks.mean()),
            "gt_safe_peak": {"n": int((yc == 0).sum()), "mean": float(peaks[yc == 0].mean()) if (yc == 0).any() else None, "p10": float(np.quantile(peaks[yc == 0], 0.1)) if (yc == 0).any() else None, "p50": float(np.median(peaks[yc == 0])) if (yc == 0).any() else None},
            "gt_stop_peak": {"n": int(stop.sum()), "mean": float(peaks[stop].mean()) if stop.any() else None, "p10": float(np.quantile(peaks[stop], 0.1)) if stop.any() else None, "p50": float(np.median(peaks[stop])) if stop.any() else None},
            "pre_fork_stop_recall": float(((yp == 2) & stop & prefork).sum() / max((stop & prefork).sum(), 1)),
            "post_fork_stop_recall": float(((yp == 2) & stop & (~prefork)).sum() / max((stop & (~prefork)).sum(), 1)),
            "n_prefork_stop": int((stop & prefork).sum()),
            "n_postfork_stop": int((stop & (~prefork)).sum()),
            "parent_correct_stop": int(((yc == 2) & (yp == 2)).sum()),
        }

    def smoke(self) -> None:
        idx = self.train_idx[:8]
        F, Ht, gt, pref, sid, cid, amb = self.windows(idx)
        H0 = self.decode_np(F)
        H0b = self.decode_np(F)
        if not np.allclose(H0, H0b, atol=1e-5):
            raise SystemExit("baseline not deterministic")
        chs = [0, 1, 2, 17]
        G = F.copy()
        G[:, 1] = 0
        if not np.allclose(G[:, 0], F[:, 0]) or np.allclose(G[:, 1], F[:, 1]):
            raise SystemExit("perturb isolation failed")
        H1 = self.decode_np(G)
        if not np.isfinite(H1).all():
            raise SystemExit("nonfinite")
        for p in self.dec.parameters():
            if p.grad is not None:
                raise SystemExit("decoder grad in smoke")
            if p.requires_grad:
                raise SystemExit("decoder requires_grad")
        fb = torch.from_numpy(F[:2]).to(self.device)
        fb.requires_grad_(True)
        h, _ = self.dec.forward_with_logits(fb)
        h.reshape(2, 25)[:, 0].sum().backward()
        if any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in self.dec.parameters()):
            raise SystemExit("decoder param grads")
        if fb.grad is None or not torch.isfinite(fb.grad).all():
            raise SystemExit("input grad bad")
        (OUT / "smoke.json").write_text(
            json.dumps({"ok": True, "hash": self.hash, "dH_ch1": float(np.abs(H0 - H1).mean())}, indent=2),
            encoding="utf-8",
        )
        self.log("SMOKE_OK")

    def bootstrap_ci(self, values, sids, masks, n_boot=200):
        """values [N,C]; CI for I on 'all' and 'gt_stop' at sample-id level."""
        rng = np.random.default_rng(SEED)
        uniq = np.unique(sids)
        out = {}
        for key in ("all", "gt_stop", "prefork_stop"):
            m = masks[key]
            if m.sum() < 10:
                continue
            scores = []
            for _ in range(n_boot):
                take = rng.choice(uniq, size=len(uniq), replace=True)
                sel = np.isin(sids, take) & m
                if sel.sum() == 0:
                    continue
                scores.append(values[sel].mean(0))
            arr = np.stack(scores)
            out[key] = {"lo": np.quantile(arr, 0.025, 0).tolist(), "hi": np.quantile(arr, 0.975, 0).tolist()}
        return out

    def group_ablate(self, F, H0, mu, groups: dict[str, list[list[int]]], masks):
        results = {}
        p0 = H0.reshape(len(H0), -1).max(1)
        for gname, glist in groups.items():
            rows = []
            for gi, chs in enumerate(glist):
                G = F.copy()
                G[:, chs] = mu[chs][None, :, None, None]
                H1 = self.decode_np(G)
                d = np.abs(H0 - H1).reshape(len(H0), 25).mean(1)
                dp = np.abs(p0 - H1.reshape(len(H0), -1).max(1))
                rec = {"group_id": gi, "n_ch": len(chs), "channels": [int(c) for c in chs]}
                for name, m in masks.items():
                    rec[name] = {
                        "I": float(d[m].mean()) if m.any() else None,
                        "I_peak": float(dp[m].mean()) if m.any() else None,
                    }
                rows.append(rec)
                self.log(f"  group {gname} {gi}/{len(glist)} n={len(chs)}")
            results[gname] = rows
        return results

    def write_report(self, payload: dict) -> None:
        tr, se = payload["train_baseline"], payload["sel_baseline"]
        ranks = payload["rankings"]
        lines = ["# Parent feature channel sensitivity\n\n", "## A. Experiment integrity\n\n"]
        lines.append(f"- Checkpoint: `{CKPT}`\n")
        lines.append(f"- Decoder hash: `{self.hash}` (expected `{EXPECTED_HASH}`)\n")
        lines.append("- Splits: train 22680 / selection 5120 / test 5120. **TEST never indexed.**\n")
        lines.append("- No student training, no optimizer, no cache writes, decoder frozen eval.\n")
        lines.append(f"- Runtime_s: {payload['runtime_s']:.1f}. Peak GPU mem GB: {payload.get('peak_gpu_mem_gb')}\n\n")
        lines.append("## B. Baseline (decoded parent F_dec)\n\n")
        for name, b in ("TRAIN", tr), ("SELECTION", se):
            lines.append(
                f"- **{name}** n={b['n']} SAFE rec={b['safe_recall']:.3f} THR={b['threat_recall']:.3f} STOP={b['stop_recall']:.3f} "
                f"STOP prec={b['stop_precision']:.3f} pre-fork STOP={b['pre_fork_stop_recall']:.3f} "
                f"post-fork STOP={b['post_fork_stop_recall']:.3f} replay MAE vs cache H={b['replay_mae']:.5g}\n"
            )
        lines.append("\n## C. Channel rankings (TRAIN mean-replacement I, plus zero / grad)\n\n")

        def top_table(title, order, k=20):
            lines.append(f"### {title}\n\n| rank | ch | I_mean | I_zero | I_grad | S_mean_peak |\n|---:|---:|---:|---:|---:|---:|\n")
            I_m = np.array(payload["train"]["mean"]["all"]["I"])
            I_z = np.array(payload["train"]["zero"]["all"]["I"])
            I_g = np.array(payload["train"]["grad"]["all"]["I_grad"])
            S = np.array(payload["train"]["mean"]["all"]["S_peak"])
            for r, c in enumerate(order[:k], 1):
                lines.append(f"| {r} | {int(c)} | {I_m[c]:.5g} | {I_z[c]:.5g} | {I_g[c]:.5g} | {S[c]:.5g} |\n")
            lines.append("\n")

        for key, title in (
            ("overall", "Top 20 overall (I_mean all)"),
            ("stop", "Top 20 STOP (I_mean GT-STOP)"),
            ("safe", "Top 20 SAFE (I_mean GT-SAFE)"),
            ("prefork_stop", "Top 20 pre-fork STOP"),
            ("parent_correct_stop", "Top 20 parent-correct STOP"),
        ):
            top_table(title, np.array(ranks[key]["order"]))
            bot = np.array(ranks[key]["order"])[-20:][::-1]
            lines.append(f"Bottom 20 `{key}` channels: {', '.join(str(int(c)) for c in bot)}\n\n")

        lines.append("## D. Stability (TRAIN vs SELECTION, I_mean)\n\n")
        st = payload["stability"]
        for k, v in st.items():
            if k == "bootstrap":
                continue
            lines.append(
                f"- **{k}**: Spearman={v['spearman']:.3f} top8={v['top8']} top16={v['top16']} top32={v['top32']}\n"
            )
        lines.append("\nBootstrap 95% CI saved in `bootstrap_ci.json` (sample-id resampling).\n\n")
        lines.append("## E. Channel interactions (mean-replace groups, SELECTION)\n\n")
        gi = payload["groups"]
        lines.append("| group | n_ch | I_all | I_stop | I_prefork_stop |\n|---|---:|---:|---:|---:|\n")
        for gname, rows in gi.items():
            # mean over random reps if any
            Iall = np.mean([r["all"]["I"] or 0 for r in rows])
            Ist = np.mean([r["gt_stop"]["I"] or 0 for r in rows])
            Ipf = np.mean([r["prefork_stop"]["I"] or 0 for r in rows])
            nch = rows[0]["n_ch"]
            lines.append(f"| {gname} | {nch} | {Iall:.5g} | {Ist:.5g} | {Ipf:.5g} |\n")
        conc = payload["concentration"]
        lines.append("\n### Concentration (TRAIN I_mean share)\n\n")
        for k, v in conc.items():
            lines.append(f"- {k}: top8={v['top8']:.3f} top16={v['top16']:.3f} top32={v['top32']:.3f} of total I\n")

        lines.append("\n## F. Interpretation\n\n")
        for q, a in payload["answers"].items():
            lines.append(f"**{q}** {a}\n\n")
        lines.append("## G. Recommendation\n\n")
        lines.append(payload["recommendation"] + "\n")
        (OUT / "REPORT.md").write_text("".join(lines), encoding="utf-8")

    def interpret(self, payload):
        I = np.array(payload["train"]["mean"]["all"]["I"])
        tot = I.sum() + 1e-12
        share8 = float(np.sort(I)[-8:].sum() / tot)
        I_stop = np.array(payload["train"]["mean"]["gt_stop"]["I"])
        I_safe = np.array(payload["train"]["mean"]["gt_safe"]["I"])
        I_pf = np.array(payload["train"]["mean"]["prefork_stop"]["I"])
        rho_ss = spearman(I_stop, I_safe)
        rho_pf = spearman(I_stop, I_pf)
        rho_zg = spearman(np.array(payload["train"]["zero"]["all"]["I"]), np.array(payload["train"]["grad"]["all"]["I_grad"]))
        rho_zm = spearman(np.array(payload["train"]["zero"]["all"]["I"]), I)
        rho_mg = spearman(I, np.array(payload["train"]["grad"]["all"]["I_grad"]))
        stab = payload["stability"]["overall"]["spearman"]
        top8_ov = payload["stability"]["overall"]["top8"]
        # interaction: top16 group vs sum of individuals on selection
        ind_sum = float(np.sort(np.array(payload["sel"]["mean"]["all"]["I"]))[-16:].sum())
        grp = payload["groups"].get("top16_overall") or payload["groups"].get("top16_I_mean_all")
        gI = None
        for name, rows in payload["groups"].items():
            if name.startswith("top16") and "overall" in name or name == "top16_overall":
                gI = rows[0]["all"]["I"]
                break
        if gI is None:
            for name, rows in payload["groups"].items():
                if "top16" in name and "stop" not in name and "random" not in name:
                    gI = rows[0]["all"]["I"]
                    break
        interact = "unknown"
        if gI is not None:
            if gI > 1.25 * ind_sum:
                interact = "synergistic (group >> sum of individuals)"
            elif gI < 0.6 * ind_sum:
                interact = "redundant (group << sum of individuals)"
            else:
                interact = "roughly additive"
        concentrated = share8 >= 0.40
        answers = {
            "1. Concentrated?": f"{'Yes, relatively concentrated' if concentrated else 'No, fairly distributed'}: top-8 channels carry {share8:.1%} of TRAIN I_mean.",
            "2. STOP-specific channels?": f"STOP vs overall Spearman={spearman(I, I_stop):.3f}. Top STOP channels: {np.argsort(-I_stop)[:8].tolist()}.",
            "3. Pre-fork STOP vs general STOP?": f"Spearman={rho_pf:.3f}. Top pre-fork: {np.argsort(-I_pf)[:8].tolist()} vs STOP {np.argsort(-I_stop)[:8].tolist()}.",
            "4. SAFE vs STOP same channels?": f"Spearman={rho_ss:.3f} ({'largely shared' if rho_ss > 0.7 else 'partly distinct' if rho_ss > 0.4 else 'different'}).",
            "5. Zero / mean / grad agree?": f"zero-mean ρ={rho_zm:.3f}, mean-grad ρ={rho_mg:.3f}, zero-grad ρ={rho_zg:.3f}.",
            "6. Stable on SELECTION?": f"Overall Spearman TRAIN-SEL={stab:.3f}, top-8 overlap {top8_ov}/8.",
            "7. Strong interactions?": f"Top-16 group vs sum of individual SEL I_mean: group={gI}, sum={ind_sum:.5g} → {interact}.",
            "8. Channel-weighted distillation justified?": None,
        }
        unstable = stab < 0.6 or top8_ov < 4
        if unstable:
            rec = "No channel weighting because rankings are too unstable on SELECTION."
            answers["8. Channel-weighted distillation justified?"] = "No — TRAIN/SELECTION rank agreement is weak."
        elif concentrated and rho_ss < 0.5:
            rec = "STOP-specific sensitivity-weighted latent distillation (SAFE and STOP load different channels)."
            answers["8. Channel-weighted distillation justified?"] = "Possibly STOP-specific weights, not a single global map."
        elif concentrated:
            rec = "Global sensitivity-weighted latent distillation, with uniform distillation kept as the control."
            answers["8. Channel-weighted distillation justified?"] = "A global weight map is reasonable; still keep uniform L1 as baseline."
        else:
            rec = "Uniform latent distillation. Sensitivity is too distributed to justify channel weights."
            answers["8. Channel-weighted distillation justified?"] = "Not really — mass is spread across many channels."
        if interact.startswith("synergistic"):
            rec = "A more structured channel-group distillation approach. " + rec
        payload["answers"] = answers
        payload["recommendation"] = rec
        return payload

    def run_full(self):
        self.log("baseline train")
        Ft, Htt, gtt, preft, sidt, cidt, ambt = self.windows(self.train_idx)
        Ht0 = self.decode_np(Ft)
        self.log("baseline sel")
        Fs, Hts, gts, prefs, sids, cids, ambs = self.windows(self.sel_idx)
        Hs0 = self.decode_np(Fs)
        tr_base = self.baseline_metrics(Ht0, Htt, gtt, preft)
        se_base = self.baseline_metrics(Hs0, Hts, gts, prefs)
        (OUT / "baseline.json").write_text(json.dumps({"train": tr_base, "selection": se_base}, indent=2), encoding="utf-8")
        self.log(f"baseline TRAIN STOP={tr_base['stop_recall']:.3f} SAFE={tr_base['safe_recall']:.3f} SEL STOP={se_base['stop_recall']:.3f}")

        mu = Ft.mean(axis=(0, 2, 3))
        np.save(OUT / "train_channel_mean.npy", mu)

        gtct = peak_class(gtt.reshape(len(gtt), -1).max(1))
        part = peak_class(Ht0.reshape(len(Ht0), -1).max(1))
        gtcs = peak_class(gts.reshape(len(gts), -1).max(1))
        pars = peak_class(Hs0.reshape(len(Hs0), -1).max(1))
        mt = subset_masks(gtct, part, preft)
        ms = subset_masks(gtcs, pars, prefs)

        def pack_exp(name, F, H0, masks, mode):
            self.log(f"exp {name} {mode} n={len(F)}")
            abs_dh, abs_dp, sgn_dp, max_dh, y1 = self.perturb_all(F, H0, mode, mu)
            y0 = peak_class(H0.reshape(len(H0), -1).max(1))
            agg = aggregate(abs_dh, abs_dp, sgn_dp, max_dh, y0, y1, masks)
            np.savez_compressed(
                OUT / f"{name}_{mode}_raw.npz",
                abs_dh=abs_dh,
                abs_dp=abs_dp,
                sgn_dp=sgn_dp,
                y1=y1,
            )
            return agg, abs_dh, abs_dp

        z_tr, z_tr_dh, z_tr_dp = pack_exp("train", Ft, Ht0, mt, "zero")
        m_tr, m_tr_dh, m_tr_dp = pack_exp("train", Ft, Ht0, mt, "mean")
        z_se, z_se_dh, z_se_dp = pack_exp("sel", Fs, Hs0, ms, "zero")
        m_se, m_se_dh, m_se_dp = pack_exp("sel", Fs, Hs0, ms, "mean")

        self.log("grads train")
        g_tr = self.grads(Ft, Ht0, mt)
        self.log("grads sel")
        g_se = self.grads(Fs, Hs0, ms)

        def ser_agg(a):
            o = {}
            for k, v in a.items():
                o[k] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
            return o

        train_pack = {"zero": ser_agg(z_tr), "mean": ser_agg(m_tr), "grad": {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()} for k, v in g_tr.items()}}
        sel_pack = {"zero": ser_agg(z_se), "mean": ser_agg(m_se), "grad": {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()} for k, v in g_se.items()}}

        def order_of(arr):
            return np.argsort(-np.asarray(arr)).tolist()

        rankings = {
            "overall": {"order": order_of(m_tr["all"]["I"]), "score": "I_mean_all"},
            "stop": {"order": order_of(m_tr["gt_stop"]["I"]), "score": "I_mean_gt_stop"},
            "safe": {"order": order_of(m_tr["gt_safe"]["I"]), "score": "I_mean_gt_safe"},
            "prefork_stop": {"order": order_of(m_tr["prefork_stop"]["I"]), "score": "I_mean_prefork_stop"},
            "parent_correct_stop": {"order": order_of(m_tr["parent_correct_stop"]["I"]), "score": "I_mean_parent_correct_stop"},
        }

        def overlap(a, b, k):
            return int(len(set(a[:k]) & set(b[:k])))

        stability = {}
        for key, sub in ("overall", "all"), ("stop", "gt_stop"), ("safe", "gt_safe"), ("prefork_stop", "prefork_stop"):
            a = np.array(m_tr[sub]["I"])
            b = np.array(m_se[sub]["I"])
            oa, ob = order_of(a), order_of(b)
            stability[key] = {
                "spearman": spearman(a, b),
                "top8": overlap(oa, ob, 8),
                "top16": overlap(oa, ob, 16),
                "top32": overlap(oa, ob, 32),
            }
        boot = {
            "mean_all": self.bootstrap_ci(m_tr_dh, sidt, mt),
            "mean_stop_peak": self.bootstrap_ci(m_tr_dp, sidt, mt),
        }
        (OUT / "bootstrap_ci.json").write_text(json.dumps({k: v for k, v in boot.items()}, indent=2, default=lambda x: x), encoding="utf-8")
        # json bootstrap already has lo/hi lists
        stability["bootstrap"] = "bootstrap_ci.json"

        rng = np.random.default_rng(SEED)
        ord_all = rankings["overall"]["order"]
        ord_st = rankings["stop"]["order"]
        ord_pf = rankings["prefork_stop"]["order"]

        def topbot(order, k, flip=False):
            return list(order[:k] if not flip else order[-k:])

        groups = {
            "top8_overall": [topbot(ord_all, 8)],
            "top16_overall": [topbot(ord_all, 16)],
            "top32_overall": [topbot(ord_all, 32)],
            "bottom8_overall": [topbot(ord_all, 8, True)],
            "bottom16_overall": [topbot(ord_all, 16, True)],
            "bottom32_overall": [topbot(ord_all, 32, True)],
            "top8_stop": [topbot(ord_st, 8)],
            "top16_stop": [topbot(ord_st, 16)],
            "top32_stop": [topbot(ord_st, 32)],
            "top8_prefork_stop": [topbot(ord_pf, 8)],
            "top16_prefork_stop": [topbot(ord_pf, 16)],
            "top32_prefork_stop": [topbot(ord_pf, 32)],
            "random8": [rng.choice(C, 8, replace=False).tolist() for _ in range(10)],
            "random16": [rng.choice(C, 16, replace=False).tolist() for _ in range(10)],
            "random32": [rng.choice(C, 32, replace=False).tolist() for _ in range(10)],
        }
        self.log("group ablation on SELECTION")
        gres = self.group_ablate(Fs, Hs0, mu, groups, ms)

        conc = {}
        for name, arr in ("overall", m_tr["all"]["I"]), ("stop", m_tr["gt_stop"]["I"]), ("safe", m_tr["gt_safe"]["I"]), ("prefork_stop", m_tr["prefork_stop"]["I"]):
            a = np.asarray(arr)
            s = a.sum() + 1e-12
            sa = np.sort(a)
            conc[name] = {"top8": float(sa[-8:].sum() / s), "top16": float(sa[-16:].sum() / s), "top32": float(sa[-32:].sum() / s)}

        figdir = OUT / "figures"
        figdir.mkdir(exist_ok=True)
        barplot(figdir / "overall_I_mean.png", np.array(m_tr["all"]["I"]), "TRAIN overall I_mean")
        barplot(figdir / "stop_I_mean.png", np.array(m_tr["gt_stop"]["I"]), "TRAIN GT-STOP I_mean")
        barplot(figdir / "safe_I_mean.png", np.array(m_tr["gt_safe"]["I"]), "TRAIN GT-SAFE I_mean")
        barplot(figdir / "prefork_stop_I_mean.png", np.array(m_tr["prefork_stop"]["I"]), "TRAIN pre-fork STOP I_mean")

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(np.argsort(np.argsort(-np.array(m_tr["all"]["I"]))), np.argsort(np.argsort(-np.array(m_se["all"]["I"]))), s=12, alpha=0.7)
        ax.set_xlabel("TRAIN rank (I_mean)")
        ax.set_ylabel("SELECTION rank (I_mean)")
        ax.set_title("TRAIN vs SELECTION channel ranks")
        ax.plot([0, 127], [0, 127], "k--", lw=0.8)
        fig.tight_layout()
        fig.savefig(figdir / "train_vs_sel_ranks.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        methods = [
            np.argsort(-np.array(m_tr["all"]["I"])),
            np.argsort(-np.array(z_tr["all"]["I"])),
            np.argsort(-np.array(g_tr["all"]["I_grad"])),
        ]
        ax.scatter(methods[0], np.searchsorted(methods[1], np.arange(C)), s=10, label="mean vs zero rank")
        ax.legend()
        ax.set_title("mean-replace rank vs zero-ablate rank")
        fig.tight_layout()
        fig.savefig(figdir / "method_rank_compare.png", dpi=120)
        plt.close(fig)

        sizes, topI, rndI, botI = [8, 16, 32], [], [], []
        for k, lab in (8, "8"), (16, "16"), (32, "32"):
            topI.append(gres[f"top{lab}_overall"][0]["all"]["I"])
            botI.append(gres[f"bottom{lab}_overall"][0]["all"]["I"])
            rndI.append(float(np.mean([r["all"]["I"] for r in gres[f"random{lab}"]])))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sizes, topI, "o-", label="top-k overall")
        ax.plot(sizes, botI, "s-", label="bottom-k")
        ax.plot(sizes, rndI, "d-", label="random-k mean")
        ax.set_xlabel("group size")
        ax.set_ylabel("SEL mean |dH|")
        ax.legend()
        ax.set_title("Group ablation vs size")
        fig.tight_layout()
        fig.savefig(figdir / "group_vs_size.png", dpi=120)
        plt.close(fig)

        # examples on selection
        def pick(mask, title):
            ii = np.where(mask)[0]
            if len(ii) == 0:
                return None
            # typical: median peak
            pk = Hs0.reshape(len(Hs0), -1).max(1)
            j = ii[np.argmin(np.abs(pk[ii] - np.median(pk[ii])))]
            return int(j)

        ex = {
            "gt_safe": pick(ms["gt_safe"], "safe"),
            "parent_correct_stop": pick(ms["parent_correct_stop"], "stop"),
            "prefork_correct_stop": pick(ms["parent_correct_stop"] & prefs, "pf"),
            "parent_miss_stop": pick((gtcs == 2) & (pars != 2), "miss"),
        }
        topc = ord_all[:8]
        for name, j in ex.items():
            if j is None:
                continue
            G = Fs[j : j + 1].copy()
            G[:, topc] = mu[topc][None, :, None, None]
            Habl = self.decode_np(G)[0]
            gtlab = ["SAFE", "THREAT", "STOP"][int(gtcs[j])]
            plab = ["SAFE", "THREAT", "STOP"][int(pars[j])]
            heatmap_grid(
                figdir / f"example_{name}.png",
                [Hs0[j], Habl, gts[j]],
                [f"parent {plab} peak={Hs0[j].max():.2f}", f"top8 mean-replace", f"GT {gtlab}"],
            )

        mem = None
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1e9
        payload = {
            "train_baseline": tr_base,
            "sel_baseline": se_base,
            "train": train_pack,
            "sel": sel_pack,
            "rankings": rankings,
            "stability": stability,
            "groups": gres,
            "concentration": conc,
            "runtime_s": time.perf_counter() - self.t0,
            "peak_gpu_mem_gb": mem,
            "test_accessed": False,
            "decoder_hash": self.hash,
            "checkpoint": str(CKPT),
        }
        payload = self.interpret(payload)
        (OUT / "results.json").write_text(json.dumps(payload, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o), encoding="utf-8")
        self.write_report(payload)
        (OUT / ".success").write_text("ok\n")
        self.log("DONE")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        device = pick_cuda(str(_args.device))
        run = Run(device, _args.batch, _args.pack)
        run.setup()
        run.smoke()
        if _args.smoke:
            return
        run.run_full()
    except Exception:
        (OUT / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
