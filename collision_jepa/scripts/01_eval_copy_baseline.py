"""The mandatory copy baseline: predict H(t+tau) = H(t).

This is the number every learned model must beat, especially on episodes that
contain NEAR_MISS / CRITICAL_THREAT events. Pure numpy, so it runs before torch is
installed.

Run:  python scripts/01_eval_copy_baseline.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa.config import Config  # noqa: E402
from collision_jepa.data.splits import load_split  # noqa: E402
from collision_jepa.metrics import MetricAccumulator, format_summary  # noqa: E402


def evaluate(cfg: Config, episodes: list[str], has_rare: dict) -> dict:
    cache_dir = Path(cfg.get("data.cache_dir"))
    tau = int(cfg.get("horizon.tau_frames"))
    offsets = cfg.get("student.frame_offsets")
    stride = 3
    recall_threshold = float(cfg.get("heatmap.recall_threshold", 0.5))
    warn_cfg = cfg.get("warning")

    overall = MetricAccumulator(recall_threshold, warn_cfg)
    rare = MetricAccumulator(recall_threshold, warn_cfg)

    t_lo = -min(offsets)
    for ep in episodes:
        hm_path = cache_dir / ep / "heatmaps.npy"
        if not hm_path.exists():
            continue
        heatmaps = np.load(hm_path)
        n_frames = heatmaps.shape[0]
        t_hi = n_frames - 1 - tau
        for t in range(t_lo, t_hi + 1, stride):
            pred = heatmaps[t]
            true = heatmaps[t + tau]
            overall.update(pred, true)
            if has_rare.get(ep, False):
                rare.update(pred, true)

    return {"overall": overall.summary(), "rare": rare.summary()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    cfg = Config.load(args.config)
    split = load_split(cfg.get("data.split_file"))
    has_rare = split.get("has_rare", {})

    print("=== Copy baseline  H(t+tau) = H(t) ===")
    res = evaluate(cfg, split["val"], has_rare)
    print(format_summary("copy [val all]", res["overall"]))
    if res["rare"]["n"]:
        print(format_summary("copy [val rare]", res["rare"]))
    print(
        "\nInterpretation: a learned model must beat these, especially HT-recall and "
        "STOP-F1 on the 'rare' row (episodes with NEAR_MISS/CRITICAL)."
    )


if __name__ == "__main__":
    main()
