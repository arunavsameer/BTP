"""Copy baseline: H(t+tau) = H(t). The number the student must beat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rs_jepa.config import Config  # noqa: E402
from rs_jepa.data import load_json  # noqa: E402
from rs_jepa.metrics import MetricAccumulator, format_summary  # noqa: E402


def evaluate(cfg: Config, episodes: list[str], has_rare: dict) -> dict:
    cache_dir = Path(cfg.get("data.cache_dir"))
    tau = int(cfg.get("horizon.tau_frames"))
    offsets = cfg.get("student.frame_offsets")
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
        t_hi = heatmaps.shape[0] - 1 - tau
        for t in range(t_lo, t_hi + 1, 3):
            overall.update(heatmaps[t], heatmaps[t + tau])
            if has_rare.get(ep, False):
                rare.update(heatmaps[t], heatmaps[t + tau])
    return {"overall": overall.summary(), "rare": rare.summary()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    cfg = Config.load(args.config)
    split = load_json(cfg.get("data.split_file"))
    has_rare = split.get("has_rare", {})
    print("=== Copy baseline  H(t+tau) = H(t) ===")
    for name in ("val", "test"):
        episodes = split.get(name) or []
        if not episodes:
            continue
        res = evaluate(cfg, episodes, has_rare)
        print(format_summary(f"copy [{name} all]", res["overall"]))
        if res["rare"]["n"]:
            print(format_summary(f"copy [{name} rare]", res["rare"]))


if __name__ == "__main__":
    main()
