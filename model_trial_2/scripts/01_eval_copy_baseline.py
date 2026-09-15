"""Copy baseline: H(t+1s) = H(t). Wearable nuisance/miss on the privileged current heatmap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvjepa.config import Config  # noqa: E402
from cvjepa.data import load_json  # noqa: E402
from cvjepa.engine import evaluate_wearable, set_seed  # noqa: E402
from cvjepa.metrics import format_wearable  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    cfg = Config.load(args.config)
    set_seed(int(cfg.get("seed", 1337)))
    split = load_json(cfg.get("data.split_file"))
    has_rare = split.get("has_rare", {})
    for name, eps in (("val", split["val"]), ("test", split.get("test") or [])):
        if not eps:
            continue
        wear = evaluate_wearable(cfg, eps, has_rare, "cpu", copy_baseline=True)
        print(f"[copy] {name}  {format_wearable('wear', wear['overall'])}")
        print(f"[copy] {name}-rare  {format_wearable('wear', wear['rare'])}")


if __name__ == "__main__":
    main()
