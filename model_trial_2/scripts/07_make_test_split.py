"""Carve a balanced test split from the current val set. Train is left unchanged."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvjepa.config import Config  # noqa: E402
from cvjepa.data import carve_test_from_holdout, load_json, save_json  # noqa: E402


def _counts(keys: list[str], split: dict) -> dict:
    fam = Counter(split["families"][k] for k in keys)
    src = Counter(split["sources"][k] for k in keys)
    rare = sum(1 for k in keys if split["has_rare"].get(k))
    return {"n": len(keys), "rare": rare, "family": dict(fam), "source": dict(src)}


def _print(title: str, c: dict) -> None:
    print(f"{title}: n={c['n']}  rare={c['rare']} ({100 * c['rare'] / max(c['n'], 1):.0f}%)")
    print(f"  family {c['family']}")
    print(f"  source {c['source']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1354)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    path = Path(cfg.get("data.split_file"))
    split = load_json(path)
    if split.get("test"):
        print(f"[split] test already present ({len(split['test'])} episodes). leave {path}")
        _print("train", _counts(split["train"], split))
        _print("val", _counts(split["val"], split))
        _print("test", _counts(split["test"], split))
        return

    train = list(split["train"])
    old_val = list(split["val"])
    overlap = set(train) & set(old_val)
    if overlap:
        raise SystemExit(f"train/val overlap: {len(overlap)}")

    val, test = carve_test_from_holdout(
        old_val,
        split["families"],
        split["sources"],
        split["has_rare"],
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    if set(val) & set(test):
        raise SystemExit("val/test overlap")
    if set(val) | set(test) != set(old_val):
        raise SystemExit("val+test does not recover previous val")
    if set(test) & set(train):
        raise SystemExit("test leaked into train")

    bak = path.with_name("split_train_val_only.json")
    shutil.copy2(path, bak)
    split["val_legacy"] = old_val
    split["val"] = val
    split["test"] = test
    split["test_fraction_of_legacy_val"] = args.test_fraction
    split["test_seed"] = args.seed
    save_json(split, path)

    print(f"[split] wrote {path}")
    print(f"[split] backup {bak} (original 80/20 val, {len(old_val)} episodes)")
    _print("train (unchanged)", _counts(train, split))
    _print("val", _counts(val, split))
    _print("test", _counts(test, split))


if __name__ == "__main__":
    main()
