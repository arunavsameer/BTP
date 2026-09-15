"""Episode catalog, family labels, stratified split.

Episode directory names collide across the three dataset packs, so every episode
is identified by ``source/episode_dir`` (e.g. ``mixed/episode_0000_jaywalker_turn_toward``).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

_FAMILY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("vehicle", ("car", "cyclist", "scooter", "swerve", "cube", "truck", "vehicle")),
    ("pedestrian", ("jaywalker", "pedestrian", "child", "group_crossing", "oncoming")),
    ("static", ("pothole", "parked", "tree", "crater", "debris", "barricade")),
    ("empty", ("empty_street", "safe_walk", "periph_empty")),
]


def classify_family(scenarios: list[str]) -> str:
    joined = "__".join(scenarios).lower()
    for family, keywords in _FAMILY_KEYWORDS:
        if any(k in joined for k in keywords):
            return family
    return "other"


def read_episode_meta(episode_dir: Path) -> dict:
    meta_path = episode_dir / "episode.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    scenarios = meta.get("scenarios") or [meta.get("scenario", "")]
    meta["_family"] = classify_family(scenarios)
    meta["_dir"] = episode_dir.name
    hist = meta.get("label_histogram") or {}
    meta["_has_rare"] = bool(hist.get("NEAR_MISS", 0) or hist.get("CRITICAL_THREAT", 0))
    return meta


def find_episode_dirs(dataset_root: str | Path) -> list[Path]:
    dataset_root = Path(dataset_root)
    eps = [p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("episode_")]
    return sorted(eps, key=lambda p: p.name)


def _stratified_sample(episode_dirs: list[Path], n: int, seed: int) -> list[Path]:
    """Take ``n`` episodes, keeping family proportions as well as possible."""
    if n >= len(episode_dirs):
        return list(episode_dirs)
    by_family: dict[str, list[Path]] = {}
    for ep in episode_dirs:
        fam = read_episode_meta(ep)["_family"]
        by_family.setdefault(fam, []).append(ep)

    rng = random.Random(seed)
    chosen: list[Path] = []
    leftover: list[Path] = []
    total = len(episode_dirs)
    for fam, paths in sorted(by_family.items()):
        paths = sorted(paths, key=lambda p: p.name)
        rng.shuffle(paths)
        k = int(round(n * len(paths) / total))
        k = min(len(paths), max(1 if paths else 0, k))
        chosen.extend(paths[:k])
        leftover.extend(paths[k:])

    rng.shuffle(leftover)
    if len(chosen) > n:
        rng.shuffle(chosen)
        chosen = chosen[:n]
    elif len(chosen) < n:
        chosen.extend(leftover[: n - len(chosen)])
    chosen.sort(key=lambda p: p.name)
    return chosen


def build_catalog(sources: list[dict], seed: int = 1337) -> list[dict]:
    """Build the list of episodes used for this trial.

    Each source dict: ``{name, path, take}`` where ``take`` is ``all`` or an int.
    """
    catalog: list[dict] = []
    for src in sources:
        name = str(src["name"])
        root = Path(src["path"])
        take = src.get("take", "all")
        eps = find_episode_dirs(root)
        if take != "all" and take is not None:
            extra = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16) % 1000
            eps = _stratified_sample(eps, int(take), seed + extra)
        for ep in eps:
            meta = read_episode_meta(ep)
            catalog.append(
                {
                    "key": f"{name}/{ep.name}",
                    "source": name,
                    "path": str(ep),
                    "family": meta["_family"],
                    "has_rare": bool(meta["_has_rare"]),
                    "scenario": meta.get("scenario"),
                    "scenarios": meta.get("scenarios") or [meta.get("scenario", "")],
                }
            )
    catalog.sort(key=lambda r: r["key"])
    return catalog


def make_split(
    catalog: list[dict],
    val_fraction: float = 0.2,
    seed: int = 1337,
    test_fraction: float = 0.0,
) -> dict:
    by_family: dict[str, list[str]] = {}
    families: dict[str, str] = {}
    rare: dict[str, bool] = {}
    paths: dict[str, str] = {}
    sources: dict[str, str] = {}
    for row in catalog:
        key = row["key"]
        fam = row["family"]
        by_family.setdefault(fam, []).append(key)
        families[key] = fam
        rare[key] = bool(row["has_rare"])
        paths[key] = row["path"]
        sources[key] = row["source"]

    rng = random.Random(seed)
    train: list[str] = []
    holdout: list[str] = []
    hold_frac = float(val_fraction) + float(test_fraction)
    for fam, names in sorted(by_family.items()):
        names = sorted(names)
        rng.shuffle(names)
        n_hold = int(round(len(names) * hold_frac))
        if len(names) >= 2:
            n_hold = max(1, n_hold)
        n_hold = min(n_hold, len(names) - 1) if len(names) >= 2 else 0
        holdout.extend(names[:n_hold])
        train.extend(names[n_hold:])

    train.sort()
    if test_fraction > 0 and holdout:
        inner = float(test_fraction) / max(hold_frac, 1e-8)
        val, test = carve_test_from_holdout(holdout, families, sources, rare, inner, seed + 17)
    else:
        val, test = sorted(holdout), []

    out = {
        "train": train,
        "val": val,
        "test": test,
        "families": families,
        "has_rare": rare,
        "paths": paths,
        "sources": sources,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
        "n_catalog": len(catalog),
    }
    return out


def carve_test_from_holdout(
    names: list[str],
    families: dict[str, str],
    sources: dict[str, str],
    rare: dict[str, bool],
    test_fraction: float = 0.5,
    seed: int = 1354,
) -> tuple[list[str], list[str]]:
    """Split a holdout list into val/test, stratified by family × source × rare.

    Train is never touched. Groups of size 1 are assigned with probability
    ``test_fraction`` so tiny families (e.g. empty/side) still appear on both sides
    when possible.
    """
    rng = random.Random(seed)
    groups: dict[tuple, list[str]] = {}
    for key in names:
        g = (families[key], sources.get(key, ""), bool(rare.get(key, False)))
        groups.setdefault(g, []).append(key)

    val: list[str] = []
    test: list[str] = []
    for _g, items in sorted(groups.items()):
        items = sorted(items)
        rng.shuffle(items)
        n = len(items)
        if n == 1:
            (test if rng.random() < float(test_fraction) else val).append(items[0])
            continue
        n_test = int(round(n * float(test_fraction)))
        n_test = min(max(n_test, 1), n - 1)
        test.extend(items[:n_test])
        val.extend(items[n_test:])
    val.sort()
    test.sort()
    return val, test


def save_json(obj: dict | list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
