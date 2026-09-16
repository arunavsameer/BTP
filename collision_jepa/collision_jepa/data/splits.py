"""Episode-level stratified train/val split.

Frames from the same 5-second clip are almost identical, so splitting individual
frames would leak the validation set into training. We therefore split whole
episodes, stratified by a coarse scenario *family* so each split sees a similar
mix of pedestrians / vehicles / static hazards / safe walks.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Keyword -> family. Checked in priority order (most dynamic/dangerous first) so an
# episode that mixes tags gets the most safety-relevant label for stratification.
_FAMILY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("vehicle", ("car", "cyclist", "swerve_vehicle", "cube")),
    ("pedestrian", ("jaywalker", "pedestrian")),
    ("static", ("pothole", "parked_car")),
    ("empty", ("empty_street", "safe_walk")),
]


def classify_family(scenarios: list[str]) -> str:
    """Map an episode's scenario tags to a single coarse family label."""
    joined = "__".join(scenarios).lower()
    for family, keywords in _FAMILY_KEYWORDS:
        if any(k in joined for k in keywords):
            return family
    return "other"


def read_episode_meta(episode_dir: Path) -> dict:
    """Load episode.json and attach a derived family label."""
    meta_path = episode_dir / "episode.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    scenarios = meta.get("scenarios") or [meta.get("scenario", "")]
    meta["_family"] = classify_family(scenarios)
    meta["_dir"] = episode_dir.name
    hist = meta.get("label_histogram") or {}
    meta["_has_rare"] = bool(hist.get("NEAR_MISS", 0) or hist.get("CRITICAL_THREAT", 0))
    return meta


def make_split(
    episode_dirs: list[Path],
    val_fraction: float = 0.2,
    seed: int = 1337,
    names: list[str] | None = None,
) -> dict:
    """Return a dict with 'train' and 'val' lists of episode cache ids.

    Stratified by family: within each family we shuffle deterministically and send
    ``val_fraction`` of episodes to validation (at least one when a family is
    non-empty and large enough). ``names`` must align with ``episode_dirs`` when
    folder names collide across subsets (``subset__episode_dir``).
    """
    if names is not None and len(names) != len(episode_dirs):
        raise ValueError("names must match episode_dirs")
    by_family: dict[str, list[str]] = {}
    families: dict[str, str] = {}
    rare: dict[str, bool] = {}
    for i, ep in enumerate(episode_dirs):
        meta = read_episode_meta(ep)
        fam = meta["_family"]
        name = names[i] if names is not None else ep.name
        by_family.setdefault(fam, []).append(name)
        families[name] = fam
        rare[name] = meta["_has_rare"]

    rng = random.Random(seed)
    train: list[str] = []
    val: list[str] = []
    for fam, names in sorted(by_family.items()):
        names = sorted(names)
        rng.shuffle(names)
        n_val = int(round(len(names) * val_fraction))
        if len(names) >= 2:
            n_val = max(1, n_val)
        n_val = min(n_val, len(names) - 1) if len(names) >= 2 else 0
        val.extend(names[:n_val])
        train.extend(names[n_val:])

    train.sort()
    val.sort()
    return {
        "train": train,
        "val": val,
        "test": [],
        "families": families,
        "has_rare": rare,
        "val_fraction": val_fraction,
        "seed": seed,
    }


def make_count_split(
    episode_dirs: list[Path],
    n_val: int = 50,
    n_test: int = 50,
    seed: int = 1337,
    names: list[str] | None = None,
) -> dict:
    """Random episode-level split with fixed val/test counts (no frame leakage)."""
    if names is not None and len(names) != len(episode_dirs):
        raise ValueError("names must match episode_dirs")
    families: dict[str, str] = {}
    rare: dict[str, bool] = {}
    ids: list[str] = []
    for i, ep in enumerate(episode_dirs):
        meta = read_episode_meta(ep)
        name = names[i] if names is not None else ep.name
        ids.append(name)
        families[name] = meta["_family"]
        rare[name] = meta["_has_rare"]
    if len(ids) < n_val + n_test + 1:
        raise ValueError(f"need more than {n_val + n_test} episodes, got {len(ids)}")
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    test = sorted(shuffled[:n_test])
    val = sorted(shuffled[n_test : n_test + n_val])
    train = sorted(shuffled[n_test + n_val :])
    return {
        "train": train,
        "val": val,
        "test": test,
        "families": families,
        "has_rare": rare,
        "n_val": n_val,
        "n_test": n_test,
        "seed": seed,
    }


def save_split(split: dict, split_file: str | Path) -> None:
    split_file = Path(split_file)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)


def load_split(split_file: str | Path) -> dict:
    with open(split_file, "r", encoding="utf-8") as f:
        return json.load(f)
