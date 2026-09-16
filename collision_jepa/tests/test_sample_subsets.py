"""Equal-from-folder sampling for datasets.zip.

Run:  python tests/test_sample_subsets.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa.data.unzip import sample_equal_from_subsets, sample_with_k5_bias  # noqa: E402


def test_round_robin_fills_small_folder_then_splits_rest():
    by = {
        "side40_s11_n50": [f"episode_{i:04d}_s" for i in range(50)],
        "mixed250_s67_n250": [f"episode_{i:04d}_m" for i in range(250)],
        "pack_s67_n1000": [f"episode_{i:04d}_p" for i in range(1000)],
    }
    chosen = sample_equal_from_subsets(by, total=500, seed=1337)
    counts = {k: len(v) for k, v in chosen.items()}
    assert sum(counts.values()) == 500, counts
    assert counts["side40_s11_n50"] == 50, counts
    assert counts["mixed250_s67_n250"] == 225, counts
    assert counts["pack_s67_n1000"] == 225, counts
    # No duplicates, all from the source lists.
    for k, eps in chosen.items():
        assert len(eps) == len(set(eps))
        assert set(eps) <= set(by[k])


def test_total_larger_than_archive_takes_everything():
    by = {"a": ["episode_0000_a"], "b": ["episode_0000_b"]}
    chosen = sample_equal_from_subsets(by, total=500, seed=1)
    assert sum(len(v) for v in chosen.values()) == 2


def test_k5_bias_takes_most_from_pack():
    by = {
        "side40_s11_n50": [f"episode_{i:04d}_s" for i in range(50)],
        "mixed250_s67_n250": [f"episode_{i:04d}_m" for i in range(250)],
        "pack_20260911_043022_s67_n1000": [f"episode_{i:04d}_p" for i in range(1000)],
    }
    chosen = sample_with_k5_bias(
        by,
        total=500,
        seed=1337,
        k5_subsets=["pack_20260911_043022_s67_n1000"],
        k5_fraction=0.8,
    )
    counts = {k: len(v) for k, v in chosen.items()}
    assert sum(counts.values()) == 500, counts
    assert counts["pack_20260911_043022_s67_n1000"] == 400, counts
    assert counts["side40_s11_n50"] + counts["mixed250_s67_n250"] == 100, counts


if __name__ == "__main__":
    test_round_robin_fills_small_folder_then_splits_rest()
    test_total_larger_than_archive_takes_everything()
    test_k5_bias_takes_most_from_pack()
    print("ok")
