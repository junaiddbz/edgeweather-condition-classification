"""
src/deduplicate.py
==================
Perceptual hash-based image deduplication engine.

Academic motivation
-------------------
Using random ``train_test_split`` on image datasets that contain burst
sequences, video frame derivatives, or mirror-flipped duplicates leads to
*data leakage* — the model sees near-identical images during training and
"test" evaluation, inflating reported accuracy.

This module prevents leakage by:
1. Computing 64-bit perceptual hashes (pHash + dHash) for every image.
2. Building a Hamming-distance graph of near-duplicate pairs.
3. Pruning duplicates so that *no two images within Hamming distance ≤ threshold*
   appear in different splits.
4. Generating clean, stratified CSV manifests for train / val / test splits.

Usage
-----
    python -m src.deduplicate \\
        --raw_dir data/raw \\
        --out_dir data/processed \\
        --threshold 4

References
----------
- Zauner (2010). Implementation and Benchmarking of Perceptual Image Hash Functions.
- imagehash library: https://github.com/JohannesBuchner/imagehash
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from src.utils import ensure_dir, get_logger, set_global_seed

logger = get_logger(__name__)


# ── Hash computation ─────────────────────────────────────────────────────────


def compute_image_hash(
    image_path: Path,
    hash_size: int = 8,
) -> tuple[imagehash.ImageHash, imagehash.ImageHash]:
    """Compute pHash and dHash for a single image.

    Both hashes are computed to reduce false negatives: pHash (DCT-based)
    excels at detecting photometric duplicates; dHash (gradient-based) is
    more sensitive to structural differences.

    Args:
        image_path: Path to the image file.
        hash_size: Hash resolution. ``8`` produces a 64-bit hash (8×8 grid).

    Returns:
        Tuple of ``(phash, dhash)`` for the image.

    Raises:
        OSError: If the image cannot be opened.
    """
    with Image.open(image_path).convert("RGB") as img:
        phash = imagehash.phash(img, hash_size=hash_size)
        dhash = imagehash.dhash(img, hash_size=hash_size)
    return phash, dhash


def hash_dataset(
    raw_dir: Path,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> pd.DataFrame:
    """Compute perceptual hashes for all images in ``raw_dir``.

    Expects the following flat directory structure::

        raw_dir/
        ├── cloudy/
        │   ├── cloudy_001.jpg
        │   └── ...
        ├── rain/
        └── ...

    Args:
        raw_dir: Root directory containing one subdirectory per class.
        extensions: Accepted image file extensions (case-insensitive).

    Returns:
        DataFrame with columns:
        ``path``, ``class_name``, ``phash``, ``dhash``.
    """
    records: list[dict] = []
    image_paths = [p for p in raw_dir.rglob("*") if p.suffix.lower() in extensions]

    logger.info(f"Computing hashes for {len(image_paths):,} images…")

    for img_path in tqdm(image_paths, desc="Hashing", unit="img"):
        class_name = img_path.parent.name
        try:
            phash, dhash = compute_image_hash(img_path)
            records.append(
                {
                    "path": str(img_path),
                    "class_name": class_name,
                    "phash": str(phash),
                    "dhash": str(dhash),
                }
            )
        except (OSError, Exception) as exc:
            logger.warning(f"Skipping {img_path.name}: {exc}")

    df = pd.DataFrame(records)
    logger.info(f"Hashed {len(df):,} images across {df['class_name'].nunique()} classes.")
    return df


# ── Duplicate detection ───────────────────────────────────────────────────────


def find_duplicate_groups(
    df: pd.DataFrame,
    threshold: int = 4,
) -> list[list[int]]:
    """Find groups of near-duplicate images by Hamming distance.

    Two images are considered near-duplicates if *both* their pHash AND
    dHash Hamming distances are ≤ ``threshold``.  Using both hashes reduces
    the false-positive rate.

    Args:
        df: DataFrame output from :func:`hash_dataset` with ``phash`` and
            ``dhash`` columns.
        threshold: Maximum Hamming distance to consider a duplicate.
            ``4`` (out of 64 bits) is recommended for photographic data.

    Returns:
        List of duplicate groups; each group is a list of row indices in ``df``.
    """
    phashes = [imagehash.hex_to_hash(h) for h in df["phash"]]
    dhashes = [imagehash.hex_to_hash(h) for h in df["dhash"]]

    n = len(df)
    # Union-Find (disjoint set) for grouping
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    logger.info(f"Finding near-duplicate pairs (threshold={threshold})…")
    duplicate_pairs = 0

    for i in tqdm(range(n), desc="Dedup scan", unit="img"):
        for j in range(i + 1, n):
            if (phashes[i] - phashes[j]) <= threshold and (dhashes[i] - dhashes[j]) <= threshold:
                union(i, j)
                duplicate_pairs += 1

    # Collect groups
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        groups[find(idx)].append(idx)

    dup_groups = [g for g in groups.values() if len(g) > 1]
    logger.info(f"Found {duplicate_pairs:,} near-duplicate pairs in {len(dup_groups):,} groups.")
    return dup_groups


def prune_duplicates(df: pd.DataFrame, dup_groups: list[list[int]]) -> pd.DataFrame:
    """Remove all but one representative from each duplicate group.

    Keeps the first image in each group (sorted by path for determinism)
    and drops the rest.

    Args:
        df: Full dataset DataFrame.
        dup_groups: Duplicate groups from :func:`find_duplicate_groups`.

    Returns:
        Deduplicated DataFrame.
    """
    indices_to_drop: set[int] = set()
    for group in dup_groups:
        # Sort group by path for determinism, keep first, drop rest
        sorted_group = sorted(group, key=lambda i: df.iloc[i]["path"])
        indices_to_drop.update(sorted_group[1:])

    pruned = df.drop(index=list(indices_to_drop)).reset_index(drop=True)
    logger.info(f"Pruned {len(indices_to_drop):,} duplicates. Remaining: {len(pruned):,} images.")
    return pruned


# ── Stratified splitting ──────────────────────────────────────────────────────


def stratified_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train / val / test splits.

    Splits are performed sequentially:
    1. First split: train vs. (val + test)
    2. Second split: val vs. test

    Class proportions are preserved within ±1 sample in each split.

    Args:
        df: Deduplicated dataset DataFrame with a ``class_name`` column.
        train_frac: Fraction of data for training.
        val_frac: Fraction of data for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(train_df, val_df, test_df)``.
    """
    test_frac = round(1.0 - train_frac - val_frac, 10)
    relative_val_frac = val_frac / (val_frac + test_frac)

    # Split 1: train vs. (val + test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_frac), random_state=seed)
    train_idx, temp_idx = next(sss1.split(df, df["class_name"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # Split 2: val vs. test from the held-out set
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - relative_val_frac), random_state=seed)
    val_idx, test_idx = next(sss2.split(temp_df, temp_df["class_name"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    logger.info(
        f"Split sizes — train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}"
    )
    _log_class_distribution(train_df, "Train")
    _log_class_distribution(val_df, "Val")
    _log_class_distribution(test_df, "Test")
    return train_df, val_df, test_df


def _log_class_distribution(df: pd.DataFrame, split_name: str) -> None:
    """Log per-class sample counts for a given split."""
    counts = df["class_name"].value_counts().sort_index()
    dist_str = " | ".join(f"{cls}: {n}" for cls, n in counts.items())
    logger.info(f"  {split_name}: {dist_str}")


# ── CSV export ────────────────────────────────────────────────────────────────


def save_split_csvs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Persist split manifests as CSV files.

    Outputs ``train.csv``, ``val.csv``, ``test.csv`` in ``out_dir``.
    Each CSV has columns ``path`` and ``class_name``.

    Args:
        train_df: Training split DataFrame.
        val_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        out_dir: Directory where CSVs will be written.
    """
    ensure_dir(out_dir)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        csv_path = out_dir / f"{name}.csv"
        df[["path", "class_name"]].to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        logger.info(f"Saved {csv_path} ({len(df):,} rows)")


# ── Hamming distance statistics ───────────────────────────────────────────────


def compute_hamming_stats(df: pd.DataFrame, sample_n: int = 5000) -> dict[str, float]:
    """Compute summary statistics of pairwise pHash Hamming distances.

    Used in EDA notebook to plot the hash distance distribution and justify
    the chosen deduplication threshold.

    Args:
        df: DataFrame with ``phash`` column.
        sample_n: Max images to sample for the pairwise comparison
            (full O(n²) comparison is too slow for large datasets).

    Returns:
        Dict with keys ``min``, ``p5``, ``p25``, ``median``, ``p75``,
        ``p95``, ``max`` of the pairwise Hamming distance distribution.
    """
    sample = df.sample(min(sample_n, len(df)), random_state=42)
    hashes = [imagehash.hex_to_hash(h) for h in sample["phash"]]
    distances: list[int] = []

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            distances.append(hashes[i] - hashes[j])

    arr = np.array(distances)
    return {
        "min": int(arr.min()),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(arr.max()),
    }


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EdgeWeather: Perceptual hash deduplication & stratified splitting."
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory with one subdirectory per class.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for train/val/test CSVs.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=4,
        help="Hamming distance threshold for near-duplicate detection.",
    )
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    logger.info("=" * 60)
    logger.info("EdgeWeather Deduplication Pipeline")
    logger.info("=" * 60)

    # Step 1: Hash all images
    df = hash_dataset(args.raw_dir)

    # Step 2: Find and prune duplicates
    dup_groups = find_duplicate_groups(df, threshold=args.threshold)
    df_clean = prune_duplicates(df, dup_groups)

    # Step 3: Stratified split
    train_df, val_df, test_df = stratified_split(
        df_clean,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    # Step 4: Save CSVs
    save_split_csvs(train_df, val_df, test_df, args.out_dir)

    logger.info("[bold green]✓ Phase 1 deduplication complete.[/bold green]")


def deduplicate_dataset(
    raw_dir: str | Path,
    out_dir: str | Path,
    threshold: int = 4,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute the full end-to-end deduplication and splitting pipeline.

    Args:
        raw_dir: Directory containing per-class image subdirectories.
        out_dir: Output directory where train.csv, val.csv, and test.csv are saved.
        threshold: Hamming distance threshold for near-duplicate pruning.
        train_frac: Fraction of clean images allocated to training.
        val_frac: Fraction of clean images allocated to validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    set_global_seed(seed)
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)

    df = hash_dataset(raw_path)
    dup_groups = find_duplicate_groups(df, threshold=threshold)
    df_clean = prune_duplicates(df, dup_groups)
    train_df, val_df, test_df = stratified_split(
        df_clean,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )
    save_split_csvs(train_df, val_df, test_df, out_path)
    return train_df, val_df, test_df


# Backward-compatible aliases
compute_image_hashes = compute_image_hash
find_duplicate_clusters = find_duplicate_groups
build_leak_free_splits = stratified_split


if __name__ == "__main__":
    main()
