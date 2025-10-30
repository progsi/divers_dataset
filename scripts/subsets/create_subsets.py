#!/usr/bin/env python3
import argparse
import os
import json
import torch
import random
import numpy as np
from copy import deepcopy
from tqdm import tqdm


def load_dataset(path):
    """Load dataset from JSON or Torch file"""
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
    elif path.endswith(".pt") or path.endswith(".pth"):
        data = torch.load(path, weights_only=False)
        # convert to dict if it's a tensor dataset
        if isinstance(data, torch.Tensor):
            raise ValueError("Torch Tensor dataset not supported directly")
        data = {k: v for k, v in data.items()}
    else:
        raise ValueError("Unsupported dataset format")
    return data

def drop_dvi_items(dataset, min_items_per_clique=2):
    """
    Remove all items marked as 'dvi' and drop cliques with fewer than min_items_per_clique items.

    Args:
        dataset: dict with 'info' and 'split'
        min_items_per_clique: minimum items per clique to keep

    Returns:
        new_dataset dict with same format as original
    """
    info = deepcopy(dataset["info"])
    splits = dataset.get("split", {})

    # 1. Build set of non-dvi items
    non_dvi_items = {k for k, v in info.items() if v["youtube_id"] in v["version"]}

    # 2. Filter splits
    new_splits = {}
    for split_name, split_dict in splits.items():
        new_splits[split_name] = {}
        for clique, item_keys in split_dict.items():
            # Keep only non-DVI items
            filtered_keys = [k for k in item_keys if k in non_dvi_items]
            if len(filtered_keys) >= min_items_per_clique:
                new_splits[split_name][clique] = filtered_keys

    # 3. Build new info dict from remaining splits
    items_in_splits = set()
    for split_dict in new_splits.values():
        for keys in split_dict.values():
            items_in_splits.update(keys)

    new_info = {k: info[k] for k in items_in_splits}

    return {"info": new_info, "split": new_splits}

def dedup_by_duration(dataset, T=1.0, min_items_per_clique=2, delete_dvi=False):
    """
    Remove per-clique duplicate durations within tolerance T (seconds),
    ensure minimum clique size, and optionally prevent deletion of DVI items.

    Args:
        dataset: dict with 'info' and 'split'
        T: float, duration difference tolerance to consider duplicates
        min_items_per_clique: minimum items per clique to keep
        delete_dvi: if False, DVI items are never removed; if True, they can be removed like others

    Returns:
        new_dataset dict with same format as original
    """
    info = deepcopy(dataset["info"])
    splits = dataset.get("split", {})

    new_info = {}
    new_splits = {}

    for split_name, split_dict in splits.items():
        new_splits[split_name] = {}
        for clique, item_keys in split_dict.items():
            # Gather items and durations
            items = [(k, info[k]["length"], not info[k]["youtube_id"] in info[k]["version"]) for k in item_keys if k in info]
            # Sort by duration
            items.sort(key=lambda x: x[1])
            kept = []

            i = 0
            while i < len(items):
                # Start new cluster
                cluster = [items[i]]
                j = i + 1
                while j < len(items) and abs(items[j][1] - items[i][1]) <= T:
                    cluster.append(items[j])
                    j += 1

                # Pick one from cluster
                dvi_items = [k for k, _, d in cluster if d]

                if dvi_items:
                    # Always prioritize DVI
                    kept.append(dvi_items[0])
                else:
                    # No DVI in cluster: pick first non-dvi
                    kept.append(cluster[0][0])

                i = j

            # If delete_dvi is False, add all remaining DVI items from clique
            if not delete_dvi:
                for k, _, d in items:
                    if d and k not in kept:
                        kept.append(k)

            # Enforce minimum clique size
            if len(kept) >= min_items_per_clique:
                new_splits[split_name][clique] = kept
                for k in kept:
                    new_info[k] = info[k]
            # else drop clique entirely

    return {"info": new_info, "split": new_splits}


def proportional_downsample(dataset, fraction=0.5, min_items_per_class=2, seed=None):
    """
    Downsample items per clique proportionally
    Ensures min_items_per_class is kept per clique
    """
    rng = random.Random(seed)
    new_dataset = {"info": {}, "split": {}}

    info = deepcopy(dataset["info"])
    splits = dataset.get("split", {})

    for split_name, split_dict in splits.items():
        new_dataset["split"][split_name] = {}
        for clique, item_keys in split_dict.items():
            n = len(item_keys)
            n_keep = max(min_items_per_class, int(round(n * fraction)))
            n_keep = min(n_keep, n)  # cannot exceed available
            sampled_keys = rng.sample(item_keys, n_keep)
            new_dataset["split"][split_name][clique] = sampled_keys

    # Recompute info to include only items still in splits
    items_in_splits = set()
    for split_dict in new_dataset["split"].values():
        for keys in split_dict.values():
            items_in_splits.update(keys)

    new_dataset["info"] = {k: info[k] for k in items_in_splits}

    return new_dataset


def main():
    parser = argparse.ArgumentParser(description="Subset dataset script")
    parser.add_argument("--dataset", required=True, help="Path to JSON or Torch dataset")
    parser.add_argument("--fraction", type=float, default=0.5, help="Fraction for proportional downsampling")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset(args.dataset)

    print("Creating non-dvi-only subset...")
    subset_non_dvi = drop_dvi_items(dataset)
    torch.save(subset_non_dvi, os.path.join(args.out_dir, "yvi.pt"))
    
    print("Creating deduplicated-by-duration subset...")
    subset_dedup_by_duration = dedup_by_duration(dataset, T=1.0)
    torch.save(subset_dedup_by_duration, os.path.join(args.out_dir, "diverse-dd.pt"))
    
    print("Creating non-deduplicated-non-dvi-only subset...")
    subset_dedup_by_duration_non_dvi = drop_dvi_items(subset_dedup_by_duration)
    torch.save(subset_dedup_by_duration_non_dvi, os.path.join(args.out_dir, "yvi-dd.pt"))

    # print(f"Creating proportional downsample subset (fraction={args.fraction})...")
    # subset_downsample = proportional_downsample(dataset, fraction=args.fraction, min_items_per_class=2, seed=42)
    # sample_name = str(len(subset_downsample["info"]))[:3] + "k"
    # torch.save(subset_downsample, os.path.join(args.out_dir, f"diverse{sample_name}.pt"))

    # print(f"Creating proportional deduplicated downsample subset (fraction={args.fraction})...")
    # subset_downsample = proportional_downsample(dataset, fraction=args.fraction, min_items_per_class=2, seed=42)
    # sample_name = str(len(subset_downsample["info"]))[:3] + "k"
    # torch.save(subset_downsample, os.path.join(args.out_dir, f"diverse{sample_name}.pt"))
    
    print("Done!")

if __name__ == "__main__":
    main()
