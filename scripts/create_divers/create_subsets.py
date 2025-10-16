#!/usr/bin/env python3
import argparse
import os
import json
import torch
import random
from copy import deepcopy

def load_dataset(path):
    """Load dataset from JSON or Torch file"""
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
    elif path.endswith(".pt") or path.endswith(".pth"):
        data = torch.load(path)
        # convert to dict if it's a tensor dataset
        if isinstance(data, torch.Tensor):
            raise ValueError("Torch Tensor dataset not supported directly")
        data = {k: v for k, v in data.items()}
    else:
        raise ValueError("Unsupported dataset format")
    return data

def save_dataset(data, out_path):
    """Save dataset as JSON"""
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

def drop_dvi_items(dataset):
    """
    Remove all items which are 'dvi', drop classes with less than 2 items
    Returns new dataset dict
    """
    new_dataset = {"info": {}, "split": {}}

    # Deepcopy info to avoid mutating original
    info = deepcopy(dataset["info"])
    splits = dataset.get("split", {})

    # Build set of non-dvi items
    non_dvi_items = {k for k, v in info.items() if not v.get("dvi", False)}

    # Filter info
    new_info = {k: v for k, v in info.items() if k in non_dvi_items}

    # Adjust splits
    new_splits = {}
    for split_name, split_dict in splits.items():
        new_splits[split_name] = {}
        for clique, item_keys in split_dict.items():
            filtered_keys = [k for k in item_keys if k in non_dvi_items]
            if len(filtered_keys) >= 2:
                new_splits[split_name][clique] = filtered_keys

    # Recompute info to include only items still in splits
    items_in_splits = set()
    for split_dict in new_splits.values():
        for keys in split_dict.values():
            items_in_splits.update(keys)

    new_info = {k: new_info[k] for k in items_in_splits}

    new_dataset["info"] = new_info
    new_dataset["split"] = new_splits
    return new_dataset

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
    save_dataset(subset_non_dvi, os.path.join(args.out_dir, "subset_non_dvi.json"))

    print(f"Creating proportional downsample subset (fraction={args.fraction})...")
    subset_downsample = proportional_downsample(dataset, fraction=args.fraction, min_items_per_class=2, seed=42)
    save_dataset(subset_downsample, os.path.join(args.out_dir, "subset_downsample.json"))

    print("Done!")

if __name__ == "__main__":
    main()
