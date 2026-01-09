#!/usr/bin/env python3
import argparse
import os
import json
import torch
import random
from copy import deepcopy


# ----------------------------
# Utilities
# ----------------------------

def make_item_id(clique, version):
    return f"{clique}:{version}"


def load_dataset(path):
    """Load dataset from JSON or Torch file"""
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
    elif path.endswith(".pt") or path.endswith(".pth"):
        data = torch.load(path, weights_only=False)
        if isinstance(data, torch.Tensor):
            raise ValueError("Torch Tensor dataset not supported directly")
    else:
        raise ValueError("Unsupported dataset format")
    return data


# ----------------------------
# Core filters
# ----------------------------

def drop_dvi_items(dataset, min_items_per_clique=2):
    """
    Remove all DVI items and drop small cliques
    """
    info = deepcopy(dataset["info"])
    splits = dataset["split"]

    non_dvi_items = {
        item_id
        for item_id, v in info.items()
        if v["youtube_id"] in v["version"]
    }

    new_splits = {}
    for split, split_dict in splits.items():
        new_splits[split] = {}
        for clique, versions in split_dict.items():
            kept_versions = [
                v for v in versions
                if make_item_id(clique, v) in non_dvi_items
            ]
            if len(kept_versions) >= min_items_per_clique:
                new_splits[split][clique] = kept_versions

    kept_item_ids = {
        make_item_id(clique, v)
        for split_dict in new_splits.values()
        for clique, versions in split_dict.items()
        for v in versions
    }

    new_info = {k: info[k] for k in kept_item_ids}
    return {"info": new_info, "split": new_splits}

def proportional_downsample(dataset, fraction=0.5, min_items_per_class=2, seed=None):
    """
    Proportional downsampling per clique
    """
    rng = random.Random(seed)
    info = deepcopy(dataset["info"])
    splits = dataset["split"]

    new_splits = {}
    for split, split_dict in splits.items():
        new_splits[split] = {}
        for clique, versions in split_dict.items():
            n = len(versions)
            k = max(min_items_per_class, int(round(n * fraction)))
            k = min(k, n)
            new_splits[split][clique] = rng.sample(versions, k)

    kept_item_ids = {
        make_item_id(clique, v)
        for split_dict in new_splits.values()
        for clique, versions in split_dict.items()
        for v in versions
    }

    new_info = {k: info[k] for k in kept_item_ids}
    return {"info": new_info, "split": new_splits}


def drop_non_tagged_items(
    dataset,
    tag_fields=("tags_yt_title",),
    min_items_per_clique=2,
    delete_dvi=False,
    selected_tags=None,
):
    """
    Keep items with tags (optionally selected_tags)
    """
    info = deepcopy(dataset["info"])
    splits = dataset["split"]

    if selected_tags is not None:
        selected_tags = set(selected_tags)

    tagged_items = set()
    for item_id, v in info.items():
        is_dvi = not (v["youtube_id"] in v["version"])

        keep = False
        for field in tag_fields:
            tags = v.get(field, [])
            if not tags:
                continue
            if selected_tags is None or any(t in selected_tags for t in tags):
                keep = True
                break

        if keep or (is_dvi and not delete_dvi):
            tagged_items.add(item_id)

    new_splits = {}
    for split, split_dict in splits.items():
        new_splits[split] = {}
        for clique, versions in split_dict.items():
            kept_versions = [
                v for v in versions
                if make_item_id(clique, v) in tagged_items
            ]
            if len(kept_versions) >= min_items_per_clique:
                new_splits[split][clique] = kept_versions

    kept_item_ids = {
        make_item_id(clique, v)
        for split_dict in new_splits.values()
        for clique, versions in split_dict.items()
        for v in versions
    }

    new_info = {k: info[k] for k in kept_item_ids}
    return {"info": new_info, "split": new_splits}


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Subset dataset script")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dataset = load_dataset(args.dataset)
    print(f"Initial items: {len(dataset['info'])}")

    yvi = drop_dvi_items(dataset)
    torch.save(yvi, os.path.join(args.out_dir, "yvi.pt"))
    print("yvi:", len(yvi["info"]))

    tagged = drop_non_tagged_items(dataset)
    torch.save(tagged, os.path.join(args.out_dir, "divers_small.pt"))
    print("divers_small:", len(tagged["info"]))

    yvi_small = drop_dvi_items(tagged)
    torch.save(yvi_small, os.path.join(args.out_dir, "yvi_small.pt"))
    print("yvi_small:", len(yvi_small["info"]))

    print("Done.")


if __name__ == "__main__":
    main()
