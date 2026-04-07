#!/usr/bin/env python3
"""
Stratified sampling of music dataset subsets with variable noise thresholds and exact pair control.
"""
import os
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import torch


def load_dataset(path):
    """Load dataset from JSON or Torch file, construct dvi column."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    def inverse_split_dict(split_dict):
        clique_to_split = {}
        for split_name, sub_dict in split_dict.items():
            for clique in sub_dict.keys():
                clique_to_split[clique] = split_name
        return clique_to_split

    if path.endswith(".json"):
        with open(path, "r") as f:
            meta = json.load(f)
    elif path.endswith(".pt"):
        meta = torch.load(path, weights_only=False)
    else:
        raise ValueError("Unsupported file type")

    if isinstance(meta, dict) and "info" in meta:
        info = meta["info"]
        split = meta["split"]
    else:
        info, split = meta

    df = pd.DataFrame.from_dict(info, orient="index")
    clique2split = inverse_split_dict(split)
    df["split"] = df["clique"].map(clique2split)

    if "youtube_id" not in df.columns:
        df["youtube_id"] = df.filename.apply(lambda x: x.split("/")[-1].split(".")[0])

    # dvi calculation
    df["dvi"] = df["youtube_id"] != df["version"]

    return df, meta


def parse_args():
    parser = argparse.ArgumentParser(description="Stratified sampling of dataset subsets.")
    parser.add_argument("--input", type=str, default="data/divers1m_json/rich/divers1m.json")
    parser.add_argument("--output", type=str, default="data/divers1m_json/stratified_samples")
    parser.add_argument("--m-dvi", type=int, default=1, help="Number of query items per pair.")
    parser.add_argument("--m-noisy", type=int, default=1, help="Number of noisy items per pair.")
    parser.add_argument("--n-pairs", type=int, default=500, help="Total number of pairs to sample.")
    parser.add_argument("--no-clique-reuse", action="store_false", dest="allow_clique_reuse")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--noise-thresh", type=str, default=None)
    parser.add_argument("--clean-thresh", type=float, default=0.9)
    parser.add_argument("--require-dvi", action="store_true")
    parser.add_argument("--total-size", type=int, default=None)
    return parser.parse_args()


def sample_exact_items(group, m_dvi, m_noisy, random_state=42):
    """Sample exactly m_dvi query items and m_noisy noisy items from a clique."""
    clean_items = group[group["is_clean"]]
    noisy_items = group[group["is_noisy"]]

    if len(clean_items) < m_dvi or len(noisy_items) < m_noisy:
        return pd.DataFrame(columns=group.columns)

    sampled = pd.concat([
        clean_items.sample(n=m_dvi, random_state=random_state),
        noisy_items.sample(n=m_noisy, random_state=random_state)
    ])
    return sampled


def stratified_music_sampling(
    df,
    m_dvi,
    m_noisy,
    n_pairs,
    noise_thresh,
    clean_thresh,
    require_dvi=False,
    allow_clique_reuse=True,
    random_state=42
):
    np.random.seed(random_state)
    df = df.copy()

    # Compute clean/noisy flags
    df["music_ratio"] = df["music_segment_inds"].apply(np.mean)
    df["is_clean"] = df["music_ratio"] >= clean_thresh if clean_thresh is not None else True
    df["is_noisy"] = (1 - df["music_ratio"]) >= noise_thresh
    df["is_query"] = df["is_clean"] & df["dvi"] if require_dvi else df["is_clean"]
    df["is_candidate"] = df["is_clean"] | df["is_noisy"]

    # Identify valid cliques
    valid_cliques = []
    clique_item_counts = {}
    for clique, group in df.groupby("clique"):
        n_clean = group[group["is_clean"]].shape[0]
        n_noisy = group[group["is_noisy"]].shape[0]
        if n_clean >= m_dvi and n_noisy >= m_noisy:
            valid_cliques.append(clique)
            clique_item_counts[clique] = (n_clean, n_noisy)

    if len(valid_cliques) == 0:
        raise ValueError(f"No cliques satisfy constraints for noise_thresh={noise_thresh}")

    sampled_pairs = []
    clique_list = valid_cliques.copy()

    while len(sampled_pairs) < n_pairs:
        if not clique_list:
            if allow_clique_reuse:
                clique_list = valid_cliques.copy()
            else:
                break

        # Pick a clique at random
        clique = np.random.choice(clique_list)
        group = df[df["clique"] == clique]

        # Sample exactly one pair (m_dvi clean + m_noisy noisy)
        pair = sample_exact_items(group, m_dvi, m_noisy, random_state=random_state)

        if not pair.empty:
            sampled_pairs.append(pair)

        # Remove clique if reuse is not allowed
        if not allow_clique_reuse:
            clique_list.remove(clique)

    # If we sampled more pairs than needed, truncate
    if len(sampled_pairs) > n_pairs:
        sampled_pairs = sampled_pairs[:n_pairs]

    # Concatenate all pairs into a single DataFrame
    sampled_df = pd.concat(sampled_pairs, ignore_index=True)

    return sampled_df.reset_index(drop=True)


def save_df_as_torch(df, out_path, split, split_col="split"):
    index = df["clique"].astype(str) + ":" + df["version"].astype(str)
    drop_cols = [split_col] if split_col in df.columns else []
    info_df = df.drop(columns=drop_cols)
    info_dict = {idx: dict(zip(info_df.columns, row))
                 for idx, row in zip(index, info_df.itertuples(index=False, name=None))}

    remaining_splits = [s for s in ["train", "valid", "test"] if s != split]
    split_dict = {s: {} for s in remaining_splits}
    split_dict[split] = df.groupby("clique")["version"].apply(list).to_dict()
    torch.save({"info": info_dict, "split": split_dict}, out_path)
    print(f"Saved torch file to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df, meta = load_dataset(args.input)

    # Noise thresholds
    if args.noise_thresh is None:
        noise_thresholds = np.round(np.arange(0.1, 1.0, 0.1), 2)
    else:
        noise_thresholds = [float(x) for x in args.noise_thresh.split(",")]

    # Loop over thresholds
    for nt in noise_thresholds:
        print(f"\nSampling with noise_thresh={nt}")
        sampled_df = stratified_music_sampling(
            df,
            m_dvi=args.m_dvi,
            m_noisy=args.m_noisy,
            n_pairs=args.n_pairs,
            noise_thresh=nt,
            clean_thresh=args.clean_thresh,
            require_dvi=args.require_dvi,
            allow_clique_reuse=args.allow_clique_reuse
        )

        out_file = Path(args.output) / f"stratified_{args.split}_noise{nt:.2f}.pt"
        save_df_as_torch(sampled_df, out_file, args.split)