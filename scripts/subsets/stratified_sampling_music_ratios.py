#!/usr/bin/env python3
"""
Stratified sampling of music dataset subsets with variable noise thresholds.
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

    # fixed dvi calculation
    df["dvi"] = df["youtube_id"] != df["version"]

    return df, meta

def parse_args():
    parser = argparse.ArgumentParser(description="Stratified sampling of dataset subsets.")
    parser.add_argument("--input", type=str, help="Path to input dataset (JSON or PT).", 
                        default="data/divers1m_json/rich/divers1m.json")
    parser.add_argument("--output", type=str, help="Directory to save sampled subsets.", 
                        default="data/divers1m_json/stratified_samples")
    parser.add_argument("--m-dvi", type=int, default=1, help="Number of query items per clique/class.")
    parser.add_argument("--m-noisy", type=int, default=1, help="Number of noisy items per clique/class.")
    parser.add_argument("--n-pairs", type=int, default=500, help="Total number of pairs to sample.")
    parser.add_argument("--no-clique-reuse", action="store_false",
                        dest="allow_clique_reuse", help="Do not allow using the same clique more than once.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to sample from.")
    parser.add_argument("--noise-thresh", type=str, default=None, 
                        help="Comma-separated list of noise thresholds (0.0,0.1,...). If None, defaults to 0.0 to 0.9.")
    parser.add_argument("--clean-thresh", type=float, default=0.9, help="Minimum fraction of music segments for clean/query items.")
    parser.add_argument("--require-dvi", action="store_true", help="Require query items to be DVI versions.")
    parser.add_argument("--total-size", type=int, default=None, help="Maximum total number of samples.")
    return parser.parse_args()


def stratified_music_sampling_fast(
    df,
    n_per_class=2,
    noise_thresh=0.2,
    clean_thresh=None,
    require_dvi=False,
    total_size=None,
    random_state=42
):
    np.random.seed(random_state)

    if clean_thresh is None and not require_dvi:
        raise ValueError("At least one of 'clean_thresh' or 'require_dvi' must be specified")

    df = df.copy()
    df["music_ratio"] = df["music_segment_inds"].apply(np.mean)
    df["is_clean"] = df["music_ratio"] >= clean_thresh if clean_thresh is not None else True
    df["is_noisy"] = (1 - df["music_ratio"]) >= noise_thresh
    df["is_query"] = df["is_clean"] & df["dvi"] if require_dvi else df["is_clean"]
    df["is_candidate"] = df["is_clean"] | df["is_noisy"]

    query_df = df[df["is_query"]]
    candidate_df = df[df["is_candidate"]]

    query_counts = query_df.groupby("clique").size()
    candidate_counts = candidate_df.groupby("clique").size()
    valid_cliques = query_counts.index[
        (query_counts >= 1) &
        (candidate_counts.reindex(query_counts.index, fill_value=0) >= n_per_class)
    ]
    if len(valid_cliques) == 0:
        raise ValueError(f"No cliques satisfy constraints for noise_thresh={noise_thresh}")

    df_valid = df[df["clique"].isin(valid_cliques)]

    def sample_group(group):
        clean_items = group[group["is_clean"]]
        noisy_items = group[group["is_noisy"]]

        n_clean = min(len(clean_items), max(1, n_per_class // 2))
        n_noisy = min(len(noisy_items), n_per_class - n_clean)

        samples = []
        if n_clean > 0:
            samples.append(clean_items.sample(n=n_clean, random_state=random_state))
        if n_noisy > 0:
            samples.append(noisy_items.sample(n=n_noisy, random_state=random_state))

        return pd.concat(samples)

    sampled_df = (
        df_valid
        .groupby("clique", group_keys=False)
        .apply(sample_group)
        .reset_index(drop=True)
    )

    if total_size is not None and len(sampled_df) > total_size:
        sampled_df = sampled_df.sample(n=total_size, random_state=random_state)

    return sampled_df


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

    # ---- Load dataset ----
    df, meta = load_dataset(args.input)

    # ---- Noise thresholds ----
    if args.noise_thresh is None:
        noise_thresholds = np.round(np.arange(0.0, 1.0, 0.1), 2)
    else:
        noise_thresholds = [float(x) for x in args.noise_thresh.split(",")]

    # ---- Loop over thresholds ----
    for nt in noise_thresholds:
        print(f"\nSampling with noise_thresh={nt}")
        sampled_df = stratified_music_sampling_fast(
            df,
            n_per_class=args.m_dvi + args.m_noisy,
            noise_thresh=nt,
            clean_thresh=args.clean_thresh,
            require_dvi=args.require_dvi,
            total_size=args.total_size,
        )

        out_file = output_dir / f"stratified_{args.split}_noise{nt:.2f}.pt"
        save_df_as_torch(sampled_df, out_file, args.split)