#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
import torch
from glob import glob

def load_dataset(path):
    """Load dataset from .json or .pt"""
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
        raise ValueError("Unsupported dataset format, use .json or .pt")

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

    df["dvi"] = ~df.apply(lambda x: x.youtube_id in x.version, axis=1)
    return df, meta

def process_music_segments(df, segment_inds_path):
    """Map music_segment_inds to df, remove invalid rows, drop extra columns, remove singletons"""
    with open(segment_inds_path, "r") as f:
        music_segment_inds = json.load(f)

    # Map segment indices to dataset
    df["music_segment_inds"] = df.youtube_id.map(music_segment_inds)

    # Remove rows where segment inds is NaN or contains only zeros
    def valid_segments(x):
        if x is None:
            return False
        if isinstance(x, list):
            return any(v != 0 for v in x)
        if pd.isna(x):
            return False
        return bool(x)
    df = df[df["music_segment_inds"].apply(valid_segments)].copy()

    # Remove singleton cliques (after filtering)
    df = df[df.groupby("clique")["clique"].transform("size") >= 2]

    # Drop samplerate and channels if present
    for col in ["samplerate", "channels"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df

def save_df_as_torch(df, out_path, split_col="split"):
    """Save processed dataframe back to torch dict format"""
    index = df["clique"].astype(str) + ":" + df["version"].astype(str)
    drop_cols = [split_col] if split_col in df.columns else []
    info_df = df.drop(columns=drop_cols)
    info_dict = {
        idx: dict(zip(info_df.columns, row))
        for idx, row in zip(index, info_df.itertuples(index=False, name=None))
    }

    split_dict = {}
    if split_col in df.columns:
        for split, sub_df in df.dropna(subset=[split_col]).groupby(split_col):
            clique_dict = sub_df.groupby("clique")["version"].apply(list).to_dict()
            split_dict[split] = clique_dict

    data = {"info": info_dict, "split": split_dict}
    torch.save(data, out_path)
    print(f"Saved processed dataset to {out_path} (size: {len(df):,})")
    return data

def main():
    parser = argparse.ArgumentParser(description="Process music segment indices in Torch datasets")
    parser.add_argument("dataset_dir", type=str, help="Directory containing dataset files (.pt/.json)")
    parser.add_argument("segment_inds_path", type=str, help="JSON file with segment indices (binary)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Directory to save processed datasets (default: overwrite input)")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output = args.output or dataset_dir

    dataset_files = glob(os.path.join(dataset_dir, "*.pt")) + glob(os.path.join(dataset_dir, "*.json"))
    print(f"Found {len(dataset_files)} dataset files in {dataset_dir}")

    for path in dataset_files:
        print(f"\nProcessing {path}...")
        df, meta = load_dataset(path)
        df_processed = process_music_segments(df, args.segment_inds_path)

        # Save
        filename = os.path.basename(path)
        save_path = os.path.join(output, filename)
        save_df_as_torch(df_processed, save_path)

if __name__ == "__main__":
    main()