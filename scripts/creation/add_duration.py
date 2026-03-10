#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
import torch
from glob import glob
import soundfile as sf  # pip install soundfile
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def get_duration(row, audio_dir):
    """Compute duration for a single row"""
    audio_path = os.path.join(audio_dir, row["filename"])
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    info = sf.info(audio_path)
    return info.frames / info.samplerate

def add_duration(df, audio_dir, workers=None, print_every=50):
    """Add duration using threads with reliable progress reporting"""
    rows = df.to_dict(orient="records")
    durations = [None] * len(rows)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(get_duration, row, audio_dir): idx for idx, row in enumerate(rows)}
        done_count = 0
        for future in as_completed(futures):
            idx = futures[future]
            durations[idx] = future.result()
            done_count += 1
            # simple progress print
            if done_count % print_every == 0 or done_count == len(futures):
                print(f"Processed {done_count}/{len(futures)} files")

    df["duration"] = durations
    return df

def save_df_as_torch(df, out_path, split_col="split"):
    """Save dataframe back to torch dict format"""
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
    print(f"Saved dataset with durations to {out_path} (size: {len(df):,})")
    return data

def main():
    parser = argparse.ArgumentParser(description="Add duration column to Torch dataset")
    parser.add_argument("dataset_path", type=str, help="Path to dataset file (.pt/.json)")
    parser.add_argument("audio_dir", type=str, help="Directory containing audio files")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    args = parser.parse_args()

    # Load dataset
    df, meta = load_dataset(args.dataset_path)
    n_rows_before = len(df)
    print(f"Loaded dataset with {n_rows_before} rows.")

    # Add duration with parallelization
    df = add_duration(df, args.audio_dir, workers=args.workers)

    # Sanity check
    n_rows_after = len(df)
    if n_rows_before != n_rows_after:
        raise RuntimeError(f"Row count changed! Before: {n_rows_before}, after: {n_rows_after}")
    print("Sanity check passed: row count unchanged.")

    # Save back to same file
    save_df_as_torch(df, args.dataset_path)

if __name__ == "__main__":
    main()