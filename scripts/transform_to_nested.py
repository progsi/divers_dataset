#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
import torch

LIGHT_COLS = ["id", "artist", "title", "filename", 
              "youtube_id", "dvi", "tags_yt_title"]
FULL_COLS = [
    "id",
    "artist",
    "title",
    "filename",
    "samplerate",
    "channels",
    "youtube_id",
    "dvi",
    "track_writer_names",
    "release_artist_names",
    "release_genres",
    "release_styles",
    "country",
    "labels",
    "formats",
    "released",
    "yt_title",
    "yt_tags",
    "yt_categories",
    "yt_channel",
    "yt_upload_date",
    "yt_view_count",
    "tempo",
    "tags_yt_title",
    "cues_yt_title",
    "tags_yt_description",
    "cues_yt_description",
    "tags_yt_tags",
    "cues_yt_tags"
]

def load_dataset(path):
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

    if isinstance(meta, dict) and "info" in meta:
        info = meta["info"]
        split = meta["split"]
    else:
        info, split = meta

    df = pd.DataFrame.from_dict(info, orient="index")
    clique2split = inverse_split_dict(split)
    df["split"] = df["clique"].map(clique2split)
    
    if "youtube_id" not in df.columns and "filename" in df.columns:
        df["youtube_id"] = df.filename.apply(lambda x: x.split("/")[-1].split(".")[0])
    
    df["dvi"] = ~df.apply(lambda x: x.youtube_id in x.version, axis=1)
    return df

def df_to_nested_dict(df, split_col="split", keep_cols=None):
    nested = {}
    for split, split_df in df.groupby(split_col):
        nested[split] = {}
        for clique, clique_df in split_df.groupby("clique"):
            nested[split][clique] = {}
            for _, row in clique_df.iterrows():
                version = row["version"]
                data = row.drop([split_col, "clique", "version"])
                if keep_cols is not None:
                    data = data[keep_cols]
                nested[split][clique][version] = data.to_dict()
    return nested

def make_json_serializable(obj):
    """
    Recursively convert values to JSON-serializable types.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    # handle numpy types
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    # fallback
    return str(obj)

def save_json(nested_dict, out_path):
    serializable_dict = make_json_serializable(nested_dict)
    with open(out_path, "w") as f:
        json.dump(serializable_dict, f, indent=2)
    print(f"Saved JSON: {out_path} (splits: {list(nested_dict.keys())})")

def main():
    parser = argparse.ArgumentParser(description="Transform dataset to nested dict (JSON output)")
    parser.add_argument("dataset_path", type=str, help="Path to input dataset (.pt or .json)")
    parser.add_argument("out_dir", type=str, help="Directory to save nested dicts")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_dataset(args.dataset_path)
    base_name = os.path.splitext(os.path.basename(args.dataset_path))[0]

    # Full version
    nested_full = df_to_nested_dict(df, keep_cols=FULL_COLS)
    full_path = os.path.join(args.out_dir, f"{base_name}.json")
    save_json(nested_full, full_path)

    # Light version
    nested_light = df_to_nested_dict(df, keep_cols=LIGHT_COLS)
    light_path = os.path.join(args.out_dir, f"{base_name}_light.json")
    save_json(nested_light, light_path)

if __name__ == "__main__":
    main()
