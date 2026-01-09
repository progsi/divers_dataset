import os
import json
import argparse
import pandas as pd
import torch
from glob import glob


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
    
    if "youtube_id" not in df.columns:
        df["youtube_id"] = df.filename.apply(lambda x: x.split("/")[-1].split(".")[0])
    
    df["dvi"] = ~df.apply(lambda x: x.youtube_id in x.version, axis=1)
    return df, meta

def save_df_as_torch(df, out_path, split_col="split"):
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
    print(f"Saved deduplicated dataset to {out_path} (size: {len(df):,})")
    return data

def deduplicate_df(df, duplicate_hashes):
    df['hash'] = df['youtube_id'].map(duplicate_hashes)
    df_no_hash = df[df['hash'].isna()]
    df_with_hash = df[df['hash'].notna()]

    hash_clique_counts = df_with_hash.groupby('hash')['clique'].nunique()
    hashes_across_cliques = hash_clique_counts[hash_clique_counts > 1].index
    df_filtered = df_with_hash[~df_with_hash['hash'].isin(hashes_across_cliques)].copy()

    def pick_row(group):
        dvi_true = group[group['dvi'] == True]
        if not dvi_true.empty:
            return dvi_true.iloc[[0]]
        else:
            return group.iloc[[0]]

    df_deduplicated_hashes = df_filtered.groupby('hash', group_keys=False).apply(pick_row)
    df_deduplicated = pd.concat([df_deduplicated_hashes, df_no_hash], ignore_index=True)

    # Remove singletons
    df_deduplicated = df_deduplicated[df_deduplicated.groupby("clique")["clique"].transform("size") >= 2]

    print(f"Deduplicated dataset: {len(df_deduplicated):,} rows (removed {len(df)-len(df_deduplicated):,})")
    return df_deduplicated

def main():
    parser = argparse.ArgumentParser(description="Deduplicate Torch datasets using hash mapping")
    parser.add_argument("dataset_dir", type=str, help="Directory containing dataset files (.pt/.json)")
    parser.add_argument("hashes_file", type=str, help="Path to duplicate_hashes JSON file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Directory to save deduplicated datasets (default: overwrite input)")

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output = args.output or dataset_dir

    # Load duplicate hashes
    with open(args.hashes_file, "r") as f:
        duplicate_hashes = json.load(f)

    # Find all dataset files
    dataset_files = glob(os.path.join(dataset_dir, "*.pt")) + glob(os.path.join(dataset_dir, "*.json"))
    print(f"Found {len(dataset_files)} dataset files in {dataset_dir}")

    for path in dataset_files:
        print(f"\nProcessing {path}...")
        df, meta = load_dataset(path)
        if "length" in df.columns:
            df = df.drop(columns=["length"])
        df_dedup = deduplicate_df(df, duplicate_hashes)

        # Save
        filename = os.path.basename(path)
        save_path = os.path.join(output, filename)
        save_df_as_torch(df_dedup, save_path)

if __name__ == "__main__":
    main()
