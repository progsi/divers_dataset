import os
import json
import argparse
import torch
import pandas as pd

def load_dataset(path):
    """
    Load dataset from the specified path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    def inverse_split_dict(split_dict):
        # Add split column to dvi2Torch based on dvi2_torch["split"]
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
        info =  meta["info"]
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

def add_first_version_info(df):
    """
    Add first version info to each row in the DataFrame.
    """
    cols_to_extract = [
        'track_title', 'track_artist_names',
        'track_writer_names', 'release_artist_names', 'released'
    ]
    
    # Ensure correct ordering by clique + version
    df = df.sort_values(["clique", "version"])
    
    # Get the first "V-" row per clique
    ref_rows = (
        df[df["version"].str.startswith("V-")]
        .groupby("clique")
        .first()[cols_to_extract]
    )
    
    # Map them back into new "first_" columns
    for col in cols_to_extract:
        df[f"first_{col}"] = df["clique"].map(ref_rows[col])
    
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/divers1m_torch/divers1m.pt", help="Path to dataset")
    parser.add_argument("--output", type=str, default="data/divers1m_torch/sample.pt", help="Path to dataset")
    parser.add_argument("-n", type=int, default=10_000, help="Sample size")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    df, meta = load_dataset(args.input)
    df = add_first_version_info(df)
    sample = {
        "info": df.sample(n=args.n, random_state=42).to_dict(orient="index"),
    }
    print(f"Saving sample of size {args.n} to {args.output}.")
    if args.output.endswith(".json"):
        with open(args.output, "w") as f:
            json.dump(sample, f)
    elif args.output.endswith(".pt"):
        torch.save(sample, args.output)
