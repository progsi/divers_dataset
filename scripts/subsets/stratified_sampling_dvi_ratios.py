import os
import json
import torch
import numpy as np
import pandas as pd


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

def stratified_sample(df, dvi_ratio, n_classes=1000, m_per_class=2, split="test", rnd=42, clique_col='clique', dvi_col='dvi'):
    """
    Sample a mixed dataframe from DVI and non-DVI items per clique.
    """
    np.random.seed(rnd)

    # Filter cliques that have at least m_per_class in both DVI and non-DVI
    df_filtered = df.loc[df.split == split].groupby(clique_col).filter(
        lambda g: (g[dvi_col].eq(True).sum() >= m_per_class) 
                  and (g[dvi_col].eq(False).sum() >= m_per_class)
    )

    # Sample valid cliques
    valid_cliques = df_filtered[clique_col].drop_duplicates().sample(n=n_classes, random_state=rnd).tolist()

    # Prepare candidates
    candidates_dvi = df_filtered.loc[(df_filtered[dvi_col] == True) & (df_filtered[clique_col].isin(valid_cliques))]
    candidates_dvi = candidates_dvi.groupby(clique_col).sample(n=m_per_class, random_state=rnd)

    candidates_yvi = df_filtered.loc[(df_filtered[dvi_col] == False) & (df_filtered[clique_col].isin(valid_cliques))]
    candidates_yvi = candidates_yvi.groupby(clique_col).sample(n=m_per_class, random_state=rnd)

    # Mix DVI and non-DVI per clique randomly
    mixed_rows = []
    for clique in valid_cliques:
        n_dvi = np.random.binomial(n=m_per_class, p=dvi_ratio)
        n_yvi = m_per_class - n_dvi

        sampled_dvi = candidates_dvi[candidates_dvi[clique_col] == clique].sample(n=n_dvi, random_state=rnd)
        sampled_yvi = candidates_yvi[candidates_yvi[clique_col] == clique].sample(n=n_yvi, random_state=rnd)

        mixed_rows.append(pd.concat([sampled_dvi, sampled_yvi]))

    # Combine all cliques into final dataframe
    mixed_df = pd.concat(mixed_rows).reset_index(drop=True)

    return mixed_df

def print_df_summary(df, subset="overall"):
    if subset != "overall":
        df = df.query(f"split == '{subset}'")
    print(f"Summary statistics for {subset}")

    # Column stats
    stats = []
    for col in ["clique", "version", "youtube_id"]:
        stats.append(f"{col:<12} | Unique: {df[col].nunique():>7,} | Total: {df[col].count():>7,}")
    print("\n".join(stats))

    # DVI distribution
    if "dvi" in df.columns and len(df) > 0:
        n_true = df["dvi"].sum()
        n_false = (~df["dvi"]).sum()
        total = len(df)
        print("\nDVI distribution:")
        print(f"  True:  {n_true:>7,} ({n_true/total:5.2%})")
        print(f"  False: {n_false:>7} ({n_false/total:5.2%})")

    # Version per clique stats
    if len(df) > 0:
        version_per_clique = df.groupby("clique")["version"].nunique()
        print("\nVersion per clique:")
        print(f"  Min: {version_per_clique.min():>3}  | Max: {version_per_clique.max():>3}  | "
              f"Mean: {version_per_clique.mean():>5.2f}  | Median: {version_per_clique.median():>5.2f}  | "
              f"Std: {version_per_clique.std():>5.2f}")
    print("=" * 40)

def save_df_as_torch(df, out_path, split, split_col="split"):
    """Save df as torch file with 'info' and 'split' dicts."""
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

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Stratified sampling of dataset subsets.")
    parser.add_argument("--input", type=str, help="Path to input dataset (JSON or PT).", 
                        default="data/divers1m_json/rich/divers1m.json")
    parser.add_argument("--output", type=str, help="Directory to save sampled subsets.", 
                        default="data/divers1m_json/stratified_samples")
    parser.add_argument("--m", type=int, default=3, help="Number of items per class.")
    parser.add_argument("--n", type=int, default=1000, help="Number of classes to sample.")
    parser.add_argument("--split", type=str, default="test", help="Dataset subset split to sample from.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    df, meta = load_dataset(args.input)
    df_split = df.loc[df.split == args.split]
    os.makedirs(args.output, exist_ok=True)
    
    for frac in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        subset = stratified_sample(
            df_split,
            dvi_ratio=frac,      # Desired proportion of True in 'dvi'
            m_per_class=args.m,      # Number of items per class
            n_classes=args.n,        # How many classes to sample
            split=args.split,       # Which split to sample from
            rnd=42     # For reproducibility
        )
        print(f"\n=== Summary for dvi_fraction={frac} ===")
        print_df_summary(subset, "overall")
        save_df_as_torch(subset,os.path.join(args.output, f"sample{frac}.pt"), args.split)

