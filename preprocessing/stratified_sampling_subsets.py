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
        meta = torch.load(path)
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


def stratified_split(df, n=100_000, target_ratio=0.7, min_class_size=2, random_state=42):
    """Split df into sub-sample with approx. target_ratio of dvi items
    and ensure >=2 rows per clique in final sample.
    """
    rng = np.random.default_rng(random_state)

    # Filter small cliques
    class_sizes = df.groupby("clique")["version"].count()
    valid_classes = class_sizes[class_sizes >= min_class_size].index
    df = df[df["clique"].isin(valid_classes)]

    n = min(n, len(df))

    # Partition into dvi / non-dvi
    dvi = df[df["dvi"].astype(bool)]
    non = df[~df["dvi"].astype(bool)]

    # Desired global counts
    n_dvi_target = min(int(n * target_ratio), len(dvi))
    n_non_target = min(n - n_dvi_target, len(non))

    # --- Step 1: force 2 rows per clique ---
    forced_samples = []
    for clique, g in df.groupby("clique"):
        k = min(len(g), 2)  # if clique has only 2, take both
        forced_samples.append(g.sample(n=k, random_state=random_state))
    forced_df = pd.concat(forced_samples)
    
    # Track how many dvi/non we've already used
    forced_dvi = forced_df["dvi"].sum()
    forced_non = len(forced_df) - forced_dvi

    # Remaining budget after forced picks
    n_dvi_remaining = max(n_dvi_target - forced_dvi, 0)
    n_non_remaining = max(n_non_target - forced_non, 0)

    # --- Step 2: fill remaining quota ---
    dvi_remaining = dvi.drop(forced_df.index, errors="ignore")
    non_remaining = non.drop(forced_df.index, errors="ignore")

    dvi_extra = (
        dvi_remaining.sample(n=n_dvi_remaining, replace=False, random_state=random_state)
        if n_dvi_remaining > 0 and len(dvi_remaining) > 0
        else pd.DataFrame(columns=df.columns)
    )
    non_extra = (
        non_remaining.sample(n=n_non_remaining, replace=False, random_state=random_state)
        if n_non_remaining > 0 and len(non_remaining) > 0
        else pd.DataFrame(columns=df.columns)
    )

    final_df = pd.concat([forced_df, dvi_extra, non_extra]).reset_index(drop=True)

    return final_df


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


def save_df_as_torch(df, out_path, split_col="split"):
    """Save df as torch file with 'info' and 'split' dicts."""
    index = df["clique"].astype(str) + ":" + df["version"].astype(str)
    drop_cols = [split_col] if split_col in df.columns else []
    info_df = df.drop(columns=drop_cols)
    info_dict = {idx: dict(zip(info_df.columns, row))
                 for idx, row in zip(index, info_df.itertuples(index=False, name=None))}
    split_dict = df.groupby("clique")["version"].apply(list).to_dict()
    torch.save({"info": info_dict, "split": split_dict}, out_path)
    print(f"Saved torch file to {out_path}")


if __name__ == "__main__":
    df, meta = load_dataset("data/divers1m_json/rich/divers1m.json")

    subset20 = stratified_split(df, n=100_000, target_ratio=0.2)
    subset40 = stratified_split(df, n=100_000, target_ratio=0.4)
    subset60 = stratified_split(df, n=100_000, target_ratio=0.6)
    subset80 = stratified_split(df, n=100_000, target_ratio=0.8)

    print_df_summary(df, "overall")
    print_df_summary(subset20, "overall")
    print_df_summary(subset40, "overall")
    print_df_summary(subset60, "overall")
    print_df_summary(subset80, "overall")

    os.makedirs("data/divers1m_json/stratified_samples", exist_ok=True)
    save_df_as_torch(subset20, "data/divers1m_json/stratified_samples/sample20.pt")
    save_df_as_torch(subset40, "data/divers1m_json/stratified_samples/sample40.pt")
    save_df_as_torch(subset60, "data/divers1m_json/stratified_samples/sample60.pt")
    save_df_as_torch(subset80, "data/divers1m_json/stratified_samples/sample80.pt")
