import os
import json
import torch
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
    """Split df into sub-sample with approximately target_ratio of dvi items."""
    # filter small classes
    class_sizes = df.groupby("clique")["version"].count()
    valid_classes = class_sizes[class_sizes >= min_class_size].index
    df = df[df["clique"].isin(valid_classes)]

    n = min(n, len(df))

    # partition into dvi / non-dvi
    dvi = df[df["dvi"].astype(bool)]
    non = df[~df["dvi"].astype(bool)]

    print(f"Available items: DVI={len(dvi)}, non-DVI={len(non)}, total={len(df)}")

    # desired counts
    n_dvi = min(int(n * target_ratio), len(dvi))
    n_non = min(n - n_dvi, len(non))

    if n_dvi + n_non == 0:
        raise ValueError("No items available for sampling. Reduce n or adjust target_ratio.")

    dvi_sample = dvi.sample(n=n_dvi, replace=False, random_state=random_state) if n_dvi > 0 else pd.DataFrame(columns=df.columns)
    non_sample = non.sample(n=n_non, replace=False, random_state=random_state) if n_non > 0 else pd.DataFrame(columns=df.columns)

    return pd.concat([dvi_sample, non_sample]).reset_index(drop=True)


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
    save_df_as_torch(subset20, "data/divers1m_json/stratified_samples/sample_20.pt")
    save_df_as_torch(subset40, "data/divers1m_json/stratified_samples/sample_40.pt")
    save_df_as_torch(subset60, "data/divers1m_json/stratified_samples/sample_60.pt")
    save_df_as_torch(subset80, "data/divers1m_json/stratified_samples/sample_80.pt")
