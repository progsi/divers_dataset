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

def stratified_sample(df, n_per_class=3, n_classes=None, dvi_ratio=0.5, random_state=None):
    """
    Stratified sampling function with fixed number of items per class, desired number of classes,
    and target ratio of dvi==True in the final sample. Filters out classes with fewer than n_per_class.
    
    Parameters:
        df (pd.DataFrame): Input dataframe with columns 'clique' and 'dvi'.
        n_per_class (int): Number of items to sample per class.
        n_classes (int): Number of classes to sample.
        dvi_ratio (float): Desired ratio of dvi==True in final sample (0-1).
        random_state (int): Random seed.
    
    Returns:
        pd.DataFrame: Sampled dataframe.
    """
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Filter classes that have enough items
    class_counts = df['clique'].value_counts()
    eligible_classes = class_counts[class_counts >= n_per_class].index.tolist()
    
    if not eligible_classes:
        raise ValueError(f"No classes have at least {n_per_class} items.")
    
    if n_classes is None or n_classes > len(eligible_classes):
        n_classes = len(eligible_classes)
    
    # Randomly choose classes
    chosen_classes = np.random.choice(eligible_classes, size=n_classes, replace=False)
    
    # Sample fixed number per class
    sampled_list = []
    for cl in chosen_classes:
        df_class = df[df['clique'] == cl]
        sampled_class = df_class.sample(n=n_per_class, replace=False, random_state=random_state)
        sampled_list.append(sampled_class)
    
    sample_df = pd.concat(sampled_list)
    
    # Adjust dvi ratio
    total_items = len(sample_df)
    desired_true_count = int(round(dvi_ratio * total_items))
    current_true_count = sample_df['dvi'].sum()
    
    if desired_true_count > total_items:
        raise ValueError("Desired number of True items exceeds sample size")
    
    if current_true_count != desired_true_count:
        trues = sample_df[sample_df['dvi'] == True]
        falses = sample_df[sample_df['dvi'] == False]
        
        if current_true_count < desired_true_count:
            to_switch = desired_true_count - current_true_count
            if len(falses) < to_switch:
                raise ValueError("Not enough False items to achieve desired dvi ratio")
            falses_indices = falses.sample(n=to_switch, random_state=random_state).index
            sample_df.loc[falses_indices, 'dvi'] = True
        else:
            to_switch = current_true_count - desired_true_count
            if len(trues) < to_switch:
                raise ValueError("Not enough True items to reduce to desired dvi ratio")
            trues_indices = trues.sample(n=to_switch, random_state=random_state).index
            sample_df.loc[trues_indices, 'dvi'] = False
    
    return sample_df.reset_index(drop=True)

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
    df_test = df.query("split == 'test'")
    os.makedirs("data/divers1m_json/stratified_samples", exist_ok=True)
    
    for frac in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        subset = stratified_sample(
            df_test,
            n_per_class=3,      # Number of items per class
            n_classes=1000,        # How many classes to sample
            dvi_ratio=0.2,      # Desired proportion of True in 'dvi'
            random_state=42     # For reproducibility
        )
        print(f"\n=== Summary for dvi_fraction={frac} ===")
        print_df_summary(subset, "overall")
        save_df_as_torch(subset, f"data/divers1m_json/stratified_samples/sample{frac}.pt")

