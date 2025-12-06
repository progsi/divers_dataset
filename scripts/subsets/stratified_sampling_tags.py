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

def stratified_sample(
    df,
    tags,
    m_dvi,
    m_tag,
    c,
    replace=False,
    random_state=None
):
    """
    Perform stratified sampling by class ('clique'):
      - m_dvi requires items per class with dvi=True
      - m_tag requires total items per class containing ALL of the tags in `tags`
      - c total classes sampled (randomly)
      - Classes that cannot satisfy requirements are ignored
      - If fewer than c valid classes exist, raise an error
      - Sampling without replacement unless replace=True

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ['clique', 'youtube_id', 'dvi', 'tags_yt_title']
    tags : list[str]
        Tags; we require m_tag items that contain *any* tag in this list
    m_dvi : int
        Required # of dvi=True items per class
    m_tag : int
        Required # of tag-containing items per class (across all tags)
    c : int
        Number of classes to sample
    replace : bool, default False
        Whether to sample with replacement
    random_state : int or None
        For reproducibility

    Returns
    -------
    pd.DataFrame
        Stratified sample across selected classes.
    """

    rng = np.random.default_rng(random_state)
    required_tags = set(tags)

    feasible_classes = []

    # 1. Check feasibility per class
    for clique, group in df.groupby("clique"):
        # Items with dvi=True
        dvi_items = group[group["dvi"] == True]

        tag_items = group[
            group["tags_yt_title"].apply(
                lambda lst: required_tags.issubset(lst)
            )
        ]
        # # Items containing ANY tag
        # tag_items = group[
        #     group["tags_yt_title"].apply(
        #         lambda lst: len(required_tags.intersection(lst)) > 0
        #     )
        # ]

        # Check availability
        if (len(dvi_items) >= m_dvi) and (len(tag_items) >= m_tag):
            feasible_classes.append(clique)

    # 2. Verify enough feasible classes
    if len(feasible_classes) < c:
        print(
            f"Only {len(feasible_classes)} feasible classes available, but c={c} required."
        )
        c = len(feasible_classes)

    # 3. Randomly choose c classes
    selected_classes = rng.choice(feasible_classes, size=c, replace=False)

    # 4. Sample within each selected class
    final_results = []

    for clique in selected_classes:
        group = df[df["clique"] == clique]

        # sample dvi=True items
        dvi_items = group[group["dvi"] == True].sample(
            n=m_dvi, replace=replace, random_state=random_state
        )

        # sample tag items
        tag_items = group[
            group["tags_yt_title"].apply(
                lambda lst: len(required_tags.intersection(lst)) > 0
            )
        ].sample(
            n=m_tag, replace=replace, random_state=random_state
        )

        # Combine samples & remove accidental duplicates
        combined = pd.concat([dvi_items, tag_items]).drop_duplicates("youtube_id")
        
        # If duplicates caused the total to drop below required amounts, fail
        if len(combined) < (m_dvi + m_tag) and not replace:
            raise ValueError(
                f"Class '{clique}' lost unique rows due to overlap; cannot meet sampling "
                f"requirements without replacement."
            )

        final_results.append(combined)

    return pd.concat(final_results, ignore_index=True)

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
    parser.add_argument("--m_dvi", type=int, default=1, help="Number of items per class.")
    parser.add_argument("--m_tag", type=int, default=1, help="Number of items per class.")
    parser.add_argument("--c", type=int, default=500, help="Number of classes to sample.")
    parser.add_argument("--split", type=str, default="test", help="Dataset subset split to sample from.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    df, meta = load_dataset(args.input)
    df_split = df.loc[df.split == args.split]
    os.makedirs(args.output, exist_ok=True)
    
    tag_subsets = [
        ["acoustic"],
        ["instrumental"],              # instrumental|karaoke
        ["live"],
        ["orchestra"],
        ["reaction"],          # firsttimehearing|firsttimereaction|reaction|reactsto|reactto
        ["cover"],
        ["cover", "bass"],             # cover&bass
        ["cover", "drum"],
        ["cover", "guitar"],
        ["cover", "piano"],
        ["solo"],
        ["solo", "bass"],
        ["solo", "drum"],
        ["solo", "guitar"],
        ["solo", "piano"],
        ["tutorial"],                 # howtoplay|howtosing|lesson|tutorial
        ["tutorial", "bass"],         # (howtoplay|lesson|tutorial)&bass
        ["tutorial", "drum"],
        ["tutorial", "guitar"],
        ["tutorial", "piano"],
    ]
    
    for tag_subset in tag_subsets:
        try:
            subset = stratified_sample(
                df_split,
                tags=tag_subset,
                m_dvi=args.m_dvi,          # Number of dvi=True items per class
                m_tag=args.m_tag,          # Number of tag-containing items per class
                c=args.c,                  # How many classes to sample
                random_state=42     # For reproducibility
            )
            tag_subset_str = "_".join(tag_subset)
            print(f"\n=== Summary for dvi_fraction={tag_subset_str} ===")
            print_df_summary(subset, "overall")
            save_df_as_torch(subset,os.path.join(args.output, f"sample_{tag_subset_str}.pt"), args.split)
        except ValueError as e:
            print(f"Skipping tag subset {tag_subset}, can not satisfy sampling requirements: {e}")