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

# def stratified_sample(
#     df,
#     tags,
#     m_dvi,
#     m_tag,
#     n_pairs,
#     allow_clique_reuse=True,
#     replace=False,
#     random_state=None,
# ):
#     """
#     Sample n_pairs pairs, where each pair consists of:
#       - m_dvi items with dvi=True
#       - m_tag items containing ALL required tags in `tags`

#     Sampling is performed within cliques, but:
#       - allow_clique_reuse=True allows multiple pairs from the same clique
#       - allow_clique_reuse=False allows a clique to be used only once

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Must contain: ['clique', 'youtube_id', 'dvi', 'tags_yt_title']
#     tags : list[str]
#         Items for m_tag sampling must contain ALL of these tags.
#     m_dvi : int
#         Items with dvi=True per pair.
#     m_tag : int
#         Items containing all required tags per pair.
#     n_pairs : int
#         Number of pairs to sample.
#     allow_clique_reuse : bool, default True
#         Whether a clique can be used multiple times.
#     replace : bool, default False
#         Sampling within a clique can reuse items.
#     random_state : int or None
#         For reproducibility.

#     Returns
#     -------
#     pd.DataFrame
#         All sampled items, with pair_id and clique_used columns.
#     """

#     rng = np.random.default_rng(random_state)
#     required_tags = set(tags)

#     # --- 1. Identify feasible cliques ---
#     feasible = {}

#     for clique, group in df.groupby("clique"):
#         dvi_items = group[group["dvi"] == True]

#         # ALL required tags must be present
#         tag_items = group[
#             group["tags_yt_title"].apply(lambda lst: required_tags.issubset(lst))
#         ]

#         if len(dvi_items) >= m_dvi and len(tag_items) >= m_tag:
#             feasible[clique] = {
#                 "dvi": dvi_items,
#                 "tag": tag_items
#             }

#     if not feasible:
#         raise ValueError("No clique can satisfy m_dvi and m_tag requirements (ALL tags).")

#     if not allow_clique_reuse and len(feasible) < n_pairs:
#         raise ValueError(
#             f"Only {len(feasible)} feasible cliques available; n_pairs={n_pairs} requested."
#         )

#     # --- 2. Determine clique sequence ---
#     cliques_sequence = (
#         rng.choice(list(feasible.keys()), size=n_pairs, replace=True)
#         if allow_clique_reuse
#         else rng.choice(list(feasible.keys()), size=n_pairs, replace=False)
#     )

#     # --- 3. Build each pair ---
#     results = []

#     for pair_id, clique in enumerate(cliques_sequence, start=1):
#         dvi_group = feasible[clique]["dvi"]
#         tag_group = feasible[clique]["tag"]

#         dvi_sample = dvi_group.sample(n=m_dvi, replace=replace, random_state=random_state)
#         tag_sample = tag_group.sample(n=m_tag, replace=replace, random_state=random_state)

#         combined = pd.concat([dvi_sample, tag_sample]).drop_duplicates("youtube_id")

#         # If duplicates lowered count below requirement (and no replacement allowed), fail
#         if len(combined) < m_dvi + m_tag and not replace:
#             raise ValueError(
#                 f"Pair from clique '{clique}' lost unique rows due to overlap; "
#                 f"cannot meet requirements without replacement."
#             )

#         combined = combined.copy()
#         combined["pair_id"] = pair_id
#         combined["clique_used"] = clique

#         results.append(combined)

#     return pd.concat(results, ignore_index=True)

def stratified_sample(
    df,
    tags,
    m_dvi,
    m_tag,
    n_pairs,
    allow_clique_reuse=True,
    replace=False,
    random_state=None,
):
    """
    Sample n_pairs pairs, each with:
      - m_dvi items with dvi=True
      - m_tag items containing ALL required tags
    Guarantees no overlap within a pair and retries other cliques if needed.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ['clique', 'youtube_id', 'dvi', 'tags_yt_title']
    tags : list[str]
        Items for m_tag sampling must contain ALL of these tags.
    m_dvi : int
        Number of dvi=True items per pair.
    m_tag : int
        Number of tag-matching items per pair.
    n_pairs : int
        Number of pairs to sample.
    allow_clique_reuse : bool, default True
        Whether a clique can provide multiple pairs.
    replace : bool, default False
        Whether to sample with replacement within a clique.
    random_state : int or None
        For reproducibility.

    Returns
    -------
    pd.DataFrame
        Combined sampled items with 'pair_id' and 'clique_used' columns.
    """

    rng = np.random.default_rng(random_state)
    required_tags = set(tags)

    # --- 1. Precompute feasible cliques ---
    feasible = {}
    for clique, group in df.groupby("clique"):
        dvi_items = group[group["dvi"] == True]
        tag_items = group[group["tags_yt_title"].apply(lambda lst: required_tags.issubset(lst))]

        if len(dvi_items) >= m_dvi and len(tag_items) >= m_tag:
            feasible[clique] = {
                "dvi": dvi_items,
                "tag": tag_items
            }

    if not feasible:
        raise ValueError("No clique can satisfy m_dvi and m_tag requirements (ALL tags).")

    if not allow_clique_reuse and len(feasible) < n_pairs:
        raise ValueError(f"Not enough cliques to sample {n_pairs} pairs without reuse.")

    # --- 2. Prepare clique pool ---
    available_cliques = list(feasible.keys())
    results = []
    pair_id = 1

    while pair_id <= n_pairs:
        if not available_cliques:
            raise ValueError(
                f"Cannot build {n_pairs} pairs. Only {pair_id - 1} pairs could be created."
            )

        # Select a clique randomly
        clique = rng.choice(available_cliques)

        dvi_group = feasible[clique]["dvi"]
        tag_group = feasible[clique]["tag"]

        try:
            # Sample dvi items
            dvi_sample = dvi_group.sample(n=m_dvi, replace=replace, random_state=random_state)

            # Sample tag items excluding dvi_sample to avoid overlap
            tag_pool = tag_group.drop(dvi_sample.index, errors='ignore')
            if len(tag_pool) < m_tag and not replace:
                raise ValueError("Not enough unique tag items after excluding dvi_sample.")

            tag_sample = tag_pool.sample(n=m_tag, replace=replace, random_state=random_state)

            # Combine and assign pair info
            combined = pd.concat([dvi_sample, tag_sample])
            combined = combined.drop_duplicates("youtube_id").copy()
            combined["pair_id"] = pair_id
            combined["clique_used"] = clique

            results.append(combined)
            pair_id += 1

            # Remove clique from pool if reuse is not allowed
            if not allow_clique_reuse:
                available_cliques.remove(clique)

        except ValueError:
            # Failed for this clique, remove it if reuse is disallowed
            if not allow_clique_reuse:
                available_cliques.remove(clique)
            # Otherwise, just retry another clique
            continue

    return pd.concat(results, ignore_index=True)

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
    parser.add_argument("--m-dvi", type=int, default=1, help="Number of items per class.")
    parser.add_argument("--m-tag", type=int, default=1, help="Number of items per class.")
    parser.add_argument("--n-pairs", type=int, default=500, help="Number of pairs to sample.")
    parser.add_argument("--no-clique-reuse", action="store_false",
                        dest="allow_clique_reuse", help="Do not allow using the same clique.")
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
                n_pairs=args.n_pairs,                  # How many classes to sample
                allow_clique_reuse=args.allow_clique_reuse,
                random_state=42     # For reproducibility
            )
            tag_subset_str = "_".join(tag_subset)
            print(f"\n=== Summary for tag subset={tag_subset_str} ===")
            print_df_summary(subset, "overall")
            save_df_as_torch(subset,os.path.join(args.output, f"sample_{tag_subset_str}.pt"), args.split)
        except ValueError as e:
            print(f"Skipping tag subset {tag_subset}, can not satisfy sampling requirements: {e}")
            
    ref_set = stratified_sample(
                df_split,
                tags=[],
                m_dvi=args.m_dvi + args.m_tag,          # Number of dvi=True items per class
                m_tag=0,          # Number of tag-containing items per class
                n_pairs=args.n_pairs,                  # How many classes to sample
                allow_clique_reuse=args.allow_clique_reuse,
                random_state=42     # For reproducibility
    )
    print(f"\n=== Summary for reference set ===")
    print_df_summary(ref_set, "overall")
    save_df_as_torch(ref_set,os.path.join(args.output, f"sample_reference.pt"), args.split)
