import argparse
import numpy as np
import pandas as pd
import os
import json
import torch
from pathlib import Path

# ----------------------------
# Load / Save functions
# ----------------------------
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
    print(f"Saved torch file to {out_path} ({len(df)} rows)")

# ----------------------------
# TAG LOGIC
# ----------------------------
tag_aliases = {
    "instrumental": ["instrumental", "karaoke"],
    "reaction": ["firsttimehearing", "firsttimereaction", "reaction", "reactsto", "reactto"],
    "tutorial": ["howtoplay", "howtosing", "lesson", "tutorial"],
}

tag_requirements = [
    #["acoustic"],
    #["instrumental"],
    ["live"],
    #["orchestra"],
    ["reaction"],
    ["cover"],
    ["tutorial"],
]

bins = np.arange(0.2, 1.0, 0.2)

def has_tags(row_tags, required_tags):
    row_tags = set(row_tags)
    for tag in required_tags:
        aliases = tag_aliases.get(tag, [tag])
        if not any(alias in row_tags for alias in aliases):
            return False
    return True

# ----------------------------
# MAIN FUNCTION
# ----------------------------
def main(input_path: str, output_dir: str):
    SAVE_DIR = Path(output_dir)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    df, _ = load_dataset(input_path)
    df_test = df[df["split"] == "test"]
    df_test["music_ratio"] = df_test["music_segment_inds"].apply(lambda x: sum(x)/len(x))

    # ----------------------------
    # FIND VALID CLIQUES
    # ----------------------------
    valid_cliques = []
    for clique, df_c in df_test.groupby("clique"):
        df_non_dvi = df_c[df_c["dvi"] == False]
        if len(df_non_dvi) == 0 or not all((df_non_dvi["music_ratio"] <= b).any() for b in bins):
            continue

        df_dvi = df_c[df_c["dvi"] == True]
        if len(df_dvi) < 2:
            continue

        tags_list = df_c["tags_yt_title"].tolist()
        if not all(any(has_tags(tags, req) for tags in tags_list) for req in tag_requirements):
            continue

        valid_cliques.append(clique)

    print(f"Valid cliques (will always be in subsets): {len(valid_cliques)}")

    # ----------------------------
    # AGGREGATE SUBSETS
    # ----------------------------
    all_reference_rows = []
    all_tag_rows = {tuple(req): [] for req in tag_requirements}
    all_ratio_rows = {b: [] for b in bins}
    all_divers_tag_rows = []
    all_divers_exclusive_rows = []

    for clique in valid_cliques:
        df_c = df_test[df_test["clique"] == clique]
        df_dvi = df_c[(df_c["dvi"] == True) & (df_c["music_ratio"] >= 0.9)]
        df_non_dvi = df_c[df_c["dvi"] == False]

        anchor_row = df_dvi.sample(1).iloc[0]

        # Reference
        ref_candidates = df_dvi.drop(anchor_row.name)
        if len(ref_candidates) > 0:
            ref_row = ref_candidates.sample(1).iloc[0]
            all_reference_rows.extend([anchor_row, ref_row])

        # Tag subsets
        for req in tag_requirements:
            candidates = df_c[df_c["tags_yt_title"].apply(lambda tags: has_tags(tags, req))].drop(anchor_row.name, errors="ignore")
            if len(candidates) == 0:
                continue
            row = candidates.sample(1).iloc[0]
            all_tag_rows[tuple(req)].extend([anchor_row, row])

        # Ratio subsets
        for b in bins:
            candidates = df_non_dvi[df_non_dvi["music_ratio"] <= b].drop(anchor_row.name, errors="ignore")
            if len(candidates) == 0:
                continue
            row = candidates.sample(1).iloc[0]
            all_ratio_rows[b].extend([anchor_row, row])

        # Diversity subsets
        divers_tag_candidates = df_non_dvi[df_non_dvi["tags_yt_title"].apply(lambda t: len(t) > 0)].drop(anchor_row.name, errors="ignore")
        if len(divers_tag_candidates) > 0:
            row = divers_tag_candidates.sample(1).iloc[0]
            all_divers_tag_rows.extend([anchor_row, row])

        divers_exclusive_candidates = df_non_dvi[df_non_dvi["tags_yt_title"].apply(lambda t: len(t) == 0)].drop(anchor_row.name, errors="ignore")
        if len(divers_exclusive_candidates) > 0:
            row = divers_exclusive_candidates.sample(1).iloc[0]
            all_divers_exclusive_rows.extend([anchor_row, row])

    # ----------------------------
    # SAVE AGGREGATED SUBSETS
    # ----------------------------
    if all_reference_rows:
        save_df_as_torch(pd.DataFrame(all_reference_rows), SAVE_DIR / f"all_reference.pt", split="test")
    for req, rows in all_tag_rows.items():
        if rows:
            save_df_as_torch(pd.DataFrame(rows), SAVE_DIR / f"all_tag_{'_'.join(req)}.pt", split="test")
    for b, rows in all_ratio_rows.items():
        if rows:
            save_df_as_torch(pd.DataFrame(rows), SAVE_DIR / f"all_ratio_le_{b:.1f}.pt", split="test")
    if all_divers_tag_rows:
        save_df_as_torch(pd.DataFrame(all_divers_tag_rows), SAVE_DIR / f"all_divers_tag.pt", split="test")
    if all_divers_exclusive_rows:
        save_df_as_torch(pd.DataFrame(all_divers_exclusive_rows), SAVE_DIR / f"all_divers_exclusive.pt", split="test")


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save Torch subsets")
    args = parser.parse_args()
    main(args.input, args.output_dir)