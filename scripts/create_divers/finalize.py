import argparse
import os
import re
import json
import yaml
from collections import defaultdict
from unidecode import unidecode
from pandarallel import pandarallel
import pandas as pd
import torch
from tqdm.auto import tqdm

DVI_KEYS = [
    "clique_id",
    "version_id",
    "youtube_id",
    "track_writer_names",
    "release_artist_names",
    "release_genres",
    "release_styles",
    "country",
    "labels",
    "formats",
    "released"
]

YT_KEYS = [
    "title",
    "description",
    "tags",
    "categories",
    "channel",
    "upload_date",
    "view_count"
]

def preprocess(s: str, exclude: str = "'", apostrophes: str = "'’‘`´ʻʼʽ") -> str:
    """Only retain space, latin chars and numbers. Remove attached special chars
    Args:
        s (str): 
    Returns:
        str: 
    """
    def unidecode_letters(s: str) -> str:
        def replace_with_unidecode(match):
            char = match.group(0)
            return unidecode(char)
        s = re.sub(r'[^\W\d_]', replace_with_unidecode, s)
        return s
    
    def isolate_special_chars(s: str, exclude: str) -> str:
        """Separates special chars.
        Args:
            s (str): input string
            exclude (str): string of chars to not isolate (typically ')
        Returns:
            str: string with isolated special chars
        """
        special_chars = r'([!\"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~])'.replace(exclude, "")
        s = re.sub(special_chars, r' \1 ', s)
        s = re.sub(r'\s+', ' ', s)
        s = s.strip()
        return s

    s = s.strip().lower()
    # remove apostrophe
    for a in apostrophes:
        s = s.replace(a, "")

    # to basic latin 
    s = ' '.join([unidecode_letters(w).replace(" ", "") for w in s.split()])
    # isolation
    s = isolate_special_chars(s, exclude)
    return s

def load_tags(tags_file):
    """Load a YAML file and normalize all cues to snake_case."""
    with open(tags_file, "r", encoding="utf-8") as f:
        tags = yaml.safe_load(f)

    def __preprocess(s):
        return s.lower().replace("_", " ").replace("-", " ")
        
    normalized_tags = {}
    for tag, langs_or_cues in tags.items():
        if isinstance(langs_or_cues, dict):
            normalized_tags[tag] = {
                lang: [__preprocess(c) for c in cues]
                for lang, cues in langs_or_cues.items()
            }
        elif isinstance(langs_or_cues, list):
            normalized_tags[tag] = {
                "original": [__preprocess(c) for c in langs_or_cues]
            }
    return normalized_tags

def match_tags(df, tags, name, columns):
    """
    Match tags from a YAML file to the specified DataFrame columns.

    Args:
        df (pd.DataFrame): The input dataframe.
        tags (dict): tag dict.
        name (str): Name of the tag file, used for naming the output column.
        columns (list[str]): List of column names to check for tag cues.
        preprocess (function): Function to preprocess column text.

    Returns:
        pd.DataFrame: Original DataFrame with an added `tag_matches` column.
    """
    
    def match_row(row, columns=["yt_title", "yt_description", "yt_tags"]):
        matches = defaultdict(dict)
        for col in columns:
            text = preprocess(str(row[col]))
            
            # make sure not to match song-related info rather than tags
            title = row["title"] if isinstance(row["title"], str) else ""
            release_artist_names = row["release_artist_names"] if isinstance(row["release_artist_names"], list) else []
            track_writer_names = row["track_writer_names"] if isinstance(row["track_writer_names"], list) else []
            stopwords = [title] + release_artist_names + track_writer_names
            stopwords = [preprocess(w) for w in stopwords if isinstance(w, str)]
            for tag, lang_map in tags.items():
                for cues in lang_map.values():  # all languages
                    for cue in cues:
                        if cue in text and tag not in matches[col] and cue not in stopwords:
                            matches[col][tag] = cue
                            break  # stop at first match for this tag
        return dict(matches)

    df[f"matched_{name}"] = df.parallel_apply(match_row, axis=1)
    return df

def parse_args():
    parser = argparse.ArgumentParser(description="Collect Discosgs and YouTube metadata.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the metadata PyTorch file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to the enriched output metadata PyTorch file.",
    )
    parser.add_argument(
        "--dvi",
        type=str,
        help="Path to the DVI jsonlines file.",
        default="data/dvi.jsonl", # NOTE: this is old, why do we need this again?
    )
    parser.add_argument(
        "--yt",
        type=str,
        help="Path to the YouTube parquet file.",
        default="data/yt.parquet",
    )
    parser.add_argument(
        "--tags_file",
        type=str,
        help="Path to the directory with yaml files with tags.",
        default="data/tags/tags_multilingual.yaml",
    )
    parser.add_argument(
        "--tempo_file",
        type=str,
        help="Path to the PyTorch file with extracted tempo info.",
        default=None # "data/tempo.pt",
    )
    parser.add_argument(
        "--with_onsets",
        help="Whether to include onsets in the tempo data.",
        action="store_true",
    )
    return parser.parse_args()

def filename_to_youtube_id(filename):
    """Extract YouTube ID from a filename."""
    return filename.split("/")[-1].split(".")[0]

def clean_path_keys(tempo_dict):
    """Ensure that in tempo dict the paths only contain filename and last parent dir."""
    tempo2 = {}
    for path, val in tempo_dict.items():
        parts = path.strip("/").split("/")
        if len(parts) >= 2:
            new_key = f"{parts[-2]}/{parts[-1]}"
        else:
            new_key = parts[-1]
        tempo2[new_key] = val
    return tempo2

def deduplicate_youtube_versions(df):
    # Identify duplicated combinations of youtube_id and clique_id
    duplicated_mask = df.duplicated(subset=['youtube_id', 'clique'], keep='first')

    # Drop the duplicates, keeping the first occurrence
    df_cleaned = df[~duplicated_mask].copy()

    return df_cleaned


def filter_non_singleton_cliques(df):
    """
    Filter out cliques that have only one entry.
    """
    clique_counts = df.groupby("clique").size()
    non_singleton_cliques = clique_counts[clique_counts > 1].index
    return df[df["clique"].isin(non_singleton_cliques)]

def deduplicate_false_cliques(df):
    """here we remove false clique splits where multiple cliques have the same youtube_id"""
    # Result holder for cleaned rows
    keep_rows = []
    grouped = df.groupby('youtube_id', sort=False)

    for youtube_id, group in grouped:
        if len(group) == 1:
            keep_rows.append(group)
        else:
            clique_counts = group['clique'].value_counts()
            max_count = clique_counts.max()
            top_cliques = clique_counts[clique_counts == max_count].index
            first_clique = group[group['clique'].isin(top_cliques)].iloc[0]['clique']
            filtered = group[group['clique'] == first_clique]
            keep_rows.append(filtered)

    # Concatenate all kept groups
    df_cleaned = pd.concat(keep_rows).reset_index(drop=True)

    return df_cleaned

def get_split_dict(df):
    """Get the splitdict from the DataFrame."""
    split_dict = {}
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            clique, version = idx.split(":")
        except ValueError:
            raise ValueError(f"Invalid index format: '{idx}' (expected 'clique:version')")
        split = row["split"]
        split_dict.setdefault(split, {}).setdefault(clique, []).append(version)
    return split_dict

def print_df_summary(df, subset="overall"):
    if subset != "overall":
        df = df.query(f"split == '{subset}'")
    print(f"Summary statistics for {subset}")

    # Column stats
    stats = []
    for col in ["clique", "version", "youtube_id"]:
        unique_count = df[col].nunique()
        total_count = df[col].count()
        stats.append(f"{col:<12} | Unique: {unique_count:>7,} | Total: {total_count:>7,}")
    print("\n".join(stats))

    # Version per clique stats
    version_per_clique = df.groupby("clique")["version"].nunique()
    print("\nVersion per clique:")
    print(f"  Min: {version_per_clique.min():>3}  | Max: {version_per_clique.max():>3}  | "
          f"Mean: {version_per_clique.mean():>5.2f}  | Median: {version_per_clique.median():>5.2f}  | "
          f"Std: {version_per_clique.std():>5.2f}")
    print("=" * 40)
    
def main() -> None:
    args = parse_args()

    # Dataset metadata
    input_data = torch.load(args.input)
    if isinstance(input_data, dict) and "info" in input_data and "split" in input_data:
        metadata = input_data["info"]
        split = input_data["split"]
    else:
        metadata, split = input_data
    df = pd.DataFrame(metadata).T
    df["youtube_id"] = df["filename"].apply(filename_to_youtube_id)
    print(f"Loaded metadata with {len(df)} entries.")

    # YouTube 
    df_yt = pd.read_parquet(args.yt)
    print(f"Loaded YouTube metadata with {len(df_yt)} entries.")

    # Discogs
    df_dvi = pd.read_json(args.dvi, lines=True)
    print(f"Loaded Discogs metadata with {len(df_dvi)} entries.")

    # extracted tempo info
    has_tempo = os.path.exists(args.tempo_file) if args.tempo_file else False
    if has_tempo:
        try:
            pt_tempo = torch.load(args.tempo_file)
            print(f"Loaded tempo data with {len(pt_tempo)} entries.")
            pt_tempo = clean_path_keys(pt_tempo)
            df_tempo = pd.DataFrame.from_dict(pt_tempo).T.reset_index().rename(
                columns={"index": "filename"}
            )
            df_tempo["youtube_id"] = df_tempo["filename"].apply(filename_to_youtube_id)
            
            if args.with_onsets:
                df_tempo["onset_env"] = df_tempo["onset_env"].apply(lambda x: torch.stack(x) if isinstance(x, list) else x)
            else:
                df_tempo = df_tempo.drop(columns=["onset_env"])
            
        except Exception as e:
            print(f"Error loading tempo data: {e}")
            has_tempo = False
            df_tempo = pd.DataFrame([])
    else:
        pt_tempo = pd.DataFrame([])

    tqdm.pandas()
    
    print("Joining Discogs metadata...")
    df = df.merge(
        df_dvi[DVI_KEYS],
        how="left",
        left_on=["version", "youtube_id"],
        right_on=["version_id", "youtube_id"],
    ).progress_apply(lambda x: x)
    
    print("Joining YouTube metadata...")
    df = df.merge(
        df_yt[YT_KEYS].add_prefix("yt_"),
        how="left",
        on="youtube_id",
    ).progress_apply(lambda x: x)
    
    if has_tempo:
        print("Joining extracted tempo information...")
        df = df.merge(
            df_tempo.drop(columns=["filename"]),
            how="left",
            on="youtube_id",
        ).progress_apply(lambda x: x)
        
    # Match tags from YAML files
    pandarallel.initialize(progress_bar=True)
    print("Matching tags from YAML files...")
    print(f"Processing tags file: {args.tags_file}")
    tags = load_tags(args.tags_file)
        
    df = match_tags(df, tags, 
                        name=args.tags_file.split(os.sep)[-1].split('.')[0],
                        columns=["yt_title", "yt_description", "yt_tags"])
    
    # Convert DataFrame to list of dicts and save as torch file
    df["cid:vid"] = df.apply(lambda row: row["clique"] + ':' + row["version"], axis=1)
    df = df.drop_duplicates(subset=["cid:vid", "youtube_id"])
    df = df.drop(columns=["clique_id", "version_id"], errors='ignore')
     # final cleanups
    df = deduplicate_youtube_versions(df)
    df = filter_non_singleton_cliques(df)
    
    metadata_dicts = df.set_index("cid:vid").to_dict(orient="index")
    
    torch.save({
        "info": metadata_dicts,
        "split": split
    }, args.output)
    print(f"Enriched metadata saved to {args.output}.")
    print_df_summary(df, subset="overall")

if __name__ == "__main__":
    main()