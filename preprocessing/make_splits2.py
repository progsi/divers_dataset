import argparse
import json
import os
from collections import defaultdict

BASE_FILENAME = "Discogs-VI-YT-20240701-light.json"

def read_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def read_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def mp4_file_exists(youtube_id: str, dir: str) -> bool:
    """
    Check if the file for the given YouTube ID exists.
    """
    return os.path.exists(os.path.join(dir, youtube_id[:2], f"{youtube_id}.mp4"))

def get_cliques_to_split(discogs_dir: str) -> dict:
    """
    Get mapping from clique_id to split.
    """
    cliques_to_split = {}
    for split in ["train", "valid", "test"]:
        filename = f"{BASE_FILENAME}.{split}"
        path = os.path.join(discogs_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected file '{filename}' not found in {discogs_dir}")
        split_cliques = read_json(path)
        for clique_id, versions in split_cliques.items():
            cliques_to_split[clique_id] = {
                "split": split,
                "versions": {v["version_id"]: v for v in versions}
            }
    return cliques_to_split

def assign_cliques(new_dataset: list, cliques_to_split: dict, use_split_content: bool, mp4_dir: str) -> tuple[dict, int]:
    """
    Assign new cliques to train/val/test splits and format content.
    """
    split_map = {"train": defaultdict(list), "valid": defaultdict(list), "test": defaultdict(list)}
    dropped_singleton = 0
    dropped_not_downloaded = 0
    
    for clique in new_dataset:
        new_clique_id = clique["clique_id"]
        old_clique_id = new_clique_id.split("_")[0]
        assigned_versions = []

        for version in clique["versions"]:
            version_id = version["version_id"]

            split = cliques_to_split[old_clique_id]["split"]
            if use_split_content:
                content = cliques_to_split[old_clique_id]["versions"].get(version_id, None)
                # this case should cover the new versions that are not in the original splits
                if not content:
                    content = {
                        "version_id": version_id, 
                        "track_title": None,
                        "youtube_id": version_id,
                        }
            else:
                content = version
            
            if mp4_dir and not mp4_file_exists(content["youtube_id"], mp4_dir):
                dropped_not_downloaded += 1
                continue

            split_map[split][new_clique_id].append(content)
            assigned_versions.append((split, new_clique_id))

        # Drop cliques with < 2 versions in each split
        for split, cid in assigned_versions:
            if len(split_map[split][cid]) < 2:
                del split_map[split][cid]
                dropped_singleton += 1

    return split_map, dropped_singleton, dropped_not_downloaded

def write_splits(splits: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for split, cliques in splits.items():
        path = os.path.join(output_dir, f"{split}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cliques, f, ensure_ascii=False, indent=2)
        num_cliques = len(cliques)
        num_versions = sum(len(v) for v in cliques.values())
        print(f"Saved {num_cliques:,} cliques with {num_versions:,} versions to: {path}")

def main():
    parser = argparse.ArgumentParser(description="Split the new dataset into train/val/test with formatted content.")
    parser.add_argument("input", type=str, 
                        help="Path to new dataset (JSONL format, output of regrouping).")
    parser.add_argument("discogs_dir", type=str, 
                        help="Directory containing original train/val/test files.")
    parser.add_argument("output_dir", type=str, 
                        help="Directory to save new split files.")
    parser.add_argument("--use-split-content", action="store_true", 
                        help="Use content from original split files instead of input dataset.")
    parser.add_argument("--mp4-dir", type=str, default=None,
                        help="Directory of mp4 files.")
    args = parser.parse_args()

    assert args.mp4_dir is None or os.path.exists(args.mp4_dir), f"Input file {args.mp4_dir} does not exist."
    
    print("Loading new dataset...")
    new_dataset = read_jsonl(args.input)

    print("Assigning cliques to splits...")
    new_splits, nsingleton, nmissing = assign_cliques(new_dataset, 
                                         get_cliques_to_split(args.discogs_dir), 
                                         args.use_split_content,
                                         args.mp4_dir)

    print(f"Total dropped cliques (not matched or <2 versions): {nsingleton}")
    print(f"Total dropped versions (not downloaded): {nmissing}")

    print("Saving split files...")
    write_splits(new_splits, args.output_dir)

if __name__ == "__main__":
    main()
