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
    
def get_cliques_to_split(discogs_dir: str) -> dict:
    """
    Get mapping from clique_id to split.
    """
    cliques_to_split = {}
    for split in ["train", "val", "test"]:
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

def assign_cliques(new_dataset: list, cliques_to_split: dict, use_split_content: bool) -> tuple[dict, int]:
    """
    Assign new cliques to train/val/test splits and format content.
    """
    split_map = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    dropped = 0
    
    for clique in new_dataset:
        new_clique_id = clique["clique_id"]
        old_clique_id = new_clique_id.split("_")[0]
        assigned_versions = []

        for version in clique["versions"]:
            version_id = version["version_id"]

            split = cliques_to_split[old_clique_id]["split"]
            split_version = cliques_to_split[old_clique_id]["versions"].get(version_id, None)
            if use_split_content and not split_version is None:
                content = split_version
            else:
                content = version

            split_map[split][new_clique_id].append(content)
            assigned_versions.append((split, new_clique_id))

        # Drop cliques with < 2 versions in each split
        for split, cid in assigned_versions:
            if len(split_map[split][cid]) < 2:
                del split_map[split][cid]
                dropped += 1

    return split_map, dropped

def write_splits(splits: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for split, cliques in splits.items():
        path = os.path.join(output_dir, f"{BASE_FILENAME}.{split}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cliques, f, ensure_ascii=False, indent=2)
        num_cliques = len(cliques)
        num_versions = sum(len(v) for v in cliques.values())
        print(f"Saved {num_cliques:,} cliques with {num_versions:,} versions to: {path}")

def main():
    parser = argparse.ArgumentParser(description="Split the new dataset into train/val/test with formatted content.")
    parser.add_argument("input", type=str, help="Path to new dataset (JSONL format, output of regrouping).")
    parser.add_argument("discogs_dir", type=str, help="Directory containing original train/val/test files.")
    parser.add_argument("output_dir", type=str, help="Directory to save new split files.")
    parser.add_argument("--use-split-content", action="store_true", help="Use content from original split files instead of input dataset.")
    args = parser.parse_args()

    print("Loading new dataset...")
    new_dataset = read_jsonl(args.input)

    print("Assigning cliques to splits...")
    new_splits, dropped = assign_cliques(new_dataset, get_cliques_to_split(args.discogs_dir), args.use_split_content, input_version_lookup)

    print(f"Total dropped cliques (not matched or <2 versions): {dropped}")

    print("Saving split files...")
    write_splits(new_splits, args.output_dir)

if __name__ == "__main__":
    main()
