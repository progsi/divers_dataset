import argparse
import json
from collections import defaultdict

def read_jsonl(path: str, k: int = None) -> list:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if k and i >= k:
                break
            results.append(json.loads(line))
    return results

def map_versions_by_id(data: list) -> dict:
    """
    Flatten all versions into a dict: version_id → full version metadata
    """
    version_map = {}
    for clique in data:
        for version in clique["versions"]:
            version_map[version["version_id"]] = version
    return version_map

def main() -> None:
    parser = argparse.ArgumentParser(description="Restructure dataset using new clique_id2 values.")
    parser.add_argument('input', type=str, help="Original dataset in JSONL format.")
    parser.add_argument('new_clique_ids', type=str, help="New clique assignments (JSONL).")
    parser.add_argument('output', type=str, help="Output JSONL file.")

    args = parser.parse_args()

    print("Reading input dataset...")
    data = read_jsonl(args.input)
    print("Reading new clique ID assignments...")
    new_clique_ids = read_jsonl(args.new_clique_ids)

    # Step 1: Build lookup of version_id → full version metadata
    version_map = map_versions_by_id(data)

    # Step 2: Reorganize by clique_id2
    new_cliques = defaultdict(list)
    for entry in new_clique_ids:
        version_id = entry["version_id"]
        clique_id2 = entry["clique_id2"]
        version = version_map.get(version_id)

        if version:
            version_copy = version.copy()
            version_copy["clique_id2"] = clique_id2
            if "clique_subgroup" in entry:
                version_copy["clique_subgroup"] = entry["clique_subgroup"]
            new_cliques[clique_id2].append(version_copy)

    # Step 3: Separate kept and dropped cliques
    kept = []
    dropped = []

    for clique_id2, versions in new_cliques.items():
        clique_obj = {
            "clique_id": clique_id2,
            "versions": versions
        }
        if len(versions) >= 2:
            kept.append(clique_obj)
        else:
            dropped.append(clique_obj)

    print(f"Total cliques after regrouping: {len(new_cliques):,}")
    print(f"Kept: {len(kept):,}, Dropped: {len(dropped):,}")

    # Step 4: Write to output files
    with open(args.output, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    dropped_path = args.output.replace(".jsonl", ".dropped.jsonl")
    with open(dropped_path, "w", encoding="utf-8") as f:
        for item in dropped:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Written kept cliques to {args.output}")
    print(f"Written dropped cliques to {dropped_path}")

if __name__ == "__main__":
    main()
