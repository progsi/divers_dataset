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

def main() -> None:
    parser = argparse.ArgumentParser(description="Reassign clique_ids without duplicating versions.")
    parser.add_argument('input', type=str, help="Original dataset in JSONL format.")
    parser.add_argument('new_clique_ids', type=str, help="New clique assignments (JSON file).")
    parser.add_argument('output', type=str, help="Output JSONL file.")
    parser.add_argument('--drop_slingleton', action='store_true',)

    args = parser.parse_args()

    print("Reading input dataset...")
    data = read_jsonl(args.input)

    print("Reading new clique ID assignments (JSON)...")
    with open(args.new_clique_ids, "r", encoding="utf-8") as f:
        new_clique_ids = json.load(f)

    # Build lookup from version_id → version
    version_map = {}
    for clique in data:
        for version in clique["versions"]:
            version_map[version["version_id"]] = version

    # Group versions by new clique_id (ignore old clique_id)
    new_cliques = defaultdict(list)

    for old_clique_id, versions_dict in new_clique_ids.items():
        for version_id, new_clique_id2 in versions_dict.items():
            version = version_map.get(version_id)
            if not version:
                continue  # skip unknown version IDs

            version_copy = version.copy()
            # Remove old clique_id from version copy, do NOT include new clique_id in version
            if "clique_id" in version_copy:
                del version_copy["clique_id"]

            new_cliques[new_clique_id2].append(version_copy)

    # Separate kept and dropped cliques
    kept = []
    dropped = []

    for new_clique_id2, versions in new_cliques.items():
        clique_entry = {
            "clique_id": new_clique_id2,
            "versions": versions
        }
        if len(versions) >= 2 or not args.drop_slingleton:
            kept.append(clique_entry)
        else:
            dropped.append(clique_entry)

    print(f"Total cliques after regrouping: {len(new_cliques):,}")
    print(f"Kept cliques: {len(kept):,}")
    print(f"Dropped cliques: {len(dropped):,}")

    # Write kept cliques to output
    with open(args.output, "w", encoding="utf-8") as f:
        for clique in kept:
            f.write(json.dumps(clique, ensure_ascii=False) + "\n")

    print(f"Written kept cliques to {args.output}")
    
    if args.drop_slingleton:
        # Write dropped cliques to separate file with '.dropped' appended
        dropped_path = args.output + ".dropped"
        with open(dropped_path, "w", encoding="utf-8") as f:
            for clique in dropped:
                f.write(json.dumps(clique, ensure_ascii=False) + "\n")
        print(f"Written dropped cliques to {dropped_path}")

if __name__ == "__main__":
    main()
