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
    Flatten all versions into a dict: version_id → version metadata
    """
    version_map = {}
    for clique in data:
        for version in clique["versions"]:
            version_map[version["version_id"]] = version
    return version_map

def main() -> None:
    parser = argparse.ArgumentParser(description="Reassign clique_ids without duplicating versions.")
    parser.add_argument('input', type=str, help="Original dataset in JSONL format.")
    parser.add_argument('new_clique_ids', type=str, help="New clique assignments (JSONL).")
    parser.add_argument('output', type=str, help="Output JSONL file.")

    args = parser.parse_args()

    print("Reading input dataset...")
    data = read_jsonl(args.input)
    print("Reading new clique ID assignments...")
    new_clique_ids = read_jsonl(args.new_clique_ids)

    # Step 1: Build lookup from version_id → version
    version_map = map_versions_by_id(data)

    # Step 2: Build mapping from version_id → new clique_id2
    version_to_clique = {}
    for entry in new_clique_ids:
        version_to_clique[entry["version_id"]] = {
            "clique_id2": entry["clique_id2"],
            "clique_subgroup": entry.get("clique_subgroup")
        }

    # Step 3: Group versions by old clique_id, then by new clique_id2
    # We'll create a nested dict: old_clique_id -> new_clique_id2 -> list of versions
    regrouped = defaultdict(lambda: defaultdict(list))

    for version_id, assignment in version_to_clique.items():
        version = version_map.get(version_id)
        if not version:
            continue  # skip unknown version IDs
        old_clique_id = version["clique_id"]  # old clique id from original data
        version_copy = version.copy()
        version_copy["clique_id"] = assignment["clique_id2"]  # update to new clique id
        if assignment["clique_subgroup"] is not None:
            version_copy["clique_subgroup"] = assignment["clique_subgroup"]
        regrouped[old_clique_id][assignment["clique_id2"]].append(version_copy)

    # Step 4: Build final output structure like the old dataset,
    # but with versions regrouped and sorted by new clique_id2
    output_cliques = []

    for old_clique_id, new_cliques_dict in regrouped.items():
        # Flatten versions for this old clique, sorting by new clique_id2
        versions_sorted = []
        for new_clique_id2 in sorted(new_cliques_dict.keys()):
            versions_sorted.extend(new_cliques_dict[new_clique_id2])

        clique_entry = {
            "clique_id": old_clique_id,
            "versions": versions_sorted
        }
        output_cliques.append(clique_entry)

    print(f"Rebuilt {len(output_cliques):,} cliques with regrouped versions.")

    # Step 5: Write output to JSONL file
    with open(args.output, "w", encoding="utf-8") as f:
        for clique in output_cliques:
            f.write(json.dumps(clique, ensure_ascii=False) + "\n")

    print(f"Written regrouped cliques to {args.output}")

if __name__ == "__main__":
    main()
