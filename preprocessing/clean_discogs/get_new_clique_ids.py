import argparse
import json
import pandas as pd
import networkx as nx
from collections import defaultdict

clique_id = "clique_id"
version_id = "version_id"
track_title = "track_title"
track_title_cleaned = "track_title_cleaned"
track_writer_names = "track_writer_names"
track_writer_ids = "track_writer_ids"
released = "released"

def read_jsonl(path: str, k: int = None) -> list:
    """
    Read the first k lines from a JSONL file.
    """
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if k and i >= k:
                break
            results.append(json.loads(line))
    return results

def yield_clique_metadata_all(data: list):
    """
    Loop through the data and extract all tracks from all versions in each clique.
    """
    for clique in data:
        for version in clique["versions"]:
            track = version["tracks"][0]  # Assuming we only want the first track in each version
            clique_metadata = {}
            clique_metadata[clique_id] = clique[clique_id]
            clique_metadata[version_id] = version[version_id]
            clique_metadata[track_title] = track[track_title]
            clique_metadata[track_title_cleaned] = track[track_title_cleaned]
            clique_metadata[track_writer_names] = track[track_writer_names]
            clique_metadata[released] = track[released]

            yield clique_metadata

def add_normalized_writers(writers: list, cleaned_writers: dict) -> list:
    """
    Add normalized writers to the list of writers.
    """
    combined = []
    for writer in writers:
        combined.append(writer)
        if writer in cleaned_writers:
            combined.extend(cleaned_writers[writer])
    return combined

def assign_connected_components(group, track_writer_col: str = "track_writer_names2"):
    """Graph construction based on shared writers. Versions are connected if they share at least one writer.
    Based on the connected components of the graph, we assign a new column 'clique_subgroup' to each row."""
    G = nx.Graph()

    # Add nodes with their index
    for idx, row in group.iterrows():
        G.add_node(idx, writer_names=set(row[track_writer_col]))

    # Add edges if writer sets intersect
    idx_list = list(group.index)
    for i in range(len(idx_list)):
        for j in range(i + 1, len(idx_list)):
            a, b = idx_list[i], idx_list[j]
            if G.nodes[a]['writer_names'] & G.nodes[b]['writer_names']:
                G.add_edge(a, b)

    # Build mapping of row index → component ID (just 0, 1, 2, ...)
    components = list(nx.connected_components(G))
    mapping = {}
    for i, comp in enumerate(components):
        for idx in comp:
            mapping[idx] = i

    # Assign new column
    group['clique_subgroup'] = group.index.map(mapping)
    return group
           
def get_new_clique_info(df: pd.DataFrame) -> dict:
    """
    Build a nested JSON from a DataFrame based on cleaned track titles, clique IDs,
    and subgroups of writer names.
    """
    output = {}

    grouped = df.groupby("track_title_cleaned")

    for title_cleaned, group in grouped:
        clique_id = group["clique_id"].iloc[0]
        
        new_cliques = defaultdict(list)
        for _, row in group.iterrows():
            key = row["clique_id2"]
            val = row["track_writer_names2"]
            new_cliques[key].append(val)

        if len(new_cliques) >= 2:
            output[title_cleaned] = {
                "clique_id": clique_id,
                "new_cliques": dict(new_cliques)
            }

    return output

def main() -> None:
    """
    Main function to parse command-line arguments and call the appropriate function.
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Assign new clique IDs by cleaned writer names.")
    parser.add_argument('input1', type=str, nargs='?', help="Directory containing Discogs-VI-YT metadata file.")
    parser.add_argument('input2', type=str, nargs='?', help="Directory containing cleaned writers (LLM output).")

    parser.add_argument('output', type=str, help="Output JSON file to save the new cliques.")

    args = parser.parse_args()
    
    print("Reading dataset...")
    data = read_jsonl(args.input1)
    print(f"Read {len(data):,} cliques from {args.input1}.")
    
    print("Reading cleaned writers...")
    cleaned_writers = {k: v for d in read_jsonl(args.input1) for k, v in d.items()}
    
    print("Collecting unique metadata from cliques...")
    data_all = []

    for entry in yield_clique_metadata_all(data):
        data_all.append(entry)

    print(f"Collected {len(data_all):,} unique entries from {len(data):,} cliques.")

    df = pd.DataFrame(data_all)
    
    print("Add normalized writers...")
    df["track_writer_names2"] = df.track_writer_names.apply(lambda x: add_normalized_writers(x, cleaned_writers))

    # Apply per clique_id group
    print("Find connected versions...")
    df2 = df.groupby('clique_id', group_keys=False).apply(assign_connected_components)
    df2["clique_id2"] = df2["clique_id"].astype(str) + "_" + df2["clique_subgroup"].astype(str)
    print(f"Number of unique cliques increased from: {df2['clique_id'].nunique():,} to {df2['clique_id2'].nunique():,}")
    
    print(f"Writing new clique IDs to {args.output}...")
    df2[["clique_id", "version_id", "clique_subgroup", "clique_id2"]].to_json(
        args.output,   
        orient="records", 
        lines=True, 
        force_ascii=False
    )

    print("Get new clique info...")
    output = get_new_clique_info(df2)
    
    print(f"Writing new clique info to {args.output + '.info'}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    
if __name__ == "__main__":
    main()