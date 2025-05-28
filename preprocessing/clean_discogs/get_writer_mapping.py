import argparse
from typing import List, Dict
import json
import copy
from tqdm import tqdm

FEAT_TOKEN = " [FEAT]"

def read_json_lines(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

def get_writer_dict(data: List[dict]) -> dict:
    writers = {}

    for clique in tqdm(data):
        version = clique["versions"][0]
        track = version["tracks"][0]
        writer = " [FEAT] ".join(track["track_writer_names"])

        metadata_light = {
                "clique_id": clique["clique_id"],
                "version_id": version["version_id"],
                "track_title": track["track_title"],
                "track_title_cleaned": track["track_title_cleaned"],
                "released": track["released"],
            }
        if writer not in writers:
            writers[writer] = [metadata_light]
        else:
            writers[writer].append(metadata_light)

    return writers

def get_contribution_map(writer_map: dict, feat_token: str) -> dict:
    """Collects all contribution versions for individual writers,
    avoiding shared references and nested lists."""
    contr_map = {}
    for writer_feat, cliques in writer_map.items():
        writers = writer_feat.split(feat_token)
        for clique in cliques:
            for writer in writers:
                writer = writer.strip()
                if writer not in contr_map:
                    contr_map[writer] = [clique]
                else:
                    contr_map[writer].append(clique)
    return contr_map

def main() -> None:
    """
    Main function to parse command-line arguments and call the appropriate function.
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Get dict mapping from writers to cliques.")
    parser.add_argument('input', type=str, nargs='?', help="Directory containing Discogs-VI-YT metadata file.")
    parser.add_argument('output1', type=str, help="Output JSON file to save the writer map.")
    parser.add_argument('output2', type=str, help="Output JSON file to save the contribution map.")


    args = parser.parse_args()
    
    print(f"Reading data from {args.input}...")
    data = read_json_lines(args.input)
    
    writer_dict = get_writer_dict(data)
    with open(args.output1, "w") as f:
        json.dump(writer_dict, f, indent=4)
    print(f"Writer dict saved to {args.output1}")
    contribution_map = get_contribution_map(writer_dict, FEAT_TOKEN)
    with open(args.output2, "w") as f:
        json.dump(contribution_map, f, indent=4)
    print(f"Contribution map saved to {args.output2}")
    
if __name__ == "__main__":
    main()