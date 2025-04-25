"""This script serves to extract unique titles from the file Discogs-VI-YT-*.jsonl.
Requires the key "track_title_cleaned". 
"""
import os
import json
import argparse


def read_json_lines(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield json.loads(line)
            
            
def main() -> None:
    """
    Main function to parse command-line arguments and call the appropriate function.
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Collect unique track titles from JSON files.")
    parser.add_argument('input', type=str, nargs='?', help="Directory containing JSON files (default: 'data/discogs')")
    parser.add_argument('output', type=str, help="Output JSON file to save the results.")
    parser.add_argument('--all', action='store_true', help="Process all JSON files in the directory.")

    args = parser.parse_args()

    assert os.path.exists(args.input), f"Input path {args.input} does not exist."
    
    count_overall = 0
    result = {}
    for line in read_json_lines(args.input):
        clique_id = line['clique_id']
        if not clique_id in result:
            result[clique_id] = []
        
        if args.all:
            for version in line['versions']:
                # add version
                if isinstance(version, dict) and version["tracks"][0].get('track_title_cleaned', False):
                    track_title = version['track_title_cleaned']
                    if not track_title in result[clique_id]:
                        result[clique_id].append(track_title)
            count = sum(len(titles) for titles in result[clique_id].values())
        else:
            track_title = line['versions'][0]["tracks"][0].get('track_title_cleaned', None)
            if track_title:
                result[clique_id] = track_title
            count = 1
        count_overall += count
        print(f"Unique track titles in total: {count_overall}")
    
    with open(args.output, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
