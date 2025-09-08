"""This script serves to extract unique titles 
from the dataset subsets (train, val, test) of Discogs-VI-YT.
"""
import os
import json
import argparse

def collect_unique_titles(data: dict, all: bool = False) -> tuple:
    """Collect unique track titles from dataset.
    Args:
        data (dict): dataset in dictionary
        all (bool, optional): Whether to collect all unique titles, or solely first. 
            Defaults to False.
    Returns:
        tuple: dataset with unique titles, count of unique titles
    """
    result = {}
    if all:
        for clique_id, versions in data.items():
            result[clique_id] = []
            for version in versions:
                # add version
                if isinstance(version, dict) and 'track_title' in version:
                    track_title = version['track_title']
                    if not track_title in result[clique_id]:
                        result[clique_id].append(track_title)
        count = sum(len(titles) for titles in result.values())
    else:
        for clique_id, versions in data.items():
            result[clique_id] = versions[0]['track_title']
        count = len(result)
    return result, count

def process_file(filepath: str, all: bool = False) -> None:
    """
    Processes single file.
        :param filepath: path to JSON file.
        :param output_file: Output file to save the result as a JSON.
    """
    # Prepare a dictionary to store results       
    print(f"Processing file: {filepath}")
    
    # Load and print
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cliques = len(data)
    print(f"Number of cliques in {filepath}: {cliques}")

    # Collect per clique
    file = os.path.basename(filepath)
    result, count = collect_unique_titles(data, all)

    print(f"Unique track titles in {file}: {count}")
    return result


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
    
    if os.path.isdir(args.input):
        result = {}
        for file in os.listdir(args.input):
            result = result | process_file(os.path.join(args.input, file), args.all) 
    else:
        result = process_file(args.input, args.all)
    
    with open(args.output, "w") as f:
        json.dump(result, f)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
