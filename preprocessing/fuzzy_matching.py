import argparse
from typing import List
import json
import pandas as pd
from tqdm import tqdm

from rapidfuzz import process, fuzz

from string_processor import StringProcessor

TITLES_KEY = "track_title_cleaned"
ARTISTS_KEY = "release_artist_names"

YOUTUBE_RESULTS_PATH = "data/metadata_filtered.jsonl"
QUERIES_PATH = "data/queries_filtered.json"

def discogs_to_dict(discogs_path: str) -> dict:
    dataset = {}
    with open(discogs_path, "r") as f:
        for line in f:
            clique = json.loads(line)
            clique_id = clique.pop("clique_id")
            dataset[clique_id] = clique
    return dataset

def read_json_lines(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    return data
    
def read_json(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def get_unique(clique: dict, key: str) -> List[str]:
    """
    Get values from dataset, given the clique_id.
    """
    result = []
    for version in clique["versions"]:
        for track in version["tracks"]:
            if isinstance(track[key], list) and key in track:
                result.extend(track[key])
            else:
                result.append(track[key])

    # make unique
    result = list(set(result))
    
    return result

def get_table(candidates: List[dict], dataset: dict, 
              title_to_clique: dict, queries: dict, max_index: int) -> List[tuple]:
    
    table = []
    for youtube_metadata in tqdm(candidates):
        yt_id = youtube_metadata["id"]
        video_title = youtube_metadata["title"]
        description = youtube_metadata.get("descriptionSnippet", "")
        
        if isinstance(description, list):
            description = " ".join([d["text"] for d in description]).replace("\n", " ").replace("\r", " ")
        
        for query, result_index in queries[yt_id].items():
            if result_index < max_index:
                # get the clique_id
                clique_id = title_to_clique[query]
                # get the metadata
                clique_metadata = dataset[clique_id]
                
                # compare title
                titles = get_unique(clique_metadata, TITLES_KEY)
                for title in titles:
                    table.append(
                        (yt_id, clique_id, "video_title", "track_title_cleaned", video_title,  title)
                    )
                    table.append(
                        (yt_id, clique_id, "description", "track_title_cleaned", description,  title)
                    )
                
                # compare all artists
                artists = get_unique(clique_metadata, ARTISTS_KEY)
                for artist in artists:
                    table.append(
                        (yt_id, clique_id, "video_title", ARTISTS_KEY, video_title,  artist)
                    )
                    table.append(
                        (yt_id, clique_id, "description", ARTISTS_KEY, description,  artist)
                    )
                
    return table

def main(discogs_path, unique_titles_path, output, max_index):
    # Your logic will go here
    print("Loading data...")
    print(f"...Discogs path: {discogs_path}")
    dataset = discogs_to_dict(discogs_path)
    
    print(f"...YouTube videos: {YOUTUBE_RESULTS_PATH}")
    youtube_videos = read_json_lines(YOUTUBE_RESULTS_PATH)
    
    print(f"...auxiliary files...")
    title_to_clique =  {v.replace(" ", "_"): k for k, v in read_json(unique_titles_path).items()}
    queries = read_json(QUERIES_PATH)
    
    print("Loaded.")
    print("Create table...")
    table = get_table(youtube_videos, dataset, title_to_clique, queries, max_index)
    print(f"Created table with {len(table)} rows.")
    
    columns = ["youtube_id", "clique_id", "youtube_attr", "discogs_attr", "youtube_text", "discogs_text"]
    df = pd.DataFrame(table, columns=columns)
    
    print(f"Fuzzy matching with {fuzz.token_ratio.__name__}...")    
    result = process.cpdist(df.youtube_text, df.discogs_text, 
               scorer=fuzz.token_ratio, 
               processor=StringProcessor(),
               workers=-1,)
    df["Score"] = result
    print("Done.")
    
    print("Saving results...")
    df = df.drop(columns=["youtube_text"])
    df.to_csv(output, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description here.")

    # Positional arguments
    parser.add_argument("discogs_path", help="Path to the Discogs data file")
    parser.add_argument("unique_titles_path", help="Path to the unique titles file")
    parser.add_argument("output", help="Path to save the output")

    # Optional argument
    parser.add_argument("--max_index", type=int, default=100, help="Maximum result index")

    args = parser.parse_args()

    main(
        discogs_path=args.discogs_path,
        unique_titles_path=args.unique_titles_path,
        output=args.output,
        max_index=args.max_index
    )
