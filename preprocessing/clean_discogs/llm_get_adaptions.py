import argparse
import json
from typing import List
from tqdm import tqdm
import pandas as pd
import re
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import RichPromptTemplate

template_str = """You are an expert in music metadata. Your task is to find alternative song titles (e.g. translations, parodies etc.) given a song title and the composer(s). 
Please return the alternative titles as a list of strings, even if there is only one alternative title. If there is none, return an empty list. Return as a JSON mapping the given song title to the alternative song titles.
---------------------
Written by: {{ written_by }}
Song Title: {{ track_title }}
---------------------
"""

LLMs = {
    "qwen": "qwen3:30b",
    "llama": "llama3.3:latest",
}

track_writer_names = "track_writer_names"
track_title = "track_title"
track_title_cleaned = "track_title_cleaned"
track_writer_ids = "track_writer_ids"
release_artist_names = "release_artist_names"
clique_id = "clique_id"
version_id = "version_id"
released = "released"

def read_json_lines(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

def yield_clique_metadata_unique(data: list):
    """
    Loop through the data and yield unique tracks per clique.
    A unique track is defined as a combination where either the writers or the cleaned track title do not match any previous track in the same clique.
    """
    for clique in data:
        seen = set()
        for version in clique["versions"]:
            for track in version["tracks"]:
                writers_tuple = tuple(sorted(track[track_writer_names]))
                cleaned_title = track[track_title_cleaned]
                key = (writers_tuple, cleaned_title)
                if key not in seen:
                    seen.add(key)
                    clique_metadata = {}
                    clique_metadata[clique_id] = clique[clique_id]
                    clique_metadata[version_id] = version[version_id]
                    clique_metadata[track_title] = track[track_title]
                    clique_metadata[track_title_cleaned] = cleaned_title
                    clique_metadata[track_writer_names] = track[track_writer_names]
                    clique_metadata[track_writer_ids] = track[track_writer_names]
                    clique_metadata[release_artist_names] = track[release_artist_names]
                    clique_metadata[released] = track[released]
                    yield clique_metadata
                    
def filter_writers_special_chars(df_explode: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out rows where 'track_writer_names' contains special characters.
    """
    latin_special_chars = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    escaped_chars = re.escape(latin_special_chars)

    # Build regex pattern: any one of these chars, e.g. [!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]
    pattern = f"[{escaped_chars}]"

    # Filter rows where 'your_text_column' contains any special char
    filtered_rows = df_explode[df_explode['track_writer_names'].str.contains(pattern, regex=True, na=False)]
    return filtered_rows

      
def main() -> None:
    """
    Main function to parse command-line arguments and call the appropriate function.
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Normalize writer names with an LLM.")
    parser.add_argument('input', type=str, nargs='?', help="Directory containing Discogs-VI-YT metadata file.")
    parser.add_argument('output', type=str, help="Output JSON file to save the writer map.")
    parser.add_argument('--llm', type=str, choices=["qwen", "llama"], 
                        help="LLM to use for normalization. Might require rewriting LLMs variable in this script.")
    args = parser.parse_args()
    
    print(f"Reading data from {args.input}...")
    data = read_json_lines(args.input)
    
    print("Collecting unique metadata from cliques...")
    data_unique = []
    for entry in yield_clique_metadata_unique(data):
        data_unique.append(entry)

    print(f"Collected {len(data_unique):,} unique entries from {len(data):,} cliques.")

    df = pd.DataFrame(data_unique)
    df.track_writer_names = df.track_writer_names.apply(lambda x: ', '.join([name.strip() for name in x if isinstance(name, str)]))
        
    df = df.drop_duplicates(subset=["track_writer_names"], keep="first")

    print(f"{df.shape[0]:,} rows.")

    template = RichPromptTemplate(template_str)
    llm = Ollama(model=LLMs[args.llm], request_timeout=120.0, json_mode=True)
    print(f"Loaded LLM: {args.llm}")

    llm_output = []
    log = []
    
    for row in tqdm(df.itertuples(index=False), total=df.shape[0], desc="Processing track writers"):
        cid = row.clique_id
        written_by = row.track_writer_names
        track_title = row.track_title_cleaned
        prompt = template.format(written_by=written_by, track_title=track_title)
        if args.llm == "qwen":
            prompt = prompt + "/nothink"
        try:
            resp = llm.complete(
            prompt=prompt, 
            response_model="json", 
            temperature=0.0, 
            max_tokens=1024, 
            stop=["\n\n"], 
            stream=False
            )
            out = json.loads(resp.text)
            out["clique_id"] = cid
            out["track_title_cleaned"] = track_title
            out["written_by"] = written_by
            respd = json.loads(resp.text)
            out["llm_resp"] = list(respd.values())[0]
            llm_output.append(out)
        except Exception as e:
            log.append(f"Error processing {written_by} for track {track_title}: {e}")
            continue
    
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in llm_output:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved adaptions to {args.output}")
    
    log_path = args.output + '.log'
    with open(log_path, "w", encoding="utf-8") as f:
        for entry in log:
            f.write(entry + "\n")
    print(f"Saved log to {log_path}")
    
if __name__ == "__main__":
    main()