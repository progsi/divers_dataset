import os
from typing import List
import json
import random
import requests
from tqdm import tqdm


BASE_URL = "https://api.secondhandsongs.com/search/"

headers = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0 Safari/537.36"
    # "X-API-Key": "ed0a8c61-4520-4317-9b95-7a260260b4c"  # Replace with your API key if needed
}


def read_json_lines(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

def get_random_proxy_from_file(file_path: str = "../proxies.txt") -> str:
    """
    Get a random proxy from a file.
    """
    with open(file_path, "r") as f:
        proxies = f.readlines()
    proxy = random.choice(proxies).strip()
    return f"http://{proxy}"

def get_random_proxy(file_path: str = "../proxies.txt") -> str:
    proxy_url = get_random_proxy_from_file(file_path)
    return {
        "http": proxy_url,
        # "https": proxy_url.replace("http://", "https://"),
    }

def get_search_url_performance(title: str, performer: str) -> str:
    title, performer = title.replace(" ", "%20"), performer.replace(" ", "%20")
    return BASE_URL + f"performance?op_title=contains&title={title}&op_performer=contains&performer={performer}&sort=simplifiedTitle&reverse=0&format=json"

def search_performance(title: str, performer: str, proxies: dict) -> dict:
    url = get_search_url_performance(title, performer)
    response = requests.get(url, headers=headers, proxies=proxies)
    return response.status_code, json.loads(response.text)

def get_performance(perf_id: int, proxies: dict) -> dict:
    url = f"https://api.secondhandsongs.com/performance/{perf_id}"
    response = requests.get(url, headers=headers)
    return response.status_code, json.loads(response.text)

def yield_clique_metadata_unique(data: list):
    """
    Loop through the data and yield unique tracks per clique.
    A unique track is defined as a combination where either the writers or the cleaned track title do not match any previous track in the same clique.
    """
    for clique in data:
        seen = set()
        for version in clique["versions"]:
            for track in version["tracks"]:
                writers_tuple = tuple(sorted(track["track_writer_names"]))
                cleaned_title = track["track_title_cleaned"]
                key = (writers_tuple, cleaned_title)
                if key not in seen:
                    seen.add(key)
                    clique_metadata = {}
                    clique_metadata["clique_id"] = clique["clique_id"]
                    clique_metadata["version_id"] = version["version_id"]
                    clique_metadata["track_title_cleaned"] = cleaned_title
                    clique_metadata["track_writer_names"] = track["track_writer_names"]
                    clique_metadata["track_writer_ids"] = track["track_writer_names"]
                    clique_metadata["release_artist_names"] = track["release_artist_names"]
                    clique_metadata["released"] = track["released"]
                    yield clique_metadata
                    
                    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Find SHS performance data.")
    parser.add_argument("dataset", type=str, help="Path to the input dataset file.")
    parser.add_argument("output", type=str, help="Output filepath.")
    return parser.parse_args()

def main():
    
    args = parse_args()
    
    # Read the input dataset
    data = read_json_lines(args.dataset)
    
    unique_cliques = list(yield_clique_metadata_unique(data))
    
    shs_enriched_cliques = []
    
    for clique in tqdm(unique_cliques, total=len(unique_cliques), desc="Searching SHS..."):
        clique_id = clique["clique_id"]
        version_id = clique["version_id"]
        title = clique["track_title_cleaned"].lower()
        performer = clique["release_artist_names"][0].lower() 
        
        collector = {
            "clique_id": clique_id,
            "version_id": version_id,
            "title": title,
            "performer": performer
        }
        
        # get search results
        status = None
        while not (status == 200):
            status, search_results = search_performance(title, performer, get_random_proxy())
        collector["shs_search"] = search_results
        
        # get performance identifier of first results
        if search_results.get("resultPage") and len(search_results.get("resultPage")) > 0:
            perf_id = search_results["resultPage"][0]["uri"].split("/")[-1]
            collector["perf_id"] = perf_id
            
            # get performance data
            status = None
            while not (status == 200):
                status, performance_data = get_performance(perf_id, get_random_proxy(mode="file"))
            collector["shs_performance"] = performance_data
            
            # Append the performance data to the list
            shs_enriched_cliques.append(collector)
            
        else:
            shs_enriched_cliques.append(collector)
            continue   
            
        
    # Write the SHS performances to the output file
    with open(args.output, 'w') as f:
        json.dump(shs_enriched_cliques, f, indent=4)

if __name__ == "__main__":
    main()
    
    