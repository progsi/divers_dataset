# load JSON Lines (one JSON object per line)
import argparse
import requests
from datetime import datetime, timedelta
import json

BASE_URL = "https://secondhandsongs.com/search/"

# Rate limit counters
request_count_minute = 0
request_count_hour = 0
request_count_day = 0
last_request_time = datetime.now()
last_hour_time = datetime.now()
last_day_time = datetime.now()

# Limits
MAX_REQUESTS_PER_MINUTE = 100
MAX_REQUESTS_PER_HOUR = 1000
MAX_REQUESTS_PER_DAY = 24_000


def get_search_url_performance(title: str, performer: str) -> str:
    title, performer = title.replace(" ", "%20"), performer.replace(" ", "%20")
    return BASE_URL + f"performance?op_title=contains&title={title}&op_performer=contains&performer={performer}&sort=simplifiedTitle&reverse=0&format=json"

def search_performance(title: str, performer: str, key: str) -> dict:
    url = get_search_url_performance(title, performer)
    if key is not None:
        response = requests.get(url, headers={"X-API-Key": key})
    else:
        response = requests.get(url)
    return json.loads(response.text)

def yield_version_metadata_unique(data: list):
    """
    Loop through the data and yield unique tracks per version.
    A unique track is defined as a combination of release_artist_names
    and cleaned track title NOT seen earlier in the same version.
    """
    for clique in data:
        for version in clique["versions"]:
            seen = set()  # uniqueness per version
            
            for track in version["tracks"]:
                artists_tuple = tuple(sorted(track["release_artist_names"]))
                cleaned_title = track["track_title_cleaned"]
                key = (artists_tuple, cleaned_title)

                if key not in seen:
                    seen.add(key)
                    yield {
                        "clique_id": clique["clique_id"],
                        "version_id": version["version_id"],
                        "track_title": track["track_title"],
                        "track_title_cleaned": cleaned_title,
                        "track_writer_names": track["track_writer_names"],
                        "track_writer_ids": track["track_writer_ids"],
                        "release_artist_names": track["release_artist_names"],
                        "released": track["released"],
                        "youtube_video": track["youtube_video"],
                    }    

def main():
    parser = argparse.ArgumentParser(
        description="Find SHS performance data from JSON Lines file."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input JSON Lines file."
    )
    parser.add_argument(
        "output_file", type=str, help="Path to the output JSON Lines file."
    )
    parser.add_argument(
        "--shs_key_file", "-k", type=str, default="../shs_key.txt"
    )
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    with open(args.shs_key_file, "r", encoding="utf-8") as kf:
        shs_key = kf.read().strip()

    with open(args.input_file, "r", encoding="utf-8") as in_f:
        with open(args.output_file, "w", encoding="utf-8") as out_f:
            for entry in yield_version_metadata_unique(data):
                print(entry)    
                # TODO: Implement rate limiting and API calls here
                
if __name__ == "__main__":
    main()