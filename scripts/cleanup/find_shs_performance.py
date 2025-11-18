# load JSON Lines (one JSON object per line)
import argparse
import requests
import time
from datetime import datetime, timedelta
import json

BASE_URL = "https://secondhandsongs.com/search/"


def get_search_url_performance(title: str, performer: str) -> str:
    title, performer = title.replace(" ", "%20"), performer.replace(" ", "%20")
    return BASE_URL + f"performance?op_title=contains&title={title}&op_performer=contains&performer={performer}&sort=simplifiedTitle&reverse=0&format=json"

def search_performance(title: str, performer: str, key: str) -> dict:
    url = get_search_url_performance(title, performer)
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/118.0.5993.90 Safari/537.36"
    }
    if key is not None:
        headers["X-API-Key"] = key

    response = requests.get(url, headers=headers)
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
    
    # Limits
    MAX_REQUESTS_PER_MINUTE = 100
    MAX_REQUESTS_PER_HOUR = 1000
    MAX_REQUESTS_PER_DAY = 24_000

    # Buffer: reduce limits slightly to be safe
    BUFFER = 10

    # Rate limit counters
    request_count_minute = 0
    request_count_hour = 0
    request_count_day = 0

    last_request_time = datetime.now()
    last_hour_time = datetime.now()
    last_day_time = datetime.now()
    total = 0
    
    with open(args.input_file, "r", encoding="utf-8") as in_f:
        with open(args.output_file, "w", encoding="utf-8") as out_f:
            for entry in yield_version_metadata_unique(data):
                now = datetime.now()
                # Reset counters if a new minute/hour/day has passed
                if (now - last_request_time) >= timedelta(minutes=1):
                    request_count_minute = 0
                    last_request_time = now
                if (now - last_hour_time) >= timedelta(hours=1):
                    request_count_hour = 0
                    last_hour_time = now
                if (now - last_day_time) >= timedelta(days=1):
                    request_count_day = 0
                    last_day_time = now

                # Wait if limits are approaching
                if request_count_minute >= (MAX_REQUESTS_PER_MINUTE - BUFFER):
                    sleep_time = 60 - (now - last_request_time).total_seconds()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    request_count_minute = 0
                    last_request_time = datetime.now()

                if request_count_hour >= (MAX_REQUESTS_PER_HOUR - BUFFER):
                    sleep_time = 3600 - (now - last_hour_time).total_seconds()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    request_count_hour = 0
                    last_hour_time = datetime.now()

                if request_count_day >= (MAX_REQUESTS_PER_DAY - BUFFER):
                    sleep_time = 86400 - (now - last_day_time).total_seconds()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    request_count_day = 0
                    last_day_time = datetime.now()
                    
                response = search_performance(
                    entry["track_title_cleaned"],
                    entry["release_artist_names"][0] if entry["release_artist_names"] else "",
                    shs_key
                )
                entry["shs_performance_search"] = response
                out_f.write(json.dumps(entry) + "\n")
                
                # Increment counters
                request_count_minute += 1
                request_count_hour += 1
                request_count_day += 1
                total += 1
                
                if total % 1000 == 0:
                    print(f"Processed {total} entries.")

                
if __name__ == "__main__":
    main()