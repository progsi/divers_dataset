"""This script takes a versioned clique file with versions that are matched to
youtube urls. Attempts to download one video per version. This video is the 
highest quality match."""

import os
import sys
import csv
import json
import time
import argparse

from download import download_audio_and_metadata
from proxy import get_random_proxy, test_proxy_connection, log_blocked_servers

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def get_youtube_id(url):
    return url.split("/watch?v=")[1]

def main(input_json, music_dir, proxy_rotate_every=None, force_failed=False):
    print("Downloading the missing YouTube IDs of the matched versions...")
    t0 = time.monotonic()
    counter, success, proxy_iteration = 0, 0, 1
    proxy = None  # Initialize proxy once outside the loop

    with open(input_json, encoding="utf-8") as in_f, open(input_json + ".log", "w") as logfile:
        logger = csv.writer(logfile, delimiter="\t")
        for jsonline in in_f:
            clique = json.loads(jsonline)
            for version in clique["versions"]:
                for video in version["youtube_video"]:
                    yt_id = get_youtube_id(video["url"])

                    # Rotate proxy only if a real download attempt happened proxy_rotate_every times
                    if proxy_rotate_every is not None and proxy_iteration % proxy_rotate_every == 0:
                        works = False
                        tries = 1
                        while not works:
                            candidate = get_random_proxy()
                            works = test_proxy_connection(candidate)
                            if works:
                                proxy = candidate
                                print(f"Using proxy: {proxy} found at try: {tries}")
                            tries += 1
                        print(f"Rotating proxy to {proxy}")

                    # Try to download
                    row = download_audio_and_metadata(
                        yt_id, music_dir, proxy=proxy, force_failed=force_failed
                    )
                    status = row[-1]
                    logger.writerow(row)

                    if status == "downloaded":
                        success += 1
                        proxy_iteration += 1  # Count this as a real download
                        break
                    elif status == "file exists":
                        break
                    elif status == "download previously failed":
                        # Optional: count it as an attempt
                        proxy_iteration += 1

                counter += 1
                print("=" * 5 + f"Processed {counter:>9,} versions" + "=" * 5)

    print(f"{success:,} YouTube IDs are downloaded.")
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.monotonic()-t0))}")
    print("Done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_json", type=str, help="Clique JSONL file with matched to Youtube URLs."
    )
    parser.add_argument(
        "music_dir",
        type=str,
        help="Directory to download Youtube videos and metadata."
        "Inside this directory, a subdirectory 'audio/' will be created to store the audio and metadata files."
        "Also a logs/ directory will be created to store the log files.",
    )
    parser.add_argument(
        "--proxy-rotate-every",
        "-p",
        type=int,
        default=100,
        help="How often to rotate the proxy (in iterations).",
    )
    parser.add_argument(
        "--force-failed",
        "-f",
        action="store_true",
        help="Force download of failed IDs.",
    )
    args = parser.parse_args()

    main(args.input_json, args.music_dir, proxy_rotate_every=args.proxy_rotate_every, force_failed=args.force_failed)