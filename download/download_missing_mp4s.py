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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def get_youtube_id(url):
    return url.split("/watch?v=")[1]

def main(input_json, music_dir, proxy=False, force_failed=False):

    print("Downloading the missing YouTube IDs of the matched versions...")
    t0 = time.monotonic()
    counter, success = 0, 0
    with open(input_json, encoding="utf-8") as in_f, open(
        input_json + ".log", "w"
    ) as logfile:
        logger = csv.writer(logfile, delimiter="\t")
        for jsonline in in_f:
            clique = json.loads(jsonline)
            for version in clique["versions"]:
                # Download the first video that is not downloaded yet
                # The videos were sorted by match quality before
                for video in version["youtube_video"]:
                    yt_id = get_youtube_id(video["url"])
                    row = download_audio_and_metadata(
                        yt_id, music_dir, proxy=proxy, force_failed=force_failed
                    )
                    status = row[-1]
                    logger.writerow(row)
                    # If the video is downloaded, we can move to the next version
                    if status == "downloaded":
                        success += 1
                        break
                    elif status == "file exists":
                        break
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
        "--proxy",
        "-p",
        action="store_true",
        help="Use a proxy server to download the videos.",
    )
    parser.add_argument(
        "--force-failed",
        "-f",
        action="store_true",
        help="Force download of failed IDs.",
    )
    args = parser.parse_args()

    main(args.input_json, args.music_dir, proxy=args.proxy, force_failed=args.force_failed)