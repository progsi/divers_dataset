import argparse
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
import pandas as pd
import torch
from tqdm import tqdm


def flatten_dvi(path, get_all_tracks=False):
    """
    Flatten the DVI dataset into a list of dictionaries.
    Args:
        path (str): Path to the DVI JSONL file.
        get_all_tracks (bool): If True, get all tracks; if False, get only the first track.
    Returns:
        list: A list of dictionaries with flattened data.
    """
    data = []
    with open(path, "r") as f:
        for line in tqdm(f):
            clique = json.loads(line.strip())
            c_id = clique["clique_id"]
            for version in clique["versions"]:
                v_id = version["version_id"]
                if version.get("tracks"): # only versions which are also in original DVI have tracks
                    for track in version["tracks"]:
                        for youtube_video in track.get("youtube_video", []):
                            version_flat = {
                                "clique_id": c_id,
                                "version_id": v_id,
                                "youtube_id": youtube_video["url"].split("watch?v=")[-1],
                                "source": youtube_video["source"],
                                "match_type": youtube_video["match_type"],
                                }
                            for k, v in track.items():
                                if k != "youtube_video":
                                    version_flat[k] = v
                            data.append(version_flat)
                        if not get_all_tracks:
                            break
    return data
    

def _load_single(
    pair: tuple[str, Dict[str, Any]],
    audio_dir: str,
    keys: List[str],
) -> Dict[str, Any] | None:
    """
    Helper that loads one *.meta file and returns a slim dict.
    Returns None on FileNotFound / JSON errors so we can skip bad rows cleanly.
    """
    identifier, version = pair
    try:
        fn = version["filename"].rsplit(".", 1)[0] + ".meta"
        with open(os.path.join(audio_dir, fn), "r", encoding="utf‑8") as f:
            youtube_meta = json.load(f)

        # keep only whitelist `keys`
        youtube_meta_light = {}
        for k in keys:
            if k not in youtube_meta:
                continue
            out_key = "youtube_id" if k == "id" else k
            youtube_meta_light[out_key] = youtube_meta[k]

        return youtube_meta_light

    except (FileNotFoundError, json.JSONDecodeError) as exc:
        # Optional: log and keep going
        print(f"[collect] skipping {identifier}: {exc}")
        return None


def collect_youtube_metadata(
    path: str,
    audio_dir: str,
    keys: list[str],
    njobs: int = 8,
) -> list[dict[str, Any]]:
    """
    Read *.meta files in parallel and return a list of lightweight dicts.

    Parameters
    ----------
    path : str
        Path to PyTorch metadata file.
    audio_dir : str
        Directory containing the *.meta files.
    keys : list[str]
        Whitelist of JSON keys to keep. `'id'` will be renamed to `'youtube_id'`.
    njobs : int, default 8
        Number of worker threads to use.

    Returns
    -------
    list[dict]
        One dict per successfully‑read *.meta file.
    """
    metadata = torch.load(path)
    items_iter = list(metadata["info"].items())  # materialize so tqdm can get length
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=njobs) as pool, tqdm(
        total=len(items_iter), desc="Collecting", unit="file"
    ) as pbar:
        futures = {
            pool.submit(_load_single, pair, audio_dir, keys): pair[0]
            for pair in items_iter
        }
        for fut in as_completed(futures):
            pbar.update(1)
            data = fut.result()
            if data:  # skip None (failed loads)
                results.append(data)

    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Collect Discosgs and YouTube metadata.")
    parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Directory containing the audio files and their metadata.",
    )
    parser.add_argument(
        "--dvi-file",
        type=str,
        required=True,
        help="Path to the DVI jsonlines file.",
    )
    parser.add_argument(
        "--meta-file",
        type=str,
        required=True,
        help="Path to the metadata PyTorch file.",
    )
    parser.add_argument(
        "--get-all-tracks",
        action="store_true",
        help="If set, get all tracks from the DVI file.",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=8,
        help="Number of parallel jobs to run.",
    )
    return parser.parse_args()

def main():
    # Example usage
    args = parse_args()

    keys = ['id', 'title', 'description', 
        'channel_id', 'channel_url', 
        'duration', 'view_count', 'average_rating', 'age_limit',
        'categories', 'tags',
        'categories', 'tags', 'release_timestamp',
        'automatic_captions', 'subtitles', 'album', 'artists', 'track', 'release_date', 'release_year', 'comment_count', 'chapters',
        'like_count', 'channel', 'channel_follower_count', 'uploader', 'uploader_id', 'uploader_url', 'upload_date', 'timestamp', 'creators', 
        'alt_title', 'availability', 'artist', 'creator', 'requested_subtitles', 'language', 'language_preference'
        ]

    
    dvi_out_path = os.path.join("data", "dvi.jsonl")
    if not os.path.isfile(dvi_out_path):
        data_dvi = flatten_dvi(args.dvi_file, get_all_tracks=args.get_all_tracks)
        
        with open(os.path.join("data", "dvi.jsonl"), "w", encoding="utf-8") as out_f:
            for item in data_dvi:
                out_f.write(json.dumps(item) + "\n")
        print(f"Flattened DVI data written to {os.path.join('data', 'dvi.jsonl')}")
    else:
        print(f"DVI already exists at {dvi_out_path}")

    yt_out_path = os.path.join("data", "yt.parquet")
    if not os.path.isfile(yt_out_path):
        data_yt = collect_youtube_metadata(
            args.meta_file, audio_dir=args.audio_dir, keys=keys, njobs=args.njobs
        )
        
        df = pd.DataFrame(data_yt)
        df = df.drop_duplicates(subset=["youtube_id", "title"])
        df = df.set_index("youtube_id", drop=False)
        df.to_parquet('data/yt.parquet', index="youtube_id")
        
        print(f"Collected YouTube metadata written to {yt_out_path}")
    else:
        print(f"YouTube metadata already exists at {yt_out_path}")
    
if __name__ == "__main__":
    main()