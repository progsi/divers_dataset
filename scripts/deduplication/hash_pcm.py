import os
import sys
import subprocess
import hashlib
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse


AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".aiff", ".alac"
}

def pcm_hash(path):
    """
    Decode audio to normalized PCM and hash the raw audio stream.
    Normalization:
      - mono
      - 44.1 kHz
      - 16-bit signed PCM
    """
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", path,
        "-ac", "1",
        "-ar", "44100",
        "-f", "s16le",
        "-"
    ]

    hasher = hashlib.sha256()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        while True:
            chunk = proc.stdout.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    finally:
        proc.stdout.close()
        proc.wait()

    if proc.returncode != 0:
        err = proc.stderr.read().decode(errors="ignore")
        raise RuntimeError(err)

    return path, hasher.hexdigest()

def collect_audio_files(input_dir):
    files = []
    for root, _, filenames in os.walk(input_dir):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
                files.append(os.path.join(root, name))
    return files

def find_duplicates(files, workers):
    hashes = defaultdict(list)

    with Pool(processes=workers) as pool:
        for result in tqdm(
            pool.imap_unordered(pcm_hash, files),
            total=len(files),
            desc="Hashing audio",
            unit="file"
        ):
            if result is None:
                continue
            path, h = result
            hashes[h].append(path)

    return {h: paths for h, paths in hashes.items() if len(paths) > 1}

def write_duplicates(duplicates, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for h, paths in duplicates.items():
            f.write(f"PCM Hash: {h}\n")
            for p in paths:
                f.write(f"  {p}\n")
            f.write("\n")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Find duplicate audio files using PCM-normalized hashing"
    )
    parser.add_argument("input_dir", help="Directory to scan")
    parser.add_argument("output_file", help="File to write duplicate groups")
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=cpu_count(),
        help="Number of CPU cores to use (default: all cores)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    workers = max(1, args.jobs)

    audio_files = collect_audio_files(args.input_dir)
    print(f"Found {len(audio_files)} audio files.")
    print(f"Using {workers} worker processes.")

    duplicates = find_duplicates(audio_files, workers)
    write_duplicates(duplicates, args.output_file)

    print(f"Done. Found {len(duplicates)} duplicate groups.")
    print(f"Results written to: {args.output_file}")

if __name__ == "__main__":
    main()
