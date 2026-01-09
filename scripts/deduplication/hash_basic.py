import os
import sys
import hashlib
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse


def file_hash(path, block_size=1024 * 1024):
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(block_size), b""):
                hasher.update(block)
        return path, hasher.hexdigest()
    except Exception as e:
        return path, None, str(e)

def collect_files(input_dir):
    files = []
    for root, _, filenames in os.walk(input_dir):
        for name in filenames:
            files.append(os.path.join(root, name))
    return files

def find_duplicates(files, workers):
    hashes = defaultdict(list)

    with Pool(processes=workers) as pool:
        for result in tqdm(
            pool.imap_unordered(file_hash, files),
            total=len(files),
            desc="Hashing files",
            unit="file"
        ):
            if len(result) == 3 and result[1] is None:
                # error
                path, _, err = result
                tqdm.write(f"Skipping {path}: {err}")
                continue
            path, h = result
            hashes[h].append(path)

    # keep only actual duplicates
    return {h: paths for h, paths in hashes.items() if len(paths) > 1}

def write_duplicates(duplicates, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for h, paths in duplicates.items():
            f.write(f"Hash: {h}\n")
            for p in paths:
                f.write(f"  {p}\n")
            f.write("\n")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Find exact duplicate files using SHA-256 hash"
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

    files = collect_files(args.input_dir)
    print(f"Found {len(files)} files.")
    print(f"Using {workers} worker processes.")

    duplicates = find_duplicates(files, workers)
    write_duplicates(duplicates, args.output_file)

    print(f"Done. Found {len(duplicates)} duplicate groups.")
    print(f"Results written to: {args.output_file}")

if __name__ == "__main__":
    main()
