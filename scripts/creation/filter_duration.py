import argparse
import json
import os
import glob
from datetime import timedelta

def parse_duration(duration_str):
    parts = duration_str.split(":")
    try:
        if len(parts) == 3:
            h, m, s = map(int, parts)
        elif len(parts) == 2:
            h = 0
            m, s = map(int, parts)
        else:
            return None
        return timedelta(hours=h, minutes=m, seconds=s).total_seconds()
    except ValueError:
        return None

def process_durations(input_dir, min_sec, max_sec):
    short_ids = set()
    long_ids = set()
    keep_ids = set()
    error_ids = set()

    for filepath in glob.glob(os.path.join(input_dir, "*.jsonl")):
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    yt_id = obj.get("id")
                    duration_str = obj.get("duration")

                    if yt_id is None:
                        continue

                    if duration_str is None:
                        error_ids.add(yt_id)
                        continue

                    duration_sec = parse_duration(duration_str)
                    if duration_sec is None:
                        error_ids.add(yt_id)
                        continue

                    if duration_sec < min_sec:
                        short_ids.add(yt_id)
                    elif duration_sec > max_sec:
                        long_ids.add(yt_id)
                    else:
                        keep_ids.add(yt_id)

                except json.JSONDecodeError:
                    continue

    return list(short_ids), list(long_ids), list(keep_ids), list(error_ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='Directory containing crawl .jsonl files')
    parser.add_argument('output_dir', help='Directory to store output')
    parser.add_argument('--min', type=int, required=True, help='Minimum duration in seconds')
    parser.add_argument('--max', type=int, required=True, help='Maximum duration in seconds')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    short_ids, long_ids, keep_ids, error_ids = process_durations(args.input_dir, args.min, args.max)

    exclude = {
        "SHORT": short_ids,
        "LONG": long_ids,
        "ERROR": error_ids
    }

    with open(os.path.join(args.output_dir, 'exclude.json'), 'w') as f:
        json.dump(exclude, f, indent=2)

    with open(os.path.join(args.output_dir, 'keep.txt'), 'w') as f:
        for _id in sorted(keep_ids):
            f.write(_id + '\n')

    print(f"Excluded SHORT: {len(short_ids)}")
    print(f"Excluded LONG: {len(long_ids)}")
    print(f"Excluded ERROR (invalid or missing duration): {len(error_ids)}")
    print(f"Total kept: {len(keep_ids)}")

if __name__ == '__main__':
    main()
