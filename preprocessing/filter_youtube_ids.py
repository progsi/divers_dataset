"""This script creates JSON files with matched YouTube IDs from different datasets and creates
a file keep.txt for IDs which are kept.
"""
import argparse
import json
import os
import glob
import csv
from collections import defaultdict


def load_crawl_ids(input_dir):
    ids = set()
    for filepath in glob.glob(os.path.join(input_dir, "*.jsonl")):
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if 'id' in obj:
                        ids.add(obj['id'])
                except json.JSONDecodeError:
                    continue
    return ids


def process_discogs(discogs_path, crawl_ids):
    result = defaultdict(list)
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        path = f"{discogs_path}.{subset}"
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)
            for entries in data.values():
                for entry in entries:
                    yt_id = entry.get("youtube_id")
                    if yt_id in crawl_ids:
                        result[subset.upper()].append(yt_id)
    for key in result:
        print(f"Discogs matches for {key}: {len(result[key])}")
    return result


def process_shs(csv_path, crawl_ids):
    result = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            subset = row.get('subset', '').upper()
            yt_id = row.get('youtube_id')
            if yt_id in crawl_ids and subset in ['TRAIN', 'VAL', 'TEST']:
                result[subset].append(yt_id)
    for key in result:
        print(f"SHS100K2 matches for {key}: {len(result[key])}")
    return result


def process_datacos(csv_path, crawl_ids):
    result = {"TEST": []}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yt_id = row.get('youtube_id')
            if yt_id in crawl_ids:
                result["TEST"].append(yt_id)
    print(f"Da-Tacos matches for TEST: {len(result['TEST'])}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='Directory containing crawl .jsonl files')
    parser.add_argument('output_dir', help='Directory to store output')
    parser.add_argument('--discogs_path', required=True, help='Base path for Discogs files (e.g. /path/to/discogs_vi_yt)')
    parser.add_argument('--shs_csv_path', required=True)
    parser.add_argument('--datacos_csv_path', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    crawl_ids = load_crawl_ids(args.input_dir)

    discogs_matches = process_discogs(args.discogs_path, crawl_ids)
    shs_matches = process_shs(args.shs_csv_path, crawl_ids)
    datacos_matches = process_datacos(args.datacos_csv_path, crawl_ids)

    matched_ids = set()
    for matches in [discogs_matches, shs_matches, datacos_matches]:
        for ids in matches.values():
            matched_ids.update(ids)

    keep_ids = sorted(crawl_ids - matched_ids)

    with open(os.path.join(args.output_dir, 'discogs_vi_yt_matches.json'), 'w') as f:
        json.dump(discogs_matches, f, indent=2)

    with open(os.path.join(args.output_dir, 'shs100k2_matches.json'), 'w') as f:
        json.dump(shs_matches, f, indent=2)

    with open(os.path.join(args.output_dir, 'datacos_matches.json'), 'w') as f:
        json.dump(datacos_matches, f, indent=2)

    with open(os.path.join(args.output_dir, 'keep.txt'), 'w') as f:
        for _id in keep_ids:
            f.write(_id + '\n')

    print(f"Total unmatched (keep) IDs: {len(keep_ids)}")


if __name__ == '__main__':
    main()
