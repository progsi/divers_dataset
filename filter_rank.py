import argparse
import json
import os
import glob


def load_crawl_ids_with_rank(input_dir, max_index):
    ids = set()
    for filepath in glob.glob(os.path.join(input_dir, "*.jsonl")):
        with open(filepath, 'r') as f:
            for index, line in enumerate(f):
                if index > max_index:
                    break
                try:
                    obj = json.loads(line)
                    if 'id' in obj:
                        ids.add(obj['id'])
                except json.JSONDecodeError:
                    continue
    return ids


def main():
    parser = argparse.ArgumentParser(description="Filter IDs from crawl files up to a max index and save to keep.txt")
    parser.add_argument('input_dir', help='Directory containing crawl .jsonl files')
    parser.add_argument('output_dir', help='Directory to store output')
    parser.add_argument('--max_index', type=int, required=True, help='Maximum index (0-based) to consider per file')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    filtered_ids = load_crawl_ids_with_rank(args.input_dir, args.max_index)

    output_path = os.path.join(args.output_dir, 'keep.txt')
    with open(output_path, 'w') as f:
        for yt_id in sorted(filtered_ids):
            f.write(f"{yt_id}\n")

    print(f"Total IDs written to keep.txt: {len(filtered_ids)}")


if __name__ == '__main__':
    main()
