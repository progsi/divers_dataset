import argparse
import json
import os
import glob

def load_keep_ids(filter_dir):
    keep_ids = None
    for root, _, files in os.walk(filter_dir):
        for file in files:
            if file == 'keep.txt':
                with open(os.path.join(root, file), 'r') as f:
                    file_ids = set(line.strip() for line in f)
                    if keep_ids is None:
                        keep_ids = file_ids
                    else:
                        keep_ids &= file_ids  # Apply AND logic
    return keep_ids if keep_ids else set()


def process_crawl(input_dir, keep_ids):
    seen_ids = set()
    id_to_metadata = {}
    id_to_queries = {}

    for filepath in glob.glob(os.path.join(input_dir, '*.jsonl')):
        query_name = os.path.basename(filepath).split('.')[0]
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    yt_id = obj.get("id")
                    if yt_id not in keep_ids:
                        continue

                    if yt_id not in seen_ids:
                        id_to_metadata[yt_id] = obj
                        seen_ids.add(yt_id)

                    if yt_id not in id_to_queries:
                        id_to_queries[yt_id] = set()
                    id_to_queries[yt_id].add(query_name)

                except json.JSONDecodeError:
                    continue

    # Convert sets to lists for JSON serialization
    id_to_queries = {k: list(v) for k, v in id_to_queries.items()}
    return id_to_metadata, id_to_queries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='Directory containing crawl .jsonl files')
    parser.add_argument('output_dir', help='Directory to store output files')
    parser.add_argument('--filter_dir', required=True, help='Directory to recursively find keep.txt files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    keep_ids = load_keep_ids(args.filter_dir)
    id_to_metadata, id_to_queries = process_crawl(args.input_dir, keep_ids)

    metadata_path = os.path.join(args.output_dir, 'metadata_filtered.jsonl')
    queries_path = os.path.join(args.output_dir, 'queries_filtered.json')

    with open(metadata_path, 'w') as meta_file:
        for obj in id_to_metadata.values():
            meta_file.write(json.dumps(obj) + '\n')

    with open(queries_path, 'w') as query_file:
        json.dump(id_to_queries, query_file, indent=2)

    print(f"Filtered metadata written to: {metadata_path}")
    print(f"Query mapping written to: {queries_path}")
    print(f"Total unique IDs kept: {len(id_to_metadata)}")

if __name__ == '__main__':
    main()