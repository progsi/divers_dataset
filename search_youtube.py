import argparse
import json
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from youtubesearchpython import VideosSearch


def search(q: str, max_results: int = 500, max_pages: int = 100) -> List[dict]:
    results = []
    search = VideosSearch(q, limit=max_results)

    cur_pages = 0
    results_left = True
    while results_left and len(results) < max_results and cur_pages < max_pages:
        new_results = search.result()["result"]
        results += new_results
        if len(new_results) == 0:
            break
        results_left = search.next()
        cur_pages += 1

    results = results[:max_results]
    print(f"Found {len(results)} results for query '{q}'")
    return results


def sanitize_filename(query: str, max_results: int) -> str:
    safe_query = query.replace(" ", "_")
    return f"{safe_query}.{max_results}.jsonl"


def process_query(
    clique_id: str,
    title: str,
    output_dir: Path,
    max_results: int,
    max_pages: int
) -> Tuple[str, bool]:
    base_filename = sanitize_filename(title, max_results)
    jsonl_path = output_dir / base_filename
    log_path = jsonl_path.with_suffix(".log")

    if jsonl_path.exists():
        return clique_id, False  # Already exists

    try:
        results = search(title, max_results=max_results, max_pages=max_pages)
        if results:
            with open(jsonl_path, "w", encoding="utf-8") as out_f:
                for item in results:
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            with open(log_path, "w", encoding="utf-8") as log_f:
                log_f.write(f"No results for query: {title}\n")
    except Exception as e:
        with open(log_path, "w", encoding="utf-8") as log_f:
            log_f.write(f"Error for query '{title}': {str(e)}\n")

    return clique_id, True


def main():
    parser = argparse.ArgumentParser(description="Batch YouTube search using queries from a JSON file.")
    parser.add_argument("input", type=str, help="Path to input JSON file with queries.")
    parser.add_argument("output", type=str, help="Directory to store results.")
    parser.add_argument("--max_results", type=int, default=500, help="Max results per query (default: 500).")
    parser.add_argument("--max_pages", type=int, default=100, help="Max number of pages (default: 100).")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4).")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_query,
                clique_id,
                title,
                output_dir,
                args.max_results,
                args.max_pages
            ): clique_id
            for clique_id, title in queries.items()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing queries"):
            pass  # tqdm just for progress

if __name__ == "__main__":
    main()
