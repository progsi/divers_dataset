import argparse
import nest_asyncio
import asyncio
from shazamio import Shazam
import os
import json
from tqdm import tqdm
import random

nest_asyncio.apply()  # allow nested loops


def get_random_proxy_with_credentials(username_file: str = "../proxy_user.txt", 
                                      pw_file: str = "../proxy_pw.txt", 
                                      servers_path: str = "../servers.txt",
                                      blocked_servers_path: str = "../blocked_servers.txt",
                                      port: int = 89) -> str:
    # get username
    with open(username_file, "r") as f:
        user = f.read().strip()

    # get password
    with open(pw_file, "r") as f:
        pw = f.read().strip()
    
    def clean_server_list(servers: list) -> list:
        servers = [s.strip().replace("\n", "") for s in servers]
        servers = [s for s in servers if s.endswith(".com")]
        return servers
    
    # get server list
    with open(servers_path, "r") as f:
        servers = clean_server_list(f.readlines())
    
    # get blocked servers
    with open(blocked_servers_path, "r") as f:
        blocked_servers = clean_server_list(f.readlines())
    
    # get random server
    servers = [s for s in servers if s not in blocked_servers]
    server = random.choice(servers)
    
    print(f"Using proxy: {server}:{port}")
    return f"https://{user}:{pw}@{server}:{port}"

async def recognize_file(file_path, api_sem, ffmpeg_sem, shazam, proxy=None, pbar=None, timeout=30):
    """Recognize a single audio file with separate concurrency limits for API and FFmpeg"""
    async with api_sem:
        if not os.path.exists(file_path):
            res = {"file": file_path, "error": "File not found"}
        else:
            try:
                async with ffmpeg_sem:  # limit FFmpeg subprocesses
                    start = asyncio.get_event_loop().time()
                    if proxy:
                        out = await asyncio.wait_for(shazam.recognize(file_path, proxy=proxy), timeout=timeout)
                    else:
                        out = await asyncio.wait_for(shazam.recognize(file_path), timeout=timeout)
                    duration = asyncio.get_event_loop().time() - start
                    res = {"file": file_path, "result": out, "time": duration}
            except asyncio.TimeoutError:
                res = {"file": file_path, "error": f"Timeout after {timeout}s"}
            except Exception as e:
                res = {"file": file_path, "error": str(e)}

        if pbar:
            pbar.update(1)
        return res


async def recognize_files(file_list, output_file=None, concurrency=16, ffmpeg_limit=16, deactivate_proxy=False, timeout=30, rotate_always=False):
    """Recognize multiple audio files, retrying on any error, with separate concurrency for API and FFmpeg"""
    api_sem = asyncio.Semaphore(concurrency)
    ffmpeg_sem = asyncio.Semaphore(ffmpeg_limit)

    # Load existing results
    processed = {}
    if output_file and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed.update(json.loads(line))
                except Exception:
                    continue
        print(f"Loaded {len(processed)} previously processed files")

    shazam = Shazam()
    results_to_retry = [f for f in file_list if f not in processed or "error" in processed.get(f, {})]

    # Initial proxy
    proxy = None
    if not deactivate_proxy:
        proxy = get_random_proxy_with_credentials()

    with tqdm(total=len(results_to_retry), desc="Recognizing", unit="file") as pbar:
        while results_to_retry:
            batch = results_to_retry[:concurrency]
            results_to_retry = results_to_retry[concurrency:]

            tasks = [recognize_file(fpath, api_sem, ffmpeg_sem, shazam, proxy, pbar, timeout) for fpath in batch]
            batch_results = await asyncio.gather(*tasks)

            failed_files = []

            for res in batch_results:
                fpath = res["file"]
                # Retry on ANY error
                if "error" in res or "result" not in res or not res.get("result"):
                    failed_files.append(fpath)
                else:
                    # Only mark success if result is valid
                    processed[fpath] = res

            # Rotate proxy if ANY failures occurred
            if (failed_files or rotate_always) and not deactivate_proxy:
                print("Rotating proxy due to errors...")
                proxy = get_random_proxy_with_credentials()

            # Re-queue failed files
            results_to_retry.extend(failed_files)

            # Save all processed results so far
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    for key, value in processed.items():
                        f.write(json.dumps({key: value}, ensure_ascii=False) + "\n")

    if output_file:
        print(f"\nAll results written to {output_file}")

    return processed

def main():
    args = parse_args()

    # Determine input files
    files_to_process = []
    if os.path.isfile(args.files) and args.files.endswith(".txt"):
        with open(args.files, "r") as f:
            files_to_process = [line.strip() for line in f if line.strip()]
    elif os.path.exists(args.files):
        if os.path.isdir(args.files):
            files_to_process = [
                os.path.join(args.files, f) for f in os.listdir(args.files)
                if not f.startswith(".")
            ]
        else:
            files_to_process = [args.files]
    else:
        print(f"No valid file or directory found: {args.files}")
        return

    asyncio.get_event_loop().run_until_complete(
        recognize_files(
            file_list=files_to_process,
            output_file=args.output,
            concurrency=args.concurrency,
            deactivate_proxy=args.noproxy,
            timeout=args.timeout,
            rotate_always=args.rotate_always
        )
    )



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'files',
        type=str,
        help='Input audio file, directory, or text file with list of audio files.'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Optional output JSON file to write results.'
    )
    parser.add_argument(
        '--concurrency', '-c',
        type=int,
        default=16,
        help='Maximum number of files to process concurrently (default: 512).'
    )
    parser.add_argument(
        '--noproxy',
        action='store_true',
        help='Deactivate proxy usage and call Shazam without proxy.'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='Timeout in seconds for each Shazam request (default: 30).'
    )
    parser.add_argument(
        '--rotate_always',
        action='store_true',
        help='Rotate proxy at every batch.'
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
