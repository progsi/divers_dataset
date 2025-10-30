#!/usr/bin/env python3
"""Feature‑extraction script with the following fixes:
1. Workers return **NumPy** arrays and Python scalars only; tensors are created in the
   parent before writing to disk (avoids pickling / ancdata crashes).
2. Robust tempo extraction – no more `.item()` on possibly empty arrays.
3. Results are checkpoint‑saved every *N* items (default : 1000) so you never lose
   more than N−1 files if the run dies midway.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import librosa  # type: ignore
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import warnings

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

###############################################################################
# Utility helpers
###############################################################################



def find_audio_files(directory: str | Path, exts=(".wav", ".flac", ".mp3", ".mp4")) -> list[str]:
    """Recursively gather audio files with the given extensions."""
    directory = Path(directory)
    return [
        str(p)
        for p in directory.rglob("*")
        if p.suffix.lower() in exts and p.is_file()
    ]


###############################################################################
# Worker function (runs in subprocesses)
###############################################################################


def _extract_features_worker(args: Tuple[str, int]) -> Tuple[str, Dict[str, Any]]:
    """Return NumPy / scalar types only – never torch objects."""
    audio_path, sr_target = args
    try:
        y, sr_actual = librosa.load(audio_path, sr=sr_target)

        # Onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr_actual)

        # Tempo: librosa returns a 1‑D array; handle empty array safely.
        tempo_arr = librosa.beat.tempo(onset_envelope=onset_env, sr=sr_actual)
        tempo = float(tempo_arr[0]) if tempo_arr.size else None

        return audio_path, {
            "onset_env": onset_env,  # NumPy array
            "tempo": tempo,          # float | None
        }

    except Exception:  # catch *everything* so the parent never crashes silently
        logging.exception("Error processing %s", audio_path)
        return audio_path, {"error": True}


###############################################################################
# Main
###############################################################################


def main() -> None:  # noqa: C901 – complex but clear
    parser = argparse.ArgumentParser(
        description="Extract rhythmic features and store them as a tensor file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio-dir", required=True, help="Directory with audio")
    parser.add_argument("--yt-file", required=True, help="Parquet with YouTube metadata")
    parser.add_argument("--output", "-o", required=True, help="Output .pt file (torch.save)")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Target sample rate")
    parser.add_argument("--njobs", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--save-every", type=int, default=10_000, help="Checkpoint every N items")
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # House‑keeping & logging
    # ---------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.njobs is None or args.njobs < 1:
        args.njobs = max(1, os.cpu_count() - 1)

    # ---------------------------------------------------------------------
    # Gather input files
    # ---------------------------------------------------------------------
    audio_files = find_audio_files(args.audio_dir)
    logging.info("Found %d candidate audio files", len(audio_files))

    df = pd.read_parquet(args.yt_file)
    audio_files = [
        f for f in audio_files if Path(f).stem in df.index
    ]
    logging.info("After metadata filter: %d files", len(audio_files))

    # ---------------------------------------------------------------------
    # Prepare for extraction
    # ---------------------------------------------------------------------
    audio_args = [(f, args.sample_rate) for f in audio_files]
    mapping: Dict[str, Dict[str, Any]] = {}
    processed = 0

    def _checkpoint(final: bool = False) -> None:
        """Save current mapping to disk (overwrites each time)."""
        if not mapping:
            return
        ckpt_path = Path(args.output)
        if not final:
            ckpt_path = ckpt_path.with_suffix(".partial.pt")
        torch_mapping = {
            k: {
                "onset_env": torch.from_numpy(v["onset_env"].copy()),
                "tempo": v["tempo"],
            }
            for k, v in mapping.items() if "error" not in v
        }
        torch.save(torch_mapping, str(ckpt_path))
        logging.info("Saved %d items → %s", len(torch_mapping), ckpt_path)

    # ---------------------------------------------------------------------
    # Extraction loop – sequential or parallel
    # ---------------------------------------------------------------------
    if args.njobs == 1:
        # Single‑process path (useful for debug)
        for arg in tqdm(audio_args, desc="Extracting", unit="file"):
            fname, res = _extract_features_worker(arg)
            mapping[fname] = res
            processed += 1
            if processed % args.save_every == 0:
                _checkpoint()
    else:
        with mp.Pool(processes=args.njobs) as pool, \
                tqdm(total=len(audio_args), desc="Extracting", unit="file") as pbar:
            for fname, res in pool.imap_unordered(_extract_features_worker, audio_args):
                mapping[fname] = res
                processed += 1
                if processed % args.save_every == 0:
                    _checkpoint()
                pbar.update(1)

    # ---------------------------------------------------------------------
    # Final save – overwrite the main output path
    # ---------------------------------------------------------------------
    _checkpoint(final=True)
    logging.info("All done – %d files processed", processed)


###############################################################################
# Entry‑point (critical for spawn‑based start methods)
###############################################################################


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # Someone already set it – safe to ignore
        pass
    main()