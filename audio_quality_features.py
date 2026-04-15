import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# -----------------------------
# Dataset loader (unchanged)
# -----------------------------
def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    def inverse_split_dict(split_dict):
        clique_to_split = {}
        for split_name, sub_dict in split_dict.items():
            for clique in sub_dict.keys():
                clique_to_split[clique] = split_name
        return clique_to_split

    if path.endswith(".json"):
        with open(path, "r") as f:
            meta = json.load(f)
    elif path.endswith(".pt"):
        meta = torch.load(path, weights_only=False)
    else:
        raise ValueError("Unsupported file type")

    if isinstance(meta, dict) and "info" in meta:
        info = meta["info"]
        split = meta["split"]
    else:
        info, split = meta

    df = pd.DataFrame.from_dict(info, orient="index")
    clique2split = inverse_split_dict(split)
    df["split"] = df["clique"].map(clique2split)

    if "youtube_id" not in df.columns:
        df["youtube_id"] = df.filename.apply(lambda x: x.split("/")[-1].split(".")[0])

    df["dvi"] = df["youtube_id"] != df["version"]

    return df, meta

# -----------------------------
# Audio Quality Feature Extractor (v1)
# -----------------------------
def audio_quality_features(path, sr=16000, n_fft=512, hop_length=256):

    y, _ = librosa.load(path, sr=sr)

    # ---- silence trimming (robust but not overly aggressive) ----
    y, _ = librosa.effects.trim(y, top_db=30)

    # ---- STFT ----
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # ---- time-domain frame features ----
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]

    # ---- spectral features ----
    spectral_flatness = librosa.feature.spectral_flatness(S=S)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]

    # ---- FIXED HF ratio (energy-based, stable) ----
    hf_mask = freqs >= 4000
    lf_mask = freqs < 4000

    hf_energy = S[hf_mask, :].sum(axis=0)
    lf_energy = S[lf_mask, :].sum(axis=0)

    hf_ratio = hf_energy / (lf_energy + 1e-8)

    # ---- dynamic range (robust percentile-based version) ----
    db_wave = librosa.amplitude_to_db(np.abs(y) + 1e-10)
    dynamic_range = np.percentile(db_wave, 95) - np.percentile(db_wave, 5)

    # ---- noise / signal proxy (IMPORTANT ADDITION) ----
    rms_sorted = np.sort(rms)

    noise_floor = np.mean(rms_sorted[:max(1, len(rms_sorted)//10)])   # bottom 10%
    signal_level = np.mean(rms_sorted[-max(1, len(rms_sorted)//10):]) # top 10%

    snr_proxy = signal_level / (noise_floor + 1e-8)

    # ---- helper: robust stats ----
    def stats(x):
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "p90": float(np.percentile(x, 90))
        }

    feats = {}

    # -----------------------------
    # Spectral + noise features
    # -----------------------------
    for name, arr in {
        "spectral_flatness": spectral_flatness,
        "spectral_rolloff": spectral_rolloff,
        "hf_ratio": hf_ratio,
        "zcr": zcr
    }.items():
        s = stats(arr)
        feats[f"{name}_mean"] = s["mean"]
        feats[f"{name}_std"] = s["std"]
        feats[f"{name}_p90"] = s["p90"]

    # -----------------------------
    # Energy features
    # -----------------------------
    rms_stats = stats(rms)
    feats["rms_mean"] = rms_stats["mean"]
    feats["rms_std"] = rms_stats["std"]
    feats["rms_p90"] = rms_stats["p90"]

    feats["dynamic_range"] = float(dynamic_range)

    # -----------------------------
    # Noise / SNR-like features (KEY ADDITION)
    # -----------------------------
    feats["noise_floor"] = float(noise_floor)
    feats["signal_level"] = float(signal_level)
    feats["snr_proxy"] = float(snr_proxy)

    return feats

# -----------------------------
# Audio file resolver
# -----------------------------
def resolve_audio_path(audio_dir, youtube_id):
    """
    Supports:
    1. audio_dir/youtube_id.mp3
    2. audio_dir/a/b/youtube_id.mp3 where a = first char, b = second char
    3. audio_dir/a/youtube_id.mp3 where a = first char only
    """
    fname = f"{youtube_id}.mp3"

    # direct
    direct = os.path.join(audio_dir, fname)
    if os.path.exists(direct):
        return direct

    # 2-level shard
    if len(youtube_id) >= 2:
        p2 = os.path.join(audio_dir, youtube_id[:2], fname)
        if os.path.exists(p2):
            return p2

    # 1-level shard
    p1 = os.path.join(audio_dir, youtube_id[0], fname)
    if os.path.exists(p1):
        return p1

    return None


# -----------------------------
# Worker init (global access)
# -----------------------------
DATA = None
AUDIO_DIR = None


def init_worker(df, audio_dir):
    global DATA, AUDIO_DIR
    DATA = df
    AUDIO_DIR = audio_dir


def process_row(idx):
    row = DATA.loc[idx]
    yt = row["youtube_id"]

    path = resolve_audio_path(AUDIO_DIR, yt)
    if path is None:
        return idx, None

    try:
        feats = audio_quality_features(path)
        return idx, feats
    except Exception as e:
        return idx, None


# -----------------------------
# Main pipeline
# -----------------------------
def main(args):
    df, meta = load_dataset(args.dataset)
    
    print(f"Loaded dataset with {len(df)} entries. Extracting audio features using {args.workers} workers...")

    indices = df.index.tolist()

    results = {}

    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(df, args.audio_dir),
    ) as pool:

        for idx, feats in tqdm(pool.imap_unordered(process_row, indices), total=len(indices)):
            if feats is not None:
                results[idx] = feats

    print(f"Extracted features for {len(results)} out of {len(df)} entries.")
    
    # convert to dataframe
    feat_df = pd.DataFrame.from_dict(results, orient="index")

    print("Sample extracted features:")
    print(feat_df.head())
    
    # merge into original
    df = df.join(feat_df)
    print("Enriched dataset:")
    print(df.head())

    # save
    torch.save(
        {
            "info": df.to_dict(orient="index"),
            "split": meta["split"] if isinstance(meta, dict) else None,
        },
        args.output,
    )

    print(f"\nSaved enriched dataset to: {args.output}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to input dataset (.json or .pt)")

    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Root directory containing mp3 files")

    parser.add_argument("--output", type=str, required=True,
                        help="Output path for enriched dataset")

    parser.add_argument("--workers", type=int, default=cpu_count(),
                        help="Number of multiprocessing workers")

    args = parser.parse_args()
    main(args)