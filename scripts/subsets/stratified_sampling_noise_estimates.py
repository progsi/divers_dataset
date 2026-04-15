import os
import json
import torch
import numpy as np
import pandas as pd
import argparse


# -----------------------------
# Load dataset
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
# Feature-based conditions
# -----------------------------
def build_conditions(df, q):
    conds = {}

    q_low = q
    q_high = 1 - q

    snr_low = df["snr_proxy"].quantile(q_low)
    snr_high = df["snr_proxy"].quantile(q_high)

    nf_low = df["noise_floor"].quantile(q_low)
    nf_high = df["noise_floor"].quantile(q_high)

    flat_low = df["spectral_flatness_mean"].quantile(q_low)
    flat_high = df["spectral_flatness_mean"].quantile(q_high)

    dyn_low = df["dynamic_range"].quantile(q_low)
    dyn_high = df["dynamic_range"].quantile(q_high)

    # --- single feature conditions ---
    conds["snr_low"] = lambda x: x["snr_proxy"] <= snr_low
    conds["snr_high"] = lambda x: x["snr_proxy"] >= snr_high

    conds["noise_floor_high"] = lambda x: x["noise_floor"] >= nf_high
    conds["noise_floor_low"] = lambda x: x["noise_floor"] <= nf_low

    conds["flatness_high"] = lambda x: x["spectral_flatness_mean"] >= flat_high
    conds["flatness_low"] = lambda x: x["spectral_flatness_mean"] <= flat_low

    conds["dynamic_low"] = lambda x: x["dynamic_range"] <= dyn_low
    conds["dynamic_high"] = lambda x: x["dynamic_range"] >= dyn_high

    # --- combined conditions ---
    conds["noisy_combined"] = lambda x: (
        (x["snr_proxy"] <= snr_low) &
        (x["noise_floor"] >= nf_high) &
        (x["spectral_flatness_mean"] >= flat_high)
    )

    conds["clean_combined"] = lambda x: (
        (x["snr_proxy"] >= snr_high) &
        (x["noise_floor"] <= nf_low) &
        (x["spectral_flatness_mean"] <= flat_low)
    )

    # --- strict ---
    conds["strict_noisy"] = lambda x: (
        (x["snr_proxy"] <= snr_low) &
        (x["noise_floor"] >= nf_high) &
        (x["spectral_flatness_mean"] >= flat_high) &
        (x["dynamic_range"] <= dyn_low)
    )

    conds["strict_clean"] = lambda x: (
        (x["snr_proxy"] >= snr_high) &
        (x["noise_floor"] <= nf_low) &
        (x["spectral_flatness_mean"] <= flat_low) &
        (x["dynamic_range"] >= dyn_high)
    )

    return conds


# -----------------------------
# Sampling logic (pairs)
# -----------------------------
def stratified_sample_feature(
    df,
    condition_fn,
    m_dvi,
    m_noise,
    n_pairs,
    allow_clique_reuse=True,
    replace=False,
    random_state=None,
):
    rng = np.random.default_rng(random_state)

    cond_mask = condition_fn(df)

    feasible = {}
    for clique, group in df.groupby("clique"):
        dvi_items = group[group["dvi"] == True]
        noise_items = group[cond_mask.loc[group.index]]

        if len(dvi_items) >= m_dvi and len(noise_items) >= m_noise:
            feasible[clique] = {
                "dvi": dvi_items,
                "noise": noise_items
            }

    if not feasible:
        raise ValueError("No clique satisfies condition.")

    if not allow_clique_reuse and len(feasible) < n_pairs:
        raise ValueError("Not enough cliques.")

    available_cliques = list(feasible.keys())
    results = []
    pair_id = 1

    while pair_id <= n_pairs:
        if not available_cliques:
            raise ValueError(f"Only built {pair_id-1} pairs.")

        clique = rng.choice(available_cliques)

        dvi_group = feasible[clique]["dvi"]
        noise_group = feasible[clique]["noise"]

        try:
            dvi_sample = dvi_group.sample(n=m_dvi, replace=replace, random_state=random_state)

            noise_pool = noise_group.drop(dvi_sample.index, errors="ignore")

            if len(noise_pool) < m_noise and not replace:
                raise ValueError

            noise_sample = noise_pool.sample(n=m_noise, replace=replace, random_state=random_state)

            combined = pd.concat([dvi_sample, noise_sample]).copy()
            combined["pair_id"] = pair_id
            combined["clique_used"] = clique

            results.append(combined)
            pair_id += 1

            if not allow_clique_reuse:
                available_cliques.remove(clique)

        except ValueError:
            if not allow_clique_reuse:
                available_cliques.remove(clique)
            continue

    return pd.concat(results, ignore_index=True)


# -----------------------------
# Save
# -----------------------------
def save_df_as_torch(df, out_path, split):
    index = df["clique"].astype(str) + ":" + df["version"].astype(str)

    info_dict = {
        idx: dict(zip(df.columns, row))
        for idx, row in zip(index, df.itertuples(index=False, name=None))
    }

    split_dict = {split: df.groupby("clique")["version"].apply(list).to_dict()}

    torch.save({"info": info_dict, "split": split_dict}, out_path)
    print(f"Saved: {out_path}")


# -----------------------------
# Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--n-pairs", type=int, default=500)
    parser.add_argument("--m-dvi", type=int, default=1)
    parser.add_argument("--m-noise", type=int, default=1)

    parser.add_argument("--quantile", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no-clique-reuse", action="store_false",
                        dest="allow_clique_reuse")

    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = parse_args()

    df, _ = load_dataset(args.input)
    df = df[df["split"] == args.split]

    os.makedirs(args.output, exist_ok=True)

    conds = build_conditions(df, args.quantile)

    for name, cond_fn in conds.items():
        try:
            subset = stratified_sample_feature(
                df,
                condition_fn=cond_fn,
                m_dvi=args.m_dvi,
                m_noise=args.m_noise,
                n_pairs=args.n_pairs,
                allow_clique_reuse=args.allow_clique_reuse,
                random_state=args.seed
            )

            save_df_as_torch(
                subset,
                os.path.join(args.output, f"{name}.pt"),
                args.split
            )

        except ValueError as e:
            print(f"Skipping {name}: {e}")