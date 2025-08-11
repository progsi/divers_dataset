# demo.py
import argparse
import os
import json
import streamlit as st
import pandas as pd
import random

# Columns to display
DISPLAY_COLS = [
    'clique', 'version', 'artist', 'title',
    'release_artist_names', 'release_genres', 'release_styles',
    'country', 'labels', 'matched_instruments_groups', 'matched_concepts',
    'subset', 'is_discogs', 'tempo'
]

@st.cache_data
def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    def inverse_split_dict(split_dict):
        clique_to_split = {}
        for split_name, sub_dict in split_dict.items():
            for clique in sub_dict.keys():
                clique_to_split[clique] = split_name
        return clique_to_split

    with open(path, "r") as f:
        meta = json.load(f)
    if isinstance(meta, dict) and "info" in meta:
        info = meta["info"]
        split = meta["split"]
    else:
        info, split = meta

    df = pd.DataFrame.from_dict(info, orient="index")
    clique2split = inverse_split_dict(split)
    df["subset"] = df["clique"].map(clique2split).str.capitalize()
    
    if "youtube_id" not in df.columns:
        df["youtube_id"] = df.filename.apply(lambda x: x.split("/")[-1].split(".")[0])
    
    df["is_discogs"] = ~df.apply(lambda x: x.youtube_id in x.version, axis=1)
    
    return df, meta  


# CLI arguments ----
parser = argparse.ArgumentParser(description="Run the Streamlit app.")
parser.add_argument('--json_file', 
                    type=str, 
                    help='JSON file with dataset.', 
                    default="data/divers1m_json/sample.json")
args = parser.parse_args()

# Load Data ----
df, meta = load_dataset(args.json_file)

st.set_page_config(page_title="Dataset Explorer", layout="wide")
st.title("🎵 Dataset Explorer")

tab1, tab2 = st.tabs(["Random Version", "Random Clique"])

# TAB 1: Random Version
with tab1:
    st.subheader("🎲 Random Version")

    if "selected_version" in st.session_state:
        # Show selected version from tab 2 click
        version_id = st.session_state.selected_version
        row = df[df["version"] == version_id].iloc[0]
    else:
        if st.button("Pick another random version"):
            st.rerun()
        random_idx = random.randint(0, len(df) - 1)
        row = df.iloc[random_idx]

    st.write(row[DISPLAY_COLS])

    youtube_id = row.get("youtube_id", None)
    if pd.notna(youtube_id):
        youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
        st.video(youtube_url)
    else:
        st.info("No YouTube video available for this entry.")

# TAB 2: Random Clique
with tab2:
    st.subheader("🎲 Random Clique")

    if st.button("Pick another random clique"):
        st.session_state.pop("selected_version", None)
        st.rerun()

    # Pick a random clique
    random_clique = random.choice(df["clique"].unique())
    clique_df = df[df["clique"] == random_clique]

    st.markdown(f"**Clique:** `{random_clique}`")
    st.markdown(f"- **Number of versions:** {len(clique_df)}")
    most_freq_title = clique_df["title"].mode().iloc[0] if not clique_df["title"].mode().empty else "N/A"
    st.markdown(f"- **Most frequent title:** {most_freq_title}")

    # Random YouTube video from clique
    sample_row = clique_df.sample(1).iloc[0]
    if pd.notna(sample_row["youtube_id"]):
        st.video(f"https://www.youtube.com/watch?v={sample_row['youtube_id']}")

    # Select version to view in tab1
    selected_version = st.selectbox(
        "Select a version to view in detail:",
        clique_df["version"].tolist()
    )

    st.dataframe(clique_df[DISPLAY_COLS])
