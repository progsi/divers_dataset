# demo.py
import argparse
import ast
import os
import json
import streamlit as st
import pandas as pd
import random

st.set_page_config(layout="wide")

# Columns to display
DISPLAY_COLS = [
    'clique', 'version', 'title', 'subset',  
    'artist', 'year', 'genres', 'styles', 'country', 
    'instruments_groups', 'matched_concepts', 'tempo',  
]

############### Preprocessing Functions ###############

def join_list(lst):
    if isinstance(lst, str):
        lst = ast.literal_eval(lst)
    elif isinstance(lst, float):
        return "?"
    return ", ".join(lst)

def join_styles(styles):
    if isinstance(styles, str):
        styles = ast.literal_eval(styles)
    elif isinstance(styles, float):
        return "?"
    styles = [style[1] for style in styles]
    return ", ".join(styles)

def get_concept_str(concept_dict, cues=False):
    concept_to_cues = {}

    for field, matches in concept_dict.items():
        for concept, cue in matches.items():
            concept_cap = concept.title()
            if concept_cap not in concept_to_cues:
                concept_to_cues[concept_cap] = set()
            if cues and cue is not None:
                concept_to_cues[concept_cap].add(cue)

    concept_strs = []
    for concept, cue_set in concept_to_cues.items():
        if cues and cue_set:
            cue_list = sorted(cue_set)  # sorted for consistent output
            concept_strs.append(f"{concept} ({', '.join(cue_list)})")
        else:
            concept_strs.append(concept)

    return ", ".join(concept_strs) if concept_strs else "(None)"

##################### Load Dataset Function ###############

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
    df["subset"] = df["clique"].map(clique2split).str.title()
    
    if "youtube_id" not in df.columns:
        df["youtube_id"] = df.filename.apply(lambda x: x.split("/")[-1].split(".")[0])
    
    is_discogs = ~df.apply(lambda x: x.youtube_id in x.version, axis=1)
    df["data_source"] = is_discogs.map({True: "Discogs", False: "YouTube"})
    
    df["year"] = df.released.fillna("?") 
    df["country"] = df.country.fillna("?")
    df["writers"] = df.track_writer_names.apply(join_list)
    df["genres"] = df.release_genres.apply(join_list)
    df["styles"] = df.release_styles.apply(join_styles)

    # additional
    df["concepts"] = df["matched_concepts"].apply(lambda x: get_concept_str(x, cues=True))
    df["instruments_groups"] = df["matched_instruments_groups"].apply(lambda x: get_concept_str(x, cues=False))
    df["segments"] = df["matched_segments"].apply(lambda x: get_concept_str(x, cues=False))

    return df, meta  

##################### Streamlit App #####################

# CLI arguments ----
parser = argparse.ArgumentParser(description="Run the Streamlit app.")
parser.add_argument('--json_file', 
                    type=str, 
                    help='JSON file with dataset.', 
                    default="data/divers1m_json/sample.json")
args = parser.parse_args()

# Load Data ----
df, meta = load_dataset(args.json_file)

st.sidebar.header("Filters")

def prettify_column_name(col_name):
    return col_name.replace("_", " ").title()

def multiselect_filter(df, column):
    options = sorted(df[column].dropna().unique())
    selected = st.sidebar.multiselect(prettify_column_name(column), options)
    if selected:
        return df[df[column].isin(selected)]
    return df

def text_filter(df, column):
    text = st.sidebar.text_input(prettify_column_name(column))
    if text:
        return df[df[column].str.contains(text, case=False, na=False)]
    return df

# Apply filters one by one
filtered_df = df.copy()
for col in ["title", "split", "data_source",
            "artist", "writers", "year", "country", "genres", "styles",
            "concepts", "instruments_groups", "segments"]:
    if col in ["title", "genres", "styles", "concepts"]:
        filtered_df = text_filter(filtered_df, col)  # free text search for these
    else:
        filtered_df = multiselect_filter(filtered_df, col)  # choose from available options

tab1, tab2 = st.tabs(["Version Explorer", "Clique Explorer"])

# TAB 1: Random Version
with tab1:
    if filtered_df.empty:
        st.warning("No entries match your filters.")
        st.stop()

    # Always show the button
    if st.button("Pick another random version"):
        # Pick a new random version and save it in session_state
        random_idx = random.randint(0, len(filtered_df) - 1)
        st.session_state.selected_version = filtered_df.iloc[random_idx]["version"]
        st.experimental_rerun()

    # Use selected_version if set, else pick a random one (once)
    if "selected_version" in st.session_state:
        version_id = st.session_state.selected_version
        row = df[df["version"] == version_id]
        if row.empty:
            st.warning("Selected version not found after filtering.")
            # Fallback: pick random
            random_idx = random.randint(0, len(filtered_df) - 1)
            row = filtered_df.iloc[[random_idx]]
            st.session_state.selected_version = row.iloc[0]["version"]
        row = row.iloc[0]
    else:
        # Initialize selected_version with a random version
        random_idx = random.randint(0, len(filtered_df) - 1)
        row = filtered_df.iloc[random_idx]
        st.session_state.selected_version = row["version"]

    # Display info
    st.subheader(f"*{row.title}*")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Clique-ID:** {row.clique}")
        st.write(f"**Version-ID:** {row.version}")
        st.write(f"**Subset:** {row.subset}")
        st.write(f"**Data Source:** {row.data_source}")

    with col2:
        st.write(f"**Artist:** {row.artist}")
        st.write(f"**Country:** {row.country}")
        st.write(f"**Genre:** {row.genres}")
        st.write(f"**Style:** {row.styles}")

    with col3:
        st.write(f"**Concept:** {row.concepts}")
        st.write(f"**Instrument/Group:** {row.instruments_groups}")
        st.write(f"**Segment:** {row.segments}")
        st.write(f"**Tempo:** {round(row.tempo, 2)} BPM")

    youtube_id = row.get("youtube_id", None)
    if pd.notna(youtube_id):
        youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
        st.video(youtube_url)
    else:
        st.info("No YouTube video available for this entry.")
        
# TAB 2: Random Clique
with tab2:
    # Initialize selected_version if missing
    if "selected_version" not in st.session_state:
        st.session_state.selected_version = df.iloc[0]["version"]

    selected_version = st.session_state.selected_version

    # Get row and clique info
    row = df[df["version"] == selected_version]
    if row.empty:
        st.error("Selected version not found in the dataset.")
        st.stop()
    row = row.iloc[0]

    selected_clique = row.clique
    clique_df = df[df["clique"] == selected_clique]

    # Find first Discogs title or fallback
    discogs_titles = clique_df[clique_df["dvi"] == True]["title"]
    clique_title = discogs_titles.iloc[0] if not discogs_titles.empty else "N/A"

    # Show clique title as subheader
    st.subheader(f"*{clique_title}*")

    st.markdown(f"**Clique:** `{selected_clique}`")
    st.markdown(f"- **Number of versions:** {len(clique_df)}")

    # Select version from the clique, default to current selected_version
    versions_list = clique_df["version"].tolist()
    try:
        default_idx = versions_list.index(selected_version)
    except ValueError:
        default_idx = 0

    display_df = clique_df[DISPLAY_COLS].rename(columns={col: prettify_column_name(col) for col in DISPLAY_COLS})
    st.dataframe(display_df)

    
