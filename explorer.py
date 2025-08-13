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
    'clique', 'version', 'title', 'data_source',
    'artist', 'year', 'genres', 'styles', 'country', 
    'concepts', 'instruments_groups', 'segments', 'tempo',  
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


def prettify_column_name(col_name):
    return col_name.replace("_", " ").title()

# Your filter functions without color param and without colored header
def multiselect_filter(df, column):
    options = sorted(df[column].dropna().unique())
    selected = st.sidebar.multiselect(column.replace('_', ' ').title(), options, key=f"multiselect_{column}")
    if selected:
        return df[df[column].isin(selected)]
    return df

def text_filter(df, column):
    text = st.sidebar.text_input(column.replace('_', ' ').title(), key=f"textinput_{column}")
    if text:
        return df[df[column].str.contains(text, case=False, na=False)]
    return df

color_general = "#D55E00"  
color_discogs = "#0072B2"  
color_enriched = "#009E73" 

# Reverse map: color → list of columns
color_groups = {
    color_general: [
        "title", "subset", "data_source"
    ],
    color_discogs: [
        "artist", "writers", "year", "country", "genres", "styles"
    ],
    color_enriched: [
        "concepts", "instruments_groups", "segments", "tempo"
    ],
}

def filter_group_box(title, color, columns, df):
    # Calculate approx height per filter (adjust as needed)
    approx_height_per_filter = 60  # px; guess based on typical widget height

    n_filters = len(columns)
    total_height = approx_height_per_filter * n_filters + 50  # extra padding for title etc

    # Colored "box" div with padding + border-radius wrapping approx the whole group
    st.sidebar.markdown(
        f"""
        <div style="
            background-color: {color};
            padding: 15px 15px 15px 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            color: white;
            font-weight: bold;
            font-size: 18px;
            position: relative;
        ">
            {title}
            <div style="
                position: absolute;
                top: 40px;
                left: 0;
                width: 100%;
                height: {total_height}px;
                pointer-events: none;  /* so clicks go through */
                border-radius: 0 0 10px 10px;
                background-color: {color};
                opacity: 0.2;
                z-index: -1;
            "></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    filtered = df.copy()
    for col in sorted(columns):
        if col in ["title", "genres", "styles", "concepts"]:
            filtered = text_filter(filtered, col)
        else:
            filtered = multiselect_filter(filtered, col)

    return filtered


filtered_df = df.copy()

filtered_df = filter_group_box("General Filters", color_general, color_groups[color_general], filtered_df)
filtered_df = filter_group_box("Discogs Filters", color_discogs, color_groups[color_discogs], filtered_df)
filtered_df = filter_group_box("Enriched Filters", color_enriched, color_groups[color_enriched], filtered_df)

tab1, tab2 = st.tabs(["Version Explorer", "Clique Explorer"])

# TAB 1: Random Version
with tab1:
    if filtered_df.empty:
        st.warning("No entries match your filters.")
        st.stop()

    # Always show the button
    if st.button("Pick another random version"):
        random_idx = random.randint(0, len(filtered_df) - 1)
        st.session_state.selected_version = filtered_df.iloc[random_idx]["version"]
        st.rerun()

    # Select version
    if "selected_version" in st.session_state:
        version_id = st.session_state.selected_version
        row = df[df["version"] == version_id]
        if row.empty:
            st.warning("Selected version not found after filtering.")
            random_idx = random.randint(0, len(filtered_df) - 1)
            row = filtered_df.iloc[[random_idx]]
            st.session_state.selected_version = row.iloc[0]["version"]
        row = row.iloc[0]
    else:
        random_idx = random.randint(0, len(filtered_df) - 1)
        row = filtered_df.iloc[random_idx]
        st.session_state.selected_version = row["version"]

    # Display info
    st.subheader(f"*{row.title}*")

    # Two columns: left for stacked info boxes, right for video
    left_col, right_col = st.columns([1, 1])

    with left_col:
        # Row 1: General
        st.markdown(f"""
            <div style="
                background-color:{color_general};
                padding:15px;
                border-radius:10px;
                font-size:20px;
                color:white;
            ">
                <h4>General</h4>
                <b>Clique-ID:</b> {row['clique']}<br>
                <b>Version-ID:</b> {row['version']}<br>
                <b>Subset:</b> {row['subset']}<br>
                <b>Data Source:</b> {row['data_source']}
            </div>
        """, unsafe_allow_html=True)

        # Row 2: Discogs
        st.markdown(f"""
            <div style="
                background-color:{color_discogs};
                padding:15px;
                border-radius:10px;
                font-size:20px;
                color:white;
            ">
                <h4>Discogs</h4>
                <b>Artist:</b> {row['artist']}<br>
                <b>Year:</b> {row['year']}<br>
                <b>Country:</b> {row['country']}<br>
                <b>Genre:</b> {row['genres']}<br>
                <b>Style:</b> {row['styles']}
            </div>
        """, unsafe_allow_html=True)

        # Row 3: Enriched
        st.markdown(f"""
            <div style="
                background-color:{color_enriched};
                padding:15px;
                border-radius:10px;
                font-size:20px;
                color:white;
            ">
                <h4>Enriched</h4>
                <b>Concept:</b> {row['concepts']}<br>
                <b>Instrument/Group:</b> {row['instruments_groups']}<br>
                <b>Segment:</b> {row['segments']}<br>
                <b>Tempo:</b> {round(row['tempo'], 2)}<br>
            </div>
        """, unsafe_allow_html=True)

    with right_col:
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
    discogs_version = clique_df[clique_df["dvi"] == True]
    clique_title = discogs_version.iloc[0].title if not discogs_version.empty else "N/A"
    clique_writer = discogs_version.iloc[0].writers if not discogs_version.empty else "N/A"


    # Show clique title as subheader
    st.subheader(f"*{clique_title}*")

    st.markdown(f"**Clique:** `{selected_clique}`")
    st.markdown(f"**Written by:** `{clique_writer}`")
    st.markdown(f"**Subset:** `{clique_df.subset.iloc[0].title()}`")
    st.markdown(f"- **Number of versions:** {len(clique_df)}")

    # Select version from the clique, default to current selected_version
    versions_list = clique_df["version"].tolist()
    try:
        default_idx = versions_list.index(selected_version)
    except ValueError:
        default_idx = 0

    display_df = clique_df[DISPLAY_COLS].rename(columns={col: prettify_column_name(col) for col in DISPLAY_COLS})
    st.dataframe(display_df)

    
