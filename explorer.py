import json
import os
import ast
import random
import numpy as np
import pandas as pd
import streamlit as st

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
        return np.nan
    return ", ".join(lst)

def join_styles(styles):
    if isinstance(styles, str):
        styles = ast.literal_eval(styles)
    elif isinstance(styles, float):
        return np.nan
    styles = [style[1] for style in styles]
    return ", ".join(styles)

def get_concepts_and_cues(concept_dict, include_cues=False):
    """Get list of concepts and their cues from the concept dictionary.
    """
    concept_to_cues = {}

    for field, matches in concept_dict.items():
        for concept, cue in matches.items():
            concept_cap = concept.title()
            if concept_cap not in concept_to_cues:
                concept_to_cues[concept_cap] = set()
            if include_cues and cue is not None:
                concept_to_cues[concept_cap].add(cue)

    concepts = list(concept_to_cues.keys())
    cues_list = [sorted(cues) if include_cues else [] for cues in concept_to_cues.values()]
    return concepts, cues_list

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
    df["genres_list"] = df.release_genres.fillna("[]")
    df["genres"] = df.genres_list.apply(join_list)
    df["styles"] = df.release_styles.apply(join_styles)
    
    # additional
    df["concepts_list"], df["concepts_cues_list"] = zip(*df["matched_concepts"].apply(lambda x: get_concepts_and_cues(x, include_cues=True)))
    df["instruments_list"], df["instruments_cues_list"] = zip(*df["matched_instruments_groups"].apply(lambda x: get_concepts_and_cues(x, include_cues=True)))
    df["segments_list"], df["segments_cues_list"] = zip(*df["matched_segments"].apply(lambda x: get_concepts_and_cues(x, include_cues=True)))
    df["concepts_list"] = df["concepts_list"].apply(lambda x: x if isinstance(x, list) else [])
    df["instruments_list"] = df["instruments_list"].apply(lambda x: x if isinstance(x, list) else [])
    df["segments_list"] = df["segments_list"].apply(lambda x: x if isinstance(x, list) else [])
    df["genres_list"] = df["genres_list"].apply(lambda x: x if isinstance(x, list) else [])

    # String version (for display)
    def list_to_str(lst):
        return ", ".join(lst) if isinstance(lst, list) else lst
    df["concepts"] = df["concepts_list"].apply(list_to_str)
    df["concepts_cues"] = df["concepts_cues_list"].apply(lambda x: ", ".join([str(i) for i in x]))
    df["instruments_groups"] = df["instruments_list"].apply(list_to_str)
    df["instruments_cues"] = df["instruments_cues_list"].apply(lambda x: ", ".join([str(i) for i in x]))
    df["segments"] = df["segments_list"].apply(list_to_str)
    df["segments_cues"] = df["segments_cues_list"].apply(lambda x: ", ".join([str(i) for i in x]))
    
    df["tempo"] = df["tempo"].apply(lambda x: round(x) if pd.notna(x) else "?")
    
    # --- normalize "?" and empty values ---
    df = df.replace(["?", ""], np.nan)
    
    return df, meta

##################### Streamlit App ###################


def prettify_column_name(col_name):
    return col_name.replace("_", " ").title()

st.title("🎶 DiVers1M Explorer")

# Load dataset
DATA_PATH = "data/divers1m_json/sample.json"  # change if needed
df, meta = load_dataset(DATA_PATH)

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters")

subset_filter = st.sidebar.multiselect("Subset", options=sorted(df["subset"].unique()))
genre_filter = st.sidebar.multiselect(
    "Genres", options=sorted({g for lst in df["genres_list"] for g in lst})
)

# Use _list columns for filtering
concept_filter = st.sidebar.multiselect(
    "Concepts", options=sorted({c for lst in df["concepts_list"] for c in lst})
)
instruments_filter = st.sidebar.multiselect(
    "Instruments/Groups", options=sorted({i for lst in df["instruments_list"] for i in lst})
)
segments_filter = st.sidebar.multiselect(
    "Segments", options=sorted({s for lst in df["segments_list"] for s in lst})
)

artist_filter = st.sidebar.text_input("Search Artist")
title_filter = st.sidebar.text_input("Search Title")

filtered_df = df.copy()

if subset_filter:
    filtered_df = filtered_df[filtered_df["subset"].isin(subset_filter)]
if concept_filter:
    mask = filtered_df["concepts_list"].apply(lambda x: any(c in x for c in concept_filter) if isinstance(x, list) else False)
    filtered_df = filtered_df[mask]
if instruments_filter:
    mask = filtered_df["instruments_list"].apply(lambda x: any(i in x for i in instruments_filter) if isinstance(x, list) else False)
    filtered_df = filtered_df[mask]
if segments_filter:
    mask = filtered_df["segments_list"].apply(lambda x: any(s in x for s in segments_filter) if isinstance(x, list) else False)
    filtered_df = filtered_df[mask]
if genre_filter:
    mask = filtered_df["genres_list"].apply(lambda x: any(g in x for g in genre_filter) if isinstance(x, list) else False)
    filtered_df = filtered_df[mask]
if artist_filter:
    filtered_df = filtered_df[filtered_df["artist"].str.contains(artist_filter, case=False, na=False)]
if title_filter:
    filtered_df = filtered_df[filtered_df["title"].str.contains(title_filter, case=False, na=False)]


# ---------------- Tabs ----------------
tab1, tab2 = st.tabs(["Version View", "List View"])


# ---------------- Detail View Tab -----------
with tab1:
    if filtered_df.empty:
        st.info("No items match the current filters.")
    else:
        filters_hash = hash(tuple(filtered_df.index.tolist()))
        if "last_filters_hash" not in st.session_state or st.session_state.last_filters_hash != filters_hash:
            st.session_state.detail_idx = random.choice(filtered_df.index.tolist())
            st.session_state.last_filters_hash = filters_hash

        def get_random_row(df_subset, current_idx=None):
            available = df_subset.index.tolist()
            if current_idx in available and len(available) > 1:
                available.remove(current_idx)
            return df_subset.loc[random.choice(available)]

        row = filtered_df.loc[st.session_state.detail_idx]

        col1, col2 = st.columns([2, 3])
        def render_tags(label, values, fontsize="18px", row_spacing="2px"):
            if pd.isna(values) or values is None:
                st.markdown(f'<div style="font-size:{fontsize}; margin-bottom:{row_spacing};"><b>{label}:</b> <i>none</i></div>', unsafe_allow_html=True)
                return

            # If it's a single string, split on commas
            if isinstance(values, str):
                values = [v.strip() for v in values.split(",") if v.strip()]

            # If it's already a list/iterable
            if isinstance(values, (list, tuple)):
                values = [str(v).strip() for v in values if pd.notna(v) and str(v).strip()]

            if values:
                tags_html = " ".join([f"<code>{v}</code>" for v in values])
                st.markdown(f'<div style="font-size:{fontsize}; margin-bottom:{row_spacing};"><b>{label}:</b> {tags_html}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="font-size:{fontsize}; margin-bottom:{row_spacing};"><b>{label}:</b> <i>none</i></div>', unsafe_allow_html=True)

        with col1:
            st.markdown(f"### {row['title']}")
            artist = row.get("artist", "Unknown")
            year = row.get("year", "Unknown")
            title = row.get("title", "")
            st.markdown(f'<div style="font-size:20px; margin-bottom:4px;">by <b>{artist}</b> from <b>{year}</b></div>', unsafe_allow_html=True)

            # Row 1: core IDs
            st.markdown(
                f'<div style="font-size:18px; margin-bottom:2px;"><b>Clique:</b> {row["clique"]}  |  <b>Version:</b> {row["version"]}  |  <b>Youtube ID:</b> {row["youtube_id"]}</div>',
                unsafe_allow_html=True
            )

            # Row 2: meta
            st.markdown(
                f'<div style="font-size:18px; margin-bottom:2px;"><b>Subset:</b> {row["subset"]}  |  <b>Data Source:</b> {row["data_source"]}</div>',
                unsafe_allow_html=True
            )

            # Extra info
            st.markdown(f'<div style="font-size:18px; margin-bottom:8px;"><b>Tempo:</b> {row.get("tempo", "Unknown")}</div>', unsafe_allow_html=True)

            # --- Tags Section ---
            st.markdown('<div style="margin-top:10px;"><b>Tags:</b></div>', unsafe_allow_html=True)
            render_tags("Genres", row["genres"], fontsize="18px", row_spacing="2px")
            render_tags("Styles", row["styles"], fontsize="18px", row_spacing="2px")
            render_tags("Concepts", row["concepts"], fontsize="18px", row_spacing="2px")
            render_tags("Instrument Groups", row["instruments_groups"], fontsize="18px", row_spacing="2px")
            render_tags("Segments", row["segments"], fontsize="18px", row_spacing="2px")

            if st.button("🎲 Random Version"):
                random_row = get_random_row(filtered_df, current_idx=row.name)
                st.session_state.detail_idx = random_row.name
                st.rerun()

        with col2:
            if row['youtube_id']:
                youtube_id = row['youtube_id']
                # Set width=100% to fit column, height smaller than default
                st.markdown(f"""
                <iframe 
                    width="100%" 
                    height="450" 
                    src="https://www.youtube.com/embed/{youtube_id}" 
                    frameborder="0" 
                    allowfullscreen>
                </iframe>
                """, unsafe_allow_html=True)


# ----------- Overview Tab -----------
with tab2:
    st.subheader("Dataset Overview")

    display_df = filtered_df[DISPLAY_COLS + ["youtube_id"]].copy()
    
    # Sort by clique and version
    display_df = display_df.sort_values(by=["clique", "version"])
    
    display_df = display_df.rename(columns={col: prettify_column_name(col) for col in display_df.columns})

    # Store selection in session_state
    if "selected_idx" not in st.session_state:
        st.session_state.selected_idx = None

    # Use st.data_editor for row selection
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
    )
    
