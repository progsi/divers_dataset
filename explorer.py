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
    
    df["concepts_list"], df["concepts_cues_list"] = zip(*df["matched_concepts"].apply(lambda x: get_concepts_and_cues(x, include_cues=True)))
    df["instruments_list"], df["instruments_cues_list"] = zip(*df["matched_instruments_groups"].apply(lambda x: get_concepts_and_cues(x, include_cues=True)))
    df["segments_list"], df["segments_cues_list"] = zip(*df["matched_segments"].apply(lambda x: get_concepts_and_cues(x, include_cues=True)))
    
    # Ensure lists
    for col in ["concepts_list", "instruments_list", "segments_list", "genres_list"]:
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
    
    # String version for display
    def list_to_str(lst):
        return ", ".join(lst) if isinstance(lst, list) else lst
    df["concepts"] = df["concepts_list"].apply(list_to_str)
    df["concepts_cues"] = df["concepts_cues_list"].apply(lambda x: ", ".join([str(i) for i in x]))
    df["instruments_groups"] = df["instruments_list"].apply(list_to_str)
    df["instruments_cues"] = df["instruments_cues_list"].apply(lambda x: ", ".join([str(i) for i in x]))
    df["segments"] = df["segments_list"].apply(list_to_str)
    df["segments_cues"] = df["segments_cues_list"].apply(lambda x: ", ".join([str(i) for i in x]))
    df["tempo"] = df["tempo"].apply(lambda x: round(x) if pd.notna(x) else "?")
    
    df = df.replace(["?", ""], np.nan)
    return df, meta

##################### Streamlit App ###################

st.title("🎶 DiVers1M Explorer")

# Load dataset
DATA_PATH = "data/divers1m_json/sample.json"
df, meta = load_dataset(DATA_PATH)



# ---------------- Filtered DataFrame ----------------
filtered_df = df.copy()
# --- Random Version Button on top ---
if st.sidebar.button("🎲 Random Version"):
    if not filtered_df.empty:
        st.session_state.detail_idx = random.choice(filtered_df.index.tolist())
        # Switch to Version View tab if currently on List View
        st.session_state.active_tab = 0
        st.rerun()
        
# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters")

# --- Filters below the button ---
subset_filter = st.sidebar.multiselect("Subset", options=sorted(df["subset"].dropna().unique()))
genre_filter = st.sidebar.multiselect(
    "Genres", options=sorted({g for lst in df["genres_list"] for g in lst})
)
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
        
# ---------------- Tabs ----------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0
tab1, tab2 = st.tabs(["Version View", "List View"])

# ---------------- Version View ----------------
with tab1:
    if filtered_df.empty:
        st.info("No items match the current filters.")
    else:
        if "detail_idx" not in st.session_state or st.session_state.detail_idx not in filtered_df.index:
            st.session_state.detail_idx = random.choice(filtered_df.index.tolist())
        row = filtered_df.loc[st.session_state.detail_idx]

        col1, col2 = st.columns([2, 3])

        def render_tags_with_cues_dict(label, tags, cues_list, fontsize="18px", row_spacing="2px"):
            """
            Render tags and show a dictionary mapping each tag to its cues.
            Ignores NaNs and None tags.
            """
            import pandas as pd

            # Ensure tags and cues are lists
            if not isinstance(tags, (list, tuple)):
                tags = [tags] if tags is not None else []
            if not isinstance(cues_list, (list, tuple)):
                cues_list = [cues_list] if cues_list is not None else []

            # Filter out NaNs or None tags
            valid_tags = [t for t in tags if t is not None and not (isinstance(t, float) and pd.isna(t))]

            # Render main tags
            if valid_tags:
                tags_html = " ".join([f"<code>{v}</code>" for v in valid_tags])
                st.markdown(f'<div style="font-size:{fontsize}; margin-bottom:{row_spacing};"><b>{label}:</b> {tags_html}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="font-size:{fontsize}; margin-bottom:{row_spacing};"><b>{label}:</b> <i>none</i></div>', unsafe_allow_html=True)

            # Build dict mapping tag -> cues (skip invalid tags)
            tag_to_cues = {}
            for i, tag in enumerate(tags):
                if tag is None or (isinstance(tag, float) and pd.isna(tag)):
                    continue
                if i < len(cues_list) and cues_list[i] is not None:
                    tag_to_cues[tag] = cues_list[i]
                else:
                    tag_to_cues[tag] = []

            if tag_to_cues:
                st.markdown(f'<div style="font-size:{fontsize}; margin-bottom:{row_spacing}; margin-left:10px; color:gray;"><i>Cues:</i> {tag_to_cues}</div>', unsafe_allow_html=True)

        with col1:
            st.markdown(f"### {row['title']}")
            artist = row.get("artist", "Unknown")
            year = row.get("year", "Unknown")
            st.markdown(f'<div style="font-size:20px; margin-bottom:4px;">by <b>{artist}</b> from <b>{year}</b></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:18px; margin-bottom:2px;"><b>Clique:</b> {row["clique"]}  |  <b>Version:</b> {row["version"]}  |  <b>Youtube ID:</b> {row["youtube_id"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:18px; margin-bottom:2px;"><b>Subset:</b> {row["subset"]}  |  <b>Data Source:</b> {row["data_source"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:18px; margin-bottom:8px;"><b>Tempo:</b> {row.get("tempo", "Unknown")}</div>', unsafe_allow_html=True)
            st.markdown('<div style="margin-top:10px;"><b>Tags:</b></div>', unsafe_allow_html=True)

            render_tags_with_cues_dict("Genres", row["genres"], [])  # Genres don’t have cues
            render_tags_with_cues_dict("Styles", row["styles"], [])  # Styles don’t have cues
            render_tags_with_cues_dict("Concepts", row["concepts_list"], row["concepts_cues_list"])
            render_tags_with_cues_dict("Instrument Groups", row["instruments_list"], row["instruments_cues_list"])
            render_tags_with_cues_dict("Segments", row["segments_list"], row["segments_cues_list"])


        with col2:
            if row['youtube_id']:
                youtube_id = row['youtube_id']
                st.markdown(f"""
                <iframe 
                    width="100%" 
                    height="450" 
                    src="https://www.youtube.com/embed/{youtube_id}" 
                    frameborder="0" 
                    allowfullscreen>
                </iframe>
                """, unsafe_allow_html=True)

# ---------------- List View ----------------
with tab2:
    st.subheader("Dataset Overview")
    display_df = filtered_df[DISPLAY_COLS + ["youtube_id"]].copy()
    display_df = display_df.sort_values(by=["clique", "version"])
    display_df = display_df.rename(columns={col: col.replace("_", " ").title() for col in display_df.columns})
    st.dataframe(display_df, hide_index=True, use_container_width=True)
