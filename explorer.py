import json
import os
import ast
import random
import numpy as np
import pandas as pd
import streamlit as st
import datetime


st.set_page_config(layout="wide")

# Columns to display
DISPLAY_COLS = [
    'clique', 'version', 'title', 'data_source',
    'artist', 'year', 'genres', 'styles', 'country', 
    'concepts', 'instruments_groups', 'segments', 'tempo',  
]

if "mode" not in st.session_state:
    st.session_state.mode = "view"

# Initialize annotation storage
if "annotations" not in st.session_state:
    st.session_state.annotations = pd.DataFrame(
        columns=[
            "youtube_id",
            "clique",
            "version",
            "annotator",
            "timestamp",
            "note",
            "confirm_version",
            "genres_list_removed",
            "concepts_list_removed",
            "instruments_list_removed",
            "segments_list_removed",
        ]
    )
    
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
# ---------------- Filtered DataFrame ----------------
filtered_df = df.copy()

# Create a stable order for navigation
ordered_indices = filtered_df.index.sort_values().tolist()
if "detail_idx" not in st.session_state or st.session_state.detail_idx not in ordered_indices:
    st.session_state.detail_idx = ordered_indices[0]

# # --- Random button on top ---
# if st.sidebar.button("🎲 Random"):
#     if not filtered_df.empty:
#         st.session_state.detail_idx = random.choice(filtered_df.index.tolist())
#         st.session_state.active_tab = 0  # stay in Version View
#         st.rerun()

# --- Prev / Next navigation ---
col_nav1, col_nav2 = st.sidebar.columns([1, 1])

with col_nav1:
    if st.button("⬅️ Prev"):
        current_pos = ordered_indices.index(st.session_state.detail_idx)
        if current_pos > 0:
            st.session_state.detail_idx = ordered_indices[current_pos - 1]
            st.rerun()

with col_nav2:
    if st.button("Next ➡️"):
        current_pos = ordered_indices.index(st.session_state.detail_idx)
        if current_pos < len(ordered_indices) - 1:
            st.session_state.detail_idx = ordered_indices[current_pos + 1]
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

# --- Mode & annotation controls ---
# --- Annotation controls ---
st.sidebar.header("Annotation Controls")

if "annotator" not in st.session_state:
    st.session_state.annotator = ""
if "mode" not in st.session_state:
    st.session_state.mode = "view"

# Annotator name
annotator = st.sidebar.text_input("Annotator name", value=st.session_state.annotator)
st.session_state.annotator = annotator

# Mode selector – default to current session mode
chosen_mode = st.sidebar.radio(
    "Choose mode:",
    ["view", "annotate"],
    index=0 if st.session_state.mode == "view" else 1
)

# Only switch to annotate if annotator name is provided
if chosen_mode == "annotate" and not st.session_state.annotator:
    st.sidebar.error("Enter annotator name to enable annotation mode")
    st.session_state.mode = "view"
else:
    st.session_state.mode = chosen_mode
 
# ---------------- Tabs ----------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0
tab1, tab2 = st.tabs(["Version View", "List View"])

# ---------------- Version View ----------------
with tab1:
    if filtered_df.empty:
        st.info("No items match the current filters.")
    else:
        # Ensure current detail_idx exists
        if "detail_idx" not in st.session_state or st.session_state.detail_idx not in filtered_df.index:
            st.session_state.detail_idx = random.choice(filtered_df.index.tolist())
        row = filtered_df.loc[st.session_state.detail_idx]

        # --- Random button in sidebar ---
        if st.sidebar.button("🎲 Random"):
            if not filtered_df.empty:
                st.session_state.detail_idx = random.choice(filtered_df.index.tolist())
                st.session_state.active_tab = 0
                st.rerun()

        col1, col2 = st.columns([2, 3])

        # Utility function to render tags in columns
        def render_tags_columns(tags, row_key, max_cols=3):
            for i in range(0, len(tags), max_cols):
                cols = st.columns(max_cols)
                for j, tag in enumerate(tags[i:i+max_cols]):
                    col = cols[j]
                    yield col, tag, f"{row_key}_{tag}"

        # Render metadata
        with col1:
            st.markdown(f"### {row['title']}")
            artist = row.get("artist", "Unknown")
            year = row.get("year", "Unknown")
            st.markdown(f'<div style="font-size:20px; margin-bottom:4px;">by <b>{artist}</b> from <b>{year}</b></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:18px; margin-bottom:2px;"><b>Clique:</b> {row["clique"]}  |  <b>Version:</b> {row["version"]}  |  <b>Youtube ID:</b> {row["youtube_id"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:18px; margin-bottom:2px;"><b>Subset:</b> {row["subset"]}  |  <b>Data Source:</b> {row["data_source"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:18px; margin-bottom:8px;"><b>Tempo:</b> {row.get("tempo", "Unknown")}</div>', unsafe_allow_html=True)
            st.markdown('<div style="margin-top:10px;"><b>Tags:</b></div>', unsafe_allow_html=True)

            if st.session_state.mode == "annotate":
                st.markdown("#### Annotation Mode")

                # Load previous annotation if exists
                prev_annotation = st.session_state.annotations[
                    (st.session_state.annotations["youtube_id"] == row["youtube_id"]) &
                    (st.session_state.annotations["clique"] == row["clique"]) &
                    (st.session_state.annotations["version"] == row["version"]) &
                    (st.session_state.annotations["annotator"] == st.session_state.annotator)
                ]
                prev_unmarked = { "genres_list": [], "concepts_list": [], "instruments_list": [], "segments_list": [] }
                prev_note = ""
                prev_confirm = True
                if not prev_annotation.empty:
                    last = prev_annotation.iloc[-1]
                    for colname in ["genres_list", "concepts_list", "instruments_list", "segments_list"]:
                        if f"{colname}_removed" in last:
                            val = last[f"{colname}_removed"]
                            if isinstance(val, list):
                                prev_unmarked[colname] = val
                    prev_note = last.get("note", "")
                    prev_confirm = last.get("confirm_version", True)

                # Confirm Version checkbox
                confirm_key = f"confirm_{row['youtube_id']}_{row['clique']}_{row['version']}_{st.session_state.annotator}"
                confirm_version = st.checkbox("Confirm Version", value=prev_confirm, key=confirm_key)

                unmarked = {}

                # Render tags side by side in columns
                for colname, label in [
                    ("genres_list", "Genres"),
                    ("concepts_list", "Concepts"),
                    ("instruments_list", "Instrument Groups"),
                    ("segments_list", "Segments"),
                ]:
                    tags = row[colname] if isinstance(row[colname], list) else []
                    st.write(f"**{label}:**")
                    unchecked = []
                    for col, tag, key in render_tags_columns(tags, f"{row['youtube_id']}_{row['clique']}_{row['version']}_{colname}"):
                        is_checked = True
                        if tag in prev_unmarked.get(colname, []):
                            is_checked = False
                        checked = col.checkbox(tag, value=is_checked, key=key)
                        if not checked:
                            unchecked.append(tag)
                    if unchecked:
                        unmarked[colname] = unchecked

                # Single-line note below tags
                note_key = f"note_{row['youtube_id']}_{row['clique']}_{row['version']}"
                note = st.text_input("Additional note", key=note_key, value=prev_note)

                # Build annotation row
                ann_row = {
                    "clique": row["clique"],
                    "version": row["version"],
                    "youtube_id": row["youtube_id"],
                    "annotator": st.session_state.annotator,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "note": note,
                    "confirm_version": confirm_version,
                }
                for colname, unchecked_tags in unmarked.items():
                    ann_row[f"{colname}_removed"] = unchecked_tags

                # Remove previous entry for this clique/version/youtube_id/annotator
                if not st.session_state.annotations.empty:
                    st.session_state.annotations = st.session_state.annotations[
                        ~(
                            (st.session_state.annotations["youtube_id"] == row["youtube_id"]) &
                            (st.session_state.annotations["clique"] == row["clique"]) &
                            (st.session_state.annotations["version"] == row["version"]) &
                            (st.session_state.annotations["annotator"] == st.session_state.annotator)
                        )
                    ]

                # Append new annotation
                st.session_state.annotations = pd.concat(
                    [st.session_state.annotations, pd.DataFrame([ann_row])],
                    ignore_index=True
                )

                # **Auto-save on any change**
                st.session_state.annotations.to_json("annotations.json", orient="records", indent=2)

            else:
                # View mode
                def render_tags_with_cues_dict(label, tags, cues_list, fontsize="18px", row_spacing="2px"):
                    if isinstance(tags, str):
                        tags = [tags] if tags else []
                    valid_tags = [t for t in tags if t is not None and not (isinstance(t, float) and pd.isna(t))]
                    if valid_tags:
                        tags_html = " ".join([f"<code>{v}</code>" for v in valid_tags])
                        st.markdown(f'<div style="font-size:{fontsize}; margin-bottom:{row_spacing};"><b>{label}:</b> {tags_html}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="font-size:{fontsize}; margin-bottom:{row_spacing};"><b>{label}:</b> <i>none</i></div>', unsafe_allow_html=True)

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
                    
                # Fix for genres and styles: ensure list, even if NaN
                def safe_str_to_list(val):
                    if isinstance(val, str):
                        return val.split(", ") if val else []
                    elif isinstance(val, list):
                        return val
                    else:
                        return []

                genres_list = safe_str_to_list(row["genres"])
                styles_list = safe_str_to_list(row["styles"])

                render_tags_with_cues_dict("Genres", genres_list, [])
                render_tags_with_cues_dict("Styles", styles_list, [])
                render_tags_with_cues_dict("Concepts", row["concepts_list"], row["concepts_cues_list"])
                render_tags_with_cues_dict("Instrument Groups", row["instruments_list"], row["instruments_cues_list"])
                render_tags_with_cues_dict("Segments", row["segments_list"], row["segments_cues_list"])

        # ---------------- Video Column ----------------
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
