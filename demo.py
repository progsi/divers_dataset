import json
import time
import random
import argparse

import streamlit as st

st.set_page_config(page_title="Discogs-VI-YT", page_icon=":loud_sound:", layout="wide")

#### Load Data ####


# TODO: store in a dict to make it faster
@st.cache_data(ttl=36000, show_spinner=True)
def read_cliques(demo_json: str):
    """Takes a path to a json file containing cliques and returns a
    list of cliques and the number of versions. You should clean the
    cliques file beforehand with utils/prepare_demo.py.

    Parameters:
    -----------
        demo_json: str
            The json file should contain only versions that are
            matched to a YouTube ID.

    Returns:
    --------
        cliques: list [{clique0}, {clique1}, ...]
            List of all the cliques in the json file filtered by the
            number of versions.
        n_versions: list
            Unique clique sizes in the json file. Sorted in ascending
            order.
        titles : list
    """

    t0 = time.monotonic()
    cliques, n_versions, titles = [], [], []
    with open(demo_json, encoding="utf-8") as in_f:
        for jsonline in in_f:
            # Load the clique
            clique = json.loads(jsonline)
            cliques.append(clique)
            # Get the number of versions per clique for the slider
            n_versions.append(len(clique["versions"]))
            # All titles are excatly the same in a clique
            titles.append(
                f'{clique["versions"][0]["tracks"][0]["track_title"]} [{len(clique["versions"])}]'
            )
    n_versions = sorted(list(set(n_versions)))
    st.success(f"Fetched cliques in {time.monotonic()-t0:.1f} seconds.")
    return cliques, n_versions, titles


#### Clique Display ####


def display_clique(clique: dict):
    """Takes a clique dict and displays it on the webpage."""

    clique_title = clique["versions"][0]["tracks"][0]["track_title"]
    st.subheader(
        f"Work: {clique_title} - Written by: {', '.join(clique['versions'][0]['tracks'][0]['track_writer_names'])}"
    )

    st.divider()
    if len(clique["versions"]) == 2:
        N_COLS = 2
    elif len(clique["versions"]) >= 3:
        N_COLS = 3
    else:
        N_COLS = 1
    with st.container():
        columns = st.columns(N_COLS)
        for i, version in enumerate(clique["versions"]):
            # TODO: release year
            with columns[i % N_COLS]:
                # Format the track information
                track = version["tracks"][0]
                # Choose which artist names to display
                if track["track_artist_names"] != []:
                    key1 = "track_artist_names"
                    key2 = "track_artist_ids"
                else:
                    key1 = "release_artist_names"
                    key2 = "release_artist_ids"
                # Make artist names clickable
                a_names = ""
                for j, (artist, artist_id) in enumerate(zip(track[key1], track[key2])):
                    a_names += f"[{artist}](https://www.discogs.com/artist/{artist_id})"
                    if j < len(track[key1]) - 1:
                        a_names += ", "
                # Add featuring artists if available
                if track["track_feat_ids"] != []:
                    a_names += " (feat. "
                    for j, (artist, artist_id) in enumerate(
                        zip(track["track_feat_names"], track["track_feat_ids"])
                    ):
                        a_names += (
                            f"[{artist}](https://www.discogs.com/artist/{artist_id})"
                        )
                        if j < len(track["feat_artist_names"]) - 1:
                            a_names += ", "
                    a_names += ")"
                # Make writer names clickable
                w_names = ""
                for j, (writer, writer_id) in enumerate(
                    zip(track["track_writer_names"], track["track_writer_ids"])
                ):
                    w_names += f"[{writer}](https://www.discogs.com/artist/{writer_id})"
                    if j < len(track["track_writer_names"]) - 1:
                        w_names += ", "
                # Write to the webpage
                st.write(f"{a_names} - {track['track_title']}")
                st.write(
                    f"Release Title: [{track['release_title']}](https://www.discogs.com/release/{track['release_id']})"
                )
                st.write(f"Writer Name(s): {w_names}")
                st.write(f"Genre(s): {', '.join(track['release_genres'])}")
                st.video(version["youtube_video"][0]["url"])
                st.caption(f"Source: {version['youtube_video'][0]['source']}")
                st.caption(f"Version ID: {version['version_id']}")
                st.divider()


#### Pages ####


def page_main():
    # Display the description, give an example of a clique
    st.write(
        "This interface is a demo of the Discogs-Vi-YT dataset."
        " It helps with the visualization and playback of cliques from of a large collection of music."
    )
    st.write(
        "A clique is a group of versions that share the same composition. "
        "For example, a clique can contain the original version of a song, "
        "a live version of it by the same artist and a cover by another artist."
    )
    with st.container():
        columns = st.columns(3)
        with columns[0]:
            st.video("https://www.youtube.com/watch?v=ZEHsIcsjtdI")
        with columns[1]:
            st.video("https://www.youtube.com/watch?v=7DOzITFjq70")
        with columns[2]:
            st.video("https://www.youtube.com/watch?v=td-_pUPVjdo")
    st.write(
        "The videos above are from the same clique and they are versions of each other."
    )


def page_random_clique(cliques, n_versions):

    def sample_clique_with_versions(cliques: list, n_version_choice: int):
        """Filters the cliques with the given number of versions and samples
        one of them randomly. The sampled clique is displayed on the webpage."""

        def sample_clique(cliques: list):
            """Takes a list of cliques, samples one of them randomly
            and displays the sampled clique on the webpage."""

            clique = random.choice(cliques)
            st.subheader(
                f"There are: {len(clique['versions'])} versions inside the selected clique."
            )
            st.caption(f"Clique ID: {clique['clique_id']}")
            print(
                f"Random Clique ID: {clique['clique_id']}"
            )  # To have a record in the terminal
            display_clique(clique)

        print(f"N_vesions: {n_version_choice}")
        cliques_with_n_versions = [
            clique for clique in cliques if len(clique["versions"]) == n_version_choice
        ]
        if cliques_with_n_versions == []:
            st.error(
                "No clique found containing the required number of versions. Choose Again."
            )
            return
        sample_clique(cliques_with_n_versions)

    st.header("Clique size based random search")
    st.write(
        "Choose clique size. A random clique will be sampled that contains that many versions. "
        f"Minimum: {n_versions[0]} Maximum {n_versions[-1]}."
    )
    # Slider to sample a clique
    n_version_choice = st.select_slider(
        "Choose a Number of versions",
        options=n_versions,
        value=2,  # Fatboy SLim - Praise You
        key="n_version_choice",
    )
    sample_clique_with_versions(cliques, n_version_choice)


def page_select_title(cliques, titles):
    st.header("Title Based Clique Search")
    option = st.selectbox(
        "Using this box you can select a title and get the clique that contains it. "
        "The number in the brackets indicates the number of versions in the clique.",
        index=318,  # Fatboy SLim - Praise You
        placeholder="Select a clique title...",
        options=titles,
    )

    title_idx = titles.index(option)
    print(option, title_idx)
    clique = cliques[title_idx]
    st.caption(f"Clique ID: {clique['clique_id']}")
    display_clique(clique)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("demo_json", type=str, help="Path to demo.json")
    args = parser.parse_args()

    st.title("Discogs-VI-YT Demo")

    # Load cliques and filter versions with no videos
    cliques, n_versions, titles = read_cliques(args.demo_json)

    st.sidebar.title("Interface Navigation")
    page = st.sidebar.radio(
        "Choose a page:", ["Main Page", "Random Clique", "Title based search"]
    )

    if page == "Main Page":
        page_main()
    elif page == "Random Clique":
        page_random_clique(cliques, n_versions)
    elif page == "Title based search":
        page_select_title(cliques, titles)