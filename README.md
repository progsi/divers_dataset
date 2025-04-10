# Discogs-VI-2
The 2nd Version of the Discogs-VI-YT dataset. Only the musical works are based on Discogs. However, the versions are crawled from YouTube and are not necessarily listed on any platform based on manual collection (eg. Discogs, SecondHandSongs, etc.).

# Dataset Creation

## Extract titles from Discogs-VI-YT 
To obtain one query (song title) per clique, we first create a new file. We use the cleaned song titles from Discogs-VI-YT.
```
python get_unique_titles.py data/discogs/Discogs-VI-YT-20240701.jsonl data/discogs/one_title_per_clique.json
```
## Search on YouTube
We now search for up to 500 results per clique, using the text query (song title).
```
python search_youtube.py data/discogs/one_title_per_clique.json data/youtube/ 
```
## Filtering
TBA
## Information Extraction with LLM
TBA
