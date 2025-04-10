# Discogs-VI-2
The 2nd Version of the Discogs-VI-YT dataset. Only the musical works are based on Discogs. However, the versions are crawled from YouTube and are not necessarily listed on any platform based on manual collection (eg. Discogs, SecondHandSongs, etc.).

# Dataset Creation

## Extract titles from Discogs-VI-YT 
```
python get_unique_titles.py data/discogs/Discogs-VI-YT-20240701.jsonl data/discogs/one_title_per_clique.json
```
