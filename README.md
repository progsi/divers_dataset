# Discogs-VI-2
The 2nd Version of the [*Discogs-VI-YT*](https://github.com/MTG/discogs-vi-dataset) dataset. Only the musical works are based on *Discogs*. However, the versions are crawled from YouTube and are not necessarily listed on any platform based on manual collection (eg. *Discogs*, *SecondHandSongs*, etc.).

# Dataset Creation
## Extract titles from Discogs-VI-YT 
To obtain one query (song title) per clique, we first create a new file. We use the cleaned song titles from *Discogs-VI-YT*.
```
python get_unique_titles.py data/discogs/Discogs-VI-YT-20240701.jsonl data/discogs/one_title_per_clique.json
```
## Search on YouTube
We now search for up to 500 results per clique, using the text query (song title).
```
python search_youtube.py data/discogs/one_title_per_clique.json data/youtube/ 
```
## Filtering
### Exclude dataset matches
We want to exclude videos which are also contained as versions in either *Discogs-VI-YT*, [*SHS100K2*](https://github.com/NovaFrost/SHS100K2) or [*Da-Tacos*](https://github.com/MTG/da-tacos). Please note that we have our own CSV files for this process which we can provide on request. Given these files, we run:

```
python filter_youtube_ids.py data/youtube data/filter/youtube_id --discogs_path data/discogs/Discogs-VI-YT-20240701-light.json --shs_csv_path ../data/shs100k2.csv --datacos_csv_path ../data/da-tacos.csv
```
### Exclude videos by duration
We exclude videos longer than 20 minutes (like in *Discogs-VI-YT*) and under 10 seconds.
```
python filter_duration.py data/youtube data/filter/duration --min 10 --max 1200
```
### Obtain one `jsonl` file 
```
python join_to_one.py data/youtube data --filter_dir data/filter
```
This creates the files `metadata_filtered.jsonl` where only the kept videos after filtering are contained and each video is contained only once. To not loose the information related to the queries, we also generate `queries_filtered.json`, which maps the YouTube identifiers to the text queries where they occur and the respective result index.

## Information Extraction with LLM
TBA

## Matching Extracted Data
TBA
