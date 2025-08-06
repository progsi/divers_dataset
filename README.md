# Discogs-VI-2

The metadata can be found on the [Zenodo page](https://zenodo.org/records/16743529?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjkyOTkxYzE5LWRlY2QtNGQyNS1iZGY5LTA4YzZkMTlhMDdhZCIsImRhdGEiOnt9LCJyYW5kb20iOiIyMTQwMjg5MzQxOTRkMWYwMzBiYzQ0Y2E3NTkzMjM2MSJ9.Enin4GOFGnHDTqKUFKTjlSEW8I_ScJTLr8IvHGO1WRAHIz4aMkGnQKmd7Py_QHSflKOX2A9s-aq2OPlHVcc-6Q)

A suite of two datasets, based on the *DVI* ([*Discogs-VI-YT*](https://github.com/MTG/discogs-vi-dataset)) dataset. Namely:
- *Discogs-VI-YT-2* or *DVI2*: a cleaned version of *DVI* which contains slightly less versions and cliques, but a cleaner assignment of versions to cliques. 
- *DiVers-1M*: contains versions found on YouTube without being constrained to listings on *Discogs* or *Secondhandsongs*.  

# Requirements
- the original *DVI* dataset, especially the file `Discogs-VI-YT-20240701.jsonl`
- installation of the conda environment by running  `conda env create -f env.yaml`

# Dataset
Provided in a separate ressource (TBA).
We provide the following subsets:

## Prior versions
- `dvi1.5`: *DVI1.5* partially cleaned *DVI* (only writers with special chars)
- `dvi`: *DVI* and its matched version

# Dataset Creation
## Cleanup *Discogs-VI-YT*
We do the following to cleanup. This should result in a cleaner version of *Discogs-VI-YT* with new clique assignments. It can be considered a second version of the dataset but with less versions.
### Reduce false positives
#### Normalizing the writers with an LLM
Here is an example with the LLM *Qwen3*. The mapping from the CLI parameter to the model is hard-coded in the script.
```
python preprocessing/clean_discogs/llm_normalize_writers.py data/discogs/Discogs-VI-YT-20240701.jsonl ndata/aux/norm_writers.jsonl --llm qwen
```
#### Finding versions of the same clique with different normalized writers
This results in new cliques.
```
python preprocessing/clean_discogs/get_new_clique_ids.py data/discogs/Discogs-VI-YT-20240701.jsonl data/aux/norm_writers_qwen.jsonl data/aux/new_clique_map.json
```
#### Create new dataset file
Remapping of cliques.
```
python preprocessing/clean_discogs/reassign_clique_ids.py data/discogs/Discogs-VI-YT-20240701.jsonl data/aux/new_clique_map.json data/dataset/dvi_cleaned.jsonl
```
### Reduce false negatives
Adaptions etc.
TBA

## Extract titles from Discogs-VI-YT 
To obtain one query (song title) per clique, we first create a new file. We use the cleaned song titles from *Discogs-VI-YT*.
```
python preprocessing/get_unique_titles.py data/dataset/dvi_cleaned.jsonl data/aux/one_title_per_clique.json
```
## Search on YouTube
We now search for up to 500 results per clique, using the text query (song title).
```
python preprocessing/search_youtube.py data/aux/one_title_per_clique.json data/youtube/ 
```
## Filtering
### Exclude dataset matches
We exclude videos which are also contained as versions in any of the datasets [*Discogs-VI-YT*](https://github.com/MTG/discogs-vi-dataset), [*SHS100K2*](https://github.com/NovaFrost/SHS100K2) or [*Da-Tacos*](https://github.com/MTG/da-tacos). Please note that we have our own CSV files for this process which we can provide upon request. Given these files, we run:

```
python preprocessing/filter_youtube_ids.py data/youtube data/filter/youtube_id --discogs_path data/discogs/Discogs-VI-YT-20240701-light.json --shs_csv_path ../data/shs100k2.csv --datacos_csv_path ../data/da-tacos.csv
```
### Exclude videos by duration
We exclude videos longer than 20 minutes (like in *Discogs-VI-YT*) and under 10 seconds.
```
python preprocessing/filter_duration.py data/youtube data/filter/duration --min 10 --max 1200
```
### Filter by result index
```
python preprocessing/filter_rank.py data/youtube data/filter/rank --max_index 100
```
### Obtain one `jsonl` file 
```
python preprocessing/join_to_one.py data/youtube data --filter_dir data/filter
```
This creates the files `metadata_filtered.jsonl` where only the kept videos after filtering are contained and each video is contained only once. To not loose the information related to the queries, we also generate `queries_filtered.json`, which maps the YouTube identifiers to the text queries where they occur and the respective result index.
### Filter by fuzzy matching
#### Method
This step aims detecting videos which are likely versions of the works in the seed dataset. For each song title and its video results, we match the respective song title and artist name by fuzzy matching.
We also apply some pre-processing steps (see `string_processor.py`) which were also used to generate [MusicUGC-NER](https://github.com/progsi/YTUnCoverLLM/tree/main?tab=readme-ov-file).
```
python preprocessing/fuzzy_matching.py data/discogs/Discogs-VI-YT-20240701.jsonl data/aux/one_title_per_clique.json data/matched/full.csv
```
#### Analysis
From the output file in `data/full.csv`, we analyze the matches with regards to pairs of attributes from *Discogs* and *YouTube* which are matched in the notebook `matching_analysis.ipynb`. Additionally, we split the data into four groups after thresholding the similarity at 80%:
- *both*: *title* and *artist* are matched
- *only_title*: the cleaned title from *Discogs-VI-YT* matches
- *only_artist*: any artist is matched using the list of arists from *Discogs*
- *none*
For the whole set, this information is written to `data/filtered_types.csv`. We additionally create a sample of 100 videos per group to check manually.

Manually checking the sample, we observed that *both* contains references to the work of interest in almost all of the cases (>95%). Additionally, *none* and *only_title* mostly contain references to other works and *only_artist* mostly contain references of works from the same artist related to the work of interest. 

**Based on our analysis we decide to only keep the *both* subset, since it is already large (1.4M videos) and the matching quality is rather high.**

## Make splits
In `make_splits.ipynb` we make two subsets. First, the full *both* dataset. We further create another filtered version were we filter some indicators of official music videos (e.g. *remastered* etc.). 
The outputs are written to `data/dataset` and contain json files containing only the new versions as well as dataset files which contain versions of `Discogs-VI-YT` and the new versions which usable to train models. 
Afterwards, we create the splits:
```
python preprocessing/make_splits.py data/dataset/dvi_fm_filtered.jsonl data/discogs/ data/dataset/ --use-split-content
```
## Make Torch File for [CLEWS](https://github.com/sony/clews)
Given a torch file like described in the [CLEWS](https://github.com/sony/clews) repo, and given the YouTube crawl and our Discogs metadata file, we can run:

`python collect_metadata.py --audio-dir /data/audio/ --dvi-file ../discogs-vi-2/data/dvi2/dataset/divers1m/dvi_fm.jsonl --meta-file ../clews/cache/metadata-dvi2fm.pt --njobs 32`

Afterwards:

`python enrich_cache_metadata.py --input ../clews/cache/metadata-dvi2fm.pt --output data/metadata-dvi2fm_aux.pt`

### Make Sub-Datasets
The dataset is very large and depending on the use case a respective subset might be enough. We can estimate the content from the YouTube video metadata. Potential use cases include:
- Version identification 
    - subset focussing on difficult versions
- Music structure analysis 
    - subset with versions only containing sub-segments 
- Fingerprinting
    - reaction videos, lyric videos?
- Other identifications
    - artist
    - genre
    - country of origin
    - year
    - type of video (e.g. studio recording, amateur, ...)
- TODO: find out which sub-datasets we have

## Download
Some tips regarding MP4 downloads are given in [*Discogs-VI-YT*](https://github.com/MTG/discogs-vi-dataset). The estimated time to download everything (when using 8 parallel downloads at a time), is around 12-18 days. 

# Analysis

In `content_analysis.ipynb` we analyze: duration, n-grams etc.

