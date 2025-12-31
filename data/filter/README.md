# Dataset Format
## Files
- `data_light.json`: all versions from DiVers-Small with the most essential metadata (title, performer, identifier, tags from title)
- `data_sample.json`: 1000 random cliques from each of the three subsets with rich metadata
## Hierarchy
First-level keys are the subsets (`train`, `valid`, `test`).
Second-level keys are the clique identifiers (same as in DiscogsVI). For example: `C-0000027`.
Third-level keys are version identifiers. These are either from DiscogsVI (e.g. `V-0000208`) or new and match the YouTube identifier (e.g. `xdZzj2BKMZE`).
So you can retrieve the version metadata as an example by:
```
import json

with open("data_light.json", "r") as f:
    data_light = json.load(f)

version = data_light["valid"]["C-0000027"]["xdZzj2BKMZE"]
```
## Version metadata in the Sample
Load the sample:
```
import json

with open("data_sample.json", "r") as f:
    data_sample = json.load(f)
```
Version metadata in `data_sample.json` has the following information:
- `id`: integer identifier of the version
- `dvi`: boolean indicating whether it is a version from DiscogsVI
- metadata from Discogs; all of these are `None` if the version is in `YVI`
    - `artist`: joined artist string
    - `title`: title string
    - `track_writer_names`: list of writer strings
    - `release_artist_names`: list of artist strings
    - `release_genres`: list of genre strings
    - `release_styles`: list of lists representing genre to style hierarchy
    - `country`: release country string
    - `labels`: list of release label strings
    - `formats`: list of release formats (e.g. CD, Vinyl, etc.).
    - `released`: release year integer
- YouTube metadata
    - `youtube_id`: YouTube identifier
    - `yt_title`: YouTube video title
    - `yt_categories`: YouTube content category
    - `yt_channel`: YouTube channel name
    - `yt_upload_date`: Upload date of the YouTube video
    - `yt_view_count`: viewcount of the YouTube video at the time of download
- technical metadata
    - `filename`: path to the audio file
    - `samplerate`: sample rate of the audio file
    - `length`: duration in seconds
    - `channels`: number of audio channels
    - `tempo`: estimated beats per minute using Librosa
- automatically detected tags
    - `tags_yt_title`, `tags_yt_description`, `tags_yt_tags`: lists of strings of detected tags in the respective YouTube metadata field (title, description or tags).
    - `cues_yt_title`, `cues_yt_description`, `cues_yt_tags`: lists of strings of cues indicating tags in the respective YouTube metadata field (title, description or tags).
