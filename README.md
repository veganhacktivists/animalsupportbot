# animalsupportbot

Reddit bot that replies with the most relevant counterargument to debunk common anti-vegan myths.

It replies to a parent comment when summoned via mentions.

## Requirements

spacy, spacy-universal-sentence-encoder, numpy, praw, pandas, scikit-learn, tqdm

# Instructions

## Get knowledge base from GDrive (optional)

(This step only applies to VH members who have been given access to the knowledge base)

To get the freshest version of the myths and examples from Google Drive, we will need the `gdrive-keyfile.json` in this directory which grants access to the spreadsheet.

Once this is in the repo directory, simply run:

```sh
python update_knowledge.py
```

This will add `myths.csv` and `myths_egs.csv` to the `./knowledge/` directory.

## Pre-compute embeddings

```sh
python argmatcher.py
```

This populates `./preload_dicts/` with pickles of the precalculated embeddings for each sentence in the examples and responses. This assumes the presence of `myths.csv` and `myths_egs.csv` in the `./knowledge/` directory.

TODO: Describe data structure of `myths.csv` and `myths_egs.csv`.

## Run reddit bot

To run this step, `./local_info.py` must exist in the repo directory. This contains the secret keys etc. to authenticate with the Reddit API. It looks something like this:

```py
USER_INFO = {
              "client_id":"XXXXXX",
	      "client_secret":"XXXXX",
	      "password":"XXXXX",
	      "user_agent":"XXXXX",
	      "username":"animalsupportbot"
	    }								               
```

Once this exists, the reddit bot can be run with the following command:

```sh
python redditbot.py <int: refresh_rate> <float: threshold>
```

- `refresh_rate` is an `int` that determines how long the bot will sleep for after checking mentions, in seconds.
- `threshold` is a `float` that determines the minimum similarity score that an input sentence must have to be declared a succesful match.

The typical command I recommend running is:
```sh
python redditbot.py 60 0.5
```




