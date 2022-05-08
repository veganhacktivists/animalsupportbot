# animalsupportbot

Reddit bot that replies with the most relevant counterargument to debunk common anti-vegan myths.

It replies to a parent comment when summoned via mentions.

# Bot Usage

The bot can be summoned using mentions on Reddit, which means replying to the comment the bot should respond to with `/u/animalsupportbot` somewhere in the reply.

For example:

```
Non-vegan user: <Comment with myth(s)>
   └── Vegan User: /u/animalsupportbot
	└── animalsupportbot: <response>
```

The bot can also be helped to match up arguments by 'hinting' in the summoning process. This is done by putting in a helpful phrase/keywords in the summoning reply (which can be separated with commas or full stops):

```
Non-vegan user: <Comment with myth(s)>
   └── Vegan User: /u/animalsupportbot plants feel pain. vegan pets
	└── animalsupportbot: <response, using hints to match more easily>
```

# [Contributing to Responses](knowledge/contributing_responses.md)

[Information for contributing to responses can be found by clicking here](knowledge/contributing_responses.md)

# Requirements

Tested on Python 3.6.9 using a virtualenv. Requirements can be found in `requirements.txt`.

```sh
pip install -r requirements.txt
```

# Deployment Instructions

## (Step 1) Pre-compute embeddings

```sh
python argmatcher.py
```

This populates `./preload_dicts/` with embeddings for each example for each myth. This saves us having to calculate them every time we want to restart the bot.

## (Step 2) Run reddit bot

To run this step, `config.yaml` must exist in the repo directory. This contains the configuration for the classifier, in addition to things like secret keys to authenticate with the Reddit API. It looks something like this:

```yaml
# Matching options
threshold: 0.8
certain_threshold: 0.9
n_neighbors: 3

# Hint matching options
hint_arg_threshold: 0.3
hint_threshold: 0.4
hint_certain_threshold: 0.8
hint_n_neighbors: 7

# Redditbot options
refresh_rate: 60

# User Info
user_info:
  client_id: "XXXXXXXX"
  client_secret: "XXXXXXXX"
  password: "XXXXXXXX"
  user_agent: "XXXXXXXX"
  username: "animalsupportbot"

# Whitelisted Subreddits
whitelisted:
  - testanimalsupportbot
  - vegan
  - debateavegan
  - vegancirclejerk
  - veganforcirclejerkers
  - veganuk

# Blacklisted Subreddits (in addition to defaults)
blacklisted:
  - depression
  - suicidewatch				               
```

- `threshold` is a `float` that determines the minimum similarity score that an input sentence must have to be declared a succesful match.
- `certain_threshold` is a `float` which determines the similarity score above which the top-1 neighbor is selected, over a weighted vote.
- `n_neighbors` is an `int` that determines how many neighbors to consider in the weighted vote argument classification.

- `hint_arg_threshold` is a `float` that determines the similarity score that the mention text must have to be matched with an argument.
- `hint_threshold` is a `float` that determines the minimum similarity score that a sentence must have to be declared a succesful match **to a hinted argument**.
- `hint_certain_threshold` and `hint_n_neighbors` are the same as above, except for hinted argument matching.

- `refresh_rate` is an `int` that determines how long the bot will sleep for after checking mentions, in seconds.
- `whitelisted` is a list of subreddits that the bot can respond to comments in
- `blacklisted` is a list of subreddits that the bot is explicitly stopped from responding to in (`depression,suicidewatch` are both hard-coded in, and not actually required in the config file)

Once this exists, the reddit bot can be run with the following command:

```sh
python redditbot.py
```

# Testing the Argument Matcher

After Step 1 has been completed, meaning embeddings have been precomputed, the argument matcher can be tested on the command-line using the following:

```sh
python argmatcher.py --test
```

This starts an interactive mode which can test various input sentences, for example:

```
Enter test sentence: but bacon though
Num neighbours with vote: 3
[{'input_sentence': 'but bacon though',
  'matched_arglabel': 2,
  'matched_argument': 'I Love How Animals Taste!',
  'matched_text': 'but bacon though',
  'reply_text': 'I think one of the most disconcerting things about the taste '
                'excuse though, is that it is an excuse that bluntly admits '
                'that the personal desires of an individual’s taste preference '
                'matter more than the morality surrounding an animal’s life '
                'and unquestionably horrific death. However, it doesn’t mean '
                'that the person using this excuse necessarily believes that '
                'their taste preferences are more important than an animal’s '
                'life (most people I talk to don’t) but because they’ve never '
                'been asked about it before they’ve never had to confront the '
                'fact that through their actions they are placing their taste '
                'higher. This is why when people say to me “I love the taste '
                'of meat.” or, “I could never give up cheese.”, I like to ask '
                'them “do you value your taste buds higher than the life of an '
                'animal?” - most people will say no, but if they do say yes '
                'make sure to ask them why. One of the big issues with this '
                'excuse is that it seeks to validate a non-vegan diet by '
                'claiming that we shouldn’t be held responsible for our '
                'immoral activities because our selfish impulses are too '
                'strong to be suppressed and as such, we can’t be held morally '
                'accountable for the actions that we make. But where do we '
                'draw the line?',
  'similarity': 1.0}]
```

# Knowledge base Overview

The information for the various arguments that the bot can match and respond to is located in `knowledge`. This folder has the following structure, with the myth "Plants Feel Pain" used as an example:

```
    └── knowledge                    
        ├── myths
	|   ├── plants_feel_pain.yaml
	|   └── ...
        └── responses
	    ├── plants_feel_pain.md
	    └── ...
```

The `.yaml` files in `knowledge/myths` contain auxiliary information about the argument, and the `.md` files in `knowledge/responses` contains the actual response text.

## Argument/Myth YAML structure

Inside `knowledge/myths` are `.yaml` files containing the following information:

```yaml
key: plants_feel_pain
title: Plants Feel Pain 
full_comment: true
enable_resp: true
link: <URL> 
examples:
- what if plants feel pain
- plants feel pain too
- how do you know plants don't feel pain
```

- `key` is the unique identifier for this argument. The response text in `knowledge/responses` must have the filename: `<key>.md`.
- `title` is the formatted title for this argument.
- `full_comment` is a boolean which indicates whether or not the full response should be posted. If this is `false` then the most similar sentence to the input in the response text is selected (along with the proceeding 5 sentences).
- `enable_resp` determines whether an argument should be responded to, if matched. This flag exists mainly to disable unfinished responses.
- `link` an optional link to highlight the argument title with in the response, such as a YouTube video. If there is no link, this must be set to `nan`.
- `examples` the example sentences/phrases which should link to this argument. These examples make up the "training set" for the nearest neighbor classifier.

## Response Editing

The responses are all stored in `.md` files in `knowledge/responses`. These should be written in the markdown style that Reddit uses.


# Performance Evaluation

Performance metrics for the bot can be obtained by using `eval.py`. If it has not been done before (or if changes to the knowledge base have been made), the pre-computed embeddings should be repopulated using:

```sh
python argmatcher.py
```

A test data `csv` file can then be evaluated by running something like the following command:

```sh
python eval.py --eval-csv <test_data.csv> --n-neighbors 3 --threshold 0.5 --certain-threshold 0.9
```

This `test_data.csv` should be a two column comma separated file with the headings: `text, label`. An example of such a file can be found [here](https://gist.githubusercontent.com/cvqluu/df0323b68f17bc255d546d4c6865e9fd/raw/a5f601d7a23101afd6f97c23a0798b227921d20c/antiveg_comments.csv).

This script will provide the following outputs in a unique `eval_logs/<eval_run_id>/` folder:

- Results summary: `results.csv`
  - Balanced accuracy
  - Averaged Precision and Recall
    - `"micro", "macro", "weighted"`: See [sklearn-docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support)
- Confusion matrices:
  - `cm_true.png`: Normalized by true label (diagonal is per-class precision)
  - `cm_pred.png`: Normalized by pred label (diagonal is per-class recall)
  - `cm_none.png`: Un-normalized
- Every example evaluated: `raw_results.csv`:
  - Contains predicted label, true label