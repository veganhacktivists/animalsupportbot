# animalsupportbot

Reddit bot that replies with the most relevant counterargument to debunk common anti-vegan myths.

It replies to a parent comment when summoned via mentions.

# Bot Usage

The bot can be summoned using mentions, which means replying to the comment the bot should respond to with `/u/animalsupportbot` somewhere in the reply.

For example:

```
Non-vegan user: <Comment with myth(s)>
   └── Vegan User: /u/animalsupportbot
	└── animalsupportbot: <response>
```

## Argument hinting

The bot can be helped to match up arguments by 'hinting' in the summoning process. This is done by putting in a helpful phrase/keywords in the summoning reply (which can be separated with commas or full stops):

```
Non-vegan user: <Comment with myth(s)>
   └── Vegan User: /u/animalsupportbot plants feel pain. vegan pets
	└── animalsupportbot: <response, using hints to match more easily>
```

# Requirements

Tested on Python 3.6.9 using a virtualenv. Requirements can be found in `requirements.txt`.

```sh
pip install -r requirements.txt
```

# Instructions

## (Step 1) Pre-compute embeddings

```sh
python argmatcher.py
```

This populates `./preload_dicts/` with embeddings for each example for each myth. This saves us having to calculate them every time we want to restart the bot.

## (Step 2) Run reddit bot

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
python redditbot.py --refresh-rate <int> --threshold <float> --hint-threshold <float> --n-neighbors <int>
```

- `refresh_rate` is an `int` that determines how long the bot will sleep for after checking mentions, in seconds.
- `threshold` is a `float` that determines the minimum similarity score that an input sentence must have to be declared a succesful match.
- `hint-threshold` is a `float` that determines the minimum similarity score that an input must have to be declared a succesful match **to a hinted argument**.
- `n-neighbors` is an `int` that determines how many neighbors to consider in the weighted vote argument classification.

The typical command I recommend running is:
```sh
python redditbot.py --refresh-rate 60 --threshold 0.6 --hint-threshold 0.4 --n-neighbors 3
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


