import argparse
import datetime
import os
from pprint import pprint
import re
import string
import sys
import time
from collections import OrderedDict

import pandas as pd
import praw
import prawcore
import spacy
from praw.models import Comment, Submission
from tinydb import Query, TinyDB
import validators
import yaml

from argmatcher import ArgMatcher
from response_templates import END_TEMPLATE, FAILURE_COMMENT, FAILURE_PM


def load_config_yaml(file):
    with open(file) as fp:
        config = yaml.safe_load(fp)
    return config


def load_myth_links(file):
    df = pd.read_csv(file)
    df = df.fillna("")
    return OrderedDict(
        {k: v for k, v in zip(df["Title"].values, df["Link"].values) if v}
    )


class MentionsBot:
    def __init__(
        self, argmatch, config, db, threshold=0.6, n_neighbors=1, hint_threshold=0.4
    ):
        self.config = config
        self.reddit = praw.Reddit(check_for_async=False, **config["user_info"])
        self.inbox = praw.models.Inbox(self.reddit, _data={})
        self.argmatch = argmatch
        self.threshold = threshold
        self.hint_threshold = hint_threshold
        self.n_neighbors = n_neighbors

        self.whitelisted_subreddits = set(config["whitelisted"])
        self.blacklisted_subreddits = set(["suicidewatch", "depression"]).union(
            set(config["blacklisted"])
        )

        self.whitelisted_subreddits = set(
            [s.lower() for s in self.whitelisted_subreddits]
        )
        self.blacklisted_subreddits = set(
            [s.lower() for s in self.blacklisted_subreddits]
        )

        self.db = db
        self.replied = self.fill_replied(self.db)

        self.alphabet = string.ascii_letters

        self.END_TEMPLATE = END_TEMPLATE
        self.FAILURE_COMMENT = FAILURE_COMMENT
        self.FAILURE_PM = FAILURE_PM

    def fill_replied(self, db):
        """
        Returns a list of all parent and mention ids found in the log DB
        """
        replied = []
        for entry in db.all():
            replied.append(entry["mention_id"])
            if "parent_id" in entry:
                replied.append(entry["parent_id"])
        return replied

    def clear_already_replied(self):
        """
        Go through mentions manually to tick off if we have already replied
        """
        for mention in self.inbox.mentions(limit=None):
            if mention not in self.replied:
                if isinstance(mention, Comment):
                    parent = mention.parent()
                    reply_info = {
                        "mention_id": mention.id,
                        "mention_username": mention.author.name
                        if mention.author
                        else None,
                        "mention_text": mention.body,
                        "mention_date": mention.created_utc,
                        "subreddit": mention.subreddit.display_name.lower(),
                        "parent_id": parent.id,
                        "parent_username": parent.author.name
                        if parent.author
                        else None,
                        "outcome": "Already replied, but not found in DB",
                    }

                    if isinstance(parent, Comment):
                        parent.refresh()
                        replies = parent.replies.list()
                    elif isinstance(parent, Submission):
                        replies = parent.comments.list()
                    else:
                        replies = None

                    if replies:
                        reply_authors = [r.author for r in replies]
                        if "animalsupportbot" in reply_authors:
                            self.replied.append(mention)
                            self.replied.append(parent)
                            self.db.insert(reply_info)

    def format_response(self, resps):
        """
        Formatting responses given from the argument matcher
        """
        args = OrderedDict({})
        for r in resps:
            inp = r["input_sentence"]
            arg = r["matched_argument"]
            passage = r["reply_text"]
            sim = r["similarity"]
            link = r["link"]

            if arg not in args:
                args[arg] = {"passage": passage, "quotes": [inp], "sim": sim}
            else:
                args[arg]["quotes"].append(inp)
                if args[arg]["sim"] < sim:
                    # replace the passage if this sentence is better matched
                    args[arg]["sim"] = sim
                    args[arg]["passage"] = passage

        replies = []
        for i, arg in enumerate(args):
            parts = []
            quotes = "".join(
                [">{} \n\n".format(q) for q in args[arg]["quotes"]]
            ) + "> ^(({})^) \n\n".format(self.alphabet[i])
            passage = args[arg]["passage"]
            if i < len(args) - 1:
                # Only add dividers between args
                passage += "\n\n --- \n\n"
            parts.append(quotes)
            parts.append(passage)
            arglist = "({}): {}".format(self.alphabet[i], arg)
            if validators.url(link):
                arglist = "[({}): {}]({})".format(self.alphabet[i], arg, link)

            parts.append(self.END_TEMPLATE.format(arglist))
            replies.append("\n".join(parts))
        return replies

    def reply_mentions(self, limit=None):
        """
        Main functionality. Go through mentions and reply to parent comments
        Uses persentence argmatcher
        """
        for mention in self.inbox.mentions(limit=limit):

            reply_info = {
                "mention_id": mention.id,
                "mention_username": mention.author.name if mention.author else None,
                "mention_text": mention.body,
                "mention_date": mention.created_utc,
                "subreddit": mention.subreddit.display_name.lower(),
            }

            # Skip mention if not in whitelisted subreddits
            if (
                mention.subreddit.display_name.lower()
                not in self.whitelisted_subreddits
            ):
                reply_info[
                    "outcome"
                ] = "Mention not in whitelisted subreddit: {}".format(
                    mention.subreddit.display_name.lower()
                )
                self.replied.append(mention)
                self.db.insert(reply_info)
                continue

            # Skip mention if included in blacklisted subreddits
            # TODO: currently irrelevant as whitelist exists, enable this if whitelist is not enabled
            if mention.subreddit.display_name.lower() in self.blacklisted_subreddits:
                reply_info["outcome"] = "Blacklisted Subreddit"
                self.replied.append(mention)
                self.db.insert(reply_info)
                continue

            # Proceed if mention has not been dealt with
            if mention not in self.replied:
                if isinstance(mention, Comment):
                    parent = mention.parent()
                    reply_info["parent_id"] = parent.id
                    reply_info["parent_username"] = (
                        parent.author.name if parent.author else None
                    )

                    # Check if parent has been handled (in case of multiple mentions)
                    if parent in self.replied:
                        reply_info["outcome"] = "Parent already replied to"
                        self.replied.append(mention)
                        self.db.insert(reply_info)
                        continue

                    try:
                        if isinstance(parent, Comment):
                            input_text = self.remove_usernames(parent.body)
                        elif isinstance(parent, Submission):
                            input_text = self.remove_usernames(".".join([parent.title, parent.selftext]))
                        else:
                            input_text = None
                    except:
                        input_text = None

                    reply_info["input_text"] = input_text

                    if input_text:
                        input_text = self.replace_newlines(input_text)
                        mention_hints = self.remove_usernames(mention.body).replace(
                            ",", "."
                        )
                        resps = self.argmatch.match_text(
                            input_text,
                            threshold=self.threshold,
                            N_neighbors=self.n_neighbors,
                        )

                        if mention_hints:
                            # Use mention hints to match arguments
                            reply_info["mention_hints"] = mention_hints

                            hint_resps = self.argmatch.match_text(
                                mention_hints,
                                threshold=self.hint_threshold,
                                N_neighbors=1,
                                return_reply=False,
                            )

                            reply_info["hint_responses"] = hint_resps

                            arg_labels = set(
                                [r["matched_arglabel"] for r in hint_resps]
                            )
                            r_arg_labels = set([r["matched_arglabel"] for r in resps])

                            # Check only the hinted args which aren't matched already
                            arg_labels = arg_labels - r_arg_labels

                            if arg_labels:
                                hinted_resps = self.argmatch.match_text(
                                    input_text,
                                    arg_labels=arg_labels,
                                    threshold=self.hint_threshold,
                                    N_neighbors=self.n_neighbors,
                                )
                                rsents = [r["input_sentence"] for r in resps]

                                for r in hinted_resps:
                                    if r["input_sentence"] in rsents:
                                        # This means the sentence was matched up already with better similarity
                                        continue
                                    else:
                                        resps.append(r)

                    else:
                        resps = []

                    reply_info["responses"] = resps

                    if resps:  # Found arg match(es)
                        formatted_responses = self.format_response(resps)
                        reply_info["full_reply"] = formatted_responses

                        for response in formatted_responses:
                            try:
                                reply = parent.reply(response)
                                print(response)
                                reply_info["outcome"] = "Replied with matched argument(s)"
                                reply_info["reply_id"] = reply.id
                            except prawcore.exceptions.Forbidden:
                                reply_info[
                                    "outcome"
                                ] = "Found arguments but failed to reply: Forbidden"

                    else:  # Failed to find arg match
                        mention.reply(self.FAILURE_COMMENT)
                        try:
                            mention.author.message(
                                "We couldn't find a response!",
                                self.FAILURE_PM.format(
                                    self.argmatch.prefilter(parent.body)
                                ),
                            )
                        except:
                            # PM-ing people sometimes fails, but this is not critical
                            pass
                        reply_info["outcome"] = "Failed to find any matched arguments"

                    # Add both the mention and the parent to the replied list
                    self.replied.append(mention)
                    self.replied.append(parent)
                    self.db.insert(reply_info)

    def run(self, refresh_rate=600, timeout_retry=600, check_replied=True):
        """
        Run the bot, checking for mentions every refresh_rate seconds
        In case of timeout, waits for timeout_retry seconds

        If check_replied is True, will check all previous mentions to see if we have already replied
        """
        if check_replied:
            print("Checking previous mentions to see if we have replied already...")
            self.clear_already_replied()

        while True:
            try:
                self.reply_mentions()
                print(
                    "{}\tReplied to mentions, sleeping for {} seconds...".format(
                        time.ctime(), refresh_rate
                    )
                )
                time.sleep(refresh_rate)
            except prawcore.exceptions.ServerError or prawcore.exceptions.ResponseException:
                print(
                    "Got a ServerError, sleeping for {} seconds before trying again...".format(
                        timeout_retry
                    )
                )
                time.sleep(timeout_retry)

    @staticmethod
    def remove_usernames(text):
        """
        Removes any /u/username or u/username strings
        """
        newtext = re.sub("\/u\/[A-Za-z0-9_-]+", "", text)
        newtext = re.sub("u\/[A-Za-z0-9_-]+", "", newtext)
        return newtext

    @staticmethod
    def replace_newlines(text):
        """
        Replaces newline symbols with periods
        """
        return text.replace("\n", ". ")


if __name__ == "__main__":
    config = load_config_yaml("./config.yaml")
    pprint(config)

    threshold = float(config["threshold"])
    hint_threshold = float(config["hint_threshold"])
    n_neighbors = int(config["n_neighbors"])

    refresh_rate = int(config["refresh_rate"])

    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("universal_sentence_encoder", config={"model_name": "en_use_lg"})

    db = TinyDB("./log_db.json")

    argm = ArgMatcher(nlp, None, None, preload=True)
    mb = MentionsBot(
        argm,
        config,
        db,
        threshold=threshold,
        hint_threshold=hint_threshold,
        n_neighbors=n_neighbors,
    )

    mb.run(refresh_rate=refresh_rate)
