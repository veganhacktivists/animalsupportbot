"""
Reddit bot that checks mentions, and launches brain.py when new mentions are detected
"""

import argparse
import os
from pprint import pprint
import re
import time

import praw
import prawcore
from praw.models import Comment, Submission
from tinydb import TinyDB
import yaml

import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-replied",
        help="Check if we have replied already first, only necessary once",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--config",
        help="Config file path (default: ./config.yaml)",
        type=str,
        default="./config.yaml",
    )
    parser.add_argument(
        "--log-db",
        help="Log db path (default: ./lob_db.json",
        type=str,
        default="./log_db.json",
    )
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    assert os.path.isfile(args.log_db)

    return args


def load_config_yaml(file):
    with open(file) as fp:
        config = yaml.safe_load(fp)
    return config


class MentionsBot:
    def __init__(
        self,
        config,
        config_file,
        db_file,
    ):
        self.config = config
        self.config_file = config_file
        self.reddit = praw.Reddit(check_for_async=False, **config["user_info"])
        self.inbox = praw.models.Inbox(self.reddit, _data={})

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

        self.db = TinyDB(db_file)
        self.db_file = db_file
        self.replied = set()
        self.fill_replied()

    def fill_replied(self):
        """
        Populates the replied set
        """
        for entry in self.db.all():
            self.replied.add(entry["mention_id"])
            if "parent_id" in entry:
                self.replied.add(entry["parent_id"])

    def clear_already_replied(self):
        """
        Go through mentions manually to tick off if we have already replied
        """
        for mention in self.inbox.mentions(limit=None):
            if mention.id not in self.replied:
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
                            self.replied.add(mention.id)
                            self.replied.add(parent.id)
                            self.db.insert(reply_info)

    def check_mentions(self, limit=None):
        """
        Check for mentions, run brain.py if a valid mention is detected
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
                self.replied.add(mention.id)
                self.db.insert(reply_info)
                continue

            # Skip mention if included in blacklisted subreddits
            # TODO: currently irrelevant as whitelist exists, enable this if whitelist is not enabled
            if mention.subreddit.display_name.lower() in self.blacklisted_subreddits:
                reply_info["outcome"] = "Blacklisted Subreddit"
                self.replied.add(mention.id)
                self.db.insert(reply_info)
                continue

            # Proceed if mention has not been dealt with
            if mention.id not in self.replied:
                self.launch_brain()
                self.replied.add(mention.id)

    def launch_brain(self):
        """
        Closes db, runs brain.py, opens db again
        """
        self.db.close()
        p = subprocess.call(["python3", "brain.py", "--log-db", self.db_file, "--config", self.config_file])
        self.db = TinyDB(self.db_file)

    def run(self, refresh_rate=600, timeout_retry=600, check_replied=True, limit=None):
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
                self.check_mentions(limit=limit)
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


if __name__ == "__main__":
    args = parse_args()
    config = load_config_yaml(args.config)
    pprint(config)

    refresh_rate = int(config["refresh_rate"])

    mb = MentionsBot(
        config,
        args.config,
        args.log_db,
    )

    mb.run(
        refresh_rate=refresh_rate,
        check_replied=args.check_replied,
    )
