import argparse
import datetime
import os
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

from argmatcher import ArgMatcher
from local_info import USER_INFO
from response_templates import (END_TEMPLATE, FAILURE_COMMENT, FAILURE_PM,
                                GFORM_LINK)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", help="Minimum similarity threshold that has to be met to match an argument (float: between 0 and 1)",
                        type=float, default=0.6)
    parser.add_argument(
        "--refresh-rate", help="Refresh rate in seconds to check for mentions", type=int, default=60)
    parser.add_argument(
        "--n-neighbors", help="Number of neighbors to consider when doing a weighted vote for classification", type=int, default=1)
    args = parser.parse_args()
    return args


def load_myth_links(file):
    df = pd.read_csv(file)
    df = df.fillna('')
    return OrderedDict({k: v for k, v in zip(df['Title'].values,
                                             df['Link'].values) if v})


class MentionsBot:

    def __init__(self, argmatch, user_info, db, threshold=0.6, n_neighbors=1):
        self.reddit = praw.Reddit(
            check_for_async=False,
            **user_info
        )
        self.inbox = praw.models.Inbox(self.reddit, _data={})
        self.argmatch = argmatch
        self.threshold = threshold
        self.n_neighbors = n_neighbors
        self.blacklisted_subreddits = set(['suicidewatch', 'depression'])

        self.db = db
        self.replied = self.fill_replied(self.db)

        self.alphabet = string.ascii_letters

        self.arg_link_dict = load_myth_links('./knowledge/myths.csv')

        self.END_TEMPLATE = END_TEMPLATE
        self.FAILURE_COMMENT = FAILURE_COMMENT
        self.GFORM_LINK = GFORM_LINK
        self.FAILURE_PM = FAILURE_PM

    def fill_replied(self, db):
        """
        Returns a list of all parent and mention ids found in the log DB
        """
        replied = []
        for entry in db.all():
            replied.append(entry['mention_id'])
            if 'parent_id' in entry:
                replied.append(entry['parent_id'])
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
                        'mention_id': mention.id,
                        'mention_username': mention.author.name if mention.author else None,
                        'mention_text': mention.body,
                        'date': mention.created_utc,
                        'subreddit': mention.subreddit.display_name.lower(),
                        'parent_id': parent.id,
                        'parent_username': parent.author.name if parent.author else None,
                        'outcome': 'Already replied, but not found in DB'
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
                        if 'animalsupportbot' in reply_authors:
                            self.replied.append(mention)
                            self.replied.append(parent)
                            self.db.insert(reply_info)


    def format_response_persentence(self, resps):
        """
        Formatting responses given from the argument matcher
        """
        args = OrderedDict({})
        for r in resps:
            inp = r['input_sentence']
            arg = r['matched_argument']
            passage = r['reply_text']
            sim = r['similarity']

            if arg not in args:
                args[arg] = {'passage': passage, 'quotes': [inp], 'sim': sim}
            else:
                args[arg]['quotes'].append(inp)
                if args[arg]['sim'] < sim:
                    # replace the passage if this sentence is better matched
                    args[arg]['sim'] = sim
                    args[arg]['passage'] = passage

        parts = []
        arglist = []

        for i, arg in enumerate(args):
            quotes = ''.join(['>{} \n\n'.format(q)
                              for q in args[arg]['quotes']]) + '> ^(({})^) \n\n'.format(self.alphabet[i])
            passage = args[arg]['passage'] + '\n'
            parts.append(quotes)
            parts.append(passage)
            if arg in self.arg_link_dict:
                arglist.append('[({}): {}]({})'.format(
                    self.alphabet[i], arg, self.arg_link_dict[arg]))
            else:
                arglist.append('({}): {}'.format(self.alphabet[i], arg))

        parts.append(self.END_TEMPLATE.format(', '.join(arglist)))
        return '\n'.join(parts)

    def reply_mentions_persentence(self, limit=None):
        """
        Main functionality. Go through mentions and reply to parent comments
        Uses persentence argmatcher
        """
        for mention in self.inbox.mentions(limit=limit):

            reply_info = {
                'mention_id': mention.id,
                'mention_username': mention.author.name if mention.author else None,
                'mention_text': mention.body,
                'date': mention.created_utc,
                'subreddit': mention.subreddit.display_name.lower(),
            }

            # Temporary restriction on only replying in test subreddit
            if mention.subreddit.display_name.lower() != 'testanimalsupportbot':
                continue

            # Skip mention if included in blacklisted subreddits
            if mention.subreddit.display_name.lower() in self.blacklisted_subreddits:
                reply_info['outcome'] = 'Blacklisted Subreddit'
                self.replied.append(mention)
                self.db.insert(reply_info)
                continue

            # Proceed if mention has not been dealt with
            if mention not in self.replied:
                if isinstance(mention, Comment):
                    parent = mention.parent()
                    reply_info['parent_id'] = parent.id
                    reply_info['parent_username'] = parent.author.name if parent.author else None

                    # Check if parent has been handled (in case of multiple mentions)
                    if parent in self.replied:
                        reply_info['outcome'] = 'Parent already replied to'
                        self.replied.append(mention)
                        self.db.insert(reply_info)
                        continue

                    try:
                        if isinstance(parent, Comment):
                            input_text = parent.body
                        elif isinstance(parent, Submission):
                            input_text = parent.selftext
                        else:
                            input_text = None
                    except:
                        input_text = None

                    reply_info['input_text'] = input_text

                    if input_text:
                        resps = self.argmatch.match_text_persentence(
                            input_text, threshold=self.threshold, N_neighbors=self.n_neighbors)
                    else:
                        resps = []

                    reply_info['responses'] = resps

                    if resps:  # Found arg match(es)
                        formatted_response = self.format_response_persentence(
                            resps)
                        parent.reply(formatted_response)
                        print(formatted_response)

                        # Add both the mention and the parent to the replied list
                        self.replied.append(mention)
                        self.replied.append(parent)
                        reply_info['outcome'] = 'Replied with matched argument(s)'
                        reply_info['full_reply'] = formatted_response
                        self.db.insert(reply_info)

                    else:  # Failed to find arg match
                        mention.reply(self.FAILURE_COMMENT)
                        try:
                            mention.author.message("We couldn't find a response!",
                                                   self.FAILURE_PM.format(self.argmatch.prefilter(parent.body), self.GFORM_LINK))
                        except:
                            # PM-ing people sometimes fails, but this is not critical
                            pass

                        # Add both the mention and the parent to the replied list
                        self.replied.append(mention)
                        self.replied.append(parent)
                        reply_info['outcome'] = "Failed to find any matched arguments"
                        self.db.insert(reply_info)

    def run(self, refresh_rate=600, timeout_retry=600, check_replied=True):
        """
        Run the bot, checking for mentions every refresh_rate seconds
        In case of timeout, waits for timeout_retry seconds

        If check_replied is True, will check all previous mentions to see if we have already replied
        """
        if check_replied:
            print('Checking previous mentions to see if we have replied already...')
            self.clear_already_replied()

        while True:
            try:
                self.reply_mentions_persentence()
                print('{}\tReplied to mentions, sleeping for {} seconds...'.format(
                    time.ctime(), refresh_rate))
                time.sleep(refresh_rate)
            except prawcore.exceptions.ServerError or prawcore.exceptions.ResponseException:
                print('Got a ServerError, sleeping for {} seconds before trying again...'.format(
                    timeout_retry))
                time.sleep(timeout_retry)


if __name__ == "__main__":
    args = parse_args()
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe('universal_sentence_encoder',
                 config={'model_name': 'en_use_lg'})

    db = TinyDB('./log_db.json')

    argm = ArgMatcher(nlp, None, None, preload=True)
    mb = MentionsBot(argm, USER_INFO, db, threshold=args.threshold,
                     n_neighbors=args.n_neighbors)

    mb.run(refresh_rate=args.refresh_rate)
