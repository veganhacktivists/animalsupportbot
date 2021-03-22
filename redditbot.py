import os
import time
import sys
from collections import OrderedDict

import pandas as pd
import praw
import spacy
from praw.models import Comment

from argmatcher import ArgMatcher
from local_info import USER_INFO

def read_list(file):
    completed = []
    with open(file) as fp:
        for line in fp:
            completed.append(line.strip())
    return completed

def load_myth_links(file):
    df = pd.read_csv(file, names=['argument', 'text', 'link'])
    return OrderedDict({k:v for k,v in zip(df['argument'].values, 
                                           df['link'].values) if v})

class MentionsBot:

    def __init__(self, argmatch, user_info, threshold=0.5):
        self.reddit = praw.Reddit(
                    check_for_async=False,
                    **user_info
                )
        self.inbox = praw.models.Inbox(self.reddit, _data={})
        self.argmatch = argmatch
        self.threshold = threshold

        self.completed = []
        self.completed_file = './completed.csv'
        if os.path.isfile(self.completed_file):
            self.completed = read_list(self.completed_file)

        self.missed = []
        self.missed_file = './missed.csv'
        if os.path.isfile(self.missed_file):
            self.missed = read_list(self.missed_file)

        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'

        self.arg_link_dict = load_myth_links('./knowledge/vegan_myths.csv')

        self.response_template = '{} \n This was an automatically generated response that I matched to the idea that "{}"'
        self.end_message = "\n *** \n This was an automatically generated response based on the idea(s)/myth(s): \n\n {} \n\n" \
                "*(Responses taken from vegan advocates like [Earthling Ed](https://www.youtube.com/channel/UCVRrGAcUc7cblUzOhI1KfFg))* \n *** \n" \
                        "**[Vegan Hacktivists](https://veganhacktivists.org/), [Vegan Bootcamp](https://veganbootcamp.org/)**"
    
    def clear_already_replied(self):
        """
        Go through mentions manually to tick off if we have already replied
        """
        for mention in self.inbox.mentions(limit=None):
            if mention not in self.completed:
                if isinstance(mention, Comment):
                    parent = mention.parent()
                    if isinstance(parent, Comment):
                        parent.refresh()
                        replies = parent.replies.list()
                        reply_authors = [r.author for r in replies]
                        if 'animalsupportbot' in reply_authors:
                            self.completed.append(mention)
                            self.append_file(self.completed_file, mention)

    def append_file(self, file, comment_id):
        with open(file, 'a') as wp:
            line = '{}\n'.format(comment_id)
            wp.write(line)


    def format_response_persentence(self, resps):
        """
        Formatting responses given from the argument matcher
        """
        args = OrderedDict({})
        for r in resps:
            inp, sim, arg, passage = r
            if arg not in args:
                args[arg] = {'passage':passage, 'quotes':[inp], 'sim':sim}
            else:
                args[arg]['quotes'].append(inp)
                if args[arg]['sim'] < sim:
                    #replace the passage if this sentence is better matched
                    args[arg]['sim'] = sim
                    args[arg]['passage'] = passage
        
        parts = []
        arglist = []
        
        for i, arg in enumerate(args):
            quotes = ''.join(['>{} \n\n'.format(q) for q in args[arg]['quotes']])
            passage = args[arg]['passage'] + '^(({})^)'.format(self.alphabet[i]) + '\n'
            parts.append(quotes)
            parts.append(passage)
            if arg in self.arg_link_dict:
                arglist.append('[({}): {}]({})'.format(self.alphabet[i], arg, self.arg_link_dict[arg]))
            else:
                arglist.append('({}): {}'.format(self.alphabet[i], arg))
        
        parts.append(self.end_message.format(', '.join(arglist)))
        return '\n'.join(parts)
        

    def reply_mentions(self, limit=25):
        """
        Main functionality. Go through mentions and reply to parent comments
        """
        for mention in self.inbox.mentions(limit=limit):
            if mention not in self.completed:
                if isinstance(mention, Comment):
                    parent = mention.parent()
                    if isinstance(parent, Comment):
                        comment_text = parent.body
                        resp = self.argmatch.match_text(comment_text, N=1)[0]
                        response_text = resp[3]
                        parent.reply(self.response_template.format(response_text, resp[0]))
                        self.completed.append(mention)
                        self.append_file(self.completed_file, mention)

    def reply_mentions_persentence(self, limit=None):
        """
        Main functionality. Go through mentions and reply to parent comments
        Uses persentence argmatcher
        """
        for mention in self.inbox.mentions(limit=limit):
            if mention.subreddit.display_name != 'testanimalsupportbot':
                continue
            if mention not in self.completed:
                if isinstance(mention, Comment):
                    parent = mention.parent()
                    if isinstance(parent, Comment):
                        comment_text = parent.body
                        resps = self.argmatch.match_text_persentence(comment_text, threshold=self.threshold)
                        if resps:
                            formatted_response = self.format_response_persentence(resps)
                            parent.reply(formatted_response)
                            print(formatted_response)
                            self.completed.append(mention)
                            self.append_file(self.completed_file, mention)
                        else:
                            self.missed.append(mention)
                            self.append_file(self.missed_file, mention)
        
    
    def run(self, refresh_rate=600):
        self.clear_already_replied()
        while True:
            self.reply_mentions_persentence()
            print('{}\tReplied to mentions, sleeping for {} seconds...'.format(time.ctime(), refresh_rate))
            time.sleep(refresh_rate)


if __name__ == "__main__":
    refresh_rate = int(sys.argv[1])
    threshold = float(sys.argv[2])
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe('universal_sentence_encoder', config={'model_name':'en_use_lg'})

    argm = ArgMatcher(nlp, None, None, preload=True)
    mb = MentionsBot(argm, USER_INFO, threshold=threshold)

    mb.run(refresh_rate=refresh_rate)



