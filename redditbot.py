import os
import time
import sys
from collections import OrderedDict

import praw
import spacy
from praw.models import Comment

from argmatcher import ArgMatcher
from local_info import USER_INFO

def read_completed_list(file):
    completed = []
    with open(file) as fp:
        for line in fp:
            completed.append(line.strip())
    return completed

class MentionsBot:

    def __init__(self, argmatch, user_info):
        self.reddit = praw.Reddit(
                    check_for_async=False,
                    **user_info
                )
        self.inbox = praw.models.Inbox(self.reddit, _data={})
        self.argmatch = argmatch
        self.completed = []
        self.completed_file = './completed.csv'
        if os.path.isfile(self.completed_file):
            self.completed = read_completed_list(self.completed_file)

        self.response_template = '{}\n\nThis was an automatically generated response that I matched to the idea that "{}"'
        self.end_message = '\n`This was an automatically generated response based on the idea(s)/myth(s):`\n `{}` \n \
                (Responses taken from vegan advocates like Earthling Ed)'
    
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
                            self.append_to_completed(mention)

    def append_to_completed(self, comment_id):
        with open(self.completed_file, 'a') as wp:
            line = '{}\n'.format(comment_id)
            wp.write(line)


    def format_response_persentence(self, resps):
        """
        TODO: Formatting responses given from the argument matcher
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
        for arg in args:
            quotes = ''.join(['>{} \n\n'.format(q) for q in args[arg]['quotes']])
            passage = args[arg]['passage'] + '\n'
            parts.append(quotes)
            parts.append(passage)
        
        parts.append(self.end_message.format(', '.join(list(args.keys()))))
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
                        self.append_to_completed(mention)

    def reply_mentions_persentence(self, limit=None):
        """
        Main functionality. Go through mentions and reply to parent comments
        Uses persentence argmatcher
        """
        for mention in self.inbox.mentions(limit=limit):
            if mention not in self.completed:
                if isinstance(mention, Comment):
                    parent = mention.parent()
                    if isinstance(parent, Comment):
                        comment_text = parent.body
                        resps = self.argmatch.match_text_persentence(comment_text)
                        formatted_response = self.format_response_persentence(resps)
                        parent.reply(formatted_response)
                        self.completed.append(mention)
                        self.append_to_completed(mention)
                        print(formatted_response)
    
    
    def run(self, refresh_rate=600):
        self.clear_already_replied()
        while True:
            self.reply_mentions_persentence()
            print('Replied to mentions, sleeping for {} seconds...'.format(refresh_rate))
            time.sleep(refresh_rate)


if __name__ == "__main__":
    refresh_rate = int(sys.argv[1])
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe('universal_sentence_encoder', config={'model_name':'en_use_lg'})

    argm = ArgMatcher(nlp, None, None, preload=True)
    mb = MentionsBot(argm, USER_INFO)

    mb.run(refresh_rate=refresh_rate)



