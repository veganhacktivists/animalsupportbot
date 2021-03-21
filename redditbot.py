import time

import praw
import spacy
from praw.models import Comment

from argmatcher import ArgMatcher


class MentionsBot:

    def __init__(self, argmatch):
        self.reddit = praw.Reddit(
            client_id="bHaig6PsHeQb6g",
            client_secret="XXXXXXXXXXXXX",
            password="XXXXXXXXXXXXX",
            user_agent="vegan response bot",
            username="animalsupportbot",
            check_for_async=False
        )
        self.inbox = praw.models.Inbox(self.reddit, _data={})
        self.argmatch = argmatch
        self.completed = []
        self.response_template = '{}\n\nThis was an automatically generated response that I matched to the idea that "{}"'

    def clear_already_replied(self):
        """
        Go through mentions manually to tick off if we have already replied,
        populates self.completed
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

    def format_response(self, resps):
        """
        TODO: Formatting responses given from the argument matcher
        """
        pass

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
                        parent.reply(self.response_template.format(
                            response_text, resp[0]))
                        self.completed.append(mention)

    def run(self, refresh_rate=600):
        self.clear_already_replied()
        while True:
            self.reply_mentions()
            time.sleep(refresh_rate)


if __name__ == "__main__":
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe('universal_sentence_encoder', config={'model_name':'en_use_lg'})

    argm = ArgMatcher(nlp, None, None, preload=True)
    mb = MentionsBot(argm)

    mb.run(refresh_rate=600)

