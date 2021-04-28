import argparse
import os
import pickle
import re
from collections import OrderedDict

import bs4
import numpy as np
import pandas as pd
import spacy
import spacy_universal_sentence_encoder
from markdown import markdown
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm

from update_knowledge import arg_dict_from_df, read_eg_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="interactive test of the argmatcher",
                        action="store_true", default=False)
    args = parser.parse_args()
    return args


class ArgMatcher:

    def __init__(self,
                 nlp,
                 myths_csv,
                 myth_examples_csv,
                 preload=False,
                 preload_dir='./preload_dicts'
                 ):
        self.nlp = nlp
        self.myths_csv = myths_csv
        self.myth_examples_csv = myth_examples_csv

        self.preload = preload
        self.preload_dir = preload_dir

        if not preload:
            self.arg_dict, self.template_dict = self.setup()
        else:
            arg_dict_path = os.path.join(preload_dir, 'arg_dict.p')
            template_dict_path = os.path.join(preload_dir, 'template_dict.p')
            assert os.path.isfile(
                arg_dict_path), "Couldn't find {}".format(arg_dict_path)
            assert os.path.isfile(template_dict_path), "Couldn't find {}".format(
                template_dict_path)

            self.arg_dict = pickle.load(open(arg_dict_path, "rb"))
            self.template_dict = pickle.load(open(template_dict_path, "rb"))

    def setup(self):
        self.arg_examples = self.load_myth_examples(self.myth_examples_csv)
        self.arg_text_df = self.load_myths(self.myths_csv)
        self.arg_dict, self.template_dict = self.populate_embed_dicts()
        return self.arg_dict, self.template_dict

    def populate_embed_dicts(self):
        """
        This function populates the embedding lookup tables
        """
        self.arg_dict = OrderedDict({})
        self.arg_dict['argument'] = self.arg_text_df['Title'].values
        self.arg_dict['text'] = self.arg_text_df['Text'].values
        self.arg_dict['full_comment'] = self.arg_text_df['Full Comment'].values.astype(
            bool)

        # Getting per sentence embeddings
        arg_s_embeds = []
        arg_sentences = []
        for a, arg in enumerate(tqdm(self.arg_dict['argument'])):
            sentence_embeds = []
            sentence_texts = []
            if not self.arg_dict['full_comment'][a]:
                for sent in self.nlp(str(self.arg_dict['text'][a])).sents:
                    sentence_embeds.append(sent.vector)
                    sentence_texts.append(sent.text)
                sentence_embeds = np.array(sentence_embeds)
                sentence_texts = np.array(sentence_texts)
            arg_s_embeds.append(sentence_embeds)
            arg_sentences.append(sentence_texts)

        self.arg_dict['sentence_embeds'] = arg_s_embeds
        self.arg_dict['sentences'] = arg_sentences

        # Labelled example embeddings
        template_embeds, template_labels, template_text = [], [], []
        for i, a in enumerate(self.arg_examples):
            # Argument title
            template_embeds.append(self.nlp(a).vector)
            template_text.append('<ARGUMENT TITLE>')
            template_labels.append(i)
            # Response text
            template_embeds.append(
                self.nlp(str(self.arg_dict['text'][i])).vector)
            template_text.append('<ARGUMENT TEXT>')
            template_labels.append(i)

            for text in self.arg_examples[a]:
                # Argument examples
                template_embeds.append(self.nlp(text).vector)
                template_text.append(text)
                template_labels.append(i)

        self.template_dict = OrderedDict({})
        self.template_dict['embeds'] = np.array(template_embeds)  # X
        self.template_dict['labels'] = np.array(template_labels)  # y
        self.template_dict['text'] = np.array(template_text)  # X_text

        # writing dicts to pickle
        os.makedirs(self.preload_dir, exist_ok=True)
        pickle.dump(self.arg_dict, open(os.path.join(
            self.preload_dir, 'arg_dict.p'), "wb"))
        pickle.dump(self.template_dict, open(os.path.join(
            self.preload_dir, 'template_dict.p'), "wb"))
        return self.arg_dict, self.template_dict

    @staticmethod
    def load_myths(file):
        df = pd.read_csv(file)
        return df

    @staticmethod
    def load_myth_examples(file):
        egs_df = read_eg_df(file)
        arg_examples = arg_dict_from_df(egs_df)
        return arg_examples

    def prefilter(self, text):
        """
        prefilter text:
            e.g. strip markdown and characters that mess up formatting
        """
        html = markdown(text)
        soup = bs4.BeautifulSoup(html, features='html.parser')
        only_text = ' '.join(soup.findAll(text=True))
        only_text = re.sub('\n', '. ', only_text)
        return only_text

    def classify_relevant(self, text):
        """
        Classifies whether user input text is vegan relevant

        input: text
        output: True/False
        """
        pass

    def classify_response(self, text):
        """
        Classifiers whether a user response is agreeing or disagreeing

        input: text
        output: True/False
        """
        pass

    def catch_special_cases(self, text):
        """
        Function where special cases can be caught and dealt with, e.g.
        if embedding similarity fails regularly with certain topics

        input: text
        output: TODO
        """
        pass

    def match_text_persentence(self, text,
                               passage_length=5,
                               threshold=0.5):
        """
        Splits input into sentences and then performs similarity scoring
        Returns:
        list of sentences which match threshold:
            (
                input sentence,
                similarity score,
                argument title,
                best matched sentence in argument + passage_length subsequent sentences
            )
        """
        text = str(self.prefilter(text))
        t = self.nlp(text)
        input_sentences = []
        input_vector = t.vector
        input_sentence_vectors = []

        if len([s.text for s in t.sents]) > 2:
            for s in t.sents:
                input_sentences.append(s.text)
                input_sentence_vectors.append(s.vector)
        else:
            input_sentences.append(text)
            input_sentence_vectors.append(input_vector)

        input_sentence_vectors = np.array(input_sentence_vectors)
        sent_cs = cosine_similarity(
            input_sentence_vectors, self.template_dict['embeds'])

        best_args = np.argmax(sent_cs, axis=1)
        responses = []

        for i, a in enumerate(best_args):
            arg = self.template_dict['labels'][a]
            sim = np.max(sent_cs[i])
            inp = input_sentences[i]
            if sim >= threshold:
                if not self.arg_dict['full_comment'][arg]:
                    cs_argsent = cosine_similarity(
                        input_vector[np.newaxis, :], self.arg_dict['sentence_embeds'][arg])
                    best_sent = np.argmax(cs_argsent[0])
                    best_passage = ' '.join(
                        self.arg_dict['sentences'][arg][best_sent:best_sent+passage_length])
                else:
                    best_passage = self.arg_dict['text'][arg]

                info = {"sim": sim,
                        "matched_text": self.template_dict['text'][a]}
                responses.append(
                    (inp, info, self.arg_dict['argument'][arg], best_passage))

        return responses


if __name__ == "__main__":
    args = parse_args()

    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe('universal_sentence_encoder',
                 config={'model_name': 'en_use_lg'})

    if not args.test:
        argm = ArgMatcher(nlp, './knowledge/myths.csv',
                          './knowledge/myths_egs.csv', preload=False)
        print('Finished populating embed dicts, saved to preload_dicts')
    else:
        argm = ArgMatcher(nlp, None, None, preload=True)
        while True:
            test_input = input("Enter test sentence: ")
            print(argm.match_text_persentence(test_input))
