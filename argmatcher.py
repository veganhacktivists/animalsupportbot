import argparse
import glob
import os
import pickle
import pprint
import re
from collections import OrderedDict

import bs4
import numpy as np
import pandas as pd
import spacy
import spacy_universal_sentence_encoder
import yaml
from markdown import markdown
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        help="interactive test of the argmatcher",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    return args


class ArgMatcher:
    def __init__(
        self,
        nlp,
        myths_csv,
        myth_examples_csv,
        n_neighbors=1,
        preload=False,
        preload_dir="./preload_dicts",
    ):
        self.nlp = nlp
        self.myths_csv = myths_csv
        self.myth_examples_csv = myth_examples_csv

        self.n_neighbors = n_neighbors
        self.preload = preload
        self.preload_dir = preload_dir

        if not preload:
            self.arg_dict, self.template_dict = self.setup()
        else:
            arg_dict_path = os.path.join(preload_dir, "arg_dict.p")
            template_dict_path = os.path.join(preload_dir, "template_dict.p")
            assert os.path.isfile(arg_dict_path), "Couldn't find {}".format(
                arg_dict_path
            )
            assert os.path.isfile(template_dict_path), "Couldn't find {}".format(
                template_dict_path
            )

            self.arg_dict = pickle.load(open(arg_dict_path, "rb"))
            self.template_dict = pickle.load(open(template_dict_path, "rb"))

        # Mapping of encoded label to text
        # E.g. {plants_feel_pain: 4}
        self.key_label_map = OrderedDict(
            {v: k for k, v in enumerate(self.arg_dict["key"])}
        )
        # E.g. {4: plants_feel_pain}
        self.label_key_map = OrderedDict({v: k for k, v in self.key_label_map.items()})

        self.eye = np.eye(len(self.arg_dict["argument"]) + 1)
        self.clf = KNeighborsClassifier(
            n_neighbors=self.n_neighbors, weights="distance", metric="cosine"
        )
        self.fit_classifier()

    @staticmethod
    def get_myths(myth_dir):
        """
        Get a dict containing all the arguments, examples, responses
        """
        myth_dict = OrderedDict({})
        yamls = sorted(glob.glob(os.path.join(myth_dir, "myths/*.yaml")))
        for file in yamls:
            with open(file) as fp:
                arg_dict = yaml.safe_load(fp)
            resp_md_path = os.path.join(
                myth_dir, "responses/{}.md".format(arg_dict["key"])
            )

            assert os.path.isfile(resp_md_path), "Couldn't find {}".format(resp_md_path)
            with open(resp_md_path) as fp:
                arg_dict["text"] = "".join(fp.readlines())

            key = arg_dict["key"]
            myth_dict[key] = arg_dict

        # Move the n/a class to the front, ensures class label is 0
        myth_dict.move_to_end("_na_", last=False)

        return myth_dict

    def setup(self):
        self.myth_dict = self.get_myths("./knowledge/")
        self.arg_dict, self.template_dict = self.populate_embed_dicts()
        return self.arg_dict, self.template_dict

    def populate_embed_dicts(self):
        """
        This function populates the embedding lookup tables

        TODO: clean this up - this became quite messy after refactoring how the knowledge was stored
        """
        self.arg_dict = OrderedDict(
            {
                "key": [],
                "argument": [],
                "text": [],
                "full_comment": [],
                "enable_resp": [],
                "link": [],
                "examples": [],
            }
        )

        for i, arg in enumerate(self.myth_dict):
            self.arg_dict["key"].append(arg)
            self.arg_dict["argument"].append(self.myth_dict[arg]["title"])
            self.arg_dict["text"].append(self.myth_dict[arg]["text"])
            self.arg_dict["full_comment"].append(self.myth_dict[arg]["full_comment"])
            self.arg_dict["enable_resp"].append(self.myth_dict[arg]["enable_resp"])
            self.arg_dict["link"].append(self.myth_dict[arg]["link"])
            self.arg_dict["examples"].append(self.myth_dict[arg]["examples"])

        # Getting per sentence embeddings
        arg_s_embeds = []
        arg_sentences = []
        for a, arg in enumerate(tqdm(self.arg_dict["argument"])):
            sentence_embeds = []
            sentence_texts = []
            if not self.arg_dict["full_comment"][a]:
                for sent in self.nlp(str(self.arg_dict["text"][a])).sents:
                    sentence_embeds.append(sent.vector)
                    sentence_texts.append(sent.text)
                sentence_embeds = np.array(sentence_embeds)
                sentence_texts = np.array(sentence_texts)
            arg_s_embeds.append(sentence_embeds)
            arg_sentences.append(sentence_texts)

        self.arg_dict["sentence_embeds"] = arg_s_embeds
        self.arg_dict["sentences"] = arg_sentences

        # Labelled example embeddings
        template_embeds, template_labels, template_text = [], [], []
        for i, a in enumerate(self.arg_dict["argument"]):
            # Argument title
            template_embeds.append(self.nlp(a).vector)
            template_text.append("<ARGUMENT TITLE>")
            template_labels.append(i)
            # Response text
            template_embeds.append(self.nlp(str(self.arg_dict["text"][i])).vector)
            template_text.append("<ARGUMENT TEXT>")
            template_labels.append(i)

            for text in self.arg_dict["examples"][i]:
                # Argument examples
                if text:
                    template_embeds.append(self.nlp(text).vector)
                    template_text.append(text)
                    template_labels.append(i)

        self.template_dict = OrderedDict({})
        self.template_dict["embeds"] = np.array(template_embeds)  # X
        self.template_dict["labels"] = np.array(template_labels)  # y
        self.template_dict["text"] = np.array(template_text)  # X_text

        # writing dicts to pickle
        os.makedirs(self.preload_dir, exist_ok=True)
        pickle.dump(
            self.arg_dict, open(os.path.join(self.preload_dir, "arg_dict.p"), "wb")
        )
        pickle.dump(
            self.template_dict,
            open(os.path.join(self.preload_dir, "template_dict.p"), "wb"),
        )
        return self.arg_dict, self.template_dict

    def fit_classifier(self):
        X_train = self.template_dict["embeds"]
        y_train = self.template_dict["labels"]
        self.clf.fit(X_train, y_train)

    def prefilter(self, text):
        """
        prefilter text:
            e.g. strip markdown and characters that mess up formatting
        """
        html = markdown(text)
        soup = bs4.BeautifulSoup(html, features="html.parser")
        only_text = " ".join(soup.findAll(text=True))
        only_text = re.sub("\n", " ", only_text)
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

    @staticmethod
    def remove_nan_arguments(responses):
        """
        Goes through responses and removes _na_ matched sentences

        Also removes arguments where "enable_resp" flag of the response is False
        """
        new_resps = []
        for r in responses:
            # _na_ class should have 0 class label
            if r["matched_arglabel"] != 0 and r["enable_resp"]:
                new_resps.append(r)
        return new_resps

    def match_text(self, text, **kwargs):
        """
        Match text persentence with _na_ removed
        """
        resps = self.match_text_persentence(text, **kwargs)
        return self.remove_nan_arguments(resps)

    def match_text_persentence(
        self,
        text,
        arg_labels=None,
        threshold=0.5,
        N_neighbors=1,
        return_reply=True,
        passage_length=5,
        certain_threshold=0.9,
    ):
        """
        Splits input into sentences and then performs similarity scoring

        Inputs:
            text: the input text
            arg_labels: (optional) a set of ints - the matcher will only match to these classes,
            threshold: the minimum threshold that the similarity must have to be matched,
            N_neighbors: number of neighbors with a weighted vote,
            return_reply: Boolean which determines if the response text should be returned,
            passage_length: If the reply text is not pasted in full, returns this many sentences
                                in addition to the most similar response sentence.
            certain_threshold: Threshold at which N_neighbors is ignored and the best match is picked.

        Returns:
            list of dicts with the following info:
                {
                    'input_sentence': Sentence from text,
                    'matched_argument': The argument input_sentence was matched to,
                    'matched_text': The nearest neighbour text which input_sentence matched to,
                    'matched_arglabel': The argument label (int) of matched_argument,
                    'similarity': similarity score of matched_text,
                    'reply_text': The most similar passage in the response text
                    'similarities': The similarities of the n_neighbors,
                    'neighbor_texts': The texts of the most similar n_neighbors
                }
        """
        text = str(self.prefilter(text))
        t = self.nlp(text)
        input_sentences = []
        input_vector = t.vector
        input_sentence_vectors = []

        for s in t.sents:
            input_sentences.append(s.text)
            input_sentence_vectors.append(s.vector)

        input_sentence_vectors = np.array(input_sentence_vectors)

        if not arg_labels:
            y = self.template_dict["labels"]
            y_text = self.template_dict["text"]

            neigh_dist, neigh_ind = self.clf.kneighbors(
                input_sentence_vectors, n_neighbors=N_neighbors, return_distance=True
            )
        else:
            # Getting neighbors with only arg_labels as candidates
            # This is quite inefficient: TODO: clean this up
            X_train = self.template_dict["embeds"]
            y_train = self.template_dict["labels"]
            y_text = self.template_dict["text"]
            mask = [i for i, y in enumerate(y_train) if y in arg_labels]

            X = X_train[mask]
            y = y_train[mask]
            y_text = y_text[mask]

            mini_clf = KNeighborsClassifier(
                n_neighbors=N_neighbors, weights="distance", metric="cosine"
            )

            mini_clf.fit(X, y)

            if mini_clf.n_samples_fit_ < N_neighbors:
                # Reduce N_neighbors if we have masked too many samples
                N_neighbors = mini_clf.n_samples_fit_

            neigh_dist, neigh_ind = mini_clf.kneighbors(
                input_sentence_vectors, n_neighbors=N_neighbors, return_distance=True
            )

        neigh_sim = 1 - neigh_dist

        best_text = y_text[neigh_ind]

        # Weighted Vote Nearest Neighbour
        best_cs_labels = y[neigh_ind]
        best_cs_labels_oh = self.eye[best_cs_labels]  # onehot
        weighted_vote = np.expand_dims(neigh_sim, -1) * best_cs_labels_oh
        weighted_vote = np.argmax(np.sum(weighted_vote, axis=1), -1)

        responses = []

        for i, weighted_arg in enumerate(weighted_vote):
            sim = np.max(neigh_sim[i])
            a = neigh_ind[i, np.argmax(neigh_sim[i])]
            inp = input_sentences[i]

            if sim >= certain_threshold:
                arg = y[a]
            else:
                arg = weighted_arg

            if sim >= threshold:
                if return_reply:
                    if not self.arg_dict["full_comment"][arg]:
                        # Find the best passage if full_comment is False
                        cs_argsent = cosine_similarity(
                            input_vector[np.newaxis, :],
                            self.arg_dict["sentence_embeds"][arg],
                        )
                        best_sent = np.argmax(cs_argsent[0])
                        best_passage = " ".join(
                            self.arg_dict["sentences"][arg][
                                best_sent : best_sent + passage_length
                            ]
                        )
                    else:
                        best_passage = self.arg_dict["text"][arg]
                else:
                    best_passage = ""

                resp = {
                    "input_sentence": inp,
                    "matched_argument": self.arg_dict["argument"][arg],
                    "enable_resp": self.arg_dict["enable_resp"][arg],
                    "matched_text": y_text[a],
                    "matched_arglabel": int(arg),
                    "similarity": float(sim),
                    "reply_text": best_passage,
                    "similarities": list(map(float, neigh_sim[i])),
                    "neighbor_texts": list(map(str, best_text[i])),
                    "certain_threshold": certain_threshold,
                    "link": self.arg_dict["link"][arg],
                }

                responses.append(resp)

        return responses

    def match_batch_text(
        self, texts, threshold=0.5, N_neighbors=1, certain_threshold=0.9
    ):
        """
        !Eval Use Only!

        Match list of text in a batch, return matched labels
        """
        X = []
        for t in tqdm(texts):
            processed_t = str(self.prefilter(t))
            X.append(self.nlp(processed_t).vector)

        X = np.array(X)  # (N, embedding_dim)
        y = self.template_dict["labels"]  # Num classes

        # Both neigh_dist, neigh_sim (N, N_neighbors)
        neigh_dist, neigh_ind = self.clf.kneighbors(
            X, n_neighbors=N_neighbors, return_distance=True
        )

        neigh_sim = 1 - neigh_dist
        best_cs_labels = y[neigh_ind]
        best_cs_labels_oh = self.eye[best_cs_labels]  # onehot
        weighted_vote = np.expand_dims(neigh_sim, -1) * best_cs_labels_oh
        weighted_vote = np.argmax(np.sum(weighted_vote, axis=1), -1)

        # Get the top 1 prediction
        top1 = y[neigh_ind[:, 0]]

        # Get indexes of examples which meet certainty thresh
        certain_indexes = np.max(neigh_sim, axis=-1) >= certain_threshold

        # Combine certain examples with weighted ones
        # Zeroes out relevant values, and then recombines
        cert_preds = certain_indexes.astype(int) * top1
        weight_preds = (1 - certain_indexes.astype(int)) * weighted_vote
        out = cert_preds + weight_preds

        # Finally, zero out examples which don't meet threshold condition
        # This turns the prediction to _na_
        below_thresh_indexes = np.max(neigh_sim, axis=-1) >= threshold
        out = out * below_thresh_indexes.astype(int)
        return out


if __name__ == "__main__":
    args = parse_args()

    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("universal_sentence_encoder", config={"model_name": "en_use_lg"})

    if not args.test:
        argm = ArgMatcher(
            nlp, "./knowledge/myths.csv", "./knowledge/myths_egs.csv", preload=False
        )
        print("Finished populating embed dicts, saved to preload_dicts")
    else:
        argm = ArgMatcher(nlp, None, None, preload=True)
        while True:
            test_input = input("Enter test sentence: ")
            num_n = int(input("Num neighbours with vote: "))
            threshold = float(input("Threshold: "))
            certain_threshold = float(input("Certain threshold: "))
            output = argm.match_text_persentence(test_input, N_neighbors=num_n, threshold=threshold, certain_threshold=certain_threshold)

            # Replacing the newline characters to make printing a little nicer
            for o in output:
                o["reply_text"] = o["reply_text"].replace("\n", "")

            pprint.pprint(output)
