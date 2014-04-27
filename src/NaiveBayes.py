# -*- coding: utf-8 -*-
"""
Naive Bayes Classifer for Text
Based on http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html

Feature Selection sone using Information Gain
@author: Satyajit Gupte
"""
from __future__ import division
import math
from collections import defaultdict
from operator import itemgetter
from FeatureSelection import IGFeatureSelection

class NaiveBayesTrain:
    def __init__(self, num_features, min_df=10):
        self.priors = {}
        self.condprobs = defaultdict(lambda: defaultdict(float))
        self.class_label_counts = defaultdict(int)
        self.token_counts = defaultdict(lambda: defaultdict(int))
        self.feature_selection = IGFeatureSelection(num_features, min_df)

    def addDocument(self, tokens, class_label):
        tokens = set(tokens)
        self.class_label_counts[class_label]+=1
        for token in tokens:
            self.token_counts[token][class_label]+=1



    def train(self):
        #calculate class priors
        doc_count = sum(self.class_label_counts.values())
        for class_label, count in self.class_label_counts.items():
            self.priors[class_label] = count/doc_count

        #calcualte count of each token in everey class. If token never seen in class then token_counts[token][class_label] = 0
        token_counts = defaultdict(lambda: defaultdict(int))
        for token, class_counts in self.token_counts.items():
            for class_label in self.class_label_counts.keys():
                try:
                    token_counts[token][class_label] = self.token_counts[token][class_label]
                except KeyError:
                    token_counts[token][class_label] = 0

        selected_tokens = self.feature_selection.selectFeatures(token_counts, self.class_label_counts)

        #calculate conditional probabilities for selected tokens
        for token in selected_tokens.keys():
            for class_label in self.class_label_counts.keys():
                self.condprobs[token][class_label] = (token_counts[token][class_label] + 1)/(self.class_label_counts[class_label] +2)
        return self.priors, self.condprobs


class NaiveBayesClassify:
    def __init__(self, priors, condprobs):
        self.priors = priors
        self.condprobs = condprobs

    def classify(self, tokens):
        scores = []
        tokens = set(tokens)
        for class_label, prior in self.priors.items():
            cat_score = 0
            cat_score+=math.log(prior)
            for t in self.condprobs.keys():
                if t in tokens:
                    cat_score+=math.log(self.condprobs[t][class_label])
                else:
                    cat_score+=math.log(1-self.condprobs[t][class_label])
            scores.append((class_label, cat_score))
        return max(scores, key = itemgetter(1))

    def classify_fast(self, tokens): # Faster but not exact scoring implementation
        scores = []
        tokens = set(tokens)
        for class_label, prior in self.priors.items():
            cat_score = 0
            cat_score+=math.log(prior)
            for token in tokens:
                if token in self.condprobs:
                    cat_score+=math.log(self.condprobs[token][class_label])
            scores.append((class_label, cat_score))
        return max(scores, key = itemgetter(1))
