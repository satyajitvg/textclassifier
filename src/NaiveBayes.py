# -*- coding: utf-8 -*-
"""
Naive Bayes Classifer for Text
Based on http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html
@author: Satyajit Gupte
"""
from __future__ import division
import math
from collections import defaultdict

class NaiveBayesTrain:
    def __init__(self):
        self.priors = {}
        self.condprobs = defaultdict(lambda: defaultdict(float))
        self.class_label_counts = defaultdict(int)
        self.token_counts = defaultdict(lambda: defaultdict(int))

    def addDocument(self, tokens, class_label):
        tokens = set(tokens)
        self.class_label_counts[class_label]+=1
        for token in tokens:
            self.token_counts[token][class_label]+=1

    def train(self):
        doc_count = sum(self.class_label_counts.values())
        for class_label, count in self.class_label_counts.items():
            self.priors[class_label] = count/doc_count
        for token, classcounts in self.token_counts.items():
            for class_label in self.class_label_counts.keys():
                try:
                    self.condprobs[token][class_label] = (self.token_counts[token][class_label] + 1)/(self.class_label_counts[class_label] +2)
                except KeyError:
                    self.condprobs[token][class_label] = 1/(self.class_label_counts[class_label] +2)
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
            for token in tokens:
                if token in self.condprobs:
                    cat_score+=math.log(self.condprobs[token][class_label])
            scores.append((class_label, cat_score))
        sorted_scores = sorted(scores, key = lambda tup:    tup[1], reverse=True)
        return sorted_scores[0]