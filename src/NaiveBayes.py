# -*- coding: utf-8 -*-
"""
Naive Bayes Classifer for Text
Based on http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html
@author: Satyajit Gupte
"""
from __future__ import division
import math
from collections import defaultdict
from operator import itemgetter

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

    def _xlx(self, x):
        if x == 0:
            return 0
        else:
            return x*math.log(x,2)

    def _entropy(self, x,y):
        return -1*self._xlx(x/(x+y)) -1*self._xlx(y/(x+y))

    def _informationGain(self, pos,neg,tp,fp):
        fn = pos-tp
        tn = neg-fp
        p_word = (tp+fp)/(pos+neg)
        ig = self._entropy(pos,neg)-(p_word*self._entropy(tp,fp)+(1-p_word)*self._entropy(fn,tn))
        return ig

    def selectFeatures(self, token_counts):
        token_infogains = defaultdict(lambda: defaultdict(float))
        for token, class_counts in token_counts.items():
            max_infogain = 0
            for class_label, class_count in class_counts.items():
                N1 = self.class_label_counts[class_label]
                N2 = sum(self.class_label_counts[x] for x in self.class_label_counts.keys() if x != class_label)
                df1 = class_count
                df2 = sum(token_counts[token][x] for x in self.class_label_counts.keys() if x!= class_label)
                if df1 + df2 > 10:
                    ig = self._informationGain(N1, N2, df1, df2)
                    if ig > max_infogain:
                        max_infogain = ig
                #print token, class_label, df1, df2, ig
            token_infogains[token] =  max_infogain
        sorted_token_infogains = sorted(token_infogains.items(), key = itemgetter(1), reverse = True)[:500]
        return dict(sorted_token_infogains)

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

        selected_tokens = self.selectFeatures(token_counts)

        #calculate conditional probabilities
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
            for token in tokens:
                if token in self.condprobs:
                    cat_score+=math.log(self.condprobs[token][class_label])
            scores.append((class_label, cat_score))
        sorted_scores = sorted(scores, key = lambda tup:    tup[1], reverse=True)
        return sorted_scores[0]

    def classify2(self, tokens):
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
        sorted_scores = sorted(scores, key = lambda tup:    tup[1], reverse=True)
        return sorted_scores[0]