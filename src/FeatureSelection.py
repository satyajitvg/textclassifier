# -*- coding: utf-8 -*-
"""
Feature Selection for Text Classifiers using Information Gain
For Multi Class Classification, info gain for
Reference : http://machinelearning.wustl.edu/mlpapers/paper_files/Forman03.pdf
@author: Satyajit Gupte
"""
from __future__ import division
import math
from collections import defaultdict
from operator import itemgetter

class IGFeatureSelection:
    def __init__(self, num_features, min_df):
        self.num_features = num_features
        self.min_df = min_df

    def xlx(self, x):
        if x == 0:
            return 0
        else:
            return x*math.log(x,2)

    def entropy(self, x,y):
        return -1*self.xlx(x/(x+y)) -1*self.xlx(y/(x+y))

    def informationGain(self, pos,neg,tp,fp):
        fn = pos-tp
        tn = neg-fp
        p_word = (tp+fp)/(pos+neg)
        ig = self.entropy(pos,neg)-(p_word*self.entropy(tp,fp)+(1-p_word)*self.entropy(fn,tn))
        return ig

    def selectFeatures(self, token_counts, class_label_counts):
        token_infogains = defaultdict(lambda: defaultdict(float))
        for token, class_counts in token_counts.items():
            max_infogain = 0
            for class_label, class_count in class_counts.items():
                N1 = class_label_counts[class_label]
                N2 = sum(class_label_counts[x] for x in class_label_counts.keys() if x != class_label)
                df1 = class_count
                df2 = sum(token_counts[token][x] for x in class_label_counts.keys() if x!= class_label)
                if df1 + df2 > self.min_df:
                    ig = self.informationGain(N1, N2, df1, df2)
                    if ig > max_infogain:
                        max_infogain = ig
            token_infogains[token] =  max_infogain
        sorted_token_infogains = sorted(token_infogains.items(), key = itemgetter(1), reverse = True)[:self.num_features]
        return dict(sorted_token_infogains)
