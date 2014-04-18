# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 19:49:27 2014

@author: Z080465
"""

import os
import sys
lib_path = os.path.abspath('../')
sys.path.insert(0, lib_path)

from src.NaiveBayes import NaiveBayesTrain
from src.NaiveBayes import NaiveBayesClassify

NB = NaiveBayesTrain()


traindocs = [("Chinese Beijing Chinese",1),("Chinese Chinese Shanghai",1),("Chinese Macao",1),("Chinese Tokyo",0)]
testdocs = ["Chinese Chinese Chinese Tokyo Japan"]

for traindoc in traindocs:
    NB.addDocument(traindoc[0].split(), traindoc[1])

#print NB.class_label_counts
#print NB.token_counts
priors, condprobs = NB.train()
NB_classify = NaiveBayesClassify(priors, condprobs)

print priors
print condprobs

for testdoc in testdocs:
    print NB_classify.classify(testdoc.split())




