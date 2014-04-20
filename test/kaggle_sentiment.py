# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 10:33:24 2014

@author: Z080465
"""

import os
import sys
import re

lib_path = os.path.abspath('../')
sys.path.insert(0, lib_path)

from src.NaiveBayes import NaiveBayesTrain
from src.NaiveBayes import NaiveBayesClassify
from src.NegateText import NegateText

puncts = [',','.','!',';']
ng = NegateText(puncts)


trainfile = open(sys.argv[1],'r')
trainfile.next()
testfile = open(sys.argv[2],'r')
testfile.next()

NB_t = NaiveBayesTrain()

def ngrams(tokens, n):
    return [' '.join(tokens[i:(i+n)]) for i in range(len(tokens) - (n-1))]


for line in trainfile:
    toks = line.split('\t')
    text = toks[2]
    tokens =re.findall(r"[\w']+|[.,!?;]", text.lower())
    tokens = ng.tokensWithNeg(tokens)
    #tokens.extend(ngrams(tokens,2))
    sentiment_label = int(toks[3])
    NB_t.addDocument(ngrams(tokens,2), sentiment_label)
    #print text, sentiment_label

priors, condprobs = NB_t.train()
#print priors

#sys.exit()
NB_c = NaiveBayesClassify(priors, condprobs)

print "phraseId,Sentiment"
for line in testfile:
    toks = line.split('\t')
    phrase_id = toks[0]
    text = toks[2]
    tokens =re.findall(r"[\w']+|[.,!?;]", text.lower())
    tokens = ng.tokensWithNeg(tokens)
    label, score =  NB_c.classify2(ngrams(tokens,2))
    print "%s,%s"%(phrase_id,label)


