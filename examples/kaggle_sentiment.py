# -*- coding: utf-8 -*-
"""
Kaggle Movie Reviews Sentiment contest
"""

import os
import sys
import re

lib_path = os.path.abspath('../')
sys.path.insert(0, lib_path)

from src.MultinomialNB import NBTrain, NBClassify
from src.NegateText import NegateText

negate = NegateText()


trainfile = open('train.tsv','r')
trainfile.next() # skip header
testfile = open('test.tsv','r')
testfile.next()

train = NBTrain(num_features=1000, min_df=2)

def ngrams(tokens, n):
    return [' '.join(tokens[i:(i+n)]) for i in range(len(tokens) - (n-1))]


for line in trainfile:
    toks = line.split('\t')
    text = toks[2]
    tokens =re.findall(r"[\w']+|[.,!?;]", text.lower())
    tokens = negate.tokensWithNeg(tokens)
    bigrams = ngrams(tokens,2)
    trigrams = ngrams(tokens,3)
    sentiment_label = int(toks[3])
    train.addDocument(bigrams + trigrams , sentiment_label)
    #print text, sentiment_label

priors, condprobs = train.train()
clf = NBClassify(priors, condprobs)

print "phraseId,Sentiment"
for line in testfile:
    toks = line.split('\t')
    phrase_id = toks[0]
    text = toks[2]
    tokens =re.findall(r"[\w']+|[.,!?;]", text.lower())
    tokens = negate.tokensWithNeg(tokens)
    bigrams = ngrams(tokens,2)
    trigrams = ngrams(tokens,3)
    label, score =  clf.classify(bigrams + trigrams)
    print "%s,%s"%(phrase_id,label)


