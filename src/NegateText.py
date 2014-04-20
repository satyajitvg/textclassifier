# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 12:22:03 2014

@author: Satyajit Gupte
"""

import re

class NegateText:
    def __init__(self, puncts):
        self.puncts = puncts

    def tokensWithNeg(self, words):
        new_tokens = []
        negated = False
        for word in words:
            transformed_word = word
            if negated == True:
                transformed_word = 'NOT_'+word
            if word in self.puncts:
                negated = False
            if word == 'not' or word.endswith("n't"):
                negated = not negated
            new_tokens.append(transformed_word)
        return new_tokens

    def getTextWithNeg(self, text):
        words = re.findall(r"[\w']+|[.,!?;]", text)
        new_words = self.tokensWithNeg(words)
        return " ".join(new_words)


