# -*- coding: utf-8 -*-
"""
Word Negations. Often useful for sentiment classification tasks.
Words that occur between a negation operator (ex : not, isn't, didnt't) and punctutation are modified
input --> This is not a good movie.
output --> This is not NOT_a NOT_good NOT_movie
@author: Satyajit Gupte
"""

import re

class NegateText:
    def __init__(self, puncts = [',','.','!',';']):
        self.puncts = puncts

    def tokensWithNeg(self, words): # return tokens with negation modifiers
        new_tokens = []
        negated = False
        for word in words:
            transformed_word = word
            if negated == True:
                transformed_word = 'NOT_'+word
            if word in self.puncts:
                negated = False
            if word == 'not' or word == 'never' or word.endswith("n't"):
                negated = not negated
            new_tokens.append(transformed_word)
        return new_tokens

    def getTextWithNeg(self, text): #return ' ' joined tokens
        words = re.findall(r"[\w']+|[.,!?;]", text)
        new_words = self.tokensWithNeg(words)
        return " ".join(new_words)


