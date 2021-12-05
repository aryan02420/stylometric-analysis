import nltk
import numpy as np
import functools
from .constants import *

class Basic(object):
    def __init__(self, label: str, text: str) -> None:
        super().__init__()
        self.raw = text.lower()
        self.label = label


    @functools.cached_property
    def tokens(self):
        return nltk.word_tokenize(self.raw)

    @functools.cached_property
    def words(self):
        return [token for token in self.tokens if token.isalpha()]

    @functools.cached_property
    def num_tokens(self):
        return len(self.tokens)

    @functools.cached_property
    def num_words(self):
        return len(self.words)

