import nltk
import numpy as np
import functools
from .constants import *
from .Lexical import Lexical


stop_words = nltk.corpus.stopwords.words('english')
ignore_words = set(stop_words + common_words)

class Syntactic(Lexical):
    def __init__(self, label: str, text: str) -> None:
        super().__init__(label, text)


    @functools.cached_property
    def tagged_tokens(self):
        return [(token, tag) for (token, tag) in nltk.pos_tag(self.tokens) if token in self.unique_words]

    @functools.cached_property
    def tags(self):
        return [tag for (token, tag) in self.tagged_tokens]

    @functools.cached_property
    def tag_freq(self):
        return nltk.probability.FreqDist(self.tags).most_common()

    @functools.cached_property
    def tag_prob(self):
        return [(tag, freq/len(self.tags)) for (tag, freq) in self.tag_freq]


    @functools.cached_property
    def bigrams(self):
        big = nltk.bigrams(self.words)
        return [(word1, word2) for (word1, word2) in list(big) if not (word1 in ignore_words and word2 in ignore_words)]

    @functools.cached_property
    def bigram_freq(self):
        return nltk.probability.FreqDist(self.bigrams).most_common()

    @functools.cached_property
    def bigram_prob(self):
        return [(big, freq/len(self.bigrams)) for (big, freq) in self.bigram_freq]


    @functools.cached_property
    def trigrams(self):
        tig = nltk.trigrams(self.words)
        return [(word1, word2, word3) for (word1, word2, word3) in list(tig) if not (int(word1 in ignore_words) + int(word2 in ignore_words) + int(word3 in ignore_words) <= 1)]

    @functools.cached_property
    def trigram_freq(self):
        return nltk.probability.FreqDist(self.trigrams).most_common()

    @functools.cached_property
    def trigram_prob(self):
        return [(tig, freq/len(self.trigrams)) for (tig, freq) in self.trigrams]
