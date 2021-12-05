import functools
from .Syntactic import Syntactic
from .constants import *


class StyloAnalyzer(Syntactic):
    def __init__(self, label: str, text: str) -> None:
        super().__init__(label, text)

    @functools.cached_property
    def vector(self):
        return [self.num_words, self.type_token_ratio] \
            + [prob for word, prob in self.common_words_prob] \
            + [prob for word, prob in self.common_puncts_prob] \
            + [self.av_word_len] \
            + [self.word_len_prob_dict.get(len, 0) for len in range(1, 16)] \
            + [self.av_filtered_word_len] \
            + [self.filtered_word_len_prob_dict.get(len, 0) for len in range(1, 16)] \
            + [self.num_sentences, self.av_words_per_sent, self.num_paragraphs, self.av_words_per_para] \
            + [self.tag_prob_dict.get(tag, 0) for tag in all_tags]
        pass
