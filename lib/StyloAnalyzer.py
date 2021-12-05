import functools
import nltk
from .Syntactic import Syntactic
from .constants import *


class StyloAnalyzer(Syntactic):
    def __init__(self, label: str, text: str) -> None:
        super().__init__(label, text)

    @functools.cached_property
    def common_words_distribution(self):
        return [prob for word, prob in self.common_words_prob]

    @functools.cached_property
    def punct_distribution(self):
        return [prob for word, prob in self.common_puncts_prob]

    @functools.cached_property
    def stop_words_distribution(self):
        return [self.stop_word_prob_dict.get(word, 0) for word in stop_words]

    @functools.cached_property
    def word_len_distribution(self):
        return [self.word_len_prob_dict.get(len, 0) for len in range(1, 16)]

    @functools.cached_property
    def filtered_word_len_distribution(self):
        return [self.filtered_word_len_prob_dict.get(len, 0) for len in range(1, 16)]

    @functools.cached_property
    def stop_word_len_distribution(self):
        return [self.stop_word_len_prob_dict.get(len, 0) for len in range(1, 16)]

    @functools.cached_property
    def tag_distribution(self):
        return [self.tag_prob_dict.get(tag, 0) for tag in all_tags]

    @functools.cached_property
    def vector(self):
        return [self.num_words, self.type_token_ratio] \
            + self.common_words_distribution\
            + self.punct_distribution \
            + self.stop_words_distribution \
            + [self.av_word_len] \
            + self.word_len_distribution \
            + [self.av_filtered_word_len] \
            + self.filtered_word_len_distribution \
            + [self.av_stop_word_len] \
            + self.stop_word_len_distribution \
            + [self.num_sentences, self.av_words_per_sent, self.num_paragraphs, self.av_words_per_para] \
            + self.tag_distribution
    
    @functools.cached_property
    def json(self):
        return {
            'common_words_distribution': self.common_words_distribution,
            'punct_distribution': self.punct_distribution,
            'stop_words_distribution': self.stop_words_distribution,
            'word_len_distribution': self.word_len_distribution,
            'filtered_word_len_distribution': self.filtered_word_len_distribution,
            'stop_word_len_distribution': self.stop_word_len_distribution,
            'tag_distribution': self.tag_distribution,
            'av_words_per_sent': self.av_words_per_sent,
            'av_words_per_para': self.av_words_per_para,
        }
