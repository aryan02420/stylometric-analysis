import nltk
import numpy as np
import functools
from .constants import *
from .Basic import Basic


class Lexical(Basic):
    def __init__(self, label: str, text: str) -> None:
        super().__init__(label, text)

    @functools.cached_property
    def unique_words(self):
        return set(self.words)

    @functools.cached_property
    def num_unique_words(self):
        return len(self.unique_words)

    @functools.cached_property
    def type_token_ratio(self):
        return self.num_unique_words / self.num_words

    # all words and tokens

    @functools.cached_property
    def token_freq(self):
        return nltk.probability.FreqDist(self.tokens).most_common()

    @functools.cached_property
    def word_freq(self):
        return nltk.probability.FreqDist(self.words).most_common()

    @functools.cached_property
    def token_prob(self):
        return [(token, freq/self.num_tokens) for (token, freq) in self.token_freq]

    @ functools.cached_property
    def token_prob_dict(self):
        return {token: freq for (token, freq) in self.token_freq}

    @functools.cached_property
    def word_prob(self):
        return [(word, freq/self.num_words) for (word, freq) in self.word_freq]

    @ functools.cached_property
    def word_prob_dict(self):
        return {word: freq for (word, freq) in self.word_freq}

    @ functools.cached_property
    def common_words_prob(self):
        return [(word, self.word_prob_dict.get(word, 0)/self.num_words) for word in common_words]

    @ functools.cached_property
    def common_puncts_prob(self):
        return [(word, self.token_prob_dict.get(word, 0)/self.num_tokens) for word in punct]

    # length of words

    @ functools.cached_property
    def word_len(self):
        return [len(word) for word in self.words]

    @ functools.cached_property
    def word_len_freq(self):
        return nltk.probability.FreqDist(self.word_len).most_common()

    @ functools.cached_property
    def word_len_prob(self):
        return [(len, freq/self.num_words) for (len, freq) in self.word_len_freq]

    @ functools.cached_property
    def word_len_prob_dict(self):
        return {word: freq for (word, freq) in self.word_len_prob}

    @ functools.cached_property
    def av_word_len(self):
        return np.mean(self.word_len)

    # filtered words

    @ functools.cached_property
    def filtered_words(self):
        return [word for word in self.words if word not in stop_words]

    @ functools.cached_property
    def num_filtered_words(self):
        return len(self.filtered_words)

    @ functools.cached_property
    def filtered_word_freq(self):
        return nltk.probability.FreqDist(self.filtered_words).most_common()

    @ functools.cached_property
    def filtered_word_prob(self):
        return [(word, freq/self.num_filtered_words) for (word, freq) in self.filtered_word_freq]

    @ functools.cached_property
    def filtered_word_prob_dict(self):
        return {word: freq for (word, freq) in self.filtered_word_prob}

    # stop words

    @ functools.cached_property
    def stop_words(self):
        return [word for word in self.words if word in stop_words]

    @ functools.cached_property
    def num_stop_words(self):
        return len(self.stop_words)

    @ functools.cached_property
    def stop_word_freq(self):
        return nltk.probability.FreqDist(self.stop_words).most_common()

    @ functools.cached_property
    def stop_word_prob(self):
        return [(word, freq/self.num_stop_words) for (word, freq) in self.stop_word_freq]

    @ functools.cached_property
    def stop_word_prob_dict(self):
        return {word: freq for (word, freq) in self.stop_word_prob}

    # length of filtered words

    @ functools.cached_property
    def filtered_word_len(self):
        return [len(word) for word in self.filtered_words]

    @ functools.cached_property
    def filtered_word_len_freq(self):
        return nltk.probability.FreqDist(self.filtered_word_len).most_common()

    @ functools.cached_property
    def filtered_word_len_prob(self):
        return [(len, freq/self.num_filtered_words) for (len, freq) in self.filtered_word_len_freq]

    @ functools.cached_property
    def filtered_word_len_prob_dict(self):
        return {word: freq for (word, freq) in self.filtered_word_len_prob}

    @ functools.cached_property
    def av_filtered_word_len(self):
        return np.mean(self.filtered_word_len)

    # length of stop words

    @ functools.cached_property
    def stop_word_len(self):
        return [len(word) for word in self.stop_words]

    @ functools.cached_property
    def stop_word_len_freq(self):
        return nltk.probability.FreqDist(self.stop_word_len).most_common()

    @ functools.cached_property
    def stop_word_len_prob(self):
        return [(len, freq/self.num_stop_words) for (len, freq) in self.stop_word_len_freq]

    @ functools.cached_property
    def stop_word_len_prob_dict(self):
        return {word: freq for (word, freq) in self.stop_word_len_prob}

    @ functools.cached_property
    def av_stop_word_len(self):
        return np.mean(self.stop_word_len)

    # sentences

    @ functools.cached_property
    def sentences(self):
        return [sent.rstrip() for sent in nltk.sent_tokenize(self.raw)]

    @ functools.cached_property
    def num_sentences(self):
        return len(self.sentences)

    @ functools.cached_property
    def sent_len(self):
        return [len(sent) for sent in self.sentences]

    @ functools.cached_property
    def sent_len_freq(self):
        return nltk.probability.FreqDist(self.sent_len).most_common()

    @ functools.cached_property
    def sent_len_prob(self):
        return [(len, freq/self.num_sentences) for (len, freq) in self.sent_len_freq]

    @ functools.cached_property
    def av_words_per_sent(self):
        return np.mean([len(nltk.word_tokenize(sent)) for sent in self.sentences])

    # paragraphs

    @ functools.cached_property
    def paragraphs(self):
        return [para.strip() for para in self.raw.split('\n\n') if len(para) > 0 and not para.isspace()]

    @ functools.cached_property
    def num_paragraphs(self):
        return len(self.paragraphs)

    @ functools.cached_property
    def para_len(self):
        return [len(para) for para in self.paragraphs]

    @ functools.cached_property
    def para_len_freq(self):
        return nltk.probability.FreqDist(self.para_len).most_common()

    @ functools.cached_property
    def para_len_prob(self):
        return [(len, freq/self.num_paragraphs) for (len, freq) in self.para_len_freq]

    @ functools.cached_property
    def av_words_per_para(self):
        return np.mean([len(nltk.word_tokenize(para)) for para in self.paragraphs])
