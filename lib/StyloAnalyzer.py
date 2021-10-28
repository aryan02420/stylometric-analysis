import nltk
import numpy as np
import functools

common_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make',
                'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us']
punct = [".", "?", "!", ",", ";", ":", "-", "--", "“", "’"]
# https://en.wikipedia.org/wiki/Function_word
func_tags = ['CC', 'IN', 'DT', 'PRP', 'PRP$', 'MD', 'RP', 'UH']
stop_words = nltk.corpus.stopwords.words('english')


class StyloAnalyzer(object):
    def __init__(self, corpus: str, raw: str) -> None:
        super().__init__()
        self.raw = raw.lower()
        self.corpus = corpus

    @classmethod
    def csv_header(cls):
        return "corpus\ttype token ratio\twords\tunique words\tsentences\tparagraphs\tav word len\tav non func word len\tav words per sent\tav words per para\t" + "\t".join(common_words) + "\t" + "\t".join(punct) + "\ttop"

    def __str__(self):
        basic_info = "%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (self.corpus, self.type_token_ratio, self.num_words, self.num_unique_words,
                                                                 self.num_sentences, self.num_paragraphs, self.av_word_len, self.av_non_func_words_len, self.av_words_per_sent, self.av_words_per_para)
        common_words_info = "\t".join(
            [str(count) for (word, count) in self.common_words_per_1000])
        common_punct_info = "\t".join(
            [str(count) for (token, count) in self.common_puncts_per_1000])
        top_non_func_words = " ".join(
            [word for (word, count) in self.non_func_words_freq.most_common()][:15])
        return "%s\t%s\t%s\t%s" % (basic_info, common_words_info, common_punct_info, top_non_func_words)

    ###### lexical analysis ######

    @functools.cached_property
    def tokens(self):
        '''all tokens including punctuations'''
        return nltk.word_tokenize(self.raw)

    @functools.cached_property
    def words(self):
        '''non punctuation token'''
        return [token for token in self.tokens if token not in punct]

    @functools.cached_property
    def num_words(self):
        '''count of all word tokens'''
        return len(self.words)

    @functools.cached_property
    def unique_words(self):
        '''unique words'''
        return set(self.words)

    @functools.cached_property
    def num_unique_words(self):
        '''count of all unique word tokens'''
        return len(self.unique_words)

    @functools.cached_property
    def text(self):
        ''''''
        return nltk.text.Text(self.tokens)

    @functools.cached_property
    def word_text(self):
        ''''''
        return nltk.text.Text(self.words)

    @functools.cached_property
    def type_token_ratio(self):
        ''''''
        return self.num_unique_words / self.num_words

    @functools.cached_property
    def token_freq(self):
        '''count of all tokens'''
        return nltk.probability.FreqDist(self.text)

    @functools.cached_property
    def word_freq(self):
        '''count of word tokens'''
        return nltk.probability.FreqDist(self.word_text)

    @functools.cached_property
    def av_word_len(self):
        '''average length of word'''
        return np.mean([len(word) for word in self.words])

    def token_per_1000(self, token: str):
        '''number of occurances of a token for every 100 tokens'''
        return self.token_freq[token.lower()] / len(self.text) * 1000

    def word_per_1000(self, word: str):
        '''number of occurances of a word for every 100 word'''
        return self.word_freq[word.lower()] / len(self.word_text) * 1000

    @functools.cached_property
    def common_words_per_1000(self):
        '''count of the most common english world'''
        return [(word, self.word_per_1000(word)) for word in common_words]

    @functools.cached_property
    def common_puncts_per_1000(self):
        '''number of occurances of a punctuation for every 100 tokens'''
        return [(word, self.token_per_1000(word)) for word in punct]

    @functools.cached_property
    def sentences(self):
        ''''''
        return nltk.sent_tokenize(self.raw)

    @functools.cached_property
    def num_sentences(self):
        ''''''
        return len(self.sentences)

    @functools.cached_property
    def av_words_per_sent(self):
        '''average count of word in each sentence'''
        return np.mean([len(nltk.word_tokenize(sent)) for sent in self.sentences])

    @functools.cached_property
    def tagged_tokens(self):
        '''POS tagged tokens'''
        return nltk.pos_tag(self.tokens)

    @functools.cached_property
    def tags(self):
        '''all unique tags used'''
        return [tag for (token, tag) in self.tagged_tokens if token not in punct]

    @functools.cached_property
    def tag_freq(self):
        '''count of all tags'''
        return nltk.probability.FreqDist(self.tags)

    def tag_per_100(self, tag: str):
        '''tag occurance per 100 tags'''
        return self.tag_freq[tag] / len(self.tags) * 100

    @functools.cached_property
    def paragraphs(self):
        ''''''
        return [para.strip() for para in self.raw.split('\n\n') if len(para) > 0 and not para.isspace()]

    @functools.cached_property
    def num_paragraphs(self):
        ''''''
        return len(self.paragraphs)

    @functools.cached_property
    def av_words_per_para(self):
        '''average count of word in each paragraph'''
        return np.mean([len(nltk.word_tokenize(para)) for para in self.paragraphs])

    @functools.cached_property
    def non_func_words(self):
        '''all tokens excluding function words'''
        return [token for (token, tag) in self.tagged_tokens if tag not in func_tags and token not in punct]

    @functools.cached_property
    def num_non_func_words(self):
        '''count of tokens excluding function words'''
        return len(self.non_func_words)

    @functools.cached_property
    def non_func_words_freq(self):
        '''count of all non functional words'''
        return nltk.probability.FreqDist(self.non_func_words)

    @functools.cached_property
    def av_non_func_words_len(self):
        '''average length of non functional words'''
        return np.mean([len(word) for word in self.non_func_words])

    def non_func_word_per_100(self, word: str):
        '''number of occurances of a non functional word per 100'''
        return self.non_func_words_freq[word.lower()] / self.num_non_func_words * 1000

    ###### syntactic analysis ######

    @functools.cached_property
    def bigrams(self):
        return nltk.bigrams(self.text)

    @functools.cached_property
    def bigram_cfd(self):
        return nltk.probability.ConditionalFreqDist(self.bigrams)

    @functools.cached_property
    def bigram_freq(self):
        return nltk.probability.FreqDist(self.bigrams)

    @functools.cached_property
    def bigram_top(self):
        return self.bigram_freq.most_common()
