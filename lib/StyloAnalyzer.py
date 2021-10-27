import nltk
import numpy as np
import functools

common_words = ['the','be','to','of','and','a','in','that','have','I','it','for','not','on','with','he','as','you','do','at','this','but','his','by','from','they','we','say','her','she','or','an','will','my','one','all','would','there','their','what','so','up','out','if','about','who','get','which','go','me','when','make','can','like','time','no','just','him','know','take','people','into','year','your','good','some','could','them','see','other','than','then','now','look','only','come','its','over','think','also','back','after','use','two','how','our','work','first','well','way','even','new','want','because','any','these','give','day','most','us']
punct = [".", "?", "!", ",", ";", ":", "-", "--", "“", "’"]
func_tags = ['CC', 'IN', 'DT', 'PRP', 'PRP$', 'MD', 'RP', 'UH'] # https://en.wikipedia.org/wiki/Function_word
stop_words = nltk.corpus.stopwords.words('english')


class StyloAnalyzer(object):
  def __init__(self, corpus: str, raw: str) -> None:
    super().__init__()
    self.raw = raw.lower()
    self.corpus = corpus  

  ###### lexical analysis ######

  @functools.cached_property
  def tokens(self):
    return nltk.word_tokenize(self.raw)

  @functools.cached_property
  def words(self):
    return [token for token in self.tokens if token not in punct]

  @functools.cached_property
  def unique_words(self):
    return set(self.words)

  @functools.cached_property
  def text(self):
    return nltk.text.Text(self.tokens)

  @functools.cached_property
  def word_text(self):
    return nltk.text.Text(self.words)

  @functools.cached_property
  def type_token_ratio(self):
    return len(self.unique_words) / len(self.words)

  @functools.cached_property
  def token_freq(self):
    return nltk.probability.FreqDist(self.text)

  @functools.cached_property
  def word_freq(self):
    return nltk.probability.FreqDist(self.word_text)

  @functools.cached_property
  def av_word_len(self):
    return np.mean([len(word) for word in self.words])

  def token_per_1000(self, token: str):
    return self.token_freq[token.lower()] / len(self.text) * 1000

  def word_per_1000(self, word: str):
    return self.word_freq[word.lower()] / len(self.word_text) * 1000
  
  @functools.cached_property
  def common_words_per_1000(self):
    return [(word, self.word_per_1000(word)) for word in common_words]

  @functools.cached_property
  def punct_per_1000(self):
    return [(word, self.token_per_1000(word)) for word in punct]

  @functools.cached_property
  def sentences(self):
    return nltk.sent_tokenize(self.raw)

  @functools.cached_property
  def av_words_per_sent(self):
    return np.mean([len(nltk.word_tokenize(sent)) for sent in self.sentences])

  @functools.cached_property
  def tagged_tokens(self):
    return nltk.pos_tag(self.tokens)

  @functools.cached_property
  def tags(self):
    return [tag for (token, tag) in self.tagged_tokens if token not in punct]

  @functools.cached_property
  def tag_freq(self):
    return nltk.probability.FreqDist(self.tags)

  def tag_per_100(self, tag: str):
    return self.tag_freq[tag] / len(self.tags) * 100

  @functools.cached_property
  def paragraphs(self):
    return [para.strip() for para in self.raw.split('\n\n') if len(para) > 0 and not para.isspace()]

  @functools.cached_property
  def av_words_per_para(self):
    return np.mean([len(nltk.word_tokenize(para)) for para in self.paragraphs])

  @functools.cached_property
  def non_func_words(self):
    return [token for (token, tag) in self.tagged_tokens if tag not in func_tags]

  @functools.cached_property
  def non_func_words_freq(self):
    return nltk.probability.FreqDist(self.non_func_words)

  @functools.cached_property
  def av_non_func_words_len(self):
    return np.mean([len(word) for word in self.non_func_words])

  ###### syntactic analysis ######