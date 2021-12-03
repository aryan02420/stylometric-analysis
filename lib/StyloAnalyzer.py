import nltk
import numpy as np
import functools
from .constants import *
from .Syntactic import Syntactic


class StyloAnalyzer(Syntactic):
    def __init__(self, corpus: str, raw: str) -> None:
        super().__init__(corpus, raw)
