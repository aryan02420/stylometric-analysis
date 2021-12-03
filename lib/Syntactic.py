import nltk
import numpy as np
import functools
from .constants import *
from .Lexical import Lexical


class Syntactic(Lexical):
    def __init__(self, corpus: str, raw: str) -> None:
        super().__init__(corpus, raw)
