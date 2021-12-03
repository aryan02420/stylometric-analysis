import nltk
import numpy as np
import functools
from .constants import *
from .Basic import Basic


class Lexical(Basic):
    def __init__(self, corpus: str, raw: str) -> None:
        super().__init__(corpus, raw)
