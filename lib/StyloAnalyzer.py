import functools
from .Syntactic import Syntactic


class StyloAnalyzer(Syntactic):
    def __init__(self, label: str, text: str) -> None:
        super().__init__(label, text)

    @functools.cached_property
    def vector(self):
        pass