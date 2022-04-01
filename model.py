import re
import random
from abc import abstractmethod, ABC
from typing import List, Tuple

import spacy
from spacy.tokens import Doc
import pymorphy2
from gensim.test.utils import datapath
from gensim.models import KeyedVectors


class Model(ABC):
    def __init__(self, gensim_model_path: str, spacy_name: str):
        self._gensim_model: KeyedVectors = KeyedVectors.load_word2vec_format(datapath(gensim_model_path),
                                                                             binary=True)
        self._spacy_model: spacy.Language = spacy.load(spacy_name)
        self._morph = pymorphy2.MorphAnalyzer()

    def get_synonyms(self, word: str, intensity: float) -> List[str]:
        morphed = self._morph.parse(word)[0]
        similarities: List[Tuple[str, float]] = []

        if self._gensim_model.has_index_for(word):
            similarities = self._gensim_model.most_similar(positive=word, topn=5)
        similarities += self._get_random_words(word)

        def split(word_to_split: Tuple[str, float]) -> str:
            raw_word, _ = word_to_split
            splitten_words = raw_word.split('_')
            raw_word = splitten_words[0]
            return raw_word

        similarities = list(map(split,
                                filter(lambda match: match[1] < intensity,
                                       similarities
                                       )
                                )
                            )

        denormalized = list(map(lambda word: self._denormalize_word(word, morphed.tag), similarities))
        return denormalized

    def tokenize_sentence(self, sentence: str) -> spacy.tokens.Doc:
        return self._spacy_model(sentence)

    def _get_random_words(self, word: str) -> List[Tuple[str, float]]:
        random_words: List[Tuple[str, float]] = []
        while len(random_words) <= 10:
            random_words += [(w, self._get_similarity(word, w)) for w in
                             random.sample(self._gensim_model.index_to_key, 10)]
        return random_words

    def _denormalize_word(self, word: str, tags):
        tags = str(tags)
        tags = re.sub(',[AGQSPMa-z-]+? ', ',', tags)
        tags = tags.replace("impf,", "")
        tags = re.sub('([A-Z]) (plur|masc|femn|neut|inan)', '\\1,\\2', tags)
        tags = tags.replace("Impe neut", "")
        tags = tags.split(',')
        tags_clean = []
        for t in tags:
            if t:
                if ' ' in t:
                    t1, t2 = t.split(' ')
                    t = t2
                tags_clean.append(t)
        tags = frozenset(tags_clean)
        parsed = self._morph.parse(word)[0]
        inflected = parsed.inflect(tags)
        result = inflected[0] if inflected else word
        return result

    def _get_similarity(self, w1: str, w2: str) -> float:
        if not self._gensim_model.has_index_for(w1) or \
                not self._gensim_model.has_index_for(w2):
            return 0.0

        return self._gensim_model.similarity(w1, w2)

