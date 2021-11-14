from typing import List, Tuple
import re
import random

import spacy
import pymorphy2
from gensim.test.utils import datapath
from gensim.models import KeyedVectors


class ModelHolder:
    def __init__(self, wv_location, is_wv_binary_model, spacy_name):
        self._gensim_model: KeyedVectors = KeyedVectors.load_word2vec_format(datapath(wv_location),
                                                                             binary=is_wv_binary_model)
        self._is_ru = "ru" in spacy_name
        self._morph = pymorphy2.MorphAnalyzer()
        self._spacy_model: spacy.Language = spacy.load(spacy_name)
        self._suffixes = ["_NOUN", "_VERB", "_ADJ", "_ADV", "_PRON", "_NUM", "_INTJ"]

    def get_synonyms(self, word: str, suffix: str, intensity: float) -> List[str]:
        morphed = self._morph.parse(word)[0]
        sought = morphed.normal_form + suffix if self._is_ru else morphed.normal_form
        similarities: List[str] = []
        if self._gensim_model.has_index_for(sought):
            similarities = self._gensim_model.most_similar(positive=sought, topn=5)
        similarities += self.get_any_words(suffix, word)

        def split(word_to_split: Tuple[str, float]) -> str:
            raw_word, _ = word_to_split
            splitten_words = raw_word.split('_')
            raw_word = splitten_words[0]
            return raw_word

        similarities = list(filter(lambda match: match[1] < intensity, similarities))
        similarities = list(map(split,
                                similarities))

        denormalized = list(map(lambda word: self._denormalize_word(word, morphed.tag), similarities))
        return denormalized

    def tokenize_sentence(self, sentence: str):
        return self._spacy_model(sentence)

    def get_any_words(self, suffix, word):
        random_words = []

        while len(random_words) < 10:
            if self._is_ru:
                random_words += list(map(lambda w: (w, self._get_similarity(word, w)),
                                         filter(lambda word: word.endswith(suffix),
                                                random.sample(self._gensim_model.index_to_key, 100))))
            else:
                random_words += [(w, self._get_similarity(word, w)) for w in
                                 random.sample(self._gensim_model.index_to_key, 100)]
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
