from typing import List
import re
import random

import spacy
import pymorphy2
from gensim.test.utils import datapath
from gensim.models import KeyedVectors


class ModelHolder:
    def __init__(self, wv_location, is_wv_binary_model, spacy_name):
        self._gensim_model = KeyedVectors.load_word2vec_format(datapath(wv_location), binary=is_wv_binary_model)
        self._morph = pymorphy2.MorphAnalyzer()
        self._spacy_model = spacy.load(spacy_name)
        self._suffixes = ["_NOUN", "_VERB", "_ADJ", "_ADV", "_PRON", "_NUM", "_INTJ"]

    def get_synonyms(self, word: str, suffix: str) -> List[str]:
        morphed = self._morph.parse(word)[0]
        sought = morphed.normal_form + suffix
        similarities: List[str] = []
        if self._gensim_model.has_index_for(sought):
            similarities = self._gensim_model.most_similar(positive=sought)
        similarities += self.get_any_words(suffix)
        removed_suffixes = list(map(lambda match: match[0].split('_')[0], similarities))

        denormalized = list(map(lambda word: self._denormalize_word(word, morphed.tag), removed_suffixes))
        return denormalized

    def tokenize_sentence(self, sentence: str):
        return self._spacy_model(sentence)

    def get_any_words(self, suffix):
        random_words = []
        while len(random_words) < 100:
            random_words += list(
                filter(lambda word: word.endswith(suffix), random.sample(self._gensim_model.index_to_key, 100)))
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
