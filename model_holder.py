from typing import List
import re

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
        sought = self._morph.parse(word)[0].normal_form + suffix
        if not self._gensim_model.has_index_for(sought):
            return []
        similarities = self._gensim_model.most_similar(positive=sought)
        removed_suffixes = list(map(lambda match: match[0].split('_')[0], similarities))

        # denormalized_words = list(map(lambda word: self._denormalize_word(word, sought_parsed.tag), removed_suffixes))
        return removed_suffixes

    def tokenize_sentence(self, sentence: str):
        return self._spacy_model(sentence)

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
        result = parsed.inflect(tags)[0]
        return result
