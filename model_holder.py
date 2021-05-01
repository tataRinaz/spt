import spacy
from gensim.test.utils import datapath
from gensim.models import KeyedVectors


class ModelHolder:
    def __init__(self, wv_location, is_wv_binary_model, spacy_name):
        self._gensim_model = KeyedVectors.load_word2vec_format(datapath(wv_location), binary=is_wv_binary_model)
        self._spacy_model = spacy.load(spacy_name)
        self._suffixes = ["_NOUN", "_VERB", "_ADJ", "_ADV", "_PRON", "_NUM", "_INTJ"]

    def get_synonyms(self, word: str) -> list:
        similarities = []
        for suffix in self._suffixes:
            searchment = word.lower() + suffix
            if not self._gensim_model.has_index_for(searchment):
                continue

            similars = self._gensim_model.most_similar(positive=searchment)
            similarities += list(map(lambda match: match[0].split('_')[0], similars))

        return similarities

    def tokenize_sentence(self, sentence):
        return self._spacy_model(sentence)
