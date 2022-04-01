from collections import namedtuple
from typing import List, Optional, Tuple, Dict

from spacy.tokens import Doc

from model import Model
from utils import RandomIndexIterator, join_sentence

SearchParameters = namedtuple("SearchParameters", ["dep", "suffix"])


class EmbeddingSystem:
    SUBJECT_DEP = 'nsubj'
    ROOT_DEP = 'ROOT'
    PUNCTUATION_DEP = 'punct'
    SPACE_TAG = 'SPACE'

    SUBJECT_PARAMS = SearchParameters(dep=SUBJECT_DEP, suffix="_NOUN")
    ROOT_PARAMS = SearchParameters(dep=ROOT_DEP, suffix="_VERB")

    def __init__(self, model_config: Dict):
        self._model = Model(model_config['gensim_model'], model_config['spacy_model'])
        self._minimal_sentence_length = model_config['minimal_sentence_length']
        self._random_seed = model_config['random_seed']
        self._intensity = model_config['intensity']

    def _get_search_parameters(self, sentence: Doc) -> Optional[SearchParameters]:
        has_subject = any(
            filter(lambda token: token.dep_ == self.SUBJECT_DEP and token.tag_ != self.SPACE_TAG, sentence))
        if has_subject:
            return self.SUBJECT_PARAMS

        has_predicate = any(
            filter(lambda token: token.dep_ == self.ROOT_DEP and token.tag_ != self.SPACE_TAG, sentence))
        if has_predicate:
            return self.ROOT_PARAMS

        return None

    def _update_sentence(self, sentence: Doc, bit: str, intensity: float) -> Optional[str]:
        search_parameters = self._get_search_parameters(sentence)
        if not search_parameters:
            return None

        updated_sentence = list(sentence)
        if len(updated_sentence) < self._minimal_sentence_length:
            return None

        bit_value = int(bit)
        for index in range(len(updated_sentence)):
            if updated_sentence[index].dep_ != search_parameters.dep:
                continue

            if len(updated_sentence[index].text) % 2 == bit_value:
                break

            matches = self._model.get_synonyms(word=updated_sentence[index].text,
                                               intensity=intensity)
            replacer_word = next(filter(lambda replacer: len(replacer) % 2 == bit_value, matches),
                                 None)
            if not replacer_word:
                return None

            if updated_sentence[index].text[0].isupper():
                replacer_word = replacer_word.capitalize()

            updated_sentence[index] = replacer_word
            break

        return join_sentence(updated_sentence)

    def embed(self, text: List[str], binary_watermark: str) -> Tuple[List[str], List[str]]:
        watermark_len = len(binary_watermark)

        assert watermark_len <= len(text), 'Sentences count should not be less than binary data size'

        embedded_sentences: List[str] = text.copy()

        inserted_bits = 0
        index_iterator = RandomIndexIterator(self._random_seed, len(text))
        for sentence_index in index_iterator:
            bit = binary_watermark[inserted_bits]
            sentence = embedded_sentences[sentence_index]

            tokenized_sentence = self._model.tokenize_sentence(sentence)
            updated_sentence = self._update_sentence(tokenized_sentence, bit, self._intensity)

            if not updated_sentence:
                continue

            print(f"Embedded {inserted_bits} of {watermark_len}.")
            embedded_sentences[sentence_index] = updated_sentence
            inserted_bits += 1

            if inserted_bits == watermark_len:
                break

        assert inserted_bits == watermark_len, "Not enough bits were inserted"

        return embedded_sentences

    def extract(self, text: List[str], data_size: int) -> str:
        def find_replacer(tokenized_sentence: Doc) -> Optional[str]:
            updated_sentence = list(tokenized_sentence)
            if len(updated_sentence) < self._minimal_sentence_length:
                return None

            search_parameters = self._get_search_parameters(tokenized_sentence)
            if not search_parameters:
                return None

            replacer_word = next(
                filter(lambda token: token.dep_ == search_parameters.dep,
                       tokenized_sentence), None)
            if not replacer_word:
                return None

            return replacer_word.text

        extracted_data = str()
        index_iterator = RandomIndexIterator(self._random_seed, len(text))
        for sentence_index in index_iterator:
            sentence = text[sentence_index]

            tokenized = self._model.tokenize_sentence(sentence)
            replacer = find_replacer(tokenized)
            if not replacer:
                continue

            extracted_data += str(len(replacer) % 2)
            print(f"Extracted {len(extracted_data)} of {data_size} bits")
            if len(extracted_data) == data_size:
                break

        return extracted_data
