from collections import namedtuple
from typing import List, Optional, Tuple

from spacy.tokens import Doc

from model import EnglishModel, RussianModel
from utils import STOP_SIGN

SearchParameters = namedtuple("SearchParameters", ["dep", "suffix"])


class EmbeddingSystem:
    SUBJECT_DEP = 'nsubj'
    ROOT_DEP = 'ROOT'
    PUNCTUATION_DEP = 'punct'
    SPACE_TAG = 'SPACE'

    SUBJECT_PARAMS = SearchParameters(dep=SUBJECT_DEP, suffix="_NOUN")
    ROOT_PARAMS = SearchParameters(dep=ROOT_DEP, suffix="_VERB")

    def __init__(self, is_english: bool = True):
        self._model = EnglishModel() if is_english else RussianModel()
        self._spacy_pos_to_suffix = {
            'PRON': '_PRON',
            'CCONJ': '_NOUN',
            'NOUN': '_NOUN',
            'VERB': '_VERB',
            'PROPN': '_NOUN',
            'ADJ': '_ADJ',
            'ADV': '_ADV',
            'DET': '_NOUN'
        }

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

    def _update_sentence(self, sentence: Doc, bit: str, intensity: float) -> Tuple[str, bool]:
        search_parameters = self._get_search_parameters(sentence)
        if not search_parameters:
            return str(sentence), False

        updated_sentence = list(sentence)
        bit_value = int(bit)
        bit_inserted = False
        for index in range(len(updated_sentence)):
            if updated_sentence[index].dep_ != search_parameters.dep or \
                    updated_sentence[index].pos_ not in self._spacy_pos_to_suffix:
                continue

            if len(updated_sentence[index].text) % 2 == bit_value:
                bit_inserted = True
                break

            matches = self._model.get_synonyms(word=updated_sentence[index].text,
                                               suffix=self._spacy_pos_to_suffix[updated_sentence[index].pos_],
                                               intensity=intensity)
            replacer_word = next(filter(lambda replacer: len(replacer) % 2 == bit_value, matches),
                                 None)
            if not replacer_word:
                break

            if updated_sentence[index].text[0].isupper():
                replacer_word = replacer_word.capitalize()

            updated_sentence[index] = replacer_word
            bit_inserted = True
            break

        return " ".join(map(str, updated_sentence)), bit_inserted

    def embed(self, text: List[str], binary_watermark: str, intensity: float) -> List[str]:
        assert binary_watermark.endswith(STOP_SIGN), "Watermark must be ended up with the stop sign"
        watermark_len = len(binary_watermark)

        assert watermark_len <= len(text), 'Sentences count should not be less than binary data size'

        embedded_sentences: List[str] = []

        inserted_bits = 0
        last_inserted_index = -1
        for sentence_index, sentence in enumerate(text):
            bit = binary_watermark[inserted_bits]
            tokenized_sentence = self._model.tokenize_sentence(sentence)
            updated_sentence, bit_inserted = self._update_sentence(tokenized_sentence, bit, intensity)

            if bit_inserted:
                embedded_sentences.append(updated_sentence)
                inserted_bits += 1

            if inserted_bits == watermark_len:
                last_inserted_index = sentence_index
                break

        if last_inserted_index == -1:
            raise RuntimeError("Not enough data. Probably some similarities are missed.")

        return embedded_sentences

    def extract(self, text: List[str]) -> str:
        def find_replacer(tokenized_sentence: Doc) -> Optional[str]:
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

        for sentence in text:
            tokenized = self._model.tokenize_sentence(sentence)
            replacer = find_replacer(tokenized)
            if not replacer:
                continue

            extracted_data += str(len(replacer) % 2)
            if extracted_data.endswith(STOP_SIGN):
                break

        return extracted_data
