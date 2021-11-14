import spacy.tokens
from typing import Tuple, List, Optional
from collections import namedtuple

from model_holder import ModelHolder

STOP_SIGN = '00000000'
SUBJECT_DEP = 'nsubj'
ROOT_DEP = 'ROOT'
PUNCTUATION_DEP = 'punct'
SPACE_TAG = 'SPACE'
BITS_IN_BYTE = 8

SearchParameters = namedtuple("SearchParameters", ["dep", "suffix"])
subject_params = SearchParameters(dep=SUBJECT_DEP, suffix="_NOUN")
root_params = SearchParameters(dep=ROOT_DEP, suffix="_VERB")

pos_suffix_map = {
    'PRON': '_PRON',
    'CCONJ': '_NOUN',
    'NOUN': '_NOUN',
    'VERB': '_VERB',
    'PROPN': '_NOUN',
    'ADJ': '_ADJ',
    'ADV': '_ADV',
    'DET': '_NOUN'
}


def get_search_parameters(has_subject: bool) -> SearchParameters:
    return subject_params if has_subject else root_params


def _make_binary_data(data: str) -> str:
    def unify(byted):
        binary = f'{byted: 08b}'.replace(' ', '')
        number_size = len(binary)
        if number_size < BITS_IN_BYTE:
            binary = "0" * (BITS_IN_BYTE - number_size) + binary

        return binary

    return "".join(map(unify, bytearray(data, 'utf8'))) + STOP_SIGN


def _update_sentence(bit: str, tokenized: spacy.tokens.Doc, models_holder: ModelHolder,
                     intensity: float) -> Tuple[str, bool]:
    assert bit is None or len(bit) == 1, 'Bit size is incorrect, it must be 1'

    has_subject = any(filter(lambda token: token.dep_ == SUBJECT_DEP, tokenized))
    has_predicate = any(filter(lambda token: token.dep_ == ROOT_DEP, tokenized))
    if not has_subject and not has_predicate:
        return str(tokenized), False

    search_params = get_search_parameters(has_subject)
    bit_inserted = False

    def process_token(token: spacy.tokens.Token):
        nonlocal bit_inserted
        space = ''
        if token.dep_ != PUNCTUATION_DEP:
            space = ' '
        if token.dep_ != search_params.dep or \
                bit is None or bit_inserted or \
                token.pos_ not in pos_suffix_map:
            output_word = token.text
        else:
            matches = models_holder.get_synonyms(word=token.text, suffix=pos_suffix_map[token.pos_],
                                                 intensity=intensity)
            replacer_word = next(filter(lambda replacer: len(replacer) % 2 == int(bit), matches),
                                 None)

            if replacer_word is None:
                output_word = token.text
            else:
                if token.text[0].isupper():
                    replacer_word = replacer_word.capitalize()

                output_word = replacer_word
                bit_inserted = True

        return space + output_word

    output_sentences = list(map(process_token,
                                filter(lambda token: token.tag_ != SPACE_TAG, tokenized)
                                )
                            )

    return "".join(output_sentences), bit_inserted


def embed(data_to_hide: str, sentences: List[str], models_holder: ModelHolder,
            intensity: float) -> Tuple[List[str], List[str], str]:
    binary_data = _make_binary_data(data_to_hide)

    binary_data_len = len(binary_data)
    sentences_count = len(sentences)

    assert sentences_count >= binary_data_len, 'Sentences count should not be less than binary data size'

    output_sentences: List[str] = []
    hiders: List[str] = []
    inserted_bits = 0
    sentences_index = 0
    while sentences_index < len(sentences):
        bit = binary_data[inserted_bits] if inserted_bits < binary_data_len else None
        tokenized = models_holder.tokenize_sentence(sentences[sentences_index])
        updated_sentence, is_inserted = _update_sentence(bit, tokenized, models_holder, intensity)
        output_sentences.append(updated_sentence)

        if is_inserted:
            inserted_bits += 1
            hiders.append(updated_sentence)

        if inserted_bits == binary_data_len:
            break

        sentences_index += 1

    assert inserted_bits == binary_data_len, f"Not enough data. Probably some similarities are missed. " \
                                             f"Try to extend text data.\n" \
                                             f"Bits inserted:      {inserted_bits}\n" \
                                             f"Binary data length: {binary_data_len}"

    return output_sentences, hiders, binary_data[:-8]


def _convert_binary_to_string(binary_string: str) -> str:
    return "".join(map(chr,
                       map(lambda binary_str: int(binary_str, 2),
                           map(lambda byte_index: binary_string[byte_index: byte_index + BITS_IN_BYTE],
                               range(0, len(binary_string), BITS_IN_BYTE)
                               )
                           )
                       )
                   )


def extract(sentences: List[str], models_holder: ModelHolder) -> Tuple[str, str]:
    def find_replaced(doc: spacy.tokens.Doc) -> Optional[str]:
        has_subject = any(filter(lambda token: token.dep_ == SUBJECT_DEP and token.tag_ != SPACE_TAG, tokenized))
        has_predicate = any(filter(lambda token: token.dep_ == ROOT_DEP and token.tag_ != SPACE_TAG, tokenized))
        if not has_subject and not has_predicate:
            return None

        search_params = get_search_parameters(has_subject)

        replacer = next((token for token in doc if token.dep_ == search_params.dep and token.tag_ != SPACE_TAG),
                        None)
        if replacer is None:
            return None

        return replacer.text

    encrypted_data = str()

    for index, sentence in enumerate(sentences):
        tokenized = models_holder.tokenize_sentence(sentence)
        replacer = find_replaced(tokenized)

        if replacer is None:
            continue

        encrypted_data += str(len(replacer) % 2)

        if len(encrypted_data) % BITS_IN_BYTE == 0 and encrypted_data.endswith(STOP_SIGN):
            break

    # assert len(encrypted_data) % BITS_IN_BYTE == 0, f'Data is incorrect, not enough bits {len(encrypted_data)}'
    encrypted_data = encrypted_data[:-BITS_IN_BYTE]

    return _convert_binary_to_string(encrypted_data), encrypted_data
