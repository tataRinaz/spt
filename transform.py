import spacy.tokens
from typing import Tuple

from model_holder import ModelHolder


def _split_to_sentences(text: str) -> []:
    return text.split('.')


STOP_SIGN = '00000000'
SUBJECT_DEP = 'nsubj'
ROOT_DEP = 'ROOT'


def _make_binary_data(data: str) -> str:
    def unify(byted):
        binary = f'{byted: 08b}'.replace(' ', '')
        number_size = len(binary)
        if number_size < 8:
            binary = "0" * (8 - number_size) + binary

        return binary

    return "".join(map(unify, bytearray(data, 'utf8'))) + STOP_SIGN


def _update_sentence(bit: str, tokenized: spacy.tokens.Doc, models_holder: ModelHolder) -> str:
    assert len(bit) == 1, 'Bit size is incorrect, it must be 1'

    output_sentence = []
    has_subject = any(filter(lambda token: token.dep_ == SUBJECT_DEP, tokenized))

    search_part = SUBJECT_DEP if has_subject else ROOT_DEP

    for index, token in enumerate(tokenized):
        if token.tag_ == 'SPACE':
            continue
        if index != 0 and token.dep_ != 'punct':
            output_sentence.append(' ')
        if token.dep_ != search_part or \
                len(token.text) % 2 == int(bit):
            output_sentence.append(token.text)
        else:
            matches = models_holder.get_synonyms(word=token.text)
            target_similarities = list(filter(lambda word: len(word) % 2 == int(bit), matches))
            replacer_word: str = target_similarities[0] if len(target_similarities) > 0 else token.text + token.text[-1]
            if token.text[0].isupper():
                replacer_word = replacer_word.capitalize()
            output_sentence.append(replacer_word)

    return "".join(output_sentence)


def encrypt(data_to_hide: str, hiding_data: str, models_holder: ModelHolder) -> Tuple[str, str]:
    binary_data = _make_binary_data(data_to_hide)
    sentences = _split_to_sentences(hiding_data)

    binary_data_len = len(binary_data)
    sentences_count = len(sentences)

    assert sentences_count >= binary_data_len, 'Sentences count should not be less than binary data size'

    output_sentences = []
    index = 0
    for bit, sentence in zip(binary_data, sentences):
        tokenized = models_holder.tokenize_sentence(sentence)
        output_sentences.append(_update_sentence(bit, tokenized, models_holder))
        index += 1

    output_sentences += sentences[binary_data_len:]

    return ".".join(output_sentences), binary_data[:-8]


def _convert_binary_to_string(binary_string: str) -> str:
    def binary_to_decimal(binary: str) -> int:
        return int(binary, 2)

    output = str()
    for i in range(0, len(binary_string), 8):
        decimal_data = binary_to_decimal(binary_string[i:i + 8])
        output += chr(decimal_data)

    return output


def decrypt(hiding_data: str, models_holder: ModelHolder) -> Tuple[str, str]:
    def find_replaced(doc: spacy.tokens.Doc) -> str:
        has_subject = any(filter(lambda token: token.dep_ == SUBJECT_DEP, tokenized))

        search_part = SUBJECT_DEP if has_subject else ROOT_DEP
        for token in doc:
            if token.dep_ == search_part and token.tag_ != 'SPACE':
                return token.text

        return str()

    encrypted_data = str()
    sentences = _split_to_sentences(hiding_data)

    for index, sentence in enumerate(sentences):
        tokenized = models_holder.tokenize_sentence(sentence)
        encrypted_data += str(len(find_replaced(tokenized)) % 2)

        if len(encrypted_data) % 8 == 0 and encrypted_data.endswith(STOP_SIGN):
            break

    assert len(encrypted_data) % 8 == 0, f'Data is incorrect, not enough bits {len(encrypted_data)}'
    encrypted_data = encrypted_data[:-8]

    return _convert_binary_to_string(encrypted_data), encrypted_data
