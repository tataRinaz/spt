import json
import random
from string import punctuation
from typing import List, Dict

from nltk import tokenize, download


# download('punkt')


def read_text(filename: str, language: str) -> List[str]:
    def read_file() -> str:
        with open(filename, 'r', encoding='utf-8') as file:
            return "".join(file.readlines())

    def split_to_sentences(text: str) -> List[str]:
        return tokenize.sent_tokenize(text, language=language)

    def discard_empty_lines(texts: List[str]) -> List[str]:
        return list(filter(lambda s: len(s) != 0, texts))

    return discard_empty_lines(split_to_sentences(read_file()))


BITS_IN_BYTE: int = 8


def to_binary_str(data: str) -> str:
    def unify(byte_data):
        binary = f'{byte_data: 08b}'.replace(' ', '')
        number_size = len(binary)
        if number_size < BITS_IN_BYTE:
            binary = "0" * (BITS_IN_BYTE - number_size) + binary

        return binary

    return "".join(map(unify, bytearray(data, 'utf8')))


def from_binary_str(binary_data: str) -> str:
    return "".join(map(chr,
                       map(lambda binary_str: int(binary_str, 2),
                           map(lambda byte_index: binary_data[byte_index: byte_index + BITS_IN_BYTE],
                               range(0, len(binary_data), BITS_IN_BYTE)
                               )
                           )
                       )
                   )


def read_config(path: str, target_model: str) -> Dict:
    with open(path, 'r') as config_file:
        config = json.load(config_file)
        model_config = next(filter(lambda model: model["name"] == "russian", config["models"]), None)
        if not model_config:
            print(f"Target config is not found: {target_model}. Config: {config}")
            raise RuntimeError(f"Target config is not found: {target_model}. Config: {config}")

        return model_config


def join_sentence(sentence: List) -> str:
    output = str()
    for index, _ in enumerate(sentence):
        word = str(sentence[index])
        if word not in punctuation and index != 0:
            output += " "
        output += word

    return output


class RandomIndexIterator:
    def __init__(self, seed: int, container_size: int):
        self._index = 0
        random.seed(seed)
        self._values = [i for i in range(container_size)]
        random.shuffle(self._values)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == len(self._values):
            raise StopIteration

        index = self._index
        self._index += 1
        return index
