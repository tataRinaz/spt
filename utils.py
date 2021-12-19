import gzip
import re
from typing import List


def read_text(filename: str, is_gzip=False) -> List[str]:
    def read_file() -> str:
        if is_gzip:
            with gzip.open(filename, 'r') as file:
                return "".join(file.readlines())
        else:
            with open(filename, 'r', encoding='utf-8') as file:
                return "".join(file.readlines())

    def split_to_sentences(text: str) -> List[str]:
        def remove_spaces(sentence: str):
            sentence = sentence.replace('\n', '')
            return sentence if not sentence.startswith(' ') else sentence[1:]

        return list(map(lambda sentence: remove_spaces(sentence), re.split(r'. |\? |!', text)))

    def discard_empty_lines(texts: List[str]) -> List[str]:
        return list(filter(lambda s: len(s) != 0, texts))

    return discard_empty_lines(split_to_sentences(read_file()))


STOP_SIGN: str = '00000000'
BITS_IN_BYTE: int = 8


def to_binary_str(data: str) -> str:
    def unify(byte_data):
        binary = f'{byte_data: 08b}'.replace(' ', '')
        number_size = len(binary)
        if number_size < BITS_IN_BYTE:
            binary = "0" * (BITS_IN_BYTE - number_size) + binary

        return binary

    return "".join(map(unify, bytearray(data, 'utf8'))) + STOP_SIGN


def from_binary_str(binary_data: str) -> str:
    return "".join(map(chr,
                       map(lambda binary_str: int(binary_str, 2),
                           map(lambda byte_index: binary_data[byte_index: byte_index + BITS_IN_BYTE],
                               range(0, len(binary_data), BITS_IN_BYTE)
                               )
                           )
                       )
                   )
