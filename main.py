import logging
import traceback
import gzip
from typing import List

import model_holder
from transform import embed, extract

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.DEBUG)


def read_text(filename: str, is_gzip=False):
    if is_gzip:
        with gzip.open(filename, 'r') as file:
            return "".join(file.readlines())
    else:
        with open(filename, 'r', encoding='utf-8') as file:
            return "".join(file.readlines())


def _split_to_sentences(text: str) -> List[str]:
    def remove_spaces(sentence: str):
        sentence = sentence.replace('\n', '')
        return sentence if not sentence.startswith(' ') else sentence[1:]

    return list(map(lambda sentence: remove_spaces(sentence), text.split('.')))


def calculate_bit_error_rate(original: str, transformed: str) -> float:
    return sum(x != y for x, y in zip(original, transformed)) / len(original)


def main():
    cwd = 'C:\\gitproj\\spt'
    path = f'{cwd}\\models\\en_model.bin'
    ru_spacy = "ru_core_news_lg"
    en_spacy = "en_core_web_trf"
    models_holder = model_holder.ModelHolder(wv_location=path,
                                             is_wv_binary_model=path.endswith('bin'),
                                             spacy_name=en_spacy)

    while True:
        try:
            filename = input('Text filename: ')
            text = _split_to_sentences(read_text(f'{cwd}\\texts\\{filename}'))
            text = list(filter(lambda s: len(s) != 0, text))
            intensity_threshold = float(input('Type intensity: '))
            data = input('Data to hide: ')
            result, hiders, hidden_data = embed(data_to_hide=data,
                                                sentences=text,
                                                models_holder=models_holder,
                                                intensity=intensity_threshold)
            decrypted, decrypted_binary = extract(sentences=hiders, models_holder=models_holder)

            print(f"Bit error rate: {calculate_bit_error_rate(hidden_data, decrypted_binary)}")
            if hidden_data != decrypted_binary:
                print(f'Decrypted binary data differs from original hidden data.\n'
                      f'Expected: {hidden_data}\n'
                      f'Actual:   {decrypted_binary}')
        except FileNotFoundError as fnfe:
            print(traceback.format_exc())


if __name__ == '__main__':
    main()
