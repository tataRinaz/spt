import logging
import gzip
from typing import List

import model_holder
from transform import encrypt, decrypt

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.DEBUG)


def read_text(filename: str, is_gzip=False):
    if is_gzip:
        with gzip.open(filename, 'r') as file:
            return "".join(file.readlines())
    else:
        with open(filename, 'r') as file:
            return "".join(file.readlines())


def _split_to_sentences(text: str) -> List[str]:
    def remove_spaces(sentence: str):
        sentence = sentence.replace('\n', '')
        return sentence if not sentence.startswith(' ') else sentence[1:]
    return list(map(lambda sentence: remove_spaces(sentence), text.split('.')))


def main():
    cwd = '/home/tatar/projects/spt'
    models_holder = model_holder.ModelHolder(wv_location=f'{cwd}/models/model.bin',
                                             is_wv_binary_model=True,
                                             spacy_name="ru_core_news_lg")
    text = _split_to_sentences(read_text(f'{cwd}/texts/text3.txt'))
    result, hiders, hidden_data = encrypt(data_to_hide="10", sentences=text, models_holder=models_holder)
    decrypted, decrypted_binary = decrypt(sentences=hiders, models_holder=models_holder)

    print("Original: " + "\n".join(text))
    print("Encrypted: " + "\n".join(result))
    print("Hiders: " + "\n".join(hiders))
    print("Decrypted: " + decrypted)

    assert hidden_data == decrypted_binary, f'Decrypted binary data differs from original hidden data.\n' \
                                            f'Expected: {hidden_data}\n' \
                                            f'Actual:   {decrypted_binary}'


if __name__ == '__main__':
    main()
