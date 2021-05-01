import logging
import gzip

import model_holder
from transform import encrypt, decrypt

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)


def read_text(filename: str, is_gzip=False):
    if is_gzip:
        with gzip.open(filename, 'r') as file:
            return "".join(file.readlines())
    else:
        with open(filename, 'r') as file:
            return "".join(file.readlines())


def main():
    cwd = '/home/tatar/projects/spt'
    models_holder = model_holder.ModelHolder(wv_location=f'{cwd}/models/model.bin',
                                             is_wv_binary_model=True,
                                             spacy_name="ru_core_news_lg")
    text = read_text(f'{cwd}/texts/text3.txt')
    result, hidden_data = encrypt(data_to_hide="rinaz", hiding_data=text, models_holder=models_holder)
    decrypted, decrypted_binary = decrypt(hiding_data=result, models_holder=models_holder)

    print("Original: " + text)
    print("Encrypted: " + result)
    print("Decrypted: " + decrypted)

    assert hidden_data == decrypted_binary, f'Decrypted binary data differs from original hidden data.\n' \
                                            f'Expected: {hidden_data}\n' \
                                            f'Actual:   {decrypted_binary}'


if __name__ == '__main__':
    main()
