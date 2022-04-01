import logging
import traceback
from typing import List, Callable

from embedding_system import EmbeddingSystem
from utils import read_text, to_binary_str, read_config, from_binary_str

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.DEBUG)


def calculate_bit_error_rate(original: str, transformed: str) -> float:
    return sum(x != y for x, y in zip(original, transformed)) / len(original)


def print_if_not(original_text: List[str], updated_text: List[str], hidden_data: str, extracted_data: str,
                 predicate: Callable[[str, str], bool]):
    for index, pair in enumerate(zip(original_text, updated_text)):
        if predicate(hidden_data[index], extracted_data[index]):
            continue

        print(f"Extraction miss found at index {index}.\nOrigin:   {pair[0]}\nEmbedded: {pair[1]}")


def main():
    target_model = "russian"
    config = read_config("D:\\Git\\spt\\config\\config.json", target_model)
    embedding_system = EmbeddingSystem(config)
    while True:
        try:
            text = read_text('.\\texts\\ru_gamer.txt', target_model)
            data = input('Data to hide: ')
            data_to_hide = to_binary_str(data)

            text_with_embedding = embedding_system.embed(text, data_to_hide)
            extracted_watermark = embedding_system.extract(text_with_embedding, len(data_to_hide))

            print(f"Bit error rate: {calculate_bit_error_rate(data_to_hide, extracted_watermark)}")
            print(f"Embedded:  {data_to_hide}\nExtracted: {extracted_watermark}")
        except FileNotFoundError as fnfe:
            print(traceback.format_exc())


if __name__ == '__main__':
    main()
