import logging
import traceback

from embedding_system import EmbeddingSystem
from utils import read_text, to_binary_str

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.DEBUG)


def calculate_bit_error_rate(original: str, transformed: str) -> float:
    return sum(x != y for x, y in zip(original, transformed)) / len(original)


def main():
    embedding_system = EmbeddingSystem(is_english=True)
    while True:
        try:
            text = read_text('.\\texts\\en_cpp.txt')
            intensity_threshold = float(input('Type intensity: '))
            data = input('Data to hide: ')
            data_to_hide = to_binary_str(data)

            text_with_embedding = embedding_system.embed(text, data_to_hide, intensity_threshold)
            extracted_watermark = embedding_system.extract(text_with_embedding)

            print(f"Bit error rate: {calculate_bit_error_rate(data_to_hide, extracted_watermark)}")
        except FileNotFoundError as fnfe:
            print(traceback.format_exc())


if __name__ == '__main__':
    main()
