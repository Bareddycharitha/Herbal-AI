"""
Run inference on a single herb image
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ai.herb.inference import herb_predictor


def main():

    image_path = input("Enter Image Path: ").strip()

    result = herb_predictor.predict(image_path)

    print()

    print(result)


if __name__ == "__main__":
    main()