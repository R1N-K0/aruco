"""images/ の画像を順に読み、bbox を出力して out/ に可視化を書く。"""

from pathlib import Path

import cv2
import numpy as np

from visualize import draw
from white_board_roi import MarkerNotFound, find_board, to_gray

IMAGES = Path('images')
OUT = Path('out')
EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.npy'}


def load(path):
    a = np.load(path) if path.suffix == '.npy' else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    return to_gray(a)


def main(images=IMAGES):
    OUT.mkdir(exist_ok=True)
    for path in sorted(p for p in images.iterdir() if p.suffix.lower() in EXTS):
        gray = load(path)
        try:
            box = find_board(gray)
        except MarkerNotFound as e:
            print(f'{path.name}: {e}')
            continue
        print(f'{path.name}: {box}')
        cv2.imwrite(str(OUT / f'{path.stem}.png'), draw(gray, box))


if __name__ == '__main__':
    import sys

    main(Path(sys.argv[1]) if len(sys.argv) > 1 else IMAGES)
