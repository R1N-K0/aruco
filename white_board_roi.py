"""ArUco マーカー2つから白板の ROI を矩形で返す。"""

import cv2
import numpy as np

__all__ = ['MarkerNotFound', 'to_gray', 'find_board']

IDS = (0, 1)
GAP = 1.6       # マーカー上端からROI下端まで（マーカー1辺の倍数）
HEIGHT = 2.2    # ROIの高さ（同上）
MARGIN = -0.3   # 左右を内側に寄せる量（同上）
SHIFT = 0.10    # ROIを右へずらす量（同上、負で左）

DICTIONARY = cv2.aruco.DICT_4X4_50

_detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(DICTIONARY))


class MarkerNotFound(Exception):
    pass


def to_gray(a):
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = a.astype(np.float32)
    a -= a.min()
    if a.max() > 0:
        a /= a.max()
    return (a * 255).astype(np.uint8)


def find_board(gray, gap=GAP, height=HEIGHT, margin=MARGIN, shift=SHIFT):
    corners, ids, _ = _detector.detectMarkers(gray)
    found = {} if ids is None else {int(i): c[0] for i, c in zip(ids.flatten(), corners)}

    missing = [i for i in IDS if i not in found]
    if missing:
        raise MarkerNotFound(f'marker {missing} not found (detected: {sorted(found)})')

    pts = np.vstack([found[i] for i in IDS])
    side = np.mean([np.linalg.norm(found[i][0] - found[i][1]) for i in IDS])

    y1 = pts[:, 1].min() - gap * side
    y0 = y1 - height * side
    x0 = pts[:, 0].min() + (margin + shift) * side
    x1 = pts[:, 0].max() - (margin - shift) * side

    h, w = gray.shape
    return (int(max(x0, 0)), int(max(y0, 0)), int(min(x1, w)), int(min(y1, h)))
