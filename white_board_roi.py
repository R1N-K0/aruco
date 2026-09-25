"""ArUco マーカー2つから白板の ROI を矩形で返す。"""

import cv2
import numpy as np

__all__ = ['MarkerNotFound', 'to_gray', 'find_board']

IDS = (0, 1)
GAP = 1.6       # マーカー上端からROI下端まで（マーカー1辺の倍数）
HEIGHT = 2.2    # ROIの高さ（同上）
MARGIN = -0.3   # 左右を内側に寄せる量（同上）
SHIFT = 0.10    # ROIを右へずらす量（同上、負で左）

OFFSET = 2.46   # マーカー0→1 の距離（同上）

KEEP = 0.2      # ROIのうち画面中心側に残す幅の割合（端ほどスペクトルが不安定なため）

SMALL = 0.25    # 2段目の検出の縮小率

BOARD_ROI = (0.083, 0.289, 0.873, 0.819)  # 白板の外形に対する ROI
BOARD_FILL = 0.8
BOARD_ASPECT = (1.1, 1.45)
BOARD_AREA = 0.01

DICTIONARY = cv2.aruco.DICT_4X4_50

_dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
_detector = cv2.aruco.ArucoDetector(_dictionary)
_small_params = cv2.aruco.DetectorParameters()
_small_params.adaptiveThreshWinSizeMax = 53
_small_detector = cv2.aruco.ArucoDetector(_dictionary, _small_params)


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


def _markers(detector, gray):
    corners, ids, _ = detector.detectMarkers(gray)
    return {} if ids is None else {int(i): c[0] for i, c in zip(ids.flatten(), corners)}


def _small_markers(gray):
    small = cv2.resize(gray, None, fx=SMALL, fy=SMALL, interpolation=cv2.INTER_AREA)
    win = int(1 / SMALL) + 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    return {i: cv2.cornerSubPix(gray, (c / SMALL).reshape(-1, 1, 2), (win, win), (-1, -1), criteria).reshape(4, 2)
            for i, c in _markers(_small_detector, small).items()}


def _detect(gray):
    found = _markers(_detector, gray)
    if any(i not in found for i in IDS):
        for i, c in _small_markers(gray).items():
            found.setdefault(i, c)
    return found


def _complete(found):
    a, b = IDS
    if a in found and b not in found:
        m = found[a]
        found[b] = m + OFFSET * (m[1] - m[0])
    elif b in found and a not in found:
        m = found[b]
        found[a] = m - OFFSET * (m[1] - m[0])
    return found


def _roi_from_markers(found, gap, height, margin, shift):
    pts = np.vstack([found[i] for i in IDS])
    side = np.mean([np.linalg.norm(found[i][0] - found[i][1]) for i in IDS])

    y1 = pts[:, 1].min() - gap * side
    y0 = y1 - height * side
    x0 = pts[:, 0].min() + (margin + shift) * side
    x1 = pts[:, 0].max() - (margin - shift) * side
    return x0, y0, x1, y1


def _roi_from_binary(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    best = None
    for x, y, w, h, area in stats[1:n]:
        if (area / (w * h) >= BOARD_FILL and BOARD_ASPECT[0] <= w / h <= BOARD_ASPECT[1]
                and area >= BOARD_AREA * gray.size and (best is None or area > best[4])):
            best = (x, y, w, h, area)
    if best is None:
        return None
    x, y, w, h, _ = best
    rx0, ry0, rx1, ry1 = BOARD_ROI
    return x + rx0 * w, y + ry0 * h, x + rx1 * w, y + ry1 * h


def _keep_center_side(x0, x1, w, keep):
    width = (x1 - x0) * keep
    if (x0 + x1) / 2 > w / 2:
        return x0, x0 + width
    return x1 - width, x1


def find_board(gray, gap=GAP, height=HEIGHT, margin=MARGIN, shift=SHIFT, keep=KEEP):
    found = _complete(_detect(gray))
    if all(i in found for i in IDS):
        x0, y0, x1, y1 = _roi_from_markers(found, gap, height, margin, shift)
    else:
        roi = _roi_from_binary(gray)
        if roi is None:
            raise MarkerNotFound(f'marker {list(IDS)} not found and no board by binarization')
        x0, y0, x1, y1 = roi

    h, w = gray.shape
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    x0, x1 = _keep_center_side(x0, x1, w, keep)
    return (int(x0), int(y0), int(x1), int(y1))
