"""bbox を画像に描く。"""

import cv2

__all__ = ['draw']


def draw(image, box, color=(0, 0, 255), thickness=2):
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    cv2.rectangle(vis, box[:2], box[2:], color, thickness)
    return vis
