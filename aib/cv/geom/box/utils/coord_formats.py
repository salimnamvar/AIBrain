"""Computer Vision - Geometry - Box - Coordinate Format Utilities"""

from typing import Tuple

import numpy as np


def xywh_to_xyxy(
    a_x: float | int, a_y: float | int, a_w: float | int, a_h: float | int
) -> Tuple[float | int, float | int, float | int, float | int]:
    """Convert bounding box from (x, y, width, height) to (x1, y1, x2, y2) format.

    Args:
        a_x (float | int): x-coordinate of the top-left corner.
        a_y (float | int): y-coordinate of the top-left corner.
        a_w (float | int): width of the bounding box.
        a_h (float | int): height of the bounding box.

    Returns:
        Tuple[float | int, float | int, float | int, float | int]:
            (x1, y1, x2, y2) coordinates of the bounding box.
    """
    x1 = a_x
    y1 = a_y
    x2 = a_x + a_w
    y2 = a_y + a_h
    return x1, y1, x2, y2


def xyxy_to_xywh(
    a_x1: float | int, a_y1: float | int, a_x2: float | int, a_y2: float | int
) -> Tuple[float | int, float | int, float | int, float | int]:
    """Convert bounding box from (x1, y1, x2, y2) to (x, y, width, height) format.

    Args:
        a_x1 (float | int): x-coordinate of the top-left corner.
        a_y1 (float | int): y-coordinate of the top-left corner.
        a_x2 (float | int): x-coordinate of the bottom-right corner.
        a_y2 (float | int): y-coordinate of the bottom-right corner.

    Returns:
        Tuple[float | int, float | int, float | int, float | int]:
            (x, y, width, height) of the bounding box.
    """
    x = a_x1
    y = a_y1
    w = a_x2 - a_x1
    h = a_y2 - a_y1
    return x, y, w, h


def xyxy_to_cxywh(
    a_x1: float | int, a_y1: float | int, a_x2: float | int, a_y2: float | int
) -> Tuple[float | int, float | int, float | int, float | int]:
    """Convert bounding box from (x1, y1, x2, y2) to (center_x, center_y, width, height) format.

    Args:
        a_x1 (float | int): x-coordinate of the top-left corner.
        a_y1 (float | int): y-coordinate of the top-left corner.
        a_x2 (float | int): x-coordinate of the bottom-right corner.
        a_y2 (float | int): y-coordinate of the bottom-right corner.

    Returns:
        Tuple[float | int, float | int, float | int, float | int]:
            (center_x, center_y, width, height) of the bounding box.
    """
    width = a_x2 - a_x1
    height = a_y2 - a_y1
    center_x = a_x1 + 0.5 * width
    center_y = a_y1 + 0.5 * height
    return center_x, center_y, width, height


def cxywh_to_xyxy(
    a_cx: float | int, a_cy: float | int, a_w: float | int, a_h: float | int
) -> Tuple[float | int, float | int, float | int, float | int]:
    """Convert bounding box from (center_x, center_y, width, height) to (x1, y1, x2, y2) format.

    Args:
        a_cx (float | int): x-coordinate of the center.
        a_cy (float | int): y-coordinate of the center.
        a_w (float | int): width of the bounding box.
        a_h (float | int): height of the bounding box.

    Returns:
        Tuple[float | int, float | int, float | int, float | int]:
            (x1, y1, x2, y2) coordinates of the bounding box.
    """
    x1 = a_cx - 0.5 * a_w
    y1 = a_cy - 0.5 * a_h
    x2 = a_cx + 0.5 * a_w
    y2 = a_cy + 0.5 * a_h
    return x1, y1, x2, y2


def cxyar_to_xyxy(
    a_cx: float | int, a_cy: float | int, a_area: float | int, a_aspect_ratio: float | int
) -> Tuple[float | int, float | int, float | int, float | int]:
    """Convert bounding box from (center_x, center_y, area, aspect_ratio) to (x1, y1, x2, y2) format.

    Args:
        a_cx (float | int): x-coordinate of the center.
        a_cy (float | int): y-coordinate of the center.
        a_area (float | int): area of the bounding box.
        a_aspect_ratio (float | int): aspect ratio of the bounding box (width / height).

    Returns:
        Tuple[float | int, float | int, float | int, float | int]:
            (x1, y1, x2, y2) coordinates of the bounding box.
    """
    width = np.sqrt(a_area * a_aspect_ratio)
    height = a_area / width
    x1 = a_cx - (width / 2.0)
    y1 = a_cy - (height / 2.0)
    x2 = a_cx + (width / 2.0)
    y2 = a_cy + (height / 2.0)
    return x1, y1, x2, y2
