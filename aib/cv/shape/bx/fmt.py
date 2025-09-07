""" Bounding Box Formatting Utilities

This module provides utility functions for converting bounding box coordinates between different formats.
"""

# region Imported Dependencies
import numpy as np

# endregion Imported Dependencies


def xywh_to_xyxy(a_x, a_y, a_w, a_h):
    """
    Convert a bounding box from (x, y, w, h) format to (x1, y1, x2, y2) format.

    Args:
        a_x (float): The x-coordinate of the top-left corner of the bounding box.
        a_y (float): The y-coordinate of the top-left corner of the bounding box.
        a_w (float): The width of the bounding box.
        a_h (float): The height of the bounding box.

    Returns:
    tuple: A tuple representing the bounding box in (x1, y1, x2, y2) format,
           where (x1, y1) are the coordinates of the top-left corner,
           and (x2, y2) are the coordinates of the bottom-right corner.
    """
    x1 = a_x
    y1 = a_y
    x2 = a_x + a_w
    y2 = a_y + a_h
    return x1, y1, x2, y2


def xyxy_to_xywh(a_x1, a_y1, a_x2, a_y2):
    """
    Convert a bounding box from (x1, y1, x2, y2) format to (x, y, w, h) format.

    Args:
        a_x1 (float): The x-coordinate of the top-left corner of the bounding box.
        a_y1 (float): The y-coordinate of the top-left corner of the bounding box.
        a_x2 (float): The x-coordinate of the bottom-right corner of the bounding box.
        a_y2 (float): The y-coordinate of the bottom-right corner of the bounding box.

    Returns:
    tuple: A tuple representing the bounding box in (x, y, w, h) format,
           where (x, y) are the coordinates of the top-left corner,
           and (w, h) are the width and height of the bounding box.
    """
    x = a_x1
    y = a_y1
    w = a_x2 - a_x1
    h = a_y2 - a_y1
    return x, y, w, h


def xyxy_to_cxywh(a_x1, a_y1, a_x2, a_y2):
    """
    Convert bounding box from xyxy format to (center-x, center-y, width, height).

    Args:
        a_x1 (float): The x-coordinate of the top-left corner of the bounding box.
        a_y1 (float): The y-coordinate of the top-left corner of the bounding box.
        a_x2 (float): The x-coordinate of the bottom-right corner of the bounding box.
        a_y2 (float): The y-coordinate of the bottom-right corner of the bounding box.

    Returns:
    - Tuple (center_x, center_y, width, height).
    """
    width = a_x2 - a_x1
    height = a_y2 - a_y1
    center_x = a_x1 + 0.5 * width
    center_y = a_y1 + 0.5 * height

    return center_x, center_y, width, height


def cxywh_to_xyxy(a_cx, a_cy, a_w, a_h):
    """
    Convert bounding box from (center-x, center-y, width, height) format to (x1, y1, x2, y2) format.

    Args:
        a_cx (float): The x-coordinate of the center point of the bounding box.
        a_cy (float): The y-coordinate of the center point of the bounding box.
        a_w (float): The width of the bounding box.
        a_h (float): The height of the bounding box.

    Returns:
    - Tuple (x1, y1, x2, y2).
    """
    x1 = a_cx - 0.5 * a_w
    y1 = a_cy - 0.5 * a_h
    x2 = a_cx + 0.5 * a_w
    y2 = a_cy + 0.5 * a_h

    return x1, y1, x2, y2


def cxyar_to_xyxy(cx, cy, area, aspect_ratio):
    """
    Calculates the bounding box coordinates in xyxy format from center coordinates, area, and aspect ratio.

    Args:
        cx (float): X-coordinate of the center.
        cy (float): Y-coordinate of the center.
        area (float): Area of the bounding box.
        aspect_ratio (float): Width-to-height ratio of the bounding box.

    Returns:
        list: A tuple of [x1, y1, x2, y2] representing the bounding box in xyxy format.
            x1 (float): X-coordinate of the top-left corner.
            y1 (float): Y-coordinate of the top-left corner.
            x2 (float): X-coordinate of the bottom-right corner.
            y2 (float): Y-coordinate of the bottom-right corner.
    """
    # Calculate width and height from area and aspect ratio
    width = np.sqrt(area * aspect_ratio)
    height = area / width

    # Calculate coordinates of the bounding box
    x1 = cx - (width / 2.0)
    y1 = cy - (height / 2.0)
    x2 = cx + (width / 2.0)
    y2 = cy + (height / 2.0)

    return x1, y1, x2, y2
