""" Letterbox Utility Modules

This module provides functions for handling images with different aspect ratios during the preprocessing stages.
"""


# region Imported Dependencies
from typing import Tuple

import cv2
import numpy as np

from aib.cv.img import Image2D
from aib.cv.shape import Size


# endregion Imported Dependencies


def letterbox(
    a_image: Image2D,
    a_new_shape: Size,
    a_color: Tuple[int, int, int] = (114, 114, 114),
    a_auto: bool = False,
    a_scale_fill: bool = False,
    a_scaleup: bool = False,
    a_stride: int = 32,
) -> Tuple[Image2D, Size, Size]:
    """
    Resize image and padding for detection. Takes image as input, resizes image to fit into new shape with saving
    original aspect ratio and pads it to meet stride-multiple constraints.

    Args:
        a_image (Image2D): Image for preprocessing.
        a_new_shape (Size): Image size after preprocessing in format.
        a_color (Tuple[int, int, int]): Color for filling padded area.
        a_auto (bool): Use dynamic input size, only padding for stride constraints applied.
        a_scale_fill (bool): Scale image to fill new_shape.
        a_scaleup (bool): Allow scale image if it is lower than the desired input size; can affect model accuracy.
        a_stride (int): Input padding stride.

    Returns:
        Tuple[Image2D, Size, Size]:
            Tuple containing the processed image, height and width scaling ratio, and padding size.
    """

    # Resize and pad image while meeting stride-multiple constraints
    old_shape = a_image.size  # current shape [height, width]

    # Scale Ratio (new / old)
    r = min(a_new_shape.height / old_shape.height, a_new_shape.width / old_shape.width)
    if not a_scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute Padding Size
    ratio = Size(a_width=r, a_height=r)  # width, height ratios
    new_unpad = Size(
        a_width=int(round(old_shape.width * r)),
        a_height=int(round(old_shape.height * r)),
    )
    dw, dh = (
        a_new_shape.width - new_unpad.width,
        a_new_shape.height - new_unpad.height,
    )  # wh padding
    if a_auto:  # minimum rectangle
        dw, dh = np.mod(dw, a_stride), np.mod(dh, a_stride)  # wh padding
    elif a_scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = Size(a_width=a_new_shape.width, a_height=a_new_shape.height)
        ratio = Size(
            a_width=a_new_shape.width / old_shape.width,
            a_height=a_new_shape.height / old_shape.height,
        )  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # Resize
    if old_shape != new_unpad:
        new_image = cv2.resize(
            a_image.data, new_unpad.to_tuple(), interpolation=cv2.INTER_LINEAR
        )
    else:
        new_image = a_image.data

    # Add Border
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    new_image = cv2.copyMakeBorder(
        new_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=a_color
    )

    new_image = Image2D(
        a_data=new_image, a_filename=a_image.filename, a_name=a_image.name
    )
    return new_image, ratio, Size(a_width=dw, a_height=dh)
