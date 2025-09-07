"""
Machine Learning - Object Detection - Preprocessing Utilities

This module contains utility functions for preprocessing images for object detection,
instance segmentation, and pose estimation tasks. The primary functionality includes
resizing and padding images to a target size while preserving their aspect ratio (letterboxing),
which is essential for preparing images for model inference.

Functions:
    letterbox: Resize and pad an image to a target size while maintaining aspect ratio.
"""

import cv2
import numpy as np
import numpy.typing as npt

from src.utils.cv.geom.size import IntSize


def letterbox(
    a_image: npt.NDArray[np.uint8],
    a_target_size: IntSize,
    a_pad_to_stride: bool = False,
    a_scale_fill: bool = False,
    a_scaleup: bool = True,
    a_center_pad: bool = True,
    a_stride: int = 32,
    a_pad_value: int = 114,
    a_interpolation: int = cv2.INTER_LINEAR,
) -> npt.NDArray[np.uint8]:
    """
    Resize and pad an image while preserving its aspect ratio (letterboxing).

    This function resizes the input image to fit within a specified target size while maintaining
    the original aspect ratio. If necessary, padding is added to reach the exact target dimensions.
    It supports optional scaling up or down, centering of the image within the padded area,
    stride-based padding adjustment, and stretching to completely fill the target size.

    Args:
        a_image (np.ndarray): Input image array of shape (H, W, C) or (H, W) in uint8 format.
        a_target_size (IntSize): Target size (width, height) to resize and pad the image to.
        a_pad_to_stride (bool, optional):
            If True, pad width and height to be multiples of `a_stride`. Defaults to False.
        a_scale_fill (bool, optional):
            If True, stretch image to completely fill `a_target_size` (ignores aspect ratio). Defaults to False.
        a_scaleup (bool, optional): If True, allow scaling up the image if smaller than target size. Defaults to True.
        a_center_pad (bool, optional):
            If True, padding is divided equally on both sides; otherwise, padding is added to bottom/right only.
            Defaults to True.
        a_stride (int, optional): Stride used when `a_pad_to_stride` is True to adjust padding. Defaults to 32.
        a_pad_value (int, optional): Value used for padding pixels. Defaults to 114.
        a_interpolation (int, optional):
            Interpolation method for resizing (e.g., `cv2.INTER_LINEAR`). Defaults to `cv2.INTER_LINEAR`.

    Returns:
        np.ndarray: Resized and padded image as a uint8 array of shape (target_height, target_width, C).

    Raises:
        RuntimeError: If the resizing or padding operation fails.

    Notes:
        - If `a_scale_fill` is True, aspect ratio may not be preserved.
        - Supports both grayscale (single-channel) and color (3-channel) images.
        - Padding value is applied to all channels.
        - The function does not modify labels or bounding boxes; it only operates on the image.
    """
    try:
        image = a_image
        current_size = IntSize(image.shape[1], image.shape[0])

        r = min(a_target_size.height / current_size.height, a_target_size.width / current_size.width)
        if not a_scaleup:
            r = min(r, 1.0)

        new_unpad = IntSize(int(round(current_size.width * r)), int(round(current_size.height * r)))

        dw, dh = a_target_size.width - new_unpad.width, a_target_size.height - new_unpad.height

        if a_pad_to_stride:
            dw, dh = np.mod(dw, a_stride), np.mod(dh, a_stride)
        elif a_scale_fill:
            dw, dh = 0.0, 0.0
            new_unpad = IntSize(a_target_size.width, a_target_size.height)

        if a_center_pad:
            dw /= 2
            dh /= 2

        if current_size != new_unpad:
            image = cv2.resize(a_image, new_unpad.to_tuple(), interpolation=a_interpolation)
            if image.ndim == 2:
                image = image[..., None]

        top, bottom = int(round(dh - 0.1)) if a_center_pad else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if a_center_pad else 0, int(round(dw + 0.1))

        h, w, c = image.shape
        if c == 3:
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(a_pad_value,) * 3)
        else:
            pad_image = np.full((h + top + bottom, w + left + right, c), fill_value=a_pad_value, dtype=image.dtype)
            pad_image[top : top + h, left : left + w] = image
            image = pad_image

        image = np.asarray(image, dtype=np.uint8)
        return image
    except Exception as e:
        raise RuntimeError(f"Letterbox preprocessing failed. Original error: {e}") from e
