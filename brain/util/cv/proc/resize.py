"""Resize Utility Modules

This module provides functions for resizing images.

Functions:
    resize_aspect_ratio: Resize an image to a specified aspect ratio.
"""

# region Imported Dependencies
from typing import Union

import cv2
import numpy as np

from brain.util.cv.img import Image2D
from brain.util.cv.shape import Size
from brain.util.cv.vid import Frame2D


# endregion Imported Dependencies


def resize_aspect_ratio(a_image: Union[Image2D, Frame2D], a_aspect_ratio: float) -> Union[Image2D, Frame2D]:
    """Resize an image to a specified aspect ratio.

    This method resizes the input image to match the specified aspect ratio while preserving the original content.
    If the input image's aspect ratio is larger than the target aspect ratio, it resizes the image's width, and if
    it's smaller, it resizes the image's height. If the aspect ratios match, no resizing is performed.

    Args:
        a_image (Union[Image2D, Frame2D]):
            The input 2D image to be resized.
        a_aspect_ratio (float):
            The aspect ratio to resize the image to.

    Returns:
        Union[Image2D, Frame2D]:
            A resized image with the adjusted aspect ratio. If the input image is an instance of `Frame2D`, the
            returned object will also be a `Frame2D`, otherwise, it will be an `Image2D`.

    Raises:
        TypeError:
            If the input image is not an instance of `Image2D`, or if the aspect ratio is not a float.
    """
    if a_image is None or not isinstance(a_image, Image2D):
        raise TypeError(f"`a_image` argument must be an `Image2D` but it's type is `{type(a_image)}`")
    if a_aspect_ratio is None or not isinstance(a_aspect_ratio, float):
        raise TypeError(f"`a_aspect_ratio` argument must be an `float` but it's type is `{type(a_aspect_ratio)}`")

    if a_image.aspect_ratio > a_aspect_ratio:
        new_width = int(a_image.height * a_aspect_ratio)
        img = cv2.resize(a_image.data, (new_width, a_image.height))
    elif a_image.aspect_ratio < a_aspect_ratio:
        new_height = int(a_image.width / a_aspect_ratio)
        img = cv2.resize(a_image.data, (a_image.width, new_height))
    else:
        img = a_image

    if isinstance(img, np.ndarray):
        if isinstance(a_image, Frame2D):
            img = Frame2D(
                a_time=a_image.time,
                a_data=img,
                a_filename=a_image.filename,
                a_id=a_image.id,
                a_video_id=a_image.video_id,
                a_name=a_image.name,
            )
        else:
            img = Image2D(
                a_data=img,
                a_filename=a_image.filename,
                a_name=a_image.name,
            )
    return img


def resize(
    a_image: Union[Image2D, Frame2D],
    a_size: Size,
    a_fx: int = 0,
    a_fy: int = 0,
    a_interpolation: int = cv2.INTER_LINEAR,
) -> Union[Image2D, Frame2D]:
    """Resize Image

    Resize the input image or frame to the specified size.

    Args:
        a_image (Union[Image2D, Frame2D]): The image or frame to be resized.
        a_size (Size): The desired size of the output image or frame.
        a_fx (int, optional): Scaling factor along the horizontal axis. Defaults to 0.
        a_fy (int, optional): Scaling factor along the vertical axis. Defaults to 0.
        a_interpolation (int, optional): Interpolation method used for resizing. Defaults to cv2.INTER_LINEAR.

    Returns:
        Union[Image2D, Frame2D]: The resized image or frame.

    Raises:
        TypeError: If the input arguments are of incorrect types.
    """

    # region Input Checking
    if a_image is None or not isinstance(a_image, Image2D):
        raise TypeError(f"`a_image` argument must be an `Image2D` but it's type is `{type(a_image)}`")
    if a_size is None or not isinstance(a_size, Size):
        raise TypeError(f"`a_size` argument must be an `Size` but it's type is `{type(a_size)}`")

    if not isinstance(a_fx, int):
        raise TypeError(f"`a_fx` argument must be an `int` but it's type is `{type(a_fx)}`")

    if not isinstance(a_fy, int):
        raise TypeError(f"`a_fy` argument must be an `int` but it's type is `{type(a_fy)}`")

    if not isinstance(a_interpolation, int):
        raise TypeError(f"`a_interpolation` argument must be an `int` but it's type is `{type(a_interpolation)}`")
    # endregion Input Checking

    img = cv2.resize(a_image.data, a_size.to_tuple(), a_fx, a_fy, a_interpolation)

    if isinstance(img, np.ndarray):
        if isinstance(a_image, Frame2D):
            img = Frame2D(
                a_time=a_image.time,
                a_data=img,
                a_filename=a_image.filename,
                a_id=a_image.id,
                a_video_id=a_image.video_id,
                a_name=a_image.name,
            )
        else:
            img = Image2D(
                a_data=img,
                a_filename=a_image.filename,
                a_name=a_image.name,
            )
    return img
