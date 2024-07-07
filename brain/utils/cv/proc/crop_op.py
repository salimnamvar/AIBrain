""" Cropping Utility Modules

This module provides functions for cropping images.

Functions:
    crop:
        Crops a region from the input image based on the specified 2D bounding box.
"""


# region Imported Dependencies
from brain.utils.cv.img import Image2D
from brain.utils.cv.shape.bx import Box2D

# endregion Imported Dependencies


def crop(
    a_image: Image2D,
    a_box: Box2D,
) -> Image2D:
    """Crop function

    This function takes an input 2D image and a 2D bounding box, and returns a new Image2D
    object containing the cropped region of the image.

    Parameters:
        a_image (Image2D): The input 2D image to be cropped.
        a_box (Box2D): The 2D bounding box defining the region to be cropped.

    Returns:
        Image2D: A new Image2D object containing the cropped region.

    Raises:
        TypeError: If input arguments are not of the correct types.

    """
    if a_image is None or not isinstance(a_image, Image2D):
        raise TypeError(
            f"`a_image` argument must be an `Image2D` but it's type is `{type(a_image)}`"
        )
    if a_box is None or not isinstance(a_box, Box2D):
        raise TypeError(
            f"`a_box` argument must be an `Box2D` but it's type is `{type(a_box)}`"
        )

    return Image2D(
        a_image.data[
            int(a_box.p1.y) : int(a_box.p2.y), int(a_box.p1.x) : int(a_box.p2.x)
        ]
    )
