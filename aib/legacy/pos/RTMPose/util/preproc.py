"""RTMPose Pre-Processing Utilities
"""

# region Imported Dependencies
from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt

from aib.cv.img import Image2D
from aib.cv.shape import Size
from aib.cv.shape.pt import Point2D


# endregion Imported Dependencies


# TODO(doc): Complete the document of following function
def get_3rd_point(a_p1: npt.NDArray[np.floating], a_p2: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    direction = a_p1 - a_p2
    p3 = a_p2 + np.r_[-direction[1], direction[0]]
    return p3


# TODO(doc): Complete the document of following function
def rotate_point2d(a_pt: npt.NDArray[np.floating], a_angle_rad: float) -> npt.NDArray[np.floating]:
    sn, cs = np.sin(a_angle_rad), np.cos(a_angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ a_pt


# endregion Imported Dependencies
# TODO(doc): Complete the document of following function
def get_affine_transform(
    a_src_size: Size,
    a_src_center: Point2D,
    a_dst_size: Size,
    a_rot: float,
    a_shift: Size = Size(a_width=0.0, a_height=0.0),
    a_inv: bool = False,
) -> npt.NDArray[np.floating]:
    # Compute Source Direction Vector
    rot_rad = np.deg2rad(a_rot)
    src_dir = rotate_point2d(np.array([0.0, a_src_size.width * -0.5]), rot_rad)

    # Calculate Source Rectangle Corners
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = a_src_center.to_numpy() + a_src_size.to_numpy() * a_shift.to_numpy()
    src[1, :] = a_src_center.to_numpy() + src_dir + a_src_size.to_numpy() * a_shift.to_numpy()
    src[2, :] = get_3rd_point(src[0, :], src[1, :])

    # Compute Destination Direction Vector
    dst_dir = np.array([0.0, a_dst_size.width * -0.5])

    # Calculate Destination Rectangle Corners
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [a_dst_size.width * 0.5, a_dst_size.height * 0.5]
    dst[1, :] = np.array([a_dst_size.width * 0.5, a_dst_size.height * 0.5]) + dst_dir
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    # Compute Affine Transformation Matrix
    if a_inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return warp_mat


# TODO(doc): Complete the document of following function
def wrap_affine(
    a_src_size: Size, a_src_center: Point2D, a_dst_size: Size, a_image: Image2D
) -> Tuple[npt.NDArray[np.floating], Size]:
    # Adjust Bounding Box Size to Match Destination Aspect Ratio
    if a_src_size.width > a_src_size.height * a_dst_size.aspect_ratio:
        adj_size = Size(a_width=a_src_size.width, a_height=a_src_size.width / a_dst_size.aspect_ratio)
    else:
        adj_size = Size(a_width=a_src_size.height * a_dst_size.aspect_ratio, a_height=a_src_size.height)

    # Compute Affine Transformation Matrix
    warp_mat = get_affine_transform(a_src_size=adj_size, a_src_center=a_src_center, a_dst_size=a_dst_size, a_rot=0)

    # Apply Affine Transformation
    image = cv2.warpAffine(a_image.data, warp_mat, a_dst_size.to_tuple(), flags=cv2.INTER_LINEAR)

    return image, adj_size
