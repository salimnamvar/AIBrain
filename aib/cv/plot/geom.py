"""Computer Vision - Geometry Plotters

This module provides functions to plot various geometric shapes and points on images.

Functions:
    - plot_base_point2d: Base function to plot a Point2D on an image with various options.
    - plot_base_box2d: Base function to plot a Box2D on an image with various options.
    - plot_point2d: Plot a Point2D on an image.
    - plot_keypoint2d: Plot a KeyPoint2D on an image.
    - plot_line2d: Plot a Line2D on an image.
    - plot_pose2d: Plot a Pose2D (skeleton) on an image.
    - plot_contour2d: Plot a Contour2D on an image.
    - plot_bbox2d: Plot a bounding box defined by two points on an image.
    - plot_box2d: Plot a Box2D on an image.
    - plot_bbox2d: Plot a BBox2D on an image.
    - plot_sbbox2d: Plot a SegBBox2D on an image.
"""

from typing import Optional, Tuple, TypeVar, Union, cast

import cv2
import numpy as np
import numpy.typing as npt

from aib.cv.geom.box.bbox2d import BBox2D
from aib.cv.geom.box.box2d import Box2D
from aib.cv.geom.box.sbbox2d import SegBBox2D
from aib.cv.geom.contour2d import Contour2D
from aib.cv.geom.line2d import Line2D
from aib.cv.geom.point.kpoint2d import KeyPoint2D
from aib.cv.geom.point.point2d import Point2D
from aib.cv.geom.pose.pose2d import Pose2D
from aib.cv.img.frame import Frame2D
from aib.cv.img.image import Image2D

T = TypeVar("T", bound=Union[int, float])
PT = TypeVar("PT", bound=Union[Point2D[int], Point2D[float]])
KPT = TypeVar("KPT", bound=Union[KeyPoint2D[int], KeyPoint2D[float]])
PKT = TypeVar("PKT", bound=Union[Point2D[int], Point2D[float], KeyPoint2D[int], KeyPoint2D[float]])


def plot_base_point2d(
    a_image: Image2D | Frame2D | npt.NDArray[np.uint8],
    a_x: int,
    a_y: int,
    a_color: Tuple[int, int, int] = (0, 0, 255),
    a_radius: int = 4,
    a_thickness: int = -1,
    a_alpha: float = 1.0,
    a_label: Optional[str] = None,
    a_label_color: Tuple[int, int, int] = (0, 0, 0),
    a_label_font_scale: float = 0.5,
    a_label_font_thickness: int = 1,
    a_label_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_cross: bool = False,
    a_cross_color: Tuple[int, int, int] = (0, 255, 0),
    a_cross_length: int = 8,
    a_cross_thickness: int = 1,
    a_score: Optional[float] = None,
    a_score_color: Tuple[int, int, int] = (255, 0, 0),
    a_score_font_scale: float = 0.4,
    a_score_font_thickness: int = 1,
    a_score_font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> Image2D | Frame2D | npt.NDArray[np.uint8]:
    """Plot a Point2D on an image.

    Args:
        a_image: The image to draw on (Image2D, Frame2D, or np.ndarray).
        a_x: X coordinate of the point.
        a_y: Y coordinate of the point.
        a_color: Point color (B, G, R).
        a_radius: Radius of the point circle.
        a_thickness: Thickness of the point circle.
        a_alpha: Alpha blending factor (0.0 transparent, 1.0 opaque).
        a_label: Optional label text.
        a_label_color: Label text color (B, G, R).
        a_label_font_scale: Font scale for label.
        a_label_font_thickness: Font thickness for label.
        a_label_font: Font type for label.
        a_cross: Whether to draw cross lines at the point.
        a_cross_color: Color for the cross lines (B, G, R).
        a_cross_length: Length of the cross arms.
        a_cross_thickness: Thickness of the cross lines.
        a_score: Optional score value to display.
        a_score_color: Color for the score text (B, G, R).
        a_score_font_scale: Font scale for the score text.
        a_score_font_thickness: Font thickness for the score text.
        a_score_font: Font type for the score text.

    Returns:
        Image2D | Frame2D | np.ndarray: The image with the point drawn.
    """
    if isinstance(a_image, (Image2D, Frame2D)):
        if hasattr(a_image, "data"):
            img: npt.NDArray[np.uint8] = a_image.data.copy()
        else:
            img: npt.NDArray[np.uint8] = np.array(a_image).copy()
    else:
        img: npt.NDArray[np.uint8] = np.array(a_image).copy()

    overlay = img.copy()
    cv2.circle(overlay, (a_x, a_y), a_radius, a_color, a_thickness)

    if a_cross:
        cv2.line(overlay, (a_x - a_cross_length, a_y), (a_x + a_cross_length, a_y), a_cross_color, a_cross_thickness)
        cv2.line(overlay, (a_x, a_y - a_cross_length), (a_x, a_y + a_cross_length), a_cross_color, a_cross_thickness)

    if a_label:
        ((text_w, text_h), _) = cv2.getTextSize(a_label, a_label_font, a_label_font_scale, a_label_font_thickness)
        label_bg = (a_x, a_y - a_radius - text_h - 4, a_x + text_w + 4, a_y - a_radius)
        cv2.rectangle(overlay, (label_bg[0], label_bg[1]), (label_bg[2], label_bg[3]), a_color, -1)
        cv2.putText(
            overlay,
            a_label,
            (a_x + 2, a_y - a_radius - 2),
            a_label_font,
            a_label_font_scale,
            a_label_color,
            a_label_font_thickness,
            cv2.LINE_AA,
        )

    if a_score is not None:
        score_str = f"{a_score:.2f}"
        ((score_w, score_h), _) = cv2.getTextSize(score_str, a_score_font, a_score_font_scale, a_score_font_thickness)
        score_bg = (a_x, a_y + a_radius + 4, a_x + score_w + 4, a_y + a_radius + score_h + 8)
        cv2.rectangle(overlay, (score_bg[0], score_bg[1]), (score_bg[2], score_bg[3]), a_color, -1)
        cv2.putText(
            overlay,
            score_str,
            (a_x + 2, a_y + a_radius + score_h + 2),
            a_score_font,
            a_score_font_scale,
            a_score_color,
            a_score_font_thickness,
            cv2.LINE_AA,
        )

    if a_alpha < 1.0:
        cv2.addWeighted(overlay, a_alpha, img, 1 - a_alpha, 0, img)
    else:
        img = overlay

    if isinstance(a_image, Frame2D):
        return Frame2D(img, **{k: v for k, v in a_image.__dict__.items() if k != 'data'})
    if isinstance(a_image, Image2D):
        return Image2D(img, **{k: v for k, v in a_image.__dict__.items() if k != 'data'})
    return img


def plot_point2d(
    a_image: Image2D | Frame2D | npt.NDArray[np.uint8],
    a_point: Point2D[T],
    a_color: Tuple[int, int, int] = (0, 0, 255),
    a_radius: int = 4,
    a_thickness: int = -1,
    a_alpha: float = 1.0,
    a_label: Optional[str] = None,
    a_label_color: Tuple[int, int, int] = (0, 0, 0),
    a_label_font_scale: float = 0.5,
    a_label_font_thickness: int = 1,
    a_label_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_cross: bool = False,
    a_cross_color: Tuple[int, int, int] = (0, 255, 0),
    a_cross_length: int = 8,
    a_cross_thickness: int = 1,
) -> Image2D | Frame2D | npt.NDArray[np.uint8]:
    """Plot a Point2D on an image.

    Args:
        a_image: The image to draw on (Image2D, Frame2D, or np.ndarray).
        a_point: The Point2D to plot.
        a_color: Point color (B, G, R).
        a_radius: Radius of the point circle.
        a_thickness: Thickness of the point circle.
        a_alpha: Alpha blending factor (0.0 transparent, 1.0 opaque).
        a_label: Optional label text.
        a_label_color: Label text color (B, G, R).
        a_label_font_scale: Font scale for label.
        a_label_font_thickness: Font thickness for label.
        a_label_font: Font type for label.
        a_cross: Whether to draw cross lines at the point.
        a_cross_color: Color for the cross lines (B, G, R).
        a_cross_length: Length of the cross arms.
        a_cross_thickness: Thickness of the cross lines.

    Returns:
        Image2D | Frame2D | np.ndarray: The image with the point drawn.
    """
    return plot_base_point2d(
        a_image,
        int(a_point.x),
        int(a_point.y),
        a_color=a_color,
        a_radius=a_radius,
        a_thickness=a_thickness,
        a_alpha=a_alpha,
        a_label=a_label,
        a_label_color=a_label_color,
        a_label_font_scale=a_label_font_scale,
        a_label_font_thickness=a_label_font_thickness,
        a_label_font=a_label_font,
        a_cross=a_cross,
        a_cross_color=a_cross_color,
        a_cross_length=a_cross_length,
        a_cross_thickness=a_cross_thickness,
        a_score=None,
    )


def plot_keypoint2d(
    a_image: Image2D | Frame2D | npt.NDArray[np.uint8],
    a_keypoint: KeyPoint2D[T],
    a_color: Tuple[int, int, int] = (0, 165, 255),
    a_radius: int = 5,
    a_thickness: int = -1,
    a_alpha: float = 1.0,
    a_label: Optional[str] = None,
    a_label_color: Tuple[int, int, int] = (0, 0, 0),
    a_label_font_scale: float = 0.5,
    a_label_font_thickness: int = 1,
    a_label_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_cross: bool = False,
    a_cross_color: Tuple[int, int, int] = (0, 255, 0),
    a_cross_length: int = 10,
    a_cross_thickness: int = 2,
    a_score_color: Tuple[int, int, int] = (255, 0, 0),
    a_score_font_scale: float = 0.4,
    a_score_font_thickness: int = 1,
    a_score_font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> Image2D | Frame2D | npt.NDArray[np.uint8]:
    """Plot a KeyPoint2D on an image.

    Args:
        a_image: The image to draw on (Image2D, Frame2D, or np.ndarray).
        a_keypoint: The KeyPoint2D to plot.
        a_color: KeyPoint color (B, G, R).
        a_radius: Radius of the keypoint circle.
        a_thickness: Thickness of the keypoint circle.
        a_alpha: Alpha blending factor (0.0 transparent, 1.0 opaque).
        a_label: Optional label text.
        a_label_color: Label text color (B, G, R).
        a_label_font_scale: Font scale for label.
        a_label_font_thickness: Font thickness for label.
        a_label_font: Font type for label.
        a_cross: Whether to draw cross lines at the keypoint.
        a_cross_color: Color for the cross lines (B, G, R).
        a_cross_length: Length of the cross arms.
        a_cross_thickness: Thickness of the cross lines.
        a_score_color: Color for the score text (B, G, R).
        a_score_font_scale: Font scale for the score text.
        a_score_font_thickness: Font thickness for the score text.
        a_score_font: Font type for the score text.

    Returns:
        Image2D | Frame2D | np.ndarray: The image with the keypoint drawn.
    """
    return plot_base_point2d(
        a_image,
        int(a_keypoint.x),
        int(a_keypoint.y),
        a_color=a_color,
        a_radius=a_radius,
        a_thickness=a_thickness,
        a_alpha=a_alpha,
        a_label=a_label,
        a_label_color=a_label_color,
        a_label_font_scale=a_label_font_scale,
        a_label_font_thickness=a_label_font_thickness,
        a_label_font=a_label_font,
        a_cross=a_cross,
        a_cross_color=a_cross_color,
        a_cross_length=a_cross_length,
        a_cross_thickness=a_cross_thickness,
        a_score=getattr(a_keypoint, "score", None),
        a_score_color=a_score_color,
        a_score_font_scale=a_score_font_scale,
        a_score_font_thickness=a_score_font_thickness,
        a_score_font=a_score_font,
    )


def plot_line2d(
    a_image: Image2D | Frame2D | npt.NDArray[np.uint8],
    a_line: Line2D[PKT],
    a_color: Tuple[int, int, int] = (255, 0, 0),
    a_thickness: int = 2,
    a_alpha: float = 1.0,
    a_label: Optional[str] = None,
    a_label_color: Tuple[int, int, int] = (0, 0, 0),
    a_label_font_scale: float = 0.5,
    a_label_font_thickness: int = 1,
    a_label_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_draw_p1: bool = False,
    a_draw_p2: bool = False,
    a_point_color: Tuple[int, int, int] = (0, 255, 0),
    a_point_radius: int = 4,
    a_point_thickness: int = -1,
    a_point_alpha: float = 1.0,
    a_point_label: Optional[str] = None,
    a_point_label_color: Tuple[int, int, int] = (255, 255, 255),
    a_point_label_font_scale: float = 0.5,
    a_point_label_font_thickness: int = 1,
    a_point_label_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_point_cross: bool = False,
    a_point_cross_color: Tuple[int, int, int] = (0, 255, 255),
    a_point_cross_length: int = 8,
    a_point_cross_thickness: int = 1,
    a_score_color: Tuple[int, int, int] = (255, 0, 0),
    a_score_font_scale: float = 0.4,
    a_score_font_thickness: int = 1,
    a_score_font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> Image2D | Frame2D | npt.NDArray[np.uint8]:
    """Plot a Line2D on an image.

    Args:
        a_image: The image to draw on (Image2D, Frame2D, or np.ndarray).
        a_line: The Line2D to plot.
        a_color: Line color (B, G, R).
        a_thickness: Line thickness.
        a_alpha: Alpha blending factor (0.0 transparent, 1.0 opaque).
        a_label: Optional label text.
        a_label_color: Label text color (B, G, R).
        a_label_font_scale: Font scale for label.
        a_label_font_thickness: Font thickness for label.
        a_label_font: Font type for label.
        a_draw_p1: Whether to draw the first point of the line.
        a_draw_p2: Whether to draw the second point of the line.
        a_point_color: Color for the points (B, G, R).
        a_point_radius: Radius of the points.
        a_point_thickness: Thickness of the points.
        a_point_alpha: Alpha blending factor for points.
        a_point_label: Optional label for the points.
        a_point_label_color: Color for the point labels (B, G, R).
        a_point_label_font_scale: Font scale for point labels.
        a_point_label_font_thickness: Font thickness for point labels.
        a_point_label_font: Font type for point labels.
        a_point_cross: Whether to draw cross lines at the points.
        a_point_cross_color: Color for the cross lines (B, G, R).
        a_point_cross_length: Length of the cross arms.
        a_point_cross_thickness: Thickness of the cross lines.
        a_score_color: Color for the score text (B, G, R).
        a_score_font_scale: Font scale for the score text.
        a_score_font_thickness: Font thickness for the score text.
        a_score_font: Font type for the score text.

    Returns:
        Image2D | Frame2D | np.ndarray: The image with the line drawn.
    """
    if isinstance(a_image, (Image2D, Frame2D)):
        if hasattr(a_image, "data"):
            img: npt.NDArray[np.uint8] = a_image.data.copy()
        else:
            img: npt.NDArray[np.uint8] = np.array(a_image).copy()
    else:
        img: npt.NDArray[np.uint8] = np.array(a_image).copy()

    overlay = img.copy()
    x1, y1 = int(a_line.p1.x), int(a_line.p1.y)
    x2, y2 = int(a_line.p2.x), int(a_line.p2.y)

    cv2.line(overlay, (x1, y1), (x2, y2), a_color, a_thickness)

    if a_label:
        mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
        ((text_w, text_h), _) = cv2.getTextSize(a_label, a_label_font, a_label_font_scale, a_label_font_thickness)
        label_bg = (mx, my - text_h - 4, mx + text_w + 4, my)
        cv2.rectangle(overlay, (label_bg[0], label_bg[1]), (label_bg[2], label_bg[3]), a_color, -1)
        cv2.putText(
            overlay,
            a_label,
            (mx + 2, my - 2),
            a_label_font,
            a_label_font_scale,
            a_label_color,
            a_label_font_thickness,
            cv2.LINE_AA,
        )

    if a_alpha < 1.0:
        cv2.addWeighted(overlay, a_alpha, img, 1 - a_alpha, 0, img)
    else:
        img = overlay

    if a_draw_p1:
        if isinstance(a_line.p1, KeyPoint2D):
            img = plot_keypoint2d(
                a_image=img,
                a_keypoint=cast(KeyPoint2D[int], a_line.p1),
                a_color=a_point_color,
                a_radius=a_point_radius,
                a_thickness=a_point_thickness,
                a_alpha=a_point_alpha,
                a_label=a_point_label,
                a_label_color=a_point_label_color,
                a_label_font_scale=a_point_label_font_scale,
                a_label_font_thickness=a_point_label_font_thickness,
                a_label_font=a_point_label_font,
                a_cross=a_point_cross,
                a_cross_color=a_point_cross_color,
                a_cross_length=a_point_cross_length,
                a_cross_thickness=a_point_cross_thickness,
                a_score_color=a_score_color,
                a_score_font_scale=a_score_font_scale,
                a_score_font_thickness=a_score_font_thickness,
                a_score_font=a_score_font,
            )
        else:
            img = plot_point2d(
                a_image=img,
                a_point=cast(Point2D[int], a_line.p1),
                a_color=a_point_color,
                a_radius=a_point_radius,
                a_thickness=a_point_thickness,
                a_alpha=a_point_alpha,
                a_label=a_point_label,
                a_label_color=a_point_label_color,
                a_label_font_scale=a_point_label_font_scale,
                a_label_font_thickness=a_point_label_font_thickness,
                a_label_font=a_point_label_font,
                a_cross=a_point_cross,
                a_cross_color=a_point_cross_color,
                a_cross_length=a_point_cross_length,
                a_cross_thickness=a_point_cross_thickness,
            )

    if a_draw_p2:
        if isinstance(a_line.p2, KeyPoint2D):
            img = plot_keypoint2d(
                a_image=img,
                a_keypoint=cast(KeyPoint2D[int], a_line.p2),
                a_color=a_point_color,
                a_radius=a_point_radius,
                a_thickness=a_point_thickness,
                a_alpha=a_point_alpha,
                a_label=a_point_label,
                a_label_color=a_point_label_color,
                a_label_font_scale=a_point_label_font_scale,
                a_label_font_thickness=a_point_label_font_thickness,
                a_label_font=a_point_label_font,
                a_cross=a_point_cross,
                a_cross_color=a_point_cross_color,
                a_cross_length=a_point_cross_length,
                a_cross_thickness=a_point_cross_thickness,
                a_score_color=a_score_color,
                a_score_font_scale=a_score_font_scale,
                a_score_font_thickness=a_score_font_thickness,
                a_score_font=a_score_font,
            )
        else:
            img = plot_point2d(
                a_image=img,
                a_point=cast(Point2D[int], a_line.p2),
                a_color=a_point_color,
                a_radius=a_point_radius,
                a_thickness=a_point_thickness,
                a_alpha=a_point_alpha,
                a_label=a_point_label,
                a_label_color=a_point_label_color,
                a_label_font_scale=a_point_label_font_scale,
                a_label_font_thickness=a_point_label_font_thickness,
                a_label_font=a_point_label_font,
                a_cross=a_point_cross,
                a_cross_color=a_point_cross_color,
                a_cross_length=a_point_cross_length,
                a_cross_thickness=a_point_cross_thickness,
            )

    if isinstance(a_image, Frame2D):
        return Frame2D(img, **{k: v for k, v in a_image.__dict__.items() if k != 'data'})
    if isinstance(a_image, Image2D):
        return Image2D(img, **{k: v for k, v in a_image.__dict__.items() if k != 'data'})
    return img


def plot_pose2d(
    a_image: Image2D | Frame2D | npt.NDArray[np.uint8],
    a_pose: Pose2D[KPT],
    a_limb_color: Tuple[int, int, int] = (255, 0, 0),
    a_limb_thickness: int = 2,
    a_limb_alpha: float = 1.0,
    a_limb_label: Optional[str] = None,
    a_limb_label_color: Tuple[int, int, int] = (255, 255, 255),
    a_limb_label_font_scale: float = 0.5,
    a_limb_label_font_thickness: int = 1,
    a_limb_label_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_limb_draw_p1: bool = False,
    a_limb_draw_p2: bool = False,
    a_point_color: Tuple[int, int, int] = (0, 165, 255),
    a_point_radius: int = 5,
    a_point_thickness: int = -1,
    a_point_alpha: float = 1.0,
    a_point_label: Optional[str] = None,
    a_point_label_color: Tuple[int, int, int] = (255, 255, 255),
    a_point_label_font_scale: float = 0.5,
    a_point_label_font_thickness: int = 1,
    a_point_label_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_point_cross: bool = False,
    a_point_cross_color: Tuple[int, int, int] = (0, 255, 0),
    a_point_cross_length: int = 8,
    a_point_cross_thickness: int = 1,
    a_score_color: Tuple[int, int, int] = (255, 0, 0),
    a_score_font_scale: float = 0.4,
    a_score_font_thickness: int = 1,
    a_score_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_draw_points: bool = True,
    a_draw_limbs: bool = True,
) -> Image2D | Frame2D | npt.NDArray[np.uint8]:
    """
    Plot a Pose2D (skeleton) on an image.

    Args:
        a_image: The image to draw on.
        a_pose: The Pose2D to plot.
        a_limb_color: Color for limbs (lines).
        a_limb_thickness: Thickness for limbs.
        a_limb_alpha: Alpha for limbs.
        a_limb_label: Optional label for limbs.
        a_limb_label_color: Limb label color.
        a_limb_label_font_scale: Limb label font scale.
        a_limb_label_font_thickness: Limb label font thickness.
        a_limb_label_font: Limb label font.
        a_limb_draw_p1: Draw endpoints for limb p1.
        a_limb_draw_p2: Draw endpoints for limb p2.
        a_point_color: Color for keypoints.
        a_point_radius: Radius for keypoints.
        a_point_thickness: Thickness for keypoints.
        a_point_alpha: Alpha for keypoints.
        a_point_label: Optional label for keypoints.
        a_point_label_color: Keypoint label color.
        a_point_label_font_scale: Keypoint label font scale.
        a_point_label_font_thickness: Keypoint label font thickness.
        a_point_label_font: Keypoint label font.
        a_point_cross: Draw cross at keypoints.
        a_point_cross_color: Cross color at keypoints.
        a_point_cross_length: Cross length at keypoints.
        a_point_cross_thickness: Cross thickness at keypoints.
        a_score_color: Color for keypoint score.
        a_score_font_scale: Font scale for keypoint score.
        a_score_font_thickness: Font thickness for keypoint score.
        a_score_font: Font for keypoint score.
        a_draw_points: Whether to draw keypoints.
        a_draw_limbs: Whether to draw limbs.

    Returns:
        Image2D | Frame2D | np.ndarray: The image with the pose drawn.
    """
    img = a_image

    # Draw limbs
    if a_draw_limbs and hasattr(a_pose, "limbs"):
        for limb in a_pose.limbs:
            img = plot_line2d(
                img,
                limb,
                a_color=a_limb_color,
                a_thickness=a_limb_thickness,
                a_alpha=a_limb_alpha,
                a_label=a_limb_label,
                a_label_color=a_limb_label_color,
                a_label_font_scale=a_limb_label_font_scale,
                a_label_font_thickness=a_limb_label_font_thickness,
                a_label_font=a_limb_label_font,
                a_draw_p1=a_limb_draw_p1,
                a_draw_p2=a_limb_draw_p2,
                a_point_color=a_point_color,
                a_point_radius=a_point_radius,
                a_point_thickness=a_point_thickness,
                a_point_alpha=a_point_alpha,
                a_point_label=a_point_label,
                a_point_label_color=a_point_label_color,
                a_point_label_font_scale=a_point_label_font_scale,
                a_point_label_font_thickness=a_point_label_font_thickness,
                a_point_label_font=a_point_label_font,
                a_point_cross=a_point_cross,
                a_point_cross_color=a_point_cross_color,
                a_point_cross_length=a_point_cross_length,
                a_point_cross_thickness=a_point_cross_thickness,
                a_score_color=a_score_color,
                a_score_font_scale=a_score_font_scale,
                a_score_font_thickness=a_score_font_thickness,
                a_score_font=a_score_font,
            )

    # Draw keypoints
    if a_draw_points:
        for kp in a_pose:
            img = plot_keypoint2d(
                img,
                cast(KeyPoint2D[int], kp),
                a_color=a_point_color,
                a_radius=a_point_radius,
                a_thickness=a_point_thickness,
                a_alpha=a_point_alpha,
                a_label=a_point_label,
                a_label_color=a_point_label_color,
                a_label_font_scale=a_point_label_font_scale,
                a_label_font_thickness=a_point_label_font_thickness,
                a_label_font=a_point_label_font,
                a_cross=a_point_cross,
                a_cross_color=a_point_cross_color,
                a_cross_length=a_point_cross_length,
                a_cross_thickness=a_point_cross_thickness,
                a_score_color=a_score_color,
                a_score_font_scale=a_score_font_scale,
                a_score_font_thickness=a_score_font_thickness,
                a_score_font=a_score_font,
            )

    # Return with original type and metadata if possible
    if isinstance(a_image, Frame2D):
        return Frame2D(img, **{k: v for k, v in a_image.__dict__.items() if k != 'data'})
    if isinstance(a_image, Image2D):
        return Image2D(img, **{k: v for k, v in a_image.__dict__.items() if k != 'data'})
    return img


def plot_contour2d(
    a_image: Image2D | Frame2D | npt.NDArray[np.uint8],
    a_contour: Contour2D[PT],
    a_color: Tuple[int, int, int] = (0, 255, 255),
    a_thickness: int = 2,
    a_alpha: float = 1.0,
    a_closed: bool = True,
    a_label: Optional[str] = None,
    a_label_color: Tuple[int, int, int] = (0, 0, 0),
    a_label_font_scale: float = 0.5,
    a_label_font_thickness: int = 1,
    a_label_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_draw_points: bool = False,
    a_point_color: Tuple[int, int, int] = (0, 0, 255),
    a_point_radius: int = 3,
    a_point_thickness: int = -1,
    a_point_alpha: float = 1.0,
    a_point_label: Optional[str] = None,
    a_point_label_color: Tuple[int, int, int] = (255, 255, 255),
    a_point_label_font_scale: float = 0.4,
    a_point_label_font_thickness: int = 1,
    a_point_label_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_point_cross: bool = False,
    a_point_cross_color: Tuple[int, int, int] = (0, 255, 0),
    a_point_cross_length: int = 6,
    a_point_cross_thickness: int = 1,
) -> Image2D | Frame2D | npt.NDArray[np.uint8]:
    """
    Plot a Contour2D on an image.

    Args:
        a_image: The image to draw on.
        a_contour: The Contour2D to plot.
        a_color: Contour color (B, G, R).
        a_thickness: Contour line thickness.
        a_alpha: Alpha blending for the contour.
        a_closed: Whether the contour is closed.
        a_label: Optional label for the contour (drawn at center).
        a_label_color: Label text color.
        a_label_font_scale: Font scale for label.
        a_label_font_thickness: Font thickness for label.
        a_label_font: Font for label.
        a_draw_points: Whether to plot the contour points.
        a_point_color: Color for points.
        a_point_radius: Radius for points.
        a_point_thickness: Thickness for points.
        a_point_alpha: Alpha for points.
        a_point_label: Optional label for points.
        a_point_label_color: Label color for points.
        a_point_label_font_scale: Font scale for point labels.
        a_point_label_font_thickness: Font thickness for point labels.
        a_point_label_font: Font for point labels.
        a_point_cross: Whether to draw a cross at points.
        a_point_cross_color: Cross color for points.
        a_point_cross_length: Cross length for points.
        a_point_cross_thickness: Cross thickness for points.

    Returns:
        Image2D | Frame2D | np.ndarray: The image with the contour drawn.
    """
    # Prepare image
    if isinstance(a_image, (Image2D, Frame2D)):
        if hasattr(a_image, "data"):
            img: npt.NDArray[np.uint8] = a_image.data.copy()
        else:
            img: npt.NDArray[np.uint8] = np.array(a_image).copy()
    else:
        img: npt.NDArray[np.uint8] = np.array(a_image).copy()

    overlay = img.copy()
    # Prepare contour points for cv2.polylines
    pts = np.array([[int(p.x), int(p.y)] for p in a_contour], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [pts], isClosed=a_closed, color=a_color, thickness=a_thickness)

    # Draw label at center if requested
    if a_label:
        center = a_contour.center
        cx, cy = int(center.x), int(center.y)
        ((text_w, text_h), _) = cv2.getTextSize(a_label, a_label_font, a_label_font_scale, a_label_font_thickness)
        label_bg = (cx, cy - text_h - 4, cx + text_w + 4, cy)
        cv2.rectangle(overlay, (label_bg[0], label_bg[1]), (label_bg[2], label_bg[3]), a_color, -1)
        cv2.putText(
            overlay,
            a_label,
            (cx + 2, cy - 2),
            a_label_font,
            a_label_font_scale,
            a_label_color,
            a_label_font_thickness,
            cv2.LINE_AA,
        )

    # Alpha blending for contour
    if a_alpha < 1.0:
        cv2.addWeighted(overlay, a_alpha, img, 1 - a_alpha, 0, img)
    else:
        img = overlay

    # Draw points if requested
    if a_draw_points:
        for idx, pt in enumerate(a_contour):
            label = a_point_label
            if a_point_label is not None and "{}" in a_point_label:
                label = a_point_label.format(idx)
            img = plot_point2d(
                img,
                cast(Point2D[int], pt),
                a_color=a_point_color,
                a_radius=a_point_radius,
                a_thickness=a_point_thickness,
                a_alpha=a_point_alpha,
                a_label=label,
                a_label_color=a_point_label_color,
                a_label_font_scale=a_point_label_font_scale,
                a_label_font_thickness=a_point_label_font_thickness,
                a_label_font=a_point_label_font,
                a_cross=a_point_cross,
                a_cross_color=a_point_cross_color,
                a_cross_length=a_point_cross_length,
                a_cross_thickness=a_point_cross_thickness,
            )

    # Return with original type and metadata if possible
    if isinstance(a_image, Frame2D):
        return Frame2D(img, **{k: v for k, v in a_image.__dict__.items() if k != 'data'})
    if isinstance(a_image, Image2D):
        return Image2D(img, **{k: v for k, v in a_image.__dict__.items() if k != 'data'})
    return img


def plot_base_box2d(
    a_image: Image2D | Frame2D | npt.NDArray[np.uint8],
    a_p1: Point2D[int],
    a_p2: Point2D[int],
    a_color: Tuple[int, int, int] = (0, 255, 0),
    a_thickness: int = 2,
    a_alpha: float = 1.0,
    a_label: Optional[str] = None,
    a_label_color: Tuple[int, int, int] = (0, 0, 0),
    a_label_font_scale: float = 0.5,
    a_label_font_thickness: int = 1,
    a_label_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_draw_points: bool = False,
    a_point_color: Tuple[int, int, int] = (0, 0, 255),
    a_point_radius: int = 3,
    a_point_thickness: int = -1,
    a_point_alpha: float = 1.0,
    a_point_label: Optional[str] = None,
    a_point_label_color: Tuple[int, int, int] = (255, 255, 255),
    a_point_label_font_scale: float = 0.4,
    a_point_label_font_thickness: int = 1,
    a_point_label_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_point_cross: bool = False,
    a_point_cross_color: Tuple[int, int, int] = (0, 255, 0),
    a_point_cross_length: int = 6,
    a_point_cross_thickness: int = 1,
) -> Image2D | Frame2D | npt.NDArray[np.uint8]:
    """Plot a bounding box defined by two points on an image.

    Args:
        a_image: The image to draw on (Image2D, Frame2D, or np.ndarray).
        a_p1: The first corner point of the bounding box.
        a_p2: The second corner point of the bounding box.
        a_color: Color of the bounding box (B, G, R).
        a_thickness: Thickness of the bounding box lines.
        a_alpha: Alpha blending factor (0.0 transparent, 1.0 opaque).
        a_label: Optional label text for the bounding box.
        a_label_color: Color of the label text (B, G, R).
        a_label_font_scale: Font scale for the label text.
        a_label_font_thickness: Font thickness for the label text.
        a_label_font: Font type for the label text.
        a_draw_points: Whether to draw the corners of the bounding box.
        a_point_color: Color for the corner points (B, G, R or grayscale).
        a_point_radius: Radius of the corner points.
        a_point_thickness: Thickness of the corner points.
        a_point_alpha: Alpha blending factor for the corner points.
        a_point_label: Optional label for the corner points.
        a_point_label_color: Color for the corner point labels (B, G, R).
        a_point_label_font_scale: Font scale for the corner point labels.
        a_point_label_font_thickness: Font thickness for the corner point labels.
        a_point_label_font: Font type for the corner point labels.
        a_point_cross: Whether to draw cross lines at the corner points.
        a_point_cross_color: Color for the cross lines (B, G, R).
        a_point_cross_length: Length of the cross arms.
        a_point_cross_thickness: Thickness of the cross lines.

    Returns:
        Image2D | Frame2D | np.ndarray: The image with the bounding box drawn.
    """
    if isinstance(a_image, (Image2D, Frame2D)):
        if hasattr(a_image, "data"):
            img = a_image.data.copy()
        else:
            img = np.array(a_image).copy()
    else:
        img = np.array(a_image).copy()

    overlay = img.copy()
    x1, y1 = int(a_p1.x), int(a_p1.y)
    x2, y2 = int(a_p2.x), int(a_p2.y)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), a_color, a_thickness)

    if a_label:
        ((text_w, text_h), _) = cv2.getTextSize(a_label, a_label_font, a_label_font_scale, a_label_font_thickness)
        label_bg = (x1, y1 - text_h - 4, x1 + text_w + 4, y1)
        cv2.rectangle(overlay, (label_bg[0], label_bg[1]), (label_bg[2], label_bg[3]), a_color, -1)
        cv2.putText(
            overlay,
            a_label,
            (x1 + 2, y1 - 2),
            a_label_font,
            a_label_font_scale,
            a_label_color,
            a_label_font_thickness,
            cv2.LINE_AA,
        )

    if a_alpha < 1.0:
        cv2.addWeighted(overlay, a_alpha, img, 1 - a_alpha, 0, img)
    else:
        img = overlay

    if a_draw_points:
        img = plot_point2d(
            img,
            a_p1,
            a_color=a_point_color,
            a_radius=a_point_radius,
            a_thickness=a_point_thickness,
            a_alpha=a_point_alpha,
            a_label=a_point_label,
            a_label_color=a_point_label_color,
            a_label_font_scale=a_point_label_font_scale,
            a_label_font_thickness=a_point_label_font_thickness,
            a_label_font=a_point_label_font,
            a_cross=a_point_cross,
            a_cross_color=a_point_cross_color,
            a_cross_length=a_point_cross_length,
            a_cross_thickness=a_point_cross_thickness,
        )
        img = plot_point2d(
            img,
            a_p2,
            a_color=a_point_color,
            a_radius=a_point_radius,
            a_thickness=a_point_thickness,
            a_alpha=a_point_alpha,
            a_label=a_point_label,
            a_label_color=a_point_label_color,
            a_label_font_scale=a_point_label_font_scale,
            a_label_font_thickness=a_point_label_font_thickness,
            a_label_font=a_point_label_font,
            a_cross=a_point_cross,
            a_cross_color=a_point_cross_color,
            a_cross_length=a_point_cross_length,
            a_cross_thickness=a_point_cross_thickness,
        )

    if isinstance(a_image, Frame2D):
        return Frame2D(img, **{k: v for k, v in a_image.__dict__.items() if k != 'data'})
    elif isinstance(a_image, Image2D):
        return Image2D(img, **{k: v for k, v in a_image.__dict__.items() if k != 'data'})
    else:
        return img


def plot_box2d(
    a_image: Image2D | Frame2D | npt.NDArray[np.uint8],
    a_box: Box2D[PT],
    a_color: Tuple[int, int, int] = (0, 255, 0),
    a_thickness: int = 2,
    a_alpha: float = 1.0,
    a_label: Optional[str] = None,
    a_label_color: Tuple[int, int, int] = (0, 0, 0),
    a_label_font_scale: float = 0.5,
    a_label_font_thickness: int = 1,
    a_label_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_draw_points: bool = False,
    a_point_color: Tuple[int, int, int] = (0, 0, 255),
    a_point_radius: int = 3,
    a_point_thickness: int = -1,
    a_point_alpha: float = 1.0,
    a_point_label: Optional[str] = None,
    a_point_label_color: Tuple[int, int, int] = (255, 255, 255),
    a_point_label_font_scale: float = 0.4,
    a_point_label_font_thickness: int = 1,
    a_point_label_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_point_cross: bool = False,
    a_point_cross_color: Tuple[int, int, int] = (0, 255, 0),
    a_point_cross_length: int = 6,
    a_point_cross_thickness: int = 1,
) -> Image2D | Frame2D | npt.NDArray[np.uint8]:
    """Plot a Box2D on an image.

    Args:
        a_image: The image to draw on (Image2D, Frame2D, or np.ndarray).
        a_box: The Box2D to plot.
        a_color: Color of the box (B, G, R).
        a_thickness: Thickness of the box lines.
        a_alpha: Alpha blending factor (0.0 transparent, 1.0 opaque).
        a_label: Optional label text for the box.
        a_label_color: Color of the label text (B, G, R).
        a_label_font_scale: Font scale for the label text.
        a_label_font_thickness: Font thickness for the label text.
        a_label_font: Font type for the label text.
        a_draw_points: Whether to draw the corners of the box.
        a_point_color: Color for the corner points (B, G, R).
        a_point_radius: Radius of the corner points.
        a_point_thickness: Thickness of the corner points.
        a_point_alpha: Alpha blending factor for the corner points.
        a_point_label: Optional label for the corner points.
        a_point_label_color: Color for the corner point labels (B, G, R).
        a_point_label_font_scale: Font scale for the corner point labels.
        a_point_label_font_thickness: Font thickness for the corner point labels.
        a_point_label_font: Font type for the corner point labels.
        a_point_cross: Whether to draw cross lines at the corner points.
        a_point_cross_color: Color for the cross lines (B, G, R).
        a_point_cross_length: Length of the cross arms.
        a_point_cross_thickness: Thickness of the cross lines.

    Returns:
        Image2D | Frame2D | np.ndarray: The image with the box drawn.
    """
    return plot_base_box2d(
        a_image,
        cast(Point2D[int], a_box.p1),
        cast(Point2D[int], a_box.p2),
        a_color=a_color,
        a_thickness=a_thickness,
        a_alpha=a_alpha,
        a_label=a_label,
        a_label_color=a_label_color,
        a_label_font_scale=a_label_font_scale,
        a_label_font_thickness=a_label_font_thickness,
        a_label_font=a_label_font,
        a_draw_points=a_draw_points,
        a_point_color=a_point_color,
        a_point_radius=a_point_radius,
        a_point_thickness=a_point_thickness,
        a_point_alpha=a_point_alpha,
        a_point_label=a_point_label,
        a_point_label_color=a_point_label_color,
        a_point_label_font_scale=a_point_label_font_scale,
        a_point_label_font_thickness=a_point_label_font_thickness,
        a_point_label_font=a_point_label_font,
        a_point_cross=a_point_cross,
        a_point_cross_color=a_point_cross_color,
        a_point_cross_length=a_point_cross_length,
        a_point_cross_thickness=a_point_cross_thickness,
    )


def plot_bbox2d(
    a_image: Image2D | Frame2D | npt.NDArray[np.uint8],
    a_box: BBox2D[PT],
    a_color: Tuple[int, int, int] = (0, 255, 255),
    a_thickness: int = 2,
    a_alpha: float = 1.0,
    a_label: Optional[str] = None,
    a_label_color: Tuple[int, int, int] = (0, 0, 0),
    a_label_font_scale: float = 0.5,
    a_label_font_thickness: int = 1,
    a_label_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_draw_points: bool = False,
    a_point_color: Tuple[int, int, int] = (0, 0, 255),
    a_point_radius: int = 3,
    a_point_thickness: int = -1,
    a_point_alpha: float = 1.0,
    a_point_label: Optional[str] = None,
    a_point_label_color: Tuple[int, int, int] = (255, 255, 255),
    a_point_label_font_scale: float = 0.4,
    a_point_label_font_thickness: int = 1,
    a_point_label_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_point_cross: bool = False,
    a_point_cross_color: Tuple[int, int, int] = (0, 255, 0),
    a_point_cross_length: int = 6,
    a_point_cross_thickness: int = 1,
    a_score_color: Tuple[int, int, int] = (255, 0, 0),
    a_score_font_scale: float = 0.4,
    a_score_font_thickness: int = 1,
    a_score_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_draw_score: bool = True,
) -> Image2D | Frame2D | npt.NDArray[np.uint8]:
    """Plot a BBox2D on an image.

    Args:
        a_image: The image to draw on (Image2D, Frame2D, or np.ndarray).
        a_box: The BBox2D to plot.
        a_color: Color of the bounding box (B, G, R).
        a_thickness: Thickness of the bounding box lines.
        a_alpha: Alpha blending factor (0.0 transparent, 1.0 opaque).
        a_label: Optional label text for the bounding box.
        a_label_color: Color of the label text (B, G, R).
        a_label_font_scale: Font scale for the label text.
        a_label_font_thickness: Font thickness for the label text.
        a_label_font: Font type for the label text.
        a_draw_points: Whether to draw the corners of the bounding box.
        a_point_color: Color for the corner points (B, G, R).
        a_point_radius: Radius of the corner points.
        a_point_thickness: Thickness of the corner points.
        a_point_alpha: Alpha blending factor for the corner points.
        a_point_label: Optional label for the corner points.
        a_point_label_color: Color for the corner point labels (B, G, R).
        a_point_label_font_scale: Font scale for the corner point labels.
        a_point_label_font_thickness: Font thickness for the corner point labels.
        a_point_label_font: Font type for the corner point labels.
        a_point_cross: Whether to draw cross lines at the corner points.
        a_point_cross_color: Color for the cross lines (B, G, R).
        a_point_cross_length: Length of the cross arms.
        a_point_cross_thickness: Thickness of the cross lines.
        a_score_color: Color for the score text (B, G, R).
        a_score_font_scale: Font scale for the score text.
        a_score_font_thickness: Font thickness for the score text.
        a_score_font: Font type for the score text.
        a_draw_score: Whether to draw the score at the top-right corner.

    Returns:
        Image2D | Frame2D | np.ndarray: The image with the bounding box drawn.
    """
    label = a_label
    if label is None and hasattr(a_box, "label"):
        label = f"Class:{getattr(a_box, 'label', '')}"
    img = plot_base_box2d(
        a_image,
        cast(Point2D[int], a_box.p1),
        cast(Point2D[int], a_box.p2),
        a_color=a_color,
        a_thickness=a_thickness,
        a_alpha=a_alpha,
        a_label=label,
        a_label_color=a_label_color,
        a_label_font_scale=a_label_font_scale,
        a_label_font_thickness=a_label_font_thickness,
        a_label_font=a_label_font,
        a_draw_points=a_draw_points,
        a_point_color=a_point_color,
        a_point_radius=a_point_radius,
        a_point_thickness=a_point_thickness,
        a_point_alpha=a_point_alpha,
        a_point_label=a_point_label,
        a_point_label_color=a_point_label_color,
        a_point_label_font_scale=a_point_label_font_scale,
        a_point_label_font_thickness=a_point_label_font_thickness,
        a_point_label_font=a_point_label_font,
        a_point_cross=a_point_cross,
        a_point_cross_color=a_point_cross_color,
        a_point_cross_length=a_point_cross_length,
        a_point_cross_thickness=a_point_cross_thickness,
    )
    if a_draw_score and hasattr(a_box, "score"):
        x2, y2 = int(a_box.p2.x), int(a_box.p2.y)
        score_str = f"{getattr(a_box, 'score', 0):.2f}"
        ((score_w, score_h), _) = cv2.getTextSize(score_str, a_score_font, a_score_font_scale, a_score_font_thickness)
        score_bg = (x2 - score_w - 4, y2 - score_h - 4, x2, y2)
        img_np = img if isinstance(img, np.ndarray) else img.data
        cv2.rectangle(img_np, (int(score_bg[0]), int(score_bg[1])), (int(score_bg[2]), int(score_bg[3])), a_color, -1)
        img = img_np
        cv2.putText(
            img if isinstance(img, np.ndarray) else img.data,
            score_str,
            (int(score_bg[0] + 2), int(score_bg[3] - 2)),
            a_score_font,
            a_score_font_scale,
            a_score_color,
            a_score_font_thickness,
            getattr(cv2, "LINE_AA", 16),
        )
    if isinstance(a_image, Frame2D):
        return Frame2D(img, **{k: v for k, v in a_image.__dict__.items() if k != 'data'})
    elif isinstance(a_image, Image2D):
        return Image2D(img, **{k: v for k, v in a_image.__dict__.items() if k != 'data'})
    else:
        return img


def plot_segbbox2d(
    a_image: Image2D | Frame2D | npt.NDArray[np.uint8],
    a_box: SegBBox2D[PT],
    a_color: Tuple[int, int, int] = (255, 0, 255),
    a_thickness: int = 2,
    a_alpha: float = 1.0,
    a_label: Optional[str] = None,
    a_label_color: Tuple[int, int, int] = (0, 0, 0),
    a_label_font_scale: float = 0.5,
    a_label_font_thickness: int = 1,
    a_label_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_draw_points: bool = False,
    a_point_color: Tuple[int, int, int] = (0, 0, 255),
    a_point_radius: int = 3,
    a_point_thickness: int = -1,
    a_point_alpha: float = 1.0,
    a_point_label: Optional[str] = None,
    a_point_label_color: Tuple[int, int, int] = (255, 255, 255),
    a_point_label_font_scale: float = 0.4,
    a_point_label_font_thickness: int = 1,
    a_point_label_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_point_cross: bool = False,
    a_point_cross_color: Tuple[int, int, int] = (0, 255, 0),
    a_point_cross_length: int = 6,
    a_point_cross_thickness: int = 1,
    a_score_color: Tuple[int, int, int] = (255, 0, 0),
    a_score_font_scale: float = 0.4,
    a_score_font_thickness: int = 1,
    a_score_font: int = cv2.FONT_HERSHEY_SIMPLEX,
    a_draw_score: bool = True,
    a_draw_mask: bool = True,
    a_mask_alpha: float = 0.3,
) -> Image2D | Frame2D | npt.NDArray[np.uint8]:
    """Plot a SegBBox2D on an image.

    Args:
        a_image: The image to draw on (Image2D, Frame2D, or np.ndarray).
        a_box: The SegBBox2D to plot.
        a_color: Color of the bounding box (B, G, R).
        a_thickness: Thickness of the bounding box lines.
        a_alpha: Alpha blending factor (0.0 transparent, 1.0 opaque).
        a_label: Optional label text for the bounding box.
        a_label_color: Color of the label text (B, G, R).
        a_label_font_scale: Font scale for the label text.
        a_label_font_thickness: Font thickness for the label text.
        a_label_font: Font type for the label text.
        a_draw_points: Whether to draw the corners of the bounding box.
        a_point_color: Color for the corner points (B, G, R).
        a_point_radius: Radius of the corner points.
        a_point_thickness: Thickness of the corner points.
        a_point_alpha: Alpha blending factor for the corner points.
        a_point_label: Optional label for the corner points.
        a_point_label_color: Color for the corner point labels (B, G, R).
        a_point_label_font_scale: Font scale for the corner point labels.
        a_point_label_font_thickness: Font thickness for the corner point labels.
        a_point_label_font: Font type for the corner point labels.
        a_point_cross: Whether to draw cross lines at the corner points.
        a_point_cross_color: Color for the cross lines (B, G, R).
        a_point_cross_length: Length of the cross arms.
        a_point_cross_thickness: Thickness of the cross lines.
        a_score_color: Color for the score text (B, G, R).
        a_score_font_scale: Font scale for the score text.
        a_score_font_thickness: Font thickness for the score text.
        a_score_font: Font type for the score text.
        a_draw_score: Whether to draw the score at the top-right corner.
        a_draw_mask: Whether to draw the segmentation mask.
        a_mask_alpha: Alpha blending factor for the mask (0.0 transparent, 1.0 opaque).

    Returns:
        Image2D | Frame2D | np.ndarray: The image with the SegBBox2D drawn.
    """
    img = plot_bbox2d(
        a_image,
        a_box,
        a_color=a_color,
        a_thickness=a_thickness,
        a_alpha=a_alpha,
        a_label=a_label,
        a_label_color=a_label_color,
        a_label_font_scale=a_label_font_scale,
        a_label_font_thickness=a_label_font_thickness,
        a_label_font=a_label_font,
        a_draw_points=a_draw_points,
        a_point_color=a_point_color,
        a_point_radius=a_point_radius,
        a_point_thickness=a_point_thickness,
        a_point_alpha=a_point_alpha,
        a_point_label=a_point_label,
        a_point_label_color=a_point_label_color,
        a_point_label_font_scale=a_point_label_font_scale,
        a_point_label_font_thickness=a_point_label_font_thickness,
        a_point_label_font=a_point_label_font,
        a_point_cross=a_point_cross,
        a_point_cross_color=a_point_cross_color,
        a_point_cross_length=a_point_cross_length,
        a_point_cross_thickness=a_point_cross_thickness,
        a_score_color=a_score_color,
        a_score_font_scale=a_score_font_scale,
        a_score_font_thickness=a_score_font_thickness,
        a_score_font=a_score_font,
        a_draw_score=a_draw_score,
    )
    if a_draw_mask and hasattr(a_box, "mask") and a_box.mask is not None:
        mask = a_box.mask.data
        if mask.ndim == 2:
            img_np = img if isinstance(img, np.ndarray) else img.data
            mask_color = np.zeros_like(img_np)
            mask_color[int(a_box.p1.y) : int(a_box.p2.y), int(a_box.p1.x) : int(a_box.p2.x), :] = np.array(
                a_color, dtype=np.uint8
            )
            width = int(a_box.width)
            height = int(a_box.height)
            mask_bool = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask_bool.astype(bool)
            y1, y2 = int(a_box.p1.y), int(a_box.p2.y)
            x1, x2 = int(a_box.p1.x), int(a_box.p2.x)
            roi = img_np[y1:y2, x1:x2]
            mask_roi_color = mask_color[y1:y2, x1:x2][mask_bool]
            roi[mask_bool] = cv2.addWeighted(
                roi[mask_bool],
                1 - a_mask_alpha,
                mask_roi_color,
                a_mask_alpha,
                0,
            )
            img_np[y1:y2, x1:x2] = roi
            if isinstance(img, np.ndarray):
                img = img_np
            else:
                img.data = img_np
    if isinstance(a_image, Frame2D):
        return Frame2D(img, **{k: v for k, v in a_image.__dict__.items() if k != 'data'})
    elif isinstance(a_image, Image2D):
        return Image2D(img, **{k: v for k, v in a_image.__dict__.items() if k != 'data'})
    else:
        return img
