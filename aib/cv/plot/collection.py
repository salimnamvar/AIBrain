"""Computer Vision - Unified Interface for Geometric and Plotting Utilities

This module provides a unified interface for plotting various geometric objects on images.

Functions:
    - plot_geom:
        Plot a geometric object (Point2D, KeyPoint2D, Line2D, Pose2D, Contour2D, Box2D, BBox2D, SegBBox2D) on an image.
    - plot_geoms:
        Plot a nested structure of geometric objects on an image.
"""

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, Union, cast

import numpy.typing as npt

from src.utils.cv.geom.b_geom import BaseGeom
from src.utils.cv.geom.box.bbox2d import BBox2D
from src.utils.cv.geom.box.box2d import Box2D
from src.utils.cv.geom.box.sbbox2d import SegBBox2D
from src.utils.cv.geom.contour2d import Contour2D
from src.utils.cv.geom.line2d import Line2D
from src.utils.cv.geom.point.kpoint2d import KeyPoint2D
from src.utils.cv.geom.point.point2d import Point2D
from src.utils.cv.geom.pose.pose2d import Pose2D
from src.utils.cv.img.frame import Frame2D
from src.utils.cv.img.image import Image2D

from .geom import (
    plot_bbox2d,
    plot_box2d,
    plot_contour2d,
    plot_keypoint2d,
    plot_line2d,
    plot_point2d,
    plot_pose2d,
    plot_segbbox2d,
)

T = TypeVar("T", bound=Union[int, float])
PT = TypeVar("PT", bound=Union[Point2D[int], Point2D[float]])
KPT = TypeVar("KPT", bound=Union[KeyPoint2D[int], KeyPoint2D[float]])
PKT = TypeVar("PKT", bound=Union[Point2D[int], Point2D[float], KeyPoint2D[int], KeyPoint2D[float]])


def plot_geom(
    a_image: Image2D | Frame2D | npt.NDArray[Any],
    a_geom: Union[
        BaseGeom,
        Point2D[T],
        KeyPoint2D[T],
        Line2D[PKT],
        Pose2D[KPT],
        Contour2D[PT],
        Box2D[PT],
        BBox2D[PT],
        SegBBox2D[PT],
    ],
    **kwargs: Any,
) -> Image2D | Frame2D | npt.NDArray[Any]:
    """
    Unified plotter for any geometric object.

    Args:
        a_image: The image to draw on.
        a_geom: The geometric object to plot (Point2D, KeyPoint2D, Line2D, Pose2D, Contour2D, Box2D, BBox2D, SegBBox2D).
        **kwargs: All plotting factors for the underlying plotter.

    Returns:
        Image2D | Frame2D | np.ndarray: The image with the geometry drawn.
    """
    if isinstance(a_geom, Point2D):
        return plot_point2d(a_image, cast(Point2D[T], a_geom), **kwargs)
    if isinstance(a_geom, KeyPoint2D):
        return plot_keypoint2d(a_image, cast(KeyPoint2D[T], a_geom), **kwargs)
    if isinstance(a_geom, Line2D):
        return plot_line2d(a_image, cast(Line2D[PKT], a_geom), **kwargs)
    if isinstance(a_geom, Pose2D):
        return plot_pose2d(a_image, a_geom, **kwargs)
    if isinstance(a_geom, Contour2D):
        return plot_contour2d(a_image, a_geom, **kwargs)
    if isinstance(a_geom, SegBBox2D):
        return plot_segbbox2d(a_image, cast(SegBBox2D[PT], a_geom), **kwargs)
    if isinstance(a_geom, BBox2D):
        return plot_bbox2d(a_image, cast(BBox2D[PT], a_geom), **kwargs)
    if isinstance(a_geom, Box2D):
        return plot_box2d(a_image, cast(Box2D[PT], a_geom), **kwargs)
    raise TypeError(f"Unsupported geometry type: {type(a_geom)}")


def plot_geoms(
    a_image: Image2D | Frame2D | npt.NDArray[Any],
    a_geoms: Any,
    a_use_dict_keys: bool = True,
    **kwargs: Any,
) -> Image2D | Frame2D | npt.NDArray[Any]:
    """
    Plot any nested structure (list, tuple, dict, etc.) of geometric objects on an image.
    If a dict is passed and use_dict_keys_as_labels is True, its key is used as the label for the geometry.

    Args:
        a_image: The image to draw on.
        a_geoms: A geometric object, or a (possibly nested) iterable/dict of geometric objects.
        a_use_dict_keys: Whether to use dict keys as labels.
        **kwargs: All plotting factors for the underlying plotter.

    Returns:
        Image2D | Frame2D | np.ndarray: The image with all geometries drawn.
    """

    def _plot_recursive(img: Image2D | Frame2D | npt.NDArray[Any], geoms: Any) -> Image2D | Frame2D | npt.NDArray[Any]:
        if isinstance(geoms, Mapping):
            for gkey, geom in geoms.items():
                plot_kwargs = dict(kwargs)
                if a_use_dict_keys and "a_label" not in plot_kwargs:
                    plot_kwargs["a_label"] = f"{gkey}"
                img = (
                    _plot_recursive(img, geom)
                    if isinstance(geom, (Mapping, Sequence)) and not isinstance(geom, (str, bytes))
                    else plot_geom(img, geom, **plot_kwargs)
                )
            return img
        if isinstance(geoms, Sequence) and not isinstance(geoms, (str, bytes)):
            for geom in geoms:
                img = _plot_recursive(img, geom)
            return img
        return plot_geom(img, geoms, **kwargs)

    return _plot_recursive(a_image, a_geoms)
