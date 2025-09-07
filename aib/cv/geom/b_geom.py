"""Computer Vision - Geometry - Base Geometry Module

This module provides a base class for geometric operations and representations.

Classes:
    BaseGeom: A base class for geometric operations and representations.
"""

from dataclasses import dataclass

from src.utils.cnt.b_data import BaseData


@dataclass(frozen=True)
class BaseGeom(BaseData):
    """Base Geometry Class

    A frozen data class that provides a structure for holding geometric data.
    It extends BaseData to include common geometric properties.
    """
