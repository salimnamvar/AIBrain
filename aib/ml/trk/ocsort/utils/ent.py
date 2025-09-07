"""Machine Learning - Object Tracking - OCSORT Entity Utilities

This module provides the EntityDict class, a specialized dictionary container for managing
entities in the OCSORT (Object-Centric Sorting) tracking framework.

Classes:
    EntityDict: A specialized dictionary for entity management in OCSORT tracking

Type Variables:
    _VT:
        Value type constrained to various 2D geometric objects (Box2D, BBox2D, SegBBox2D)
        with Point2D coordinates supporting both integer and float precision
"""

from typing import Dict, TypeVar

from aib.cnt.b_dict import BaseDict
from aib.cv.geom.box import AnyBox, FloatBox

BoxT = TypeVar("BoxT", bound=AnyBox, default=FloatBox)


class EntityDict(BaseDict[int, BoxT]):
    """Entity Dictionary Data Container

    This class serves as a specialized dictionary for managing entities in the OCSORT framework.

    Attributes:
        - data (Dict[int, BoxT]): The underlying dictionary storing entity data.
    """

    def __init__(
        self, a_dict: Dict[int, BoxT] | None = None, a_max_size: int | None = None, a_name: str = "EntityDict"
    ):
        """Initialize EntityDict

        Args:
            a_dict (Dict[int, BoxT] | None): Initial dictionary to populate the EntityDict.
            a_max_size (int | None): Maximum size of the EntityDict.
            a_name (str): Name of the EntityDict.
        """
        super().__init__(a_dict=a_dict, a_max_size=a_max_size, a_name=a_name)
