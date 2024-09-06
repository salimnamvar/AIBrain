""" 2D Frame List

    This Python file contains the definition of the :class:`Frame2DList` class, which serves as a container for a
    collection of 2D frame objects (:class:`Frame2D`).
"""

# region Imported Dependencies
from typing import List

from brain.cv.vid import Frame2D
from brain.obj import BaseObjectList


# endregion Imported Dependencies


class Frame2DList(BaseObjectList[Frame2D]):
    """Frame2D List

    The Frame2D List class is based on the :class:`ObjectList` class and serves as a container for a collection of
    :class:`Image2D` objects.

    Attributes:
        name (str, optional):
            A string specifying the name of the Image2DList (default is 'Image2DList').
        max_size (int, optional):
            An integer representing the maximum size of the list (default is -1, indicating no size limit).
        items (List[Image2D], optional):
            A list of Image2D objects to initialize the Image2DList (default is None).
    """

    def __init__(
        self,
        a_name: str = "Frame2DList",
        a_max_size: int = -1,
        a_items: List[Frame2D] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)
