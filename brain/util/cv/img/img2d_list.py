"""2D Image List Module

    This Python file contains the definition of the :class:`Image2DList` class, an object class that stores a
    collection of :class:`Image2D` objects.
"""

# region Imported Dependencies
from typing import List

import numpy as np

from brain.util.cv.img import Image2D
from brain.util.obj import BaseObjectList


# endregion Imported Dependencies


class Image2DList(BaseObjectList[Image2D]):
    """Image2D List

    The Image2DList class is based on the :class:`ObjectList` class and serves as a container for a collection of
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
        a_name: str = "Image2DList",
        a_max_size: int = -1,
        a_items: List[Image2D] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)

    # TODO(doc): Complete the document of following method
    def to_numpy(self) -> np.ndarray:
        if len(self.items):
            arr = np.stack([img.data for img in self.items], axis=0)
        else:
            arr = np.empty(shape=(0,))

        return arr
