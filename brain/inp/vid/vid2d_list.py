"""2D Video List

This python file contains the 2D Video List class object.
"""


# region Imported Dependencies
import uuid
from typing import List, Tuple, Union
from brain.inp.vid.frm2d_list import Frame2DList
from brain.inp.vid.vid2d import Video2D
# endregion Imported Dependencies


class Video2DList:
    def __init__(self, a_items: Union[List['Video2DList'], List[Video2D], 'Video2DList', Video2D] = None) -> None:
        # If the `a_items` is not empty
        if a_items is not None:
            if isinstance(a_items, (list, tuple, Video2D, self.__class__)):
                # If `a_items` is a a_items_type object
                if isinstance(a_items, Video2D):
                    self._items = [a_items]

                # If the `a_items` is a self.__class__ object
                elif isinstance(a_items, self.__class__):
                    self._items = a_items.items

                # If the `a_items` is a list or tuple
                elif isinstance(a_items, (list, tuple)):

                    # If all the items in the list-tuple are a_items_type objects
                    if all(isinstance(x, Video2D) for x in a_items):
                        self._items = [item for item in a_items]

                    # If all the items in the list-tuple are self.__class__ objects
                    elif all(isinstance(x, self.__class__) for x in a_items):
                        for objects_list in a_items:
                            for item in objects_list.items:
                                self._items.append(item)

                    else:
                        raise TypeError("The element of a_items is of invalid type. They must be all,"
                                        + Video2D.__name__ + " or " + self.__class__.__name__ + ")")
            else:
                raise TypeError('The `a_items` should be a list or tuple of [' + Video2D.__name__ + "," +
                                self.__class__.__name__ + '] objects')

        # If the list is empty, initialize the list
        elif a_items is None or len(a_items) == 0:
            self._items = []
        else:
            raise TypeError('The `a_items` should be a list or tuple of [' + Video2D.__name__ + "," +
                            self.__class__.__name__ + '] objects')

    def append(self, a_item: Video2D) -> None:
        self._items.append(a_item)

    @property
    def items(self) -> List[Video2D]:
        return self._items

    def __getitem__(self, a_index: int) -> Video2D:
        return self._items[a_index]

    def __len__(self) -> int:
        return len(self._items)

    def read(self) -> Tuple[bool, Frame2DList]:
        if len(self._items) == 0:
            raise IndexError('The Video2DList is empty.')

        frames: Frame2DList = Frame2DList()
        frames_ret = False
        for vid in self._items:
            ret, frame = vid.read()
            if ret:
                frames.append(frame)
                frames_ret = True
        return frames_ret, frames

    def get_ids(self) -> List[uuid.UUID]:
        return [item.id for item in self._items]
