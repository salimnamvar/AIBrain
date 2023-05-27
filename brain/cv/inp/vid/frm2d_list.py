""" 2D Frame List

    This python file contains the 2D Frame List class object.
"""


# region Imported Dependencies
from typing import List, Union
from brain.cv.inp.vid.frm2d import Frame2D
# endregion Imported Dependencies


class Frame2DList:
    def __init__(self, a_items: Union[List['Frame2DList'], List[Frame2D], 'Frame2DList', Frame2D] = None) -> None:
        # If the `a_items` is not empty
        if a_items is not None:
            if isinstance(a_items, (list, tuple, Frame2D, self.__class__)):
                # If `a_items` is a a_items_type object
                if isinstance(a_items, Frame2D):
                    self._items = [a_items]

                # If the `a_items` is a self.__class__ object
                elif isinstance(a_items, self.__class__):
                    self._items = a_items.items

                # If the `a_items` is a list or tuple
                elif isinstance(a_items, (list, tuple)):

                    # If all the items in the list-tuple are a_items_type objects
                    if all(isinstance(x, Frame2D) for x in a_items):
                        self._items = [item for item in a_items]

                    # If all the items in the list-tuple are self.__class__ objects
                    elif all(isinstance(x, self.__class__) for x in a_items):
                        for objects_list in a_items:
                            for item in objects_list.items:
                                self._items.append(item)

                    else:
                        raise TypeError("The element of a_items is of invalid type. They must be all,"
                                        + Frame2D.__name__ + " or " + self.__class__.__name__ + ")")
            else:
                raise TypeError('The `a_items` should be a list or tuple of [' + Frame2D.__name__ + "," +
                                self.__class__.__name__ + '] objects')

        # If the list is empty, initialize the list
        elif a_items is None or len(a_items) == 0:
            self._items = []
        else:
            raise TypeError('The `a_items` should be a list or tuple of [' + Frame2D.__name__ + "," +
                            self.__class__.__name__ + '] objects')

    def append(self, a_item: Frame2D) -> None:
        self._items.append(a_item)

    @property
    def items(self) -> List[Frame2D]:
        return self._items

    def __getitem__(self, a_index: int) -> Frame2D:
        return self._items[a_index]

    def __len__(self) -> int:
        return len(self._items)

