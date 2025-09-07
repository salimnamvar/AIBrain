"""Unmatched Individuals
"""

# region Imported Dependencies
import uuid
from typing import List
from aib.obj import BaseObjectList

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class UMTList(BaseObjectList[int]):
    def __init__(
        self,
        a_name: str = "Unmatched Target List",
        a_max_size: int = -1,
        a_items: List[int] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)


# TODO(doc): Complete the document of following class
class UMEList(BaseObjectList[uuid.UUID]):
    def __init__(
        self,
        a_name: str = "Unmatched Entity List",
        a_max_size: int = -1,
        a_items: List[uuid.UUID] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)
