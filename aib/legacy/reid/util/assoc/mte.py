"""Matched Target-Entity
"""

# region Imported Dependencies
import uuid
from typing import List
from aib.obj import ExtBaseObject, BaseObjectList

# endregion Imported Dependencies


# TODO(doc): Complete the document of following class
class MTE(ExtBaseObject):
    def __init__(
        self,
        a_tgt: int,
        a_ent: uuid.UUID,
        a_name: str = "Matched Target-Entity",
    ):
        super().__init__(a_name)
        self.tgt: int = a_tgt
        self.ent: uuid.UUID = a_ent

    def to_dict(self) -> dict:
        dic = {
            "name": self.name,
            "tgt": self.tgt,
            "ent": self.ent,
        }
        return dic


# TODO(doc): Complete the document of following class
class MTEList(BaseObjectList[MTE]):
    def __init__(
        self,
        a_name: str = "Matched Target-Entity List",
        a_max_size: int = -1,
        a_items: List[MTE] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)
