"""Re-identification State Modules
"""

# region Imported Dependencies
from dataclasses import dataclass
from typing import TypeVar, Dict, Union, List, Tuple, Type, Optional

from brain.util.misc import Time
from brain.util.obj import (
    BaseObjectTable,
    BaseTableRow,
    TypeTableColKey,
    TypeTableColValue,
    BaseObjectList,
    BaseTableRowList,
    BaseObjectDict,
)
from brain.util.obj.list import TypeBaseObjectList

# endregion Imported Dependencies

# TODO(doc): Complete the document of following type
TypeReidState = TypeVar("TypeReidState", bound="ReidState")
# TODO(doc): Complete the document of following type
TypeReidStateList = TypeVar("TypeReidStateList", bound="ReidStateList")
# TODO(doc): Complete the document of following type
TypeReidStateTable = TypeVar("TypeReidStateTable", bound="ReidStateTable")


@dataclass
class ReidState(BaseTableRow):
    time: Optional[Time] = None


# TODO(doc): Complete the document of following class
class ReidStateList(BaseTableRowList, BaseObjectList[ReidState]):
    def __init__(
        self,
        a_name: str = "ReidStateList",
        a_max_size: int = -1,
        a_items: Union[ReidState, List[ReidState], "ReidStateList"] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)


class ReidStateTable(BaseObjectTable[TypeReidState]):
    def __init__(
        self,
        a_data: Union[
            TypeReidState,
            List[TypeReidState],
            Tuple[TypeReidState],
            TypeBaseObjectList,
            TypeReidStateList,
            List[TypeTableColValue],
            Tuple[TypeTableColValue],
            Dict[TypeTableColKey, TypeTableColValue],
            List[Dict[TypeTableColKey, TypeTableColValue]],
            Tuple[Dict[TypeTableColKey, TypeTableColValue]],
            TypeReidStateTable,
            BaseObjectDict[TypeTableColKey, TypeBaseObjectList],
        ] = None,
        a_max_cols: int = -1,
        a_max_rows: int = -1,
        a_name: str = "ReidStateTable",
        a_row_type: Type[ReidState] = ReidState,
    ):
        super().__init__(
            a_data=a_data,
            a_max_cols=a_max_cols,
            a_max_rows=a_max_rows,
            a_name=a_name,
            a_row_type=a_row_type,
        )

    @property
    def first_state(self) -> TypeReidState:
        return self.first_row

    @property
    def last_state(self) -> TypeReidState:
        return self.last_row

    @property
    def time(self) -> BaseObjectList[float]:
        return self["time"]
