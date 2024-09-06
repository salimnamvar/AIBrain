"""Base Object Table
"""

# region Imported Dependencies
from dataclasses import dataclass, fields, asdict
import pprint
from copy import deepcopy
from typing import TypeVar, Dict, Generic, Tuple, Sequence, List, Union, get_args
import numpy as np
import numpy.typing as npt
from brain.obj import BaseObjectDict, BaseObjectList
from brain.obj.dict import TypeBaseObjectDict
from brain.obj.list import TypeBaseObjectList

# endregion Imported Dependencies


# TODO(doc): Complete the document of following type
TypeTableColKey = TypeVar("TypeTableColKey")
# TODO(doc): Complete the document of following type
TypeTableColValue = TypeVar("TypeTableColValue")
# TODO(doc): Complete the document of following type
TypeBaseTableRow = TypeVar("TypeBaseTableRow", bound="BaseTableRow")
# TODO(doc): Complete the document of following type
TypeBaseTableRowList = TypeVar("TypeBaseTableRowList", bound="BaseTableRowList")
# TODO(doc): Complete the document of following type
TypeBaseObjectTable = TypeVar("TypeBaseObjectTable", bound="BaseObjectTable")


# TODO(doc): Complete the document of following class
@dataclass
class BaseTableRow:
    NotImplementedError("Subclasses must implement `fields`")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_list(self) -> list:
        return list(self.to_dict().values())


# TODO(doc): Complete the document of following class
class BaseTableRowList(BaseObjectList[BaseTableRow]):
    def __init__(
        self,
        a_name: str = "BaseRowList",
        a_max_size: int = -1,
        a_items: Union[BaseTableRow, List[BaseTableRow], "BaseTableRowList"] = None,
    ):
        super().__init__(a_name=a_name, a_max_size=a_max_size, a_items=a_items)


# TODO(doc): Complete the document of following class
class BaseObjectTable(Generic[TypeBaseTableRow]):
    def __init__(
        self,
        a_data: Union[
            TypeBaseTableRow,
            List[TypeBaseTableRow],
            Tuple[TypeBaseTableRow],
            TypeBaseObjectList,
            TypeBaseTableRowList,
            List[TypeTableColValue],
            Tuple[TypeTableColValue],
            Dict[TypeTableColKey, TypeTableColValue],
            List[Dict[TypeTableColKey, TypeTableColValue]],
            Tuple[Dict[TypeTableColKey, TypeTableColValue]],
            TypeBaseObjectTable,
            BaseObjectDict[TypeTableColKey, TypeBaseObjectList],
        ] = None,
        a_max_cols: int = -1,
        a_max_rows: int = -1,
        a_name: str = "BaseObjectTable",
        a_row_type: type[BaseTableRow] = BaseTableRow,
    ):
        self.name: str = a_name
        self.max_cols: int = a_max_cols
        self.max_rows: int = a_max_rows
        self._row_type: type[BaseTableRow] = a_row_type
        self.__init_table(a_row_type=a_row_type)
        if a_data is not None:
            self.append(a_data=a_data)

    def __init_table(self, a_row_type: type[BaseTableRow], a_max_cols: int = -1):
        self._data: BaseObjectDict[TypeTableColKey, BaseObjectList] = BaseObjectDict[TypeTableColKey, BaseObjectList](
            a_name="Table", a_max_size=a_max_cols
        )
        for field in fields(a_row_type):
            col_key = field.name
            # Check if the field type is Optional
            if hasattr(field.type, "__origin__") and field.type.__origin__ is Union:
                col_type = get_args(field.type)
                if type(None) in col_type:
                    col_type = col_type[0] if col_type[1] is type(None) else col_type[1]
            else:
                col_type = field.type
            self._data.append(
                a_key=col_key,
                a_value=BaseObjectList[col_type](
                    a_items=[], a_name="TableColumn", a_max_size=self.max_rows, a_item_type=col_type
                ),
            )

    def __getitem__(
        self,
        a_item: Union[
            TypeTableColKey,
            List[TypeTableColKey],
            Tuple[TypeTableColKey],
            npt.NDArray[TypeTableColKey],
            Sequence[TypeTableColKey],
            int,
            List[int],
            Tuple[int],
            npt.NDArray[np.integer],
            Sequence[int],
            slice,
            List[bool],
            Tuple[bool],
            npt.NDArray[bool],
            Sequence[bool],
        ],
    ) -> Union[TypeBaseObjectTable, TypeBaseObjectList, TypeBaseTableRow]:
        if isinstance(a_item, (list, tuple, np.ndarray)):
            # Handle table-wise selection
            if len(a_item) == 2:
                col_item, row_item = a_item
                data = BaseObjectDict[TypeTableColKey, TypeBaseObjectList](a_name="SubTable")
                # Select columns
                selected_cols = self._data[col_item]
                # Select rows
                for col_key, row_values in selected_cols.items():
                    data.append(a_key=col_key, a_value=row_values[row_item])
                if not isinstance(data[0], BaseObjectList) or (
                    isinstance(data[0], BaseObjectList) and len(data[0]) == 1
                ):
                    # Handle single row
                    row = {}
                    for key, col in data.items():
                        row[key] = col[-1] if isinstance(col, BaseObjectList) else col
                    return self._row_type(**row)
                else:
                    # Handle multiple rows
                    return self.__class__(a_data=data)
            else:
                raise TypeError("Tuple index must be of length 2.")
        else:
            # Handle column-wise selection
            col_item = a_item
            data = BaseObjectDict[TypeTableColKey, TypeBaseObjectList](a_name="SubTable")
            # Select columns
            selected_cols = self._data[col_item]
            if isinstance(selected_cols, BaseObjectDict):
                # Handle multiple columns
                # Select rows
                for col_key, row_values in selected_cols:
                    data.append(a_key=col_key, a_value=row_values)
                return self.__class__(a_data=data)
            elif isinstance(selected_cols, BaseObjectList):
                # Handle single column
                return selected_cols
            else:
                raise TypeError("No index is entered.")

    def append(
        self,
        a_data: Union[
            TypeBaseTableRow,
            List[TypeBaseTableRow],
            Tuple[TypeBaseTableRow],
            TypeBaseObjectList,
            TypeBaseTableRowList,
            List[TypeTableColValue],
            Tuple[TypeTableColValue],
            Dict[TypeTableColKey, TypeTableColValue],
            List[Dict[TypeTableColKey, TypeTableColValue]],
            Tuple[Dict[TypeTableColKey, TypeTableColValue]],
            TypeBaseObjectTable,
            BaseObjectDict[TypeTableColKey, TypeBaseObjectList],
        ],
        a_removal_strategy: str = "first",
    ) -> None:
        if isinstance(a_data, (BaseTableRow, dict)):
            # Handle row/dict
            self._append_row(a_data=a_data, a_removal_strategy=a_removal_strategy)
        elif isinstance(a_data, (list, tuple, BaseTableRowList, BaseObjectList)):
            if all(isinstance(item, (BaseTableRow, dict)) for item in a_data):
                # Handle list/tuple of rows/dict
                for row in a_data:
                    self._append_row(a_data=row, a_removal_strategy=a_removal_strategy)
            else:
                # Handle list/tuple of values
                self._append_row(a_data=a_data, a_removal_strategy=a_removal_strategy)
        elif isinstance(a_data, (BaseObjectTable, BaseObjectDict)):
            # Handle table
            if isinstance(a_data, BaseObjectTable):
                items = a_data.data.items()
            elif isinstance(a_data, BaseObjectDict):
                items = a_data.items()
            else:
                raise ValueError(f"BaseObjectTable or BaseObjectDict are valid, but {type(a_data)} is entered.")
            for key, col in items:
                self._append_col(a_key=key, a_col=col)

    def _append_row(
        self,
        a_data: Union[
            TypeBaseTableRow,
            Dict[TypeTableColKey, TypeTableColValue],
            List[TypeTableColValue],
            Tuple[TypeTableColValue],
            TypeBaseObjectList,
        ],
        a_removal_strategy: str = "first",
    ):
        self._clip_rows(a_removal_strategy=a_removal_strategy)

        if isinstance(a_data, BaseTableRow):
            if len(fields(a_data)) != self.cols:
                raise ValueError(
                    f"Length of the row ({len(fields(a_data))}) must match length of row in the table ({self.cols})"
                )
            for field in fields(a_data):
                key = field.name
                value = getattr(a_data, key)
                self.data[key].append(a_item=value, a_removal_strategy=a_removal_strategy, a_merge=False)

        elif isinstance(a_data, dict):
            if len(a_data) != self.cols:
                raise ValueError(
                    f"Length of the row ({len(a_data)}) must match length of row in the table ({self.cols})"
                )
            for key, value in a_data.items():
                self.data[key].append(a_item=value, a_removal_strategy=a_removal_strategy, a_merge=False)

        elif isinstance(a_data, (list, tuple, BaseObjectList)):
            for i, value in enumerate(a_data):
                self.data[self.col_keys[i]].append(a_item=value, a_removal_strategy=a_removal_strategy, a_merge=False)

        else:
            raise ValueError(f"Row data must be `list`, `Row`, or `dict`, but it is passed as `{type(a_data)}`")

    def _append_col(
        self,
        a_key: TypeTableColKey,
        a_col: Union[TypeBaseObjectList, List[TypeTableColValue], Tuple[TypeTableColValue]],
        a_removal_strategy: str = "first",
    ):
        if a_key not in self.col_keys:
            self._clip_cols(a_removal_strategy=a_removal_strategy)

        if len(a_col) != self.rows:
            raise ValueError(f"The column length `{len(a_col)}` must match the row length of table `{self.rows}`.")

        self.data[a_key] = a_col

    def _clip_cols(self, a_removal_strategy: str = "first") -> None:
        if self.max_cols != -1 and self.cols >= self.max_cols:
            if a_removal_strategy.lower() == "first":
                self.data.pop_first()
            elif a_removal_strategy.lower() == "last":
                self.data.pop_last()
            else:
                raise ValueError("Invalid removal strategy. Use 'first' or 'last'.")

    def _clip_rows(self, a_removal_strategy: str = "first"):
        if self.max_rows != -1 and self.rows >= self.max_rows:
            if a_removal_strategy.lower() == "first":
                self.pop_row(0)
            elif a_removal_strategy.lower() == "last":
                self.pop_row(-1)
            else:
                raise ValueError("Invalid removal strategy. Use 'first' or 'last'.")

    def pop_row(self, a_index: int = -1) -> None:
        for col_key, col in self.data.items():
            col.pop(a_index=a_index)

    @property
    def cols(self) -> int:
        return len(self.data)

    @property
    def col_keys(self) -> List[TypeTableColKey]:
        return list(self.data.keys())

    @property
    def col_values(self) -> List[TypeTableColValue]:
        return list(self.data.values())

    @property
    def rows(self) -> int:
        return len(self.data[0]) if self.cols > 0 else 0

    def to_dict(self) -> TypeBaseObjectDict:
        return self._data

    def to_str(self) -> str:
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        return self.to_str()

    @property
    def data(self) -> BaseObjectDict[TypeTableColKey, TypeBaseObjectList]:
        return self._data

    def copy(self: TypeBaseObjectTable) -> TypeBaseObjectTable:
        return deepcopy(self)

    def __len__(self) -> int:
        return self.rows

    @property
    def first_row(self) -> TypeBaseTableRow:
        return self[:, 0]

    @property
    def last_row(self) -> TypeBaseTableRow:
        return self[:, -1]

    def clear(self) -> None:
        for col_key, col in self.data.items():
            col.clear()
