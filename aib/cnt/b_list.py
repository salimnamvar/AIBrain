"""Data Container - Base List Container Utilities

This module provides a base list container class that extends the functionality of UserList.

Classes:
    BaseList:
        A list-like container with additional features such as size limits, named identification,
        and enhanced item access methods.
"""

import copy
from collections import UserList
from typing import (
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Self,
    Sequence,
    SupportsIndex,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import numpy.typing as npt

from src.utils.misc.type_check import is_bool, is_int

_T = TypeVar("_T")


class BaseList(UserList[_T], Generic[_T]):
    """A base list container that extends UserList with additional features.

    Attributes:
        data (List[_T]): The underlying list data.
        name (str): The name of the BaseList for identification.
        _max_size (Optional[int]): The maximum size of the BaseList. If set, it limits the number of items it can hold.
    """

    data: list[_T]

    def __init__(
        self,
        a_iterable: Optional[Iterable[_T]] = None,
        a_max_size: Optional[int] = None,
        a_name: str = "BaseList",
    ):
        """Initialize the BaseList

        Args:
            a_iterable (Optional[Iterable[_T]]): An optional iterable to initialize the list.
            a_max_size (Optional[int]): An optional maximum size for the list.
            a_name (str): The name of the list.
        """
        super().__init__(a_iterable or [])
        self._name: str = a_name
        self._max_size: Optional[int] = a_max_size
        self._enforce_size_limit()

    @property
    def name(self) -> str:
        """Get the name of the list.

        Returns:
            str: The name of the BaseList.
        """
        return self._name

    @overload
    def __getitem__(self, a_item: int) -> _T: ...

    @overload
    def __getitem__(self, a_item: np.integer) -> _T: ...

    @overload
    def __getitem__(self, a_item: np.bool_) -> _T: ...

    @overload
    def __getitem__(self, a_item: slice) -> Self: ...

    @overload
    def __getitem__(self, a_item: List[int]) -> Self: ...

    @overload
    def __getitem__(self, a_item: List[bool]) -> Self: ...

    @overload
    def __getitem__(self, a_item: Tuple[int, ...]) -> Self: ...

    @overload
    def __getitem__(self, a_item: Sequence[bool]) -> Self: ...

    @overload
    def __getitem__(self, a_item: Sequence[int]) -> Self: ...

    @overload
    def __getitem__(self, a_item: npt.NDArray[np.integer]) -> Self: ...

    @overload
    def __getitem__(self, a_item: npt.NDArray[np.bool_]) -> Self: ...

    @overload
    def __getitem__(self, a_item: SupportsIndex) -> _T: ...

    def __getitem__(
        self,
        a_item: Union[
            SupportsIndex,
            int,
            np.integer,
            np.bool_,
            bool,
            slice,
            Sequence[int | bool],
            List[int | bool],
            Tuple[int | bool, ...],
            npt.NDArray[np.integer | np.bool_],
        ],
    ) -> _T | Self:
        """Get an item or a slice of items from the list.

        Args:
            a_item: The index, slice, or sequence of indices to access the list.

        Returns:
            _T | Self: The item at the specified index or a new BaseList containing the sliced items.
        """
        if isinstance(a_item, (int, slice)):
            return super().__getitem__(a_item)
        if isinstance(a_item, (np.integer, np.signedinteger)):
            a_item = int(a_item.item())
            return super().__getitem__(a_item)
        if isinstance(a_item, np.bool_):
            a_item = bool(a_item)
            return super().__getitem__(a_item)
        if isinstance(a_item, np.ndarray):
            a_item = a_item.tolist()
        if isinstance(a_item, (list, tuple)):
            if is_int(a_item):
                return self.__class__([self._data[i] for i in a_item])
            if is_bool(a_item):
                if len(a_item) != len(self):
                    raise ValueError("Boolean sequence length must match the list length.")
                return self.__class__([item for item, keep in zip(self.data, a_item) if keep])
            raise TypeError("All indices must be either integers or booleans.")
        raise TypeError("Index must be an int, slice, list, tuple, or numpy array of integers or booleans.")

    def append(self, item: _T, a_removal_strategy: Literal['first', 'last'] = "first") -> None:
        """Append an item to the list, enforcing the maximum size limit.

        Args:
            item (_T): The item to append.
            a_removal_strategy (Literal['first', 'last']):
                Strategy for removing items when max_size is reached. Options are `first` or `last`.
        """
        self._clip(a_removal_strategy=a_removal_strategy)
        super().append(item)

    def _clip(self, a_removal_strategy: Literal['first', 'last'] = "first") -> None:
        """Clip the list to the maximum size, removing items based on the specified strategy.

        Args:
            a_removal_strategy (Literal['first', 'last']):
                Strategy for removing items when max_size is reached. Options are `first` or `last`.

        Raises:
            ValueError: If the removal strategy is not recognized.
        """
        if self._max_size is not None and len(self) >= self._max_size:
            if a_removal_strategy.lower() == "first":
                self.pop(0)
            elif a_removal_strategy.lower() == "last":
                self.pop()
            else:
                raise ValueError("Invalid removal strategy. Use 'first' or 'last'.")

    def _enforce_size_limit(self):
        """Enforce the maximum size limit of the list.

        If the list exceeds the maximum size, it trims the oldest items.
        """
        if self._max_size is not None and len(self) > self._max_size:
            self._data = self.data[-self._max_size :]

    def __iter__(self) -> Iterator[_T]:
        """Return an iterator over the items in the BaseList.

        Returns:
            Iterator[_T]: An iterator over the items in the BaseList.
        """
        return iter(self.data)

    def filter(self, a_condition: Callable[[_T], bool]) -> Self:
        """Filter the list based on a condition.

        Args:
            a_condition (Callable[[_T], bool]):
                A function that takes an item and returns True if the item should be included.

        Returns:
            Self: A new BaseList containing only the items that satisfy the condition.
        """
        return self.__class__(
            a_iterable=[item for item in self.data if a_condition(item)], a_max_size=self._max_size, a_name=self._name
        )

    def copy(self) -> Self:
        """Create a deep copy of the list.

        Returns:
            Self: A new instance of BaseList with the same data.
        """
        return copy.deepcopy(self)
