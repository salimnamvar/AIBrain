"""Data Container - Base Dictionary Container Utilities

This module provides a base dictionary container class that extends the functionality of UserDict.

Classes:
    BaseDict:
        A dictionary-like container with additional features such as size limits, named identification,
        and enhanced item access methods.
"""

import copy
from collections import UserDict
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Self,
    SupportsIndex,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class BaseDict(UserDict[_KT, _VT]):
    """Base Dictionary Container

    This class extends UserDict to provide a dictionary-like container with additional features such as size limits,
    named identification, and enhanced item access methods. It supports clipping to a maximum size, filtering
    items based on conditions, and accessing items by key, index, or slice.

    Attributes:
        data (Dict[_KT, _VT]): The underlying dictionary data.
        name (str): The name of the BaseDict for identification.
        _max_size (Optional[int]): The maximum size of the BaseDict. If set, it limits the number of items it can hold.
    """

    _MISSING = object()
    data: Dict[_KT, _VT]

    def __init__(
        self, a_dict: Optional[Dict[_KT, _VT]] = None, a_max_size: Optional[int] = None, a_name: str = "BaseDict"
    ):
        """Initialize the BaseDict with an optional dictionary, maximum size, and name.

        Args:
            a_dict: Optional initial dictionary to populate the BaseDict.
            a_max_size: Optional maximum size for the BaseDict. If set, it will limit the number of items it can hold.
            a_name: Name of the BaseDict for identification.
        """
        self._name: str = a_name
        self._max_size: Optional[int] = a_max_size
        super().__init__(a_dict or {})
        self._enforce_size_limit()

    @property
    def name(self) -> str:
        """Get the name of the BaseDict.

        Returns:
            str: The name of the BaseDict.
        """
        return self._name

    def _clip(self, a_removal_strategy: Literal["first", "last"] = "first") -> None:
        """Clip the dictionary to the maximum size by removing items based on the removal strategy.

        Args:
            a_removal_strategy: Strategy for removing items when max_size is reached. Options are `first` or `last`.
        """
        if self._max_size is not None and len(self.data) >= self._max_size:
            if a_removal_strategy.lower() == "first":
                self.pop_first()
            elif a_removal_strategy.lower() == "last":
                self.pop_last()
            else:
                raise ValueError("Invalid removal strategy. Use 'first' or 'last'.")

    def pop_first(self) -> _VT:
        """Remove and return the first item in the dictionary.

        Raises:
            KeyError: If the dictionary is empty.

        Returns:
            _VT: The value of the first item removed from the dictionary.
        """
        if not self.data:
            raise KeyError(f"{self.name} is empty")
        first_key = next(iter(self.data))
        return self.data.pop(first_key)

    def pop_last(self) -> _VT:
        """Remove and return the last item in the dictionary.

        Raises:
            KeyError: If the dictionary is empty.

        Returns:
            _VT: The value of the last item removed from the dictionary.
        """
        if not self.data:
            raise KeyError(f"{self.name} is empty")
        last_key = next(reversed(self.data))
        return self.data.pop(last_key)

    def __iter__(self) -> Iterator[_KT]:
        """Return an iterator over the keys of the BaseDict.

        Returns:
            Iterator[_KT]: An iterator over the keys of the BaseDict.
        """
        return iter(self.data)

    def copy(self) -> Self:
        """Create a shallow copy of the BaseDict.

        Returns:
            Self: A new instance of BaseDict with the same data.
        """
        return copy.deepcopy(self)

    def filter(self, a_condition: Callable[[_VT], bool]) -> Self:
        """Filter the dictionary based on a condition function.

        Args:
            a_condition: A function that takes a value and returns True if it should be included.

        Returns:
            A new BaseDict containing only the items that satisfy the condition.
        """
        return self.__class__(
            a_dict={k: v for k, v in self.data.items() if a_condition(v)},
            a_max_size=self._max_size,
            a_name=self._name,
        )

    def _enforce_size_limit(self):
        """Ensure the dictionary does not exceed the maximum size.

        If the dictionary exceeds the maximum size, it trims the oldest items based on the removal strategy.
        """
        if self._max_size is not None and len(self.data) > self._max_size:
            trimmed = dict(list(self.data.items())[-self._max_size :])
            self.data.clear()
            self.data.update(trimmed)

    def __setitem__(self, a_key: _KT, a_value: _VT) -> None:
        """Set the value for a key in the dictionary.

        Args:
            a_key: The key to set.
            a_value: The value to associate with the key.
        """
        if a_key not in self.data and self._max_size is not None and len(self.data) >= self._max_size:
            self._clip(a_removal_strategy="first")
        self.data[a_key] = a_value

    @overload
    def get_by_key(self, a_key: _KT) -> _VT: ...
    @overload
    def get_by_key(self, a_key: _KT, a_default: _VT) -> _VT: ...
    @overload
    def get_by_key(self, a_key: List[_KT]) -> Self: ...
    @overload
    def get_by_key(self, a_key: List[_KT], a_default: Self) -> Self: ...
    @overload
    def get_by_key(self, a_key: Tuple[_KT, ...]) -> Self: ...
    @overload
    def get_by_key(self, a_key: Tuple[_KT, ...], a_default: Self) -> Self: ...
    @overload
    def get_by_key(self, a_key: npt.NDArray[Any]) -> Self: ...
    @overload
    def get_by_key(self, a_key: npt.NDArray[Any], a_default: Self) -> Self: ...

    def get_by_key(
        self, a_key: Union[_KT, List[_KT], Tuple[_KT, ...], npt.NDArray[Any]], a_default: Optional[Any] = _MISSING
    ) -> Union[_VT, Self, Any]:
        """Get items by dictionary key(s).

        Args:
            a_key (Union[_KT, List[_KT], Tuple[_KT, ...], npt.NDArray[Any]]):
                Single key or collection of keys to retrieve.
            a_default (Optional[Self | Any]):
                Default value to return if key(s) not found. If not provided, raises KeyError.

        Returns:
            Union[_VT, Self, Any]:
                Single value for single key, or new BaseDict for multiple keys, or default if not found.

        Raises:
            KeyError: If any key is not found and no default is provided.
        """
        try:
            # Handle numpy arrays
            if isinstance(a_key, np.ndarray):
                a_key = a_key.tolist()

            # Handle single key
            if not isinstance(a_key, (list, tuple)):
                if isinstance(a_key, bool):
                    raise KeyError(f"Boolean key '{a_key}' is not allowed for dictionary access.")
                return self.data[cast(_KT, a_key)]

            # Handle empty list/tuple
            if len(a_key) == 0:
                return self.__class__(
                    a_dict={},
                    a_max_size=self._max_size,
                    a_name=self._name,
                )

            # Handle multiple keys - collect found items
            found_items = {k: self.data[k] for k in a_key if k in self.data}

            # If NO keys were found and default is provided, return default
            if len(found_items) == 0 and a_default is not self._MISSING:
                return a_default

            # If NO keys were found and no default, raise error
            if len(found_items) == 0:
                missing_keys = [k for k in a_key if k not in self.data]
                raise KeyError(f"Key(s) {missing_keys} not found in dictionary.")

            # Return found items (even if partial)
            return self.__class__(
                a_dict=found_items,
                a_max_size=self._max_size,
                a_name=self._name,
            )

        except KeyError:
            if a_default is self._MISSING:
                raise
            return a_default

    @overload
    def get_by_index(self, a_index: int) -> _VT: ...
    @overload
    def get_by_index(self, a_index: int, a_default: _VT) -> _VT: ...
    @overload
    def get_by_index(self, a_index: SupportsIndex) -> _VT: ...
    @overload
    def get_by_index(self, a_index: SupportsIndex, a_default: _VT) -> _VT: ...
    @overload
    def get_by_index(self, a_index: slice) -> Self: ...
    @overload
    def get_by_index(self, a_index: slice, a_default: Self) -> Self: ...
    @overload
    def get_by_index(self, a_index: List[int]) -> Self: ...
    @overload
    def get_by_index(self, a_index: List[int], a_default: Self) -> Self: ...
    @overload
    def get_by_index(self, a_index: Tuple[int, ...]) -> Self: ...
    @overload
    def get_by_index(self, a_index: Tuple[int, ...], a_default: Self) -> Self: ...
    @overload
    def get_by_index(self, a_index: npt.NDArray[np.integer]) -> Self: ...
    @overload
    def get_by_index(self, a_index: npt.NDArray[np.integer], a_default: Self) -> Self: ...

    def get_by_index(
        self,
        a_index: int | SupportsIndex | slice | List[int] | Tuple[int, ...] | npt.NDArray[np.integer],
        a_default: Optional[Any] = _MISSING,
    ) -> _VT | Self | Any:
        """Get items by positional index/indices.

        Args:
            a_index (int | SupportsIndex | slice | List[int] | Tuple[int, ...] | npt.NDArray[np.integer]):
                Single index, slice, or collection of indices to retrieve.
            a_default (Optional[Self | Any]):
                Default value to return if index/indices out of bounds. If not provided, raises IndexError.

        Returns:
            _VT | Self | Any:
                Single value for single index, or new BaseDict for multiple indices/slice, or default if out of bounds.

        Raises:
            IndexError: If any index is out of bounds and no default is provided.
        """
        try:
            keys = list(self.data.keys())

            # Handle slice
            if isinstance(a_index, slice):
                sliced_keys = keys[a_index]
                if len(sliced_keys) == 0:
                    if a_default is not self._MISSING:
                        return a_default
                    return self.__class__(
                        a_dict={},
                        a_max_size=self._max_size,
                        a_name=self._name,
                    )
                return self.__class__(
                    a_dict={k: self.data[k] for k in sliced_keys},
                    a_max_size=self._max_size,
                    a_name=self._name,
                )

            # Handle numpy arrays
            if isinstance(a_index, np.ndarray):
                a_index = a_index.tolist()

            # Handle single index
            if not isinstance(a_index, (list, tuple)):
                index = cast(int, a_index)
                # Handle negative indices
                if index < 0:
                    index = len(keys) + index
                # Check bounds after handling negative indices
                if index < 0 or index >= len(keys):
                    raise IndexError(f"Index {index} out of range for dictionary with {len(keys)} items")
                return self.data[keys[index]]

            # Handle empty list/tuple
            if len(a_index) == 0:
                return self.__class__(
                    a_dict={},
                    a_max_size=self._max_size,
                    a_name=self._name,
                )

            # Handle multiple indices
            normalized_indices: List[int] = []
            for i in a_index:
                if i < 0:
                    i = len(keys) + i
                # Check bounds after handling negative indices
                if i < 0 or i >= len(keys):
                    raise IndexError(f"Index {i} out of range for dictionary with {len(keys)} items")
                normalized_indices.append(i)

            return self.__class__(
                a_dict={keys[i]: self.data[keys[i]] for i in normalized_indices},
                a_max_size=self._max_size,
                a_name=self._name,
            )

        except IndexError:
            if a_default is self._MISSING:
                raise
            return a_default

    @overload
    def get_by_mask(self, a_mask: List[bool]) -> Self: ...
    @overload
    def get_by_mask(self, a_mask: List[bool], a_default: Self) -> Self: ...
    @overload
    def get_by_mask(self, a_mask: Tuple[bool, ...]) -> Self: ...
    @overload
    def get_by_mask(self, a_mask: Tuple[bool, ...], a_default: Self) -> Self: ...
    @overload
    def get_by_mask(self, a_mask: npt.NDArray[np.bool_]) -> Self: ...
    @overload
    def get_by_mask(self, a_mask: npt.NDArray[np.bool_], a_default: Self) -> Self: ...

    def get_by_mask(
        self, a_mask: Union[List[bool], Tuple[bool, ...], npt.NDArray[np.bool_]], a_default: Any = _MISSING
    ) -> Union[Self, Any]:
        """Get items by boolean mask.

        Args:
            a_mask (List[bool] | Tuple[bool, ...] | npt.NDArray[np.bool_]): Boolean mask to filter items.
            a_default (Self | Any): Default value to return if mask is invalid. If not provided, raises ValueError.

        Returns:
            New BaseDict containing only items where mask is True, or default if mask is invalid.

        Raises:
            ValueError: If mask length doesn't match dictionary length and no default is provided.
        """
        try:
            # Handle numpy arrays
            if isinstance(a_mask, np.ndarray):
                a_mask = a_mask.tolist()

            # Validate mask length
            if len(a_mask) != len(self.data):
                raise ValueError("Boolean mask length must match dictionary length.")

            keys = list(self.data.keys())
            return self.__class__(
                a_dict={k: v for k, v, keep in zip(keys, self.data.values(), a_mask) if keep},
                a_max_size=self._max_size,
                a_name=self._name,
            )

        except ValueError:
            if a_default is self._MISSING:
                raise
            return a_default

    @overload
    def __getitem__(self, a_key: _KT) -> _VT: ...
    @overload
    def __getitem__(self, a_key: int) -> _VT: ...
    @overload
    def __getitem__(self, a_key: SupportsIndex) -> _VT: ...
    @overload
    def __getitem__(self, a_key: slice) -> Self: ...
    @overload
    def __getitem__(self, a_key: List[int]) -> Self: ...
    @overload
    def __getitem__(self, a_key: Tuple[int, ...]) -> Self: ...
    @overload
    def __getitem__(self, a_key: List[bool]) -> Self: ...
    @overload
    def __getitem__(self, a_key: Tuple[bool, ...]) -> Self: ...
    @overload
    def __getitem__(self, a_key: List[_KT]) -> Self: ...
    @overload
    def __getitem__(self, a_key: Tuple[_KT, ...]) -> Self: ...
    @overload
    def __getitem__(self, a_key: npt.NDArray[np.integer]) -> Self: ...
    @overload
    def __getitem__(self, a_key: npt.NDArray[np.bool_]) -> Self: ...
    @overload
    def __getitem__(self, a_key: npt.NDArray[Any]) -> Self: ...

    def __getitem__(
        self,
        a_key: (
            _KT
            | int
            | SupportsIndex
            | slice
            | List[int]
            | Tuple[int, ...]
            | List[bool]
            | Tuple[bool, ...]
            | List[_KT]
            | Tuple[_KT, ...]
            | npt.NDArray[np.integer]
            | npt.NDArray[np.bool_]
            | npt.NDArray[Any]
        ),
    ) -> _VT | Self:
        """Get item by key, index, or slice.

        Args:
            a_key (_KT | int | SupportsIndex | slice | List[int] | Tuple[int, ...] | List[bool] | Tuple[bool, ...] | List[_KT] | Tuple[_KT, ...] | npt.NDArray[np.integer] | npt.NDArray[np.bool_] | npt.NDArray[Any]):
                The key, index, or slice to retrieve.

        Returns:
            _VT | Self: The value associated with the key, or a new BaseDict if a slice or list of keys is provided.

        Raises:
            KeyError: If the key is not found.
            IndexError: If the index is out of range.
            ValueError: If a boolean mask's length does not match the dictionary length.
        """
        # Handle slice - delegate to get_by_index
        if isinstance(a_key, slice):
            return self.get_by_index(a_key)

        # Handle numpy arrays - convert and process
        if isinstance(a_key, np.ndarray):
            a_key = a_key.tolist()

        # Handle lists and tuples
        if isinstance(a_key, (list, tuple)):
            # Handle empty list/tuple
            if len(a_key) == 0:
                return self.get_by_index(a_key)

            # Type guard for boolean lists - delegate to get_by_mask
            if all(isinstance(x, bool) for x in a_key):
                return self.get_by_mask(a_key)

            # Type guard for integer index lists
            if all(isinstance(x, int) for x in a_key):
                int_key = a_key
                # Check if all integers exist as actual keys in the dictionary
                all_keys_exist = all(k in self.data for k in int_key)

                if all_keys_exist:
                    # Treat as list of keys - delegate to get_by_key
                    return self.get_by_key(int_key)
                else:
                    # Treat as list of indices - delegate to get_by_index
                    return self.get_by_index(int_key)

            # Handle as list of keys - delegate to get_by_key
            return self.get_by_key(a_key)

        # Handle integer index vs integer keys
        if isinstance(a_key, (int, np.integer)) and not isinstance(a_key, bool):
            # First check if it's an actual key in the dictionary
            try:
                return self.get_by_key(a_key)
            except KeyError:
                # If not found as a key, treat as an index - delegate to get_by_index
                return self.get_by_index(a_key)

        # Handle all other key types - delegate to get_by_key
        return self.get_by_key(a_key)

    @overload
    def get(self, a_key: _KT, a_default: _VT) -> _VT: ...
    @overload
    def get(self, a_key: _KT, a_default: None) -> _VT | None: ...
    @overload
    def get(self, a_key: int, a_default: _VT) -> _VT: ...
    @overload
    def get(self, a_key: int, a_default: None) -> _VT | None: ...
    @overload
    def get(self, a_key: SupportsIndex, a_default: _VT) -> _VT: ...
    @overload
    def get(self, a_key: SupportsIndex, a_default: None) -> _VT | None: ...
    @overload
    def get(self, a_key: slice, a_default: Self | None = None) -> Self | None: ...
    @overload
    def get(self, a_key: List[int], a_default: Self | None = None) -> Self | None: ...
    @overload
    def get(self, a_key: Tuple[int, ...], a_default: Self | None = None) -> Self | None: ...
    @overload
    def get(self, a_key: List[bool], a_default: Self | None = None) -> Self | None: ...
    @overload
    def get(self, a_key: Tuple[bool, ...], a_default: Self | None = None) -> Self | None: ...
    @overload
    def get(self, a_key: List[_KT], a_default: Self | None = None) -> Self | None: ...
    @overload
    def get(self, a_key: Tuple[_KT, ...], a_default: Self | None = None) -> Self | None: ...
    @overload
    def get(self, a_key: npt.NDArray[np.integer], a_default: Self | None = None) -> Self | None: ...
    @overload
    def get(self, a_key: npt.NDArray[np.bool_], a_default: Self | None = None) -> Self | None: ...
    @overload
    def get(self, a_key: npt.NDArray[Any], a_default: Self | None = None) -> Self | None: ...

    def get(
        self,
        a_key: (
            _KT
            | int
            | SupportsIndex
            | slice
            | List[int]
            | Tuple[int, ...]
            | List[bool]
            | Tuple[bool, ...]
            | List[_KT]
            | Tuple[_KT, ...]
            | npt.NDArray[np.integer]
            | npt.NDArray[np.bool_]
            | npt.NDArray[Any]
        ),
        a_default: Optional[Any] = _MISSING,
    ) -> Self | _VT | Any:
        """Get item by key, index, or slice with a default value if not found.

        This method follows the same intelligent dispatching strategy as __getitem__,
        but returns a default value instead of raising exceptions when items are not found.

        Args:
            a_key (_KT | int | SupportsIndex | slice | List[int] | Tuple[int, ...] | List[bool] | Tuple[bool, ...] | List[_KT] | Tuple[_KT, ...] | npt.NDArray[np.integer] | npt.NDArray[np.bool_] | npt.NDArray[Any]):
                The key, index, slice, or collection to retrieve.
            a_default (Self | Any): The default value to return if the item is not found.

        Returns:
            Self | _VT | Any: The value(s) associated with the key/index, or the default value if not found.
            - Single access returns _VT or default
            - Multiple access returns Self (new BaseDict) or default

        Raises:
            KeyError: If the key is not found and no default is provided.
            IndexError: If the index is out of bounds and no default is provided.
            ValueError: If the value is not found and no default is provided.
        """
        try:
            return self.__getitem__(a_key)
        except (KeyError, IndexError, ValueError):
            if a_default is self._MISSING:
                raise
            return a_default

    @overload
    def pop_by_index(self, a_index: int) -> _VT: ...
    @overload
    def pop_by_index(self, a_index: int, a_default: _VT) -> _VT: ...
    @overload
    def pop_by_index(self, a_index: SupportsIndex) -> _VT: ...
    @overload
    def pop_by_index(self, a_index: SupportsIndex, a_default: _VT) -> _VT: ...
    @overload
    def pop_by_index(self, a_index: slice) -> Self: ...
    @overload
    def pop_by_index(self, a_index: slice, a_default: Self) -> Self: ...
    @overload
    def pop_by_index(self, a_index: List[int]) -> Self: ...
    @overload
    def pop_by_index(self, a_index: List[int], a_default: Self) -> Self: ...
    @overload
    def pop_by_index(self, a_index: Tuple[int, ...]) -> Self: ...
    @overload
    def pop_by_index(self, a_index: Tuple[int, ...], a_default: Self) -> Self: ...
    @overload
    def pop_by_index(self, a_index: npt.NDArray[np.integer]) -> Self: ...
    @overload
    def pop_by_index(self, a_index: npt.NDArray[np.integer], a_default: Self) -> Self: ...

    def pop_by_index(
        self,
        a_index: Union[int, slice, List[int], Tuple[int, ...], npt.NDArray[np.integer]],
        a_default: Optional[Any] = _MISSING,
    ) -> _VT | Self | Any:
        """Remove and return item(s) by positional index/indices, or return default if not found.

        Args:
            a_index (Union[int, slice, List[int], Tuple[int, ...], npt.NDArray[np.integer]]):
                The index/indices to remove.
            a_default (Self | Any): The default value to return if the index/indices are not found.

        Returns:
            _VT | Self | Any: The removed item(s) or the default value if not found.
        """
        try:
            selected = self.get_by_index(a_index)
            if isinstance(a_index, (int, np.integer)):
                keys = list(self.data.keys())
                key = keys[a_index if a_index >= 0 else len(keys) + a_index]
                return self.data.pop(key)
            else:
                for key in list(selected.data.keys()):
                    self.data.pop(key)
                return selected
        except IndexError:
            if a_default is self._MISSING:
                raise
            return a_default

    @overload
    def pop_by_key(self, a_key: _KT) -> _VT: ...
    @overload
    def pop_by_key(self, a_key: List[_KT]) -> Self: ...
    @overload
    def pop_by_key(self, a_key: Tuple[_KT, ...]) -> Self: ...
    @overload
    def pop_by_key(self, a_key: npt.NDArray[Any]) -> Self: ...
    @overload
    def pop_by_key(self, a_key: _KT, a_default: _VT) -> _VT: ...
    @overload
    def pop_by_key(self, a_key: List[_KT], a_default: Self) -> Self: ...
    @overload
    def pop_by_key(self, a_key: Tuple[_KT, ...], a_default: Self) -> Self: ...
    @overload
    def pop_by_key(self, a_key: npt.NDArray[Any], a_default: Self) -> Self: ...

    def pop_by_key(
        self, a_key: Union[_KT, List[_KT], Tuple[_KT, ...], npt.NDArray[Any]], a_default: Optional[Any] = _MISSING
    ) -> _VT | Self | Any:
        """Remove and return item(s) by key(s), or return default if not found.

        Args:
            a_key (Union[_KT, List[_KT], Tuple[_KT, ...], npt.NDArray[Any]]): The key(s) to remove from the dictionary.
            a_default (Self | Any): The default value to return if the key(s) are not found.

        Returns:
            _VT | Self | Any: The removed item(s) or the default value if not found.

        Raises:
            KeyError: If the key(s) are not found and no default is provided.
        """
        try:
            selected = self.get_by_key(a_key)
            if not isinstance(a_key, (list, tuple, np.ndarray)):
                return self.data.pop(a_key)
            else:
                for key in list(selected.data.keys()):
                    self.data.pop(key)
                return selected
        except KeyError:
            if a_default is self._MISSING:
                raise
            return a_default

    @overload
    def pop_by_mask(self, a_mask: List[bool]) -> Self: ...
    @overload
    def pop_by_mask(self, a_mask: Tuple[bool, ...]) -> Self: ...
    @overload
    def pop_by_mask(self, a_mask: npt.NDArray[np.bool_]) -> Self: ...
    @overload
    def pop_by_mask(self, a_mask: List[bool], a_default: Self) -> Self: ...
    @overload
    def pop_by_mask(self, a_mask: Tuple[bool, ...], a_default: Self) -> Self: ...
    @overload
    def pop_by_mask(self, a_mask: npt.NDArray[np.bool_], a_default: Self) -> Self: ...

    def pop_by_mask(
        self, a_mask: Union[List[bool], Tuple[bool, ...], npt.NDArray[np.bool_]], a_default: Optional[Any] = _MISSING
    ) -> Self | Any:
        """Remove and return item(s) by boolean mask, or return default if mask is invalid.

        Args:
            a_mask (Union[List[bool], Tuple[bool, ...], npt.NDArray[np.bool_]]): The boolean mask to apply.

        Returns:
            Self | Any: The removed item(s) or the default value if not found.

        Raises:
            ValueError: If the mask is invalid.
        """
        try:
            selected = self.get_by_mask(a_mask)
            for key in list(selected.data.keys()):
                self.data.pop(key)
            return selected
        except ValueError:
            if a_default is self._MISSING:
                raise
            return a_default

    def __delitem__(
        self,
        a_key: (
            _KT
            | int
            | SupportsIndex
            | slice
            | List[int]
            | Tuple[int, ...]
            | List[bool]
            | Tuple[bool, ...]
            | List[_KT]
            | Tuple[_KT, ...]
            | npt.NDArray[np.integer]
            | npt.NDArray[np.bool_]
            | npt.NDArray[Any]
        ),
    ) -> None:
        """Delete item(s) by key, index, slice, mask, or collection.

        Args:
            a_key: The key(s) to remove from the dictionary.
        """
        # Handle slice - delegate to pop_by_index
        if isinstance(a_key, slice):
            self.pop_by_index(a_key)
            return

        # Handle numpy arrays - convert and process
        if isinstance(a_key, np.ndarray):
            a_key = a_key.tolist()

        # Handle lists and tuples
        if isinstance(a_key, (list, tuple)):
            # Handle empty list/tuple
            if len(a_key) == 0:
                return

            # Type guard for boolean lists - delegate to pop_by_mask
            if all(isinstance(x, bool) for x in a_key):
                self.pop_by_mask(a_key)
                return

            # Type guard for integer index lists
            if all(isinstance(x, int) for x in a_key):
                int_key = a_key
                # Check if all integers exist as actual keys in the dictionary
                all_keys_exist = all(k in self.data for k in int_key)

                if all_keys_exist:
                    # Treat as list of keys - delegate to pop_by_key
                    self.pop_by_key(int_key)
                    return
                else:
                    # Treat as list of indices - delegate to pop_by_index
                    self.pop_by_index(int_key)
                    return

            # Handle as list of keys - delegate to pop_by_key
            self.pop_by_key(a_key)
            return

        # Handle integer index vs integer keys
        if isinstance(a_key, (int, np.integer)) and not isinstance(a_key, bool):
            # First check if it's an actual key in the dictionary
            try:
                self.pop_by_key(a_key)
            except KeyError:
                # If not found as a key, treat as an index - delegate to pop_by_index
                self.pop_by_index(a_key)
            return

        # Handle all other key types - delegate to pop_by_key
        self.pop_by_key(a_key)

    @overload
    def pop(self, a_key: _KT, a_default: _VT) -> _VT: ...
    @overload
    def pop(self, a_key: _KT, a_default: None) -> _VT | None: ...
    @overload
    def pop(self, a_key: int, a_default: _VT) -> _VT: ...
    @overload
    def pop(self, a_key: int, a_default: None) -> _VT | None: ...
    @overload
    def pop(self, a_key: SupportsIndex, a_default: _VT) -> _VT: ...
    @overload
    def pop(self, a_key: SupportsIndex, a_default: None) -> _VT | None: ...
    @overload
    def pop(self, a_key: slice, a_default: Self | None = None) -> Self | None: ...
    @overload
    def pop(self, a_key: List[int], a_default: Self | None = None) -> Self | None: ...
    @overload
    def pop(self, a_key: Tuple[int, ...], a_default: Self | None = None) -> Self | None: ...
    @overload
    def pop(self, a_key: List[bool], a_default: Self | None = None) -> Self | None: ...
    @overload
    def pop(self, a_key: Tuple[bool, ...], a_default: Self | None = None) -> Self | None: ...
    @overload
    def pop(self, a_key: List[_KT], a_default: Self | None = None) -> Self | None: ...
    @overload
    def pop(self, a_key: Tuple[_KT, ...], a_default: Self | None = None) -> Self | None: ...
    @overload
    def pop(self, a_key: npt.NDArray[np.integer], a_default: Self | None = None) -> Self | None: ...
    @overload
    def pop(self, a_key: npt.NDArray[np.bool_], a_default: Self | None = None) -> Self | None: ...
    @overload
    def pop(self, a_key: npt.NDArray[Any], a_default: Self | None = None) -> Self | None: ...
    def pop(
        self,
        a_key: (
            _KT
            | int
            | SupportsIndex
            | slice
            | List[int]
            | Tuple[int, ...]
            | List[bool]
            | Tuple[bool, ...]
            | List[_KT]
            | Tuple[_KT, ...]
            | npt.NDArray[np.integer]
            | npt.NDArray[np.bool_]
            | npt.NDArray[Any]
        ),
        a_default: Optional[Any] = _MISSING,
    ) -> Self | _VT | Any:
        """Pop an item from the dictionary.

        A pop operation removes the specified key from the dictionary and returns its value.

        Args:
            a_key (_KT | int | SupportsIndex | slice | List[int] | Tuple[int, ...] | List[bool] | Tuple[bool, ...] | List[_KT] | Tuple[_KT, ...] | npt.NDArray[np.integer] | npt.NDArray[np.bool_] | npt.NDArray[Any]):
                The key to remove from the dictionary.
            a_default (Self | Any): The value to return if the key is not found.

        Returns:
            Self | None | _VT: The value associated with the key, or a_default if the key is not found.

        Raises:
            KeyError: If the key is not found and a_default is not provided.
            IndexError: If the key is an index and is out of bounds.
            ValueError: If the key is not valid.
        """
        try:
            # Handle slice - delegate to pop_by_index
            if isinstance(a_key, slice):
                return self.pop_by_index(a_key)

            # Handle numpy arrays - convert and process
            if isinstance(a_key, np.ndarray):
                a_key = a_key.tolist()

            # Handle lists and tuples
            if isinstance(a_key, (list, tuple)):
                # Handle empty list/tuple
                if len(a_key) == 0:
                    return self.__class__(a_dict={}, a_max_size=self._max_size, a_name=self._name)

                # Type guard for boolean lists - delegate to pop_by_mask
                if all(isinstance(x, bool) for x in a_key):
                    return self.pop_by_mask(a_key)

                # Type guard for integer index lists
                if all(isinstance(x, int) for x in a_key):
                    int_key = a_key
                    # Check if all integers exist as actual keys in the dictionary
                    all_keys_exist = all(k in self.data for k in int_key)

                    if all_keys_exist:
                        # Treat as list of keys - delegate to pop_by_key
                        return self.pop_by_key(int_key)
                    else:
                        # Treat as list of indices - delegate to pop_by_index
                        return self.pop_by_index(int_key)

                # Handle as list of keys - delegate to pop_by_key
                return self.pop_by_key(a_key)

            # Handle integer index vs integer keys
            if isinstance(a_key, (int, np.integer)) and not isinstance(a_key, bool):
                # First check if it's an actual key in the dictionary
                try:
                    return self.pop_by_key(a_key)
                except KeyError:
                    # If not found as a key, treat as an index - delegate to pop_by_index
                    return self.pop_by_index(a_key)

            # Handle all other key types - delegate to pop_by_key
            return self.pop_by_key(a_key)
        except (KeyError, IndexError, ValueError):
            if a_default is self._MISSING:
                raise
            return a_default

    