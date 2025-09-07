"""Base Dict Module

This module defines the `BaseDict` class, a generic class representing a dictionary of objects with specified key and
value types.

Classes:
    BaseDict (Generic[T_key, T_value]):
        A generic class representing a dictionary of objects with specified key and value types.

Variables:
    T_key (TypeVar):
        Type variable for keys in a generic context.
        This type variable is used to represent the type of keys in generic classes or functions.

    T_value (TypeVar):
        Type variable for values in a generic context.
        This type variable is used to represent the type of values in generic classes or functions.

Functions:
    None
"""

# region Imported Dependencies
import numpy as np
import numpy.typing as npt
import pprint
from copy import deepcopy
from typing import TypeVar, Generic, Union, List, Dict, ItemsView, Tuple, Sequence, Optional

from aib.misc.type import is_int, is_bool

# endregion Imported Dependencies


TypeKey = TypeVar("TypeKey")
"""
Type variable for keys in a generic context.

This type variable is used to represent the type of keys in generic classes or functions.
"""

TypeValue = TypeVar("TypeValue")
"""
Type variable for values in a generic context.

This type variable is used to represent the type of values in generic classes or functions.
"""

TypeBaseObjectDict = TypeVar("TypeBaseObjectDict", bound="BaseObjectDict")
"""
Type variable for `BaseObjectDict` in a generic context.

This type variable is used to represent the type of `BaseObjectDict` instances in generic classes or functions.
"""


class BaseObjectDict(Generic[TypeKey, TypeValue]):
    """Base Object Dictionary

    The `BaseObjectDict` class represents a dictionary of objects with specified key and value types.

    Attributes:
        name (str):
            A :type:`string` that specifies the name of the `BaseObjectDict` instance.
        max_size (int):
            An integer representing the maximum size of the dictionary (default is -1, indicating no size limit).
        items (Dict[T_key, T_value]):
            A dictionary of objects with specified key and value types contained within the `BaseObjectDict`.
    """

    def __init__(
        self,
        a_name: str = "BASE_OBJECT_DICT",
        a_max_size: int = -1,
        a_key: Optional[Union[TypeKey, List[TypeKey]]] = None,
        a_value: Optional[Union[TypeValue, List[TypeValue]]] = None,
        a_key_type: Optional[type] = None,
        a_value_type: Optional[type] = None,
    ):
        """
        Constructor for the BaseObjectDict class.

        Args:
            a_name (str, optional):
                A string specifying the name of the BaseObjectDict instance (default is 'BASE_OBJECT_DICT').
            a_max_size (int, optional):
                An integer representing the maximum size of the dictionary (default is -1, indicating no size limit).
            a_key (Union[T_key, List[T_key]], optional):
                The key or list of keys to initialize the BaseObjectDict (default is None).
            a_value (Union[T_value, List[T_value]], optional):
                The value or list of values to initialize the BaseObjectDict (default is None).
            a_key_type (type, optional):
                The key data type that specifies the type of keys in the dictionary.
            a_value_type (type, optional):
                The value data type that specifies the type of values in the dictionary.
        Returns:
            None: The constructor does not return any values.

        Raises:
            RuntimeError: If the length of keys and values in the lists is different during initialization.
        """
        self.name: str = a_name
        self._max_size: int = a_max_size
        self._items: Dict[TypeKey, TypeValue] = {}
        self._key_type: type = a_key_type
        self._value_type: type = a_value_type
        if a_key is not None and a_value is not None:
            self.append(a_key, a_value)

    @property
    def name(self) -> str:
        """Instance Name Getter

        This property specifies the name of the class object.

        Returns
            str: This property returns a :type:`string` as the name of the class object.
        """
        return self._name

    @name.setter
    def name(self, a_name: str = "BASE_OBJECT_DICT") -> None:
        """Instance Name Setter

        This setter is used to set the name of the class object.

        Args:
            a_name (str): A :type:`string` that specifies the class object's name.

        Returns:
            None
        """
        self._name = a_name.upper().replace(" ", "_")

    @property
    def max_size(self) -> int:
        """Get the maximum size of the BaseObjectDict.

        Returns:
            int: The maximum size of the BaseObjectDict.
        """
        return self._max_size

    @max_size.setter
    def max_size(self, a_max_size: int) -> None:
        """Set the maximum size of the BaseObjectDict.

        Args:
            a_max_size (int): The new maximum size for the BaseObjectDict.

        Raises:
            TypeError: If `a_max_size` is not an integer.
        """
        if a_max_size is None or not isinstance(a_max_size, float):
            raise TypeError("The `a_max_size` must be a `int`.")
        self._max_size: int = a_max_size

    def to_dict(self) -> Dict[TypeKey, TypeValue]:
        """
        Return a shallow copy of the dictionary.

        Returns:
            Dict[T_key, T_value]: A shallow copy of the dictionary.
        """
        return self._items.copy()

    def to_str(self) -> str:
        """
        Convert the `BaseObjectDict` to a formatted string.

        This method converts the `BaseObjectDict` into a human-readable string representation by
        using the :class:`pprint.pformat` function on the result of `to_dict`.

        Returns:
            str: A formatted string representing the `BaseObjectDict`.
        """
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """
        Return a string representation of the `BaseObjectDict` object.

        This method returns a string representation of the `BaseObjectDict` by calling the `to_str` method.

        Returns:
            str: A string representing the `BaseObjectDict` object.
        """
        return self.to_str()

    def items(self) -> ItemsView[TypeKey, TypeValue]:
        """
        Return a view object that displays a list of the dictionary's key-value tuple pairs.

        Returns:
            ItemsView[T_key, T_value]: A view object that displays a list of the dictionary's key-value tuple pairs.
        """
        return self._items.items()

    def __getitem__(
        self,
        a_key: Union[
            TypeKey,
            List[TypeKey],
            Tuple[TypeKey],
            npt.NDArray[TypeKey],
            Sequence[TypeKey],
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
    ) -> Union[TypeValue, TypeBaseObjectDict]:
        """
        Retrieve the value(s) associated with the given key(s), indices, or slice.

        This method supports single keys, lists or tuples of keys, numpy arrays of keys,
        integer indices, boolean sequences, and slicing.

        Args:
            a_key (Union[TypeKey, List[TypeKey], Tuple[TypeKey], np.ndarray[TypeKey], int,
            List[int], Tuple[int], np.ndarray[int], slice, List[bool], Tuple[bool], np.ndarray[bool]]):
                The key, list of keys, tuple of keys, numpy array of keys, integer index,
                list of integer indices, tuple of integer indices, numpy array of integer indices,
                slice, list of boolean values, tuple of boolean values, or numpy array of boolean values.

        Returns:
            Union[TypeValue, Dict[TypeKey, TypeValue]]:
                - If a single key is provided, returns the value associated with that key.
                - If a list, tuple, or numpy array of keys or indices is provided, returns a dictionary with
                  key-value pairs corresponding to the provided keys or indices.
                - If an integer index is provided, returns the value at that index.
                - If a slice is provided, returns a dictionary with key-value pairs corresponding to the sliced portion.
                - If a list/tuple/array of booleans is provided, returns a dictionary with key-value pairs
                  corresponding to the positions where the boolean is True.

        Raises:
            TypeError: If `a_key` is not of the supported types (key, list, tuple, numpy array of keys, integer index,
                       boolean sequence, or slice).
            IndexError: If an integer index is out of range.
            ValueError: If the boolean sequence length does not match the length of the dictionary.
        """
        if self._key_type != int and isinstance(a_key, (int, np.integer)):
            # Handle integer
            keys = list(self._items.keys())
            if a_key < 0 or a_key >= len(keys):
                raise IndexError("Integer index is out of range.")
            key = keys[a_key]
            return self._items[key]

        elif isinstance(a_key, (list, tuple, np.ndarray)):
            if self._key_type != int and is_int(a_key):
                # Handle sequence of integers
                keys = list(self._items.keys())
                if any(x < 0 or x >= len(keys) for x in a_key):
                    raise IndexError("Index is out of range.")
                return self.__class__(a_key=[keys[i] for i in a_key], a_value=[self._items[keys[i]] for i in a_key])

            elif all(isinstance(x, self._key_type) for x in a_key):
                # Handle sequence of keys
                return self.__class__(
                    a_key=[key for key in a_key if key in self._items],
                    a_value=[self._items[key] for key in a_key if key in self._items],
                )

            elif is_bool(a_key):
                # Handle sequence of booleans
                if len(a_key) != len(self._items):
                    raise ValueError("Boolean sequence length must match the length of the dictionary.")
                keys = []
                values = []
                for key, value in zip(self._items.keys(), self._items.values()):
                    if a_key[list(self._items.keys()).index(key)]:
                        keys.append(key)
                        values.append(value)
                return self.__class__(a_key=keys, a_value=values)
            else:
                raise TypeError("All elements in the list, tuple, or numpy array must be either integers or keys.")

        elif isinstance(a_key, slice):
            # Handle slice
            keys = list(self._items.keys())
            sliced_keys = keys[a_key]
            return self.__class__(a_key=[key for key in sliced_keys], a_value=[self._items[key] for key in sliced_keys])

        else:
            # Handle key
            if a_key in self._items:
                return self._items[a_key]
            else:
                raise KeyError(f"Key ({a_key}) not found in the dictionary.")

    def __setitem__(self, a_key: TypeKey, a_value: TypeValue):
        """
        Set the value associated with the given key.

        Args:
            a_key (T_key): The key for which to set the associated value.
            a_value (T_value): The value to be associated with the provided key.

        Returns:
            None
        """
        if a_key in self._items or self._max_size == -1:
            self._items[a_key] = a_value
        elif a_key not in self._items and self._max_size > 0:
            self.append(a_key=a_key, a_value=a_value)
        else:
            raise IndexError(f"The dictionary does not contain the `{a_key}` key.")

    def append(
        self,
        a_key: Union[TypeKey, List[TypeKey]],
        a_value: Union[TypeValue, List[TypeValue]],
        a_removal_strategy: str = "first",
    ):
        """
        Append key-value pairs to the BaseObjectDict.

        This method appends individual key-value pairs or lists of key-value pairs to the BaseObjectDict.
        If the maximum size is specified, it removes key-value pairs according to the removal strategy.

        Args:
            a_key (Union[T_key, List[T_key]]): The key or list of keys to append.
            a_value (Union[T_value, List[T_value]]): The value or list of values to append.
            a_removal_strategy (str, optional):
                The strategy for removing key-value pairs when the maximum size is reached.
                    Options: 'first' (default) or 'last'.

        Returns:
            None

        Raises:
            ValueError: If an invalid removal strategy is provided.
            RuntimeError: If the length of keys and values in the lists is different.
        """
        if isinstance(a_key, list) and isinstance(a_value, list):
            if len(a_key) != len(a_value):
                raise RuntimeError(
                    f"The length of keys and values must be the same, but they are {len(a_key)}, "
                    f"{len(a_value)} respectively."
                )
            for key, value in zip(a_key, a_value):
                self._append_item(key, value, a_removal_strategy)
        else:
            self._append_item(a_key, a_value, a_removal_strategy)

    def _append_item(self, a_key: TypeKey, a_value: TypeValue, a_removal_strategy: str = "first") -> None:
        """
        Append a key-value pair to the BaseObjectDict, handling size constraints.

        This internal method appends a key-value pair to the BaseObjectDict, and if a maximum size is specified,
        it removes a key-value pair according to the removal strategy when the size exceeds the maximum.

        Args:
            a_key (T_key): The key to append.
            a_value (T_value): The value to append.
            a_removal_strategy (str, optional):
                The strategy for removing key-value pairs when the maximum size is reached.
                Options: 'first' (default) or 'last'.

        Returns:
            None

        Raises:
            ValueError: If an invalid removal strategy is provided.
        """
        if a_key not in self.keys():
            self._clip(a_removal_strategy=a_removal_strategy)
        self._items[a_key] = a_value

    # TODO(doc): Complete the document of following method
    def _clip(self, a_removal_strategy: str = "first") -> None:
        if self._max_size != -1 and len(self) >= self._max_size:
            if a_removal_strategy.lower() == "first":
                first_key = next(iter(self._items))
                self._items.pop(first_key)
            elif a_removal_strategy.lower() == "last":
                self._items.popitem()
            else:
                raise ValueError("Invalid removal strategy. Use 'first' or 'last'.")

    def __delitem__(self, a_key: TypeKey):
        """
        Remove the key-value pair associated with the given key.

        Args:
            a_key (T_key): The key for which to remove the associated key-value pair.

        Returns:
            None
        """
        del self._items[a_key]

    def copy(self: TypeBaseObjectDict) -> TypeBaseObjectDict:
        """
        Create a deep copy of the BaseObjectDict.

        This method creates a deep copy of the BaseObjectDict, including a copy of all contained key-value pairs.

        Returns:
            BaseObjectDict[T_key, T_value]: A duplicated instance of the class.
        """
        return deepcopy(self)

    def __len__(self) -> int:
        """
        Get the number of key-value pairs in the BaseObjectDict.

        Returns:
            int: The number of key-value pairs in the BaseObjectDict.
        """
        return len(self._items)

    def clear(self) -> None:
        """
        Clear all key-value pairs in the BaseObjectDict.

        This method resets the dictionary of key-value pairs to an empty dictionary.

        Returns:
            None
        """
        self._items = {}

    def __contains__(self, a_key: TypeKey):
        """Check if the BaseObjectDict contains the specified key.

        Args:
            a_key (T_key): The key to check for membership in the BaseObjectDict.

        Returns:
            bool: True if the item is found in the BaseObjectDict, False otherwise.
        """
        return a_key in self._items

    def keys(self):
        """
        Return a list of all keys in the dictionary.

        Returns:
            List[T_key]: A list of all keys in the dictionary.
        """
        return self._items.keys()

    def values(self):
        """
        Return a list of all values in the dictionary.

        Returns:
            List[T_value]: A list of all values in the dictionary.
        """
        return self._items.values()

    def pop(self, a_key: TypeKey) -> None:
        """
        Remove the item with the specified key.

        This method removes the item with the specified key.

        Args:
            a_key (T_key): The key of the item to remove.

        Returns:
            None
        """
        self._items.pop(a_key)

    # TODO(doc): Complete the document of following method
    def popitem(self) -> None:
        if not self._items:
            raise KeyError("popitem(): dictionary is empty")
        key = next(reversed(self._items))
        self._items.pop(key)

    # TODO(doc): Complete the document of following method
    def pop_first(self) -> None:
        if not self._items:
            raise KeyError("pop_first(): dictionary is empty")
        first_key = next(iter(self._items))
        self._items.pop(first_key)

    # TODO(doc): Complete the document of following method
    def pop_last(self) -> None:
        if not self._items:
            raise KeyError("pop_last(): dictionary is empty")
        self.popitem()

    def update(
        self,
        a_dict: Union[Dict[TypeKey, TypeValue], ItemsView[TypeKey, TypeValue], "BaseObjectDict[TypeKey, TypeValue]"],
        a_removal_strategy: str = "first",
    ) -> None:
        """
        Update the BaseObjectDict with key-value pairs from another dictionary, iterable, or BaseObjectDict.

        This method updates the dictionary with key-value pairs from another dictionary, iterable of key-value pairs,
        or another BaseObjectDict instance. If the key already exists, its value is updated.

        Args:
            a_dict (Union[Dict[T_key, T_value], ItemsView[T_key, T_value], BaseObjectDict[T_key, T_value]]):
                The dictionary, iterable of key-value pairs, or BaseObjectDict to update the BaseObjectDict with.
            a_removal_strategy (str, optional):
                The strategy for removing key-value pairs when the maximum size is reached.
                Options: 'first' (default) or 'last'.
        Returns:
            None

        Raises:
            TypeError: If `other` is not a dictionary, iterable of key-value pairs, or BaseObjectDict.
        """
        if isinstance(a_dict, dict):
            for key, value in a_dict.items():
                self._append_item(key, value, a_removal_strategy=a_removal_strategy)
        elif isinstance(a_dict, ItemsView):
            for key, value in a_dict:
                self._append_item(key, value, a_removal_strategy=a_removal_strategy)
        elif isinstance(a_dict, BaseObjectDict):
            for key, value in a_dict.items():
                self._append_item(key, value, a_removal_strategy=a_removal_strategy)
        else:
            raise TypeError(
                "The 'other' argument must be a dictionary, iterable of key-value pairs, or BaseObjectDict."
            )
