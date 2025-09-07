"""Base List Module

This module defines the `BaseList` class, a generic class representing a list of objects of type `T`.

Classes:
    BaseList (Generic[T], ABC):
        A generic class representing a list of objects of type `T`.

Variables:
    T (TypeVar):
        A type variable used in generic classes. `T` represents the element type of the generic class,
        typically used in generic classes like :class:`BaseList`.

Functions:
    None
"""

# region Imported Dependencies
import numpy as np
import numpy.typing as npt
import pprint
from copy import deepcopy
from typing import TypeVar, Generic, List, Union, Callable, Any, Iterator, Tuple, Sequence, Optional
from aib.misc.type import is_int, is_bool

# endregion Imported Dependencies


TypeItem = TypeVar("TItem")
"""
A type variable used in generic classes to represent the element type.

`T` is a type variable that is commonly used in generic class definitions to specify the type of elements
contained within the class. It allows for flexibility and type safety when defining and working with generic
classes and methods.

Attributes:
    type (Type): Represents the type of elements that the generic class or method will work with.

Usage:
    - In a generic class, `T` defines the type of the items in the collection. For example, in a class 
    `BaseObjectList[T]`, `T` specifies the type of objects stored in the list.
    - In a generic method, `T` can represent the type of parameters or return values to ensure that the
    method operates with the expected type.

Example:
    class BaseObjectList(Generic[T]):
        def __init__(self, items: List[T]):
            self._items = items

    def get_first_item(self) -> T:
        return self._items[0]
"""

TypeBaseObjectList = TypeVar("TypeBaseObjectList", bound="BaseObjectList")
"""
A type variable used in generic classes to represent a type that is bound to the `BaseObjectList` class.

`TList` is a type variable that is specifically used to indicate a type that extends or is a subclass of the
`BaseObjectList` class. It is used in generic class definitions and method signatures to ensure type consistency
within the context of `BaseObjectList` and its subclasses.

Attributes:
    bound (Type[BaseObjectList]): The class to which `TList` is bound, ensuring that it can only be used with types
    that are derived from `BaseObjectList`.

Example:
    If you have a subclass of `BaseObjectList` such as `MyObjectList`, you can use `TList` to refer to the type
    `MyObjectList` in method signatures or other generic contexts.

    class MyObjectList(BaseObjectList):
        ...

    def some_method(self: TList) -> TList:
        ...
"""


class BaseObjectList(Generic[TypeItem]):
    """Base Object List

    The `BaseObjectList` class represents a list of objects of type `T`.

    Attributes:
        name (str):
            A :type:`string` that specifies the name of the `BaseObjectList` instance.
        max_size (int):
            An integer representing the maximum size of the list (default is -1, indicating no size limit).
        items (List[T]):
            A list of objects of type `T` contained within the `BaseObjectList`.
    """

    def __init__(
        self,
        a_items: Union[TypeItem, List[TypeItem], "BaseObjectList"] = None,
        a_name: str = "ObjectList",
        a_max_size: int = -1,
        a_item_type: Optional[type] = None,
    ):
        """
        Constructor for the `BaseObjectList` class.

        Args:
            a_name (str, optional):
                A :type:`string` that specifies the name of the `BaseObjectList` instance (default is 'Objects').
            a_max_size (int, optional):
                An :type:`int` representing the maximum size of the list (default is -1, indicating no size limit).
            a_items (List[T], optional):
                A list of objects of type :class:`T` to initialize the `BaseObjectList` (default is None).
            a_item_type (type, optional):
                The value data type that specifies the type of values in the list.
        Returns:
            None: The constructor does not return any values.
        """
        self.name: str = a_name
        self._max_size: int = a_max_size
        self._item_type: type = a_item_type
        self._items: List[TypeItem] = []

        if a_items is not None:
            self.append(a_item=a_items)

    @property
    def name(self) -> str:
        """Instance Name Getter

        This property specifies the name of the class object.

        Returns
            str: This property returns a :type:`string` as the name of the class object.
        """
        return self._name

    @name.setter
    def name(self, a_name: str = "BASE_OBJECT_LIST") -> None:
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
        """Get the maximum size of the BaseObjectList.

        Returns:
            int: The maximum size of the BaseObjectList.
        """
        return self._max_size

    @max_size.setter
    def max_size(self, a_max_size: int) -> None:
        """Set the maximum size of the BaseObjectList.

        Args:
            a_max_size (int): The new maximum size for the BaseObjectList.

        Raises:
            TypeError: If `a_max_size` is not an integer.
        """
        if a_max_size is None or not isinstance(a_max_size, int):
            raise TypeError("The `a_max_size` must be a `int`.")
        self._max_size: int = a_max_size

    def to_dict(self) -> List[dict]:
        """
        Convert the `BaseObjectList` to a list of dictionaries.

        This method iterates through the objects in the `BaseObjectList` and converts each object to a dictionary.

        Returns:
            List[dict]: A list of dictionaries, where each dictionary represents an object in the `BaseObjectList`.
        """
        dict_items = []
        for item in self._items:
            dict_items.append(item.to_dict() if hasattr(item, "to_dict") else item)
        return dict_items

    def to_str(self) -> str:
        """
        Convert the `BaseObjectList` to a formatted string.

        This method converts the `BaseObjectList` into a human-readable string representation by
        using the :class:`pprint.pformat` function on the result of `to_dict`.

        Returns:
            str: A formatted string representing the `BaseObjectList`.
        """
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """
        Return a string representation of the `BaseObjectList` object.

        This method returns a string representation of the `BaseObjectList` by calling the `to_str` method.

        Returns:
            str: A string representing the `BaseObjectList` object.
        """
        return self.to_str()

    @property
    def items(self) -> List[TypeItem]:
        """
        Get the list of items in the `BaseObjectList`.

        This property provides access to the list of items contained within the `BaseObjectList`.

        Returns:
            List[T]: A :class:`list` of objects of type :class:`T` within the `BaseObjectList`.
        """
        return self._items

    def __getitem__(
        self: TypeBaseObjectList,
        a_index: Union[
            int,
            List[int],
            Tuple[int],
            npt.NDArray[np.integer],
            Sequence[int],
            List[bool],
            Tuple[bool],
            npt.NDArray[bool],
            Sequence[bool],
            slice,
        ],
    ) -> Union[TypeItem, TypeBaseObjectList]:
        """
        Retrieve the value(s) associated with the given index or slice.

        This method supports retrieving single items, multiple items via lists or numpy arrays of indices,
        boolean indexing, and slicing similar to Python lists.

        Args:
            a_index (Union[int, List[int], Tuple[int], np.ndarray, Sequence[int], List[bool], Tuple[bool],
            np.ndarray[bool], Sequence[bool], slice]):
                The index, list of indices, tuple of indices, numpy array of indices, sequence of booleans,
                or a slice to retrieve values.

        Returns:
            Union[T, BaseObjectList]: The value associated with a single index, or a new instance containing
            the retrieved items for multiple indices, boolean indexing, or slices.

        Raises:
            TypeError: If `a_index` is not of the supported types (int, list, tuple, numpy array of integers,
            sequence of booleans, or slice).
            ValueError: If the boolean sequence length does not match the length of the list.
        """
        if isinstance(a_index, (int, np.integer)):
            return self._items[a_index]
        elif isinstance(a_index, (list, tuple, np.ndarray)):
            if is_int(a_index):
                return self.__class__(a_items=[self._items[i] for i in a_index])
            elif is_bool(a_index):
                if len(a_index) != len(self._items):
                    raise ValueError("Boolean sequence length must match the length of the list.")
                return self.__class__(a_items=[item for item, flag in zip(self._items, a_index) if flag])
            else:
                raise TypeError("All elements in the list, tuple, or numpy array must be either integers or booleans.")
        elif isinstance(a_index, slice):
            return self.__class__(a_items=self._items[a_index])
        else:
            raise TypeError("Index must be an int, list, tuple, numpy array of integers, or slice.")

    def __setitem__(self, a_index: int, a_item: TypeItem):
        """
        Get an item from the `BaseObjectList` by index.

        This method allows retrieving an item from the `BaseObjectList` by its index.

        Args:
            a_index (:type:int): The index of the item to retrieve.

        Returns:
            T: The item at the specified index.
        """
        self._items[a_index] = a_item

    def append(
        self,
        a_item: Union[TypeItem, List[TypeItem], "BaseObjectList"],
        a_removal_strategy: str = "first",
        a_merge: bool = True,
    ):
        """
        Append an item or a list of items to the `BaseObjectList`.

        This method appends an individual item or a list of items to the `BaseObjectList`.

        Args:
            a_item (Union[T, List[T]]): An item or a list of items to append.
            a_removal_strategy (str):
                The strategy for removing items when the maximum size is reached. Options: `first` (default) or `last`.
            a_merge (bool):
                The flag that indicates if the same `BaseObjectList` is passed to be merged appended as an item.

        Returns:
            None
        """
        if isinstance(a_item, (list, tuple)) or (isinstance(a_item, self.__class__) and a_merge):
            for item in a_item:
                self._append_item(item, a_removal_strategy)
        else:
            self._append_item(a_item, a_removal_strategy)

    def _append_item(self, a_item: TypeItem, a_removal_strategy: str = "first") -> None:
        """
        Append an item to the `BaseObjectList` (Internal).

        This internal method appends an item to the `BaseObjectList`, handling size constraints if `_max_size` is set.

        Args:
            a_item (T): The item to append.
            a_removal_strategy (str): The strategy for removing items when the maximum size is reached. Options:
            `first` or `last`.

        Returns:
            None
        """
        self._clip(a_removal_strategy=a_removal_strategy)
        self._items.append(a_item)

    # TODO(doc): Complete the document of following method
    def _clip(self, a_removal_strategy: str = "first") -> None:
        if self._max_size != -1 and len(self) >= self._max_size:
            if a_removal_strategy.lower() == "first":
                self._items.pop(0)
            elif a_removal_strategy.lower() == "last":
                self._items.pop()
            else:
                raise ValueError("Invalid removal strategy. Use 'first' or 'last'.")

    def __delitem__(self, a_index: int):
        """
        Delete an item from the `BaseObjectList` by index.

        This method allows deleting an item from the `BaseObjectList` by its index.

        Args:
            a_index (int): The index of the item to delete.

        Returns:
            None
        """
        del self._items[a_index]

    def copy(self: TypeBaseObjectList) -> TypeBaseObjectList:
        """
        Create a deep copy of the `BaseObjectList`.

        This method creates a deep copy of the `BaseObjectList`, including a copy of all contained items.

        Returns:
            BaseObjectList[T]: A duplicated instance of the class.
        """
        return deepcopy(self)

    def __len__(self) -> int:
        """
        Get the number of items in the `BaseObjectList`.

        This method returns the number of items contained within the `BaseObjectList`.

        Returns:
            int: The number of items.
        """
        return len(self._items)

    def clear(self) -> None:
        """Clear all items in the list.

        This method resets the list of items to an empty list.

        Returns:
            None

        """
        self._items = []

    def __contains__(self, a_item: TypeItem):
        """Check if the list contains the specified item.

        Args:
            a_item (T): The item to check for membership in the list.

        Returns:
            bool: True if the item is found in the list, False otherwise.
        """
        return a_item in self._items

    def pop(self, a_index: int) -> None:
        """
        Remove the item at the specified index.

        This method removes the item at the specified index.

        Args:
            a_index (int): The index of the item to remove.

        Returns:
            None
        """
        self._items.pop(a_index)

    # TODO(doc): Complete the document of following method
    def pop_first(self) -> None:
        if not self._items:
            raise KeyError("pop_first(): list is empty")
        self._items.pop(0)

    # TODO(doc): Complete the document of following method
    def pop_last(self) -> None:
        if not self._items:
            raise KeyError("pop_last(): list is empty")
        self._items.pop()

    def sort(self, a_key: Callable[[TypeItem], Any] = None, a_reverse: bool = False) -> None:
        """
        Sort the items in the list.

        Args:
            a_key (Callable[[T], Any], optional):
                A callable function used as the key for sorting the items. If provided, the list will be sorted based
                on the result of applying this function to each item. Defaults to None.
            a_reverse (bool, optional):
                If True, the list will be sorted in descending order; if False, it will be sorted in ascending order.
                Defaults to False.

        Returns:
            None
        """
        self._items.sort(key=a_key, reverse=a_reverse)

    def __iter__(self) -> Iterator[TypeItem]:
        """
        Return an iterator over the items in the list.

        Returns:
            Iterator[T]: An iterator over the items in the list.
        """
        return iter(self._items)

    def filter(self: TypeBaseObjectList, a_condition: Callable[[TypeItem], bool]) -> TypeBaseObjectList:
        """
        Filter items in the `BaseObjectList` based on a given condition and return a new `BaseObjectList` instance.

        Args:
            a_condition (Callable[[T], bool]): A callable function that defines the filtering condition.

        Returns:
            BaseObjectList[T]: A new `BaseObjectList` instance containing filtered items.
        """
        filtered_items = [item for item in self._items if a_condition(item)]
        return self.__class__(a_items=deepcopy(filtered_items))
