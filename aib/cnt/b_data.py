"""Data Container - Base Data Container Utilities

This module provides a base data container class that extends the functionality of UserDict.

Classes:
    BaseDict:
        A dictionary-like container with additional features such as size limits, named identification,
        and enhanced item access methods.
"""

from copy import deepcopy
from dataclasses import asdict, astuple, dataclass
from typing import Any, Self


@dataclass(frozen=True)
class BaseData:
    """Base Data Class

    A frozen data class that provides a structure for holding data in a dictionary-like format.
    """

    def to_dict(self) -> dict[str, object]:
        """Convert the data class to a dictionary.

        Returns:
            dict[str, object]: Dictionary representation of the data class.
        """
        return asdict(self)

    def to_tuple(self) -> tuple[Any, ...]:
        """Convert the data class to a tuple.

        Returns:
            tuple[Any, ...]: Tuple representation of the data class.
        """
        return astuple(self)

    def to_list(self) -> list[Any]:
        """Convert the data class to a list.

        Returns:
            list[Any]: List representation of the data class.
        """
        return list(self.to_tuple())

    def copy(self) -> Self:
        """Create a deep copy of the data class.

        Returns:
            Self: A new instance of the data class with the same data.
        """
        return deepcopy(self)
