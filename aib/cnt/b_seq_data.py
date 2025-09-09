"""Data Container - Base Sequence Data Container Utilities

This module provides a base sequence data container class that extends the functionality of BaseData.

Classes:
    BaseSeqData:
        A sequence-like container with an identifier and timestamp, designed to be immutable and comparable.
"""

from abc import ABC
from dataclasses import dataclass, field

from aib.cnt.b_data import BaseData


@dataclass(order=True, frozen=True, kw_only=True)
class BaseSeqData(BaseData, ABC):
    """Base Sequence Data Class

    This class represents a base sequence data structure that includes an identifier and a timestamp.
    It is designed to be immutable and comparable based on the identifier.

    Attributes:
        id (int): Unique identifier for the sequence data.
        timestamp (float): Timestamp associated with the sequence data.
    """

    id: int = field(compare=True)
    timestamp: float = field(compare=False)

    def __getitem__(self, a_index: int):
        """Get item by index.
        Args:
            a_index (int): Index of the item to retrieve.
        Returns:
            The item at the specified index.
        Raises:
            IndexError: If the index is out of range.
        """
        if a_index == 0:
            return self.id
        if a_index == 1:
            return self.timestamp
        raise IndexError(f"Index is out of range for the `{self.__class__.__name__}`.")
