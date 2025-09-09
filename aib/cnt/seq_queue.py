"""Asynchronous Sequence Queue

This module provides an asynchronous sequence queue implementation.

Classes:
    AsyncSeqQueue:
        An asynchronous priority queue that retrieves items in sequence order, waiting for the expected
        sequence number before returning an item.

Type Variables:
    _T: A type variable bound to a sequence (typically a tuple) representing an item in the queue.
"""

import asyncio
from typing import Any, Dict, Sequence, Tuple, TypeVar

_T = TypeVar("_T", bound=Sequence[Any], default=Tuple[int, Any])


class AsyncSeqQueue(asyncio.PriorityQueue[_T]):
    """Asynchronous Sequence Queue

    AsyncSeqQueue is an asynchronous priority queue that ensures items are retrieved in a sequential order based on
    an internal iterator.

    Attributes:
        _iter (int): The current sequence number for item retrieval.
    """

    def __init__(self, a_maxsize: int = 0, a_iter: int = 0, **kwargs: Any) -> None:
        """Initialize the asynchronous sequence queue.

        Args:
            a_maxsize (int): The maximum size of the queue.
            a_iter (int): The initial sequence number for item retrieval.
        """
        super().__init__(maxsize=a_maxsize, **kwargs)
        self._iter: int = a_iter

    async def get(self) -> _T:
        """Get Item

        Asynchronously retrieves the next item from the queue in sequence order.

        This method waits until an item with the expected sequence number (`self._iter`)
        is available at the front of the queue. If the item is not in sequence, it is
        put back into the queue and the method waits briefly before retrying.

        Returns:
            _T: The next item in sequence from the queue.

        Raises:
            RuntimeError: If an exception occurs during retrieval.
        """
        try:
            while True:
                item = await super().get()
                if item[0] == self._iter:
                    self._iter += 1
                    return item
                else:
                    await asyncio.sleep(0.0)
                    await super().put(item)
        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__}.get() failed " f"[iter={self._iter}]: {e}") from e

    async def put(self, item: _T) -> None:
        """Put Item

        Asynchronously adds an item to the queue.

        Args:
            item (_T): The item to be added to the queue.

        Raises:
            RuntimeError: If adding the item to the queue fails.
        """
        try:
            await super().put(item)
        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__}.put() failed [item={item}]: {e}") from e
