"""IO Utilities

This module provides abstract and concrete classes for input/output operations using queues.
Supports synchronous, asynchronous, and multiprocessing queue modes for flexible IO handling.

Classes:
    BaseIO: Abstract base class for IO operations.
    QueueIO: IO class using queues for input/output in sync, async, or multiprocessing modes.

Type Variables:
    IT: Type variable for input data.
    OT: Type variable for output data.
"""

import asyncio
import multiprocessing as mp
import multiprocessing.queues as mpq
import warnings
from abc import ABC, abstractmethod
from queue import Queue as StdQueue
from typing import Any, Generic, Literal, Optional, TypeVar

from src.utils.sys.b_obj import BaseObject

IT = TypeVar("IT", bound=Any, default=Any)
OT = TypeVar("OT", bound=Any, default=Any)


class BaseIO(Generic[IT, OT], BaseObject, ABC):
    """Abstract base class for input/output utilities.

    Provides a standard interface for input/output operations. Subclasses
    should implement synchronous, asynchronous, or multiprocessing queue-based
    IO by overriding the abstract methods.

    Attributes:
        _mode (str): Mode of operation ("sync", "async", "mp").
        _num_in_cons (int): Number of consumers for the input queue.
        _num_out_cons (int): Number of consumers for the output queue.
        _sentinel (object): Unique object used to signal queue completion.
        _input_done (bool): Flag indicating if input queue is done.
        _output_done (bool): Flag indicating if output queue is done.
    """

    def __init__(
        self,
        a_mode: Literal["sync", "async", "mp"] = "sync",
        a_num_in_cons: int = 1,
        a_num_out_cons: int = 1,
        a_id: Optional[int] = None,
        a_name: str = 'BaseIO',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the BaseIO object.

        Args:
            a_mode (Literal["sync", "async", "mp"]): IO mode.
            a_num_in_cons (int): Number of consumers reading from inputs.
            a_num_out_cons (int): Number of consumers reading from outputs.
            a_id (Optional[int]): Optional unique identifier.
            a_name (str): Name of the IO object.
            a_use_prof (bool): Enable profiling.
            a_use_cfg (bool): Enable configuration support.
            a_use_log (bool): Enable logging.
            **kwargs: Additional keyword arguments to pass to BaseObject.
        """
        super().__init__(
            a_id=a_id,
            a_name=a_name,
            a_use_prof=a_use_prof,
            a_use_cfg=a_use_cfg,
            a_use_log=a_use_log,
            **kwargs,
        )
        self._mode: str = a_mode
        self._sentinel = object()
        self._num_in_cons: int = a_num_in_cons
        self._num_out_cons: int = a_num_out_cons
        self._input_done: bool = False
        self._output_done: bool = False

    @property
    def num_in_cons(self) -> int:
        """Get the number of input consumers.

        Returns:
            int: Number of consumers reading from the input queue.
        """
        return self._num_in_cons

    @property
    def num_out_cons(self) -> int:
        """Get the number of output consumers.

        Returns:
            int: Number of consumers reading from the output queue.
        """
        return self._num_out_cons

    @property
    def mode(self) -> str:
        """Get mode of operation.

        Returns:
            str: Mode of operation.
        """
        return self._mode

    @abstractmethod
    def put_output(self, *args: Any, a_output: Optional[OT], **kwargs: Any) -> None:
        """Put an item into the output queue.

        Args:
            a_output (OT): The output item to put in the output queue.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError("Subclasses must implement `put_output` method.")

    @abstractmethod
    def get_output(self, *args: Any, **kwargs: Any) -> Optional[OT]:
        """Retrieve an item from the output queue.

        Returns:
            Optional[OT]: The output item.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError("Subclasses must implement `get_output` method.")

    @abstractmethod
    def put_input(self, *args: Any, a_input: IT, **kwargs: Any) -> None:
        """Put an item into the input queue.

        Args:
            a_input (IT): The input item to put in the input queue.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError("Subclasses must implement `put_input` method.")

    @abstractmethod
    def get_input(self, *args: Any, **kwargs: Any) -> Optional[IT]:
        """Retrieve an item from the input queue.

        Returns:
            Optional[IT]: The input item.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError("Subclasses must implement `get_input` method.")

    def release(self, *args: Any, **kwargs: Any) -> None:
        """Release the IO object and any resources.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError("Subclasses must implement `release` method.")

    @abstractmethod
    async def put_input_async(self, *args: Any, a_input: IT, **kwargs: Any) -> None:
        """Asynchronously put an input item into the input queue.

        Args:
            a_input (IT): The input item to put.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError("Subclasses must implement `put_input_async` method.")

    @abstractmethod
    async def get_input_async(self, *args: Any, **kwargs: Any) -> Optional[IT]:
        """Asynchronously retrieve an item from the input queue.

        Returns:
            Optional[IT]: The input item, or None if no item is available.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError("Subclasses must implement `get_input_async` method.")

    @abstractmethod
    async def put_output_async(self, *args: Any, a_output: OT, **kwargs: Any) -> None:
        """Asynchronously put an output item into the output queue.

        Args:
            a_output (OT): The output item to put.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError("Subclasses must implement `put_output_async` method.")

    @abstractmethod
    async def get_output_async(self, *args: Any, **kwargs: Any) -> Optional[OT]:
        """Asynchronously retrieve an item from the output queue.

        Returns:
            Optional[OT]: The output item, or None if no item is available.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError("Subclasses must implement `get_output_async` method.")

    @abstractmethod
    def input_done(self) -> None:
        """Mark the input queue as done.

        This signals that no more input will be provided. Subclasses
        should implement the behavior appropriate to their queue type.
        """
        raise NotImplementedError("Subclasses must implement `input_done` method.")

    @abstractmethod
    async def input_done_async(self) -> None:
        """Asynchronously mark the input queue as done.

        Subclasses should implement the async behavior for signaling that
        no more input will be provided.
        """
        raise NotImplementedError("Subclasses must implement `input_done_async` method.")

    @abstractmethod
    def output_done(self) -> None:
        """Mark the output queue as done.

        This signals that no more output will be provided. Subclasses
        should implement the behavior appropriate to their queue type.
        """
        raise NotImplementedError("Subclasses must implement `output_done` method.")

    @abstractmethod
    async def output_done_async(self) -> None:
        """Asynchronously mark the output queue as done.

        Subclasses should implement the async behavior for signaling that
        no more output will be provided.
        """
        raise NotImplementedError("Subclasses must implement `output_done_async` method.")

    @abstractmethod
    def is_input_done(self) -> bool:
        """Check whether input processing is complete.

        Returns:
            bool: True if the input queue is marked done and has no items left.
        """
        raise NotImplementedError("Subclasses must implement `is_input_done` method.")

    @abstractmethod
    def is_output_done(self) -> bool:
        """Check whether output processing is complete.

        Returns:
            bool: True if the output queue is marked done and has no items left.
        """
        raise NotImplementedError("Subclasses must implement `is_output_done` method.")

    @abstractmethod
    def reset(self, a_force: bool = False) -> None:
        """Reset the IO state.

        Args:
            a_force (bool): If True, clear queues even if they contain items.
        """
        raise NotImplementedError("Subclasses must implement `reset` method.")


class QueueIO(BaseIO[IT, OT]):
    """Queue-based IO supporting sync, async, and multiprocessing modes.

    This class allows putting and getting items from input and output
    queues in three modes:
        - "sync" : standard queue.Queue for synchronous usage
        - "async": asyncio.Queue for asynchronous usage
        - "mp"   : multiprocessing.Queue for inter-process communication

    Attributes:
        input (StdQueue[IT] | asyncio.Queue[IT] | mpq.Queue[IT]): Input queue.
        output (StdQueue[OT] | asyncio.Queue[OT] | mpq.Queue[OT]): Output queue.
        _input_done (bool): Indicates whether the input queue is finished.
        _output_done (bool): Indicates whether the output queue is finished.
    """

    def __init__(
        self,
        a_mode: Literal["sync", "async", "mp"] = "sync",
        a_in_qsize: int = 0,
        a_out_qsize: int = 0,
        a_in_queue: Optional[StdQueue[IT] | asyncio.Queue[IT] | mpq.Queue[IT]] = None,
        a_out_queue: Optional[StdQueue[OT] | asyncio.Queue[OT] | mpq.Queue[OT]] = None,
        a_num_in_cons: int = 1,
        a_num_out_cons: int = 1,
        a_id: int | None = None,
        a_name: str = 'QueueIO',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the QueueIO object.

        Args:
            a_mode (Literal["sync", "async", "mp"]): Mode of operation.
            a_in_qsize (int): Maximum size of the input queue.
            a_out_qsize (int): Maximum size of the output queue.
            a_in_queue (Optional[Queue]): Optional pre-existing input queue.
            a_out_queue (Optional[Queue]): Optional pre-existing output queue.
            a_num_in_cons (int): Number of consumers for input queue.
            a_num_out_cons (int): Number of consumers for output queue.
            a_id (Optional[int]): Optional unique identifier.
            a_name (str): Name of the IO object.
            a_use_prof (bool): Enable profiling.
            a_use_cfg (bool): Enable configuration.
            a_use_log (bool): Enable logging.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If an unsupported mode is provided.
        """
        super().__init__(
            a_mode=a_mode,
            a_num_in_cons=a_num_in_cons,
            a_num_out_cons=a_num_out_cons,
            a_id=a_id,
            a_name=a_name,
            a_use_prof=a_use_prof,
            a_use_cfg=a_use_cfg,
            a_use_log=a_use_log,
            **kwargs,
        )

        if a_mode == "sync":
            self._input: StdQueue[IT] = a_in_queue or StdQueue(maxsize=a_in_qsize)
            self._output: StdQueue[OT] = a_out_queue or StdQueue(maxsize=a_out_qsize)

        elif a_mode == "async":
            self._input: asyncio.Queue[IT] = a_in_queue or asyncio.Queue(maxsize=a_in_qsize)
            self._output: asyncio.Queue[OT] = a_out_queue or asyncio.Queue(maxsize=a_out_qsize)

        elif a_mode == "mp":
            self._input: mpq.Queue[IT] = a_in_queue or mp.Queue(maxsize=a_in_qsize)
            self._output: mpq.Queue[OT] = a_out_queue or mp.Queue(maxsize=a_out_qsize)

        else:
            raise ValueError(f"Unsupported mode: {a_mode}")

        self._in_qsize: int = self._input.qsize()
        self._out_qsize: int = self._output.qsize()

    @property
    def in_qsize(self) -> int:
        """Get the configured input queue size.

        Returns:
            int: Maximum size of the input queue.
        """
        return self._in_qsize

    @property
    def out_qsize(self) -> int:
        """Get the configured output queue size.

        Returns:
            int: Maximum size of the output queue.
        """
        return self._out_qsize

    def input_done(self) -> None:
        """Mark the input queue as done (sync/mp mode).

        Puts sentinel objects to signal consumers that no more data will arrive.

        Raises:
            RuntimeError: If called in async mode.
        """
        if self._mode == "async":
            raise RuntimeError("Use `await input_done_async()` in async mode.")

        if self._input_done:
            warnings.warn("input_done() already called, ignoring subsequent call.")
            return

        self._input_done = True
        for _ in range(self._num_in_cons):
            self._input.put(self._sentinel)

    async def input_done_async(self) -> None:
        """Mark the input queue as done (async mode).

        Puts sentinel objects asynchronously to signal consumers that no more data will arrive.

        Raises:
            RuntimeError: If not in async mode.
        """
        if self._mode != "async":
            raise RuntimeError("Async methods only valid in async mode.")

        if self._input_done:
            warnings.warn("input_done_async() already called, ignoring subsequent call.")
            return

        self._input_done = True
        for _ in range(self._num_in_cons):
            await self._input.put(self._sentinel)

    def output_done(self) -> None:
        """Mark the output queue as done (sync/mp mode).

        Puts sentinel objects to signal consumers that no more data will arrive.

        Raises:
            RuntimeError: If called in async mode.
        """
        if self._mode == "async":
            raise RuntimeError("Use `await output_done_async()` in async mode.")

        if self._output_done:
            warnings.warn("output_done() already called, ignoring subsequent call.")
            return

        self._output_done = True
        for _ in range(self._num_out_cons):
            self._output.put(self._sentinel)

    async def output_done_async(self) -> None:
        """Mark the output queue as done (async mode).

        Puts sentinel objects asynchronously to signal consumers that no more data will arrive.

        Raises:
            RuntimeError: If not in async mode.
        """
        if self._mode != "async":
            raise RuntimeError("Async methods only valid in async mode.")

        if self._output_done:
            warnings.warn("output_done_async() already called, ignoring subsequent call.")
            return

        self._output_done = True
        for _ in range(self._num_out_cons):
            await self._output.put(self._sentinel)

    def is_input_done(self) -> bool:
        """Check if the input queue is done and empty.

        Returns:
            bool: True if `input_done()` has been called and the input queue is empty, False otherwise.
        """
        return self._input_done and self._input.empty()

    def is_output_done(self) -> bool:
        """Check if the output queue is done and empty.

        Returns:
            bool: True if `output_done()` has been called and the output queue is empty, False otherwise.
        """
        return self._output_done and self._output.empty()

    @property
    def input(self) -> StdQueue[IT] | asyncio.Queue[IT] | mpq.Queue[IT]:
        """Get input queue.

        Returns:
            StdQueue[IT] | asyncio.Queue[IT] | mpq.Queue[IT]: Input queue.
        """
        return self._input

    @property
    def output(self) -> StdQueue[OT] | asyncio.Queue[OT] | mpq.Queue[OT]:
        """Get output queue.

        Returns:
            StdQueue[OT] | asyncio.Queue[OT] | mpq.Queue[OT]: Output queue.
        """
        return self._output

    def put_input(self, a_input: IT) -> None:
        """Put input data into the input queue (sync/mp mode).

        Args:
            a_input (IT): Input data to put in the queue.

        Raises:
            RuntimeError: If called in async mode or if input queue is done.
        """
        if self._mode == "async":
            raise RuntimeError("Use `await put_input_async` in async mode.")
        if self._input_done:
            raise RuntimeError("Cannot put input: input queue marked as done.")
        self._input.put(a_input)

    def get_input(self) -> Optional[IT]:
        """Get input data from the input queue (sync/mp mode).

        Returns:
            Optional[IT]: Input data from the queue or None if sentinel received.

        Raises:
            RuntimeError: If called in async mode.
        """
        if self._mode == "async":
            raise RuntimeError("Use `await get_input_async()` in async mode.")
        item = self._input.get()
        if item is self._sentinel:
            return None
        return item

    def put_output(self, a_output: OT) -> None:
        """Put output data into the output queue (sync/mp mode).

        Args:
            a_output (OT): Output data to put in the queue.

        Raises:
            RuntimeError: If called in async mode or if output queue is done.
        """
        if self._mode == "async":
            raise RuntimeError("Use `await put_output_async` in async mode.")
        if self._output_done:
            raise RuntimeError("Cannot put output: output queue marked as done.")
        self._output.put(a_output)

    def get_output(self) -> Optional[OT]:
        """Get output data from the output queue (sync/mp mode).

        Returns:
            Optional[OT]: Output data from the queue or None if sentinel received.

        Raises:
            RuntimeError: If called in async mode.
        """
        if self._mode == "async":
            raise RuntimeError("Use `await get_output_async()` in async mode.")
        item = self._output.get()
        if item is self._sentinel:
            return None
        return item

    async def put_input_async(self, a_input: IT) -> None:
        """Asynchronously put input data into the input queue (async mode).

        Args:
            a_input (IT): Input data to put in the queue.

        Raises:
            RuntimeError: If not in async mode.
        """
        if self._mode != "async":
            raise RuntimeError("Async methods only valid in async mode.")
        if self._input_done:
            raise RuntimeError("Cannot put input: input queue marked as done.")
        await self._input.put(a_input)

    async def get_input_async(self) -> Optional[IT]:
        """Asynchronously get input data from the input queue (async mode).

        Returns:
            Optional[IT]: Input data from the queue or None if sentinel received.

        Raises:
            RuntimeError: If not in async mode.
        """
        if self._mode != "async":
            raise RuntimeError("Async methods only valid in async mode.")
        item = await self._input.get()
        if item is self._sentinel:
            return None
        return item

    async def put_output_async(self, a_output: OT) -> None:
        """Asynchronously put output data into the output queue (async mode).

        Args:
            a_output (OT): Output data to put in the queue.

        Raises:
            RuntimeError: If not in async mode.
        """
        if self._mode != "async":
            raise RuntimeError("Async methods only valid in async mode.")
        if self._output_done:
            raise RuntimeError("Cannot put output: output queue marked as done.")
        await self._output.put(a_output)

    async def get_output_async(self) -> Optional[OT]:
        """Asynchronously get output data from the output queue (async mode).

        Returns:
            Optional[OT]: Output data from the queue or None if sentinel received.

        Raises:
            RuntimeError: If not in async mode.
        """
        if self._mode != "async":
            raise RuntimeError("Async methods only valid in async mode.")
        item = await self._output.get()
        if item is self._sentinel:
            return None
        return item

    def reset(self, a_force: bool = False) -> None:
        """Reset the input and output queues.

        Args:
            a_force (bool): If True, forcibly clear queues even if not empty.

        Notes:
            - If queues contain data and `a_force` is False, a warning is issued and queues are not cleared.
            - Resets both `_input_done` and `_output_done` flags to False.
            - Reinitializes queues based on current mode ("sync", "async", "mp").
        """
        in_not_empty = not self._input.empty()
        out_not_empty = not self._output.empty()

        if (in_not_empty or out_not_empty) and not a_force:
            warnings.warn("Reset called but queues are not empty. " "Use force=True to clear them.")
            return

        if a_force:
            try:
                while not self._input.empty():
                    self._input.get_nowait()
            except Exception:
                pass

            try:
                while not self._output.empty():
                    self._output.get_nowait()
            except Exception:
                pass

        self._input_done = False
        self._output_done = False

        if self._mode == "sync":
            self._input = StdQueue(maxsize=self.in_qsize)
            self._output = StdQueue(maxsize=self.out_qsize)

        elif self._mode == "async":
            self._input = asyncio.Queue(maxsize=self.in_qsize)
            self._output = asyncio.Queue(maxsize=self.out_qsize)

        elif self._mode == "mp":
            self._input = mp.Queue(maxsize=self.in_qsize)
            self._output = mp.Queue(maxsize=self.out_qsize)
