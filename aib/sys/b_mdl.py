"""System - Base Abstract Models

This module defines the base abstract model classes for the system, providing a foundation
for model implementations.

Classes:
    BaseModel: Base abstract model class with common functionality.

Type Variables:
    - IOT: Type variable bound to BaseIO for input/output handling.

Type Aliases:
    - StopEvent: Union type for different event objects used in stopping execution.
"""

import os
from abc import ABC
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Generic, Literal, Optional, TypeVar

from src.utils.cnt.io import BaseIO
from src.utils.misc.common_types import StopEvent
from src.utils.sys.b_obj import BaseObject

IOT = TypeVar("IOT", bound=BaseIO, default=BaseIO[Any, Any])


class BaseModel(Generic[IOT], BaseObject, ABC):
    """Base Abstract Model Class.

    Provides a common foundation for all model implementations, including configuration,
    logging, profiling, iteration tracking, execution mode, and input/output handling.

    Attributes:
        iter (int): Current iteration count of the model.
        call_mode (str): Execution mode, either "sync" or "async".
        io_mode (str): Input feeding mode, e.g., "args", "queue", or "ipc".
        proc_mode (str): Processing mode, either "batch" or "online".
        backend (str): Backend engine for model execution, e.g., "ovms", "openvino", "sys".
        io (Optional[IOT]): IO object for feeding or retrieving data.
        stop_event (Optional[StopEvent]): Stop event for graceful shutdown.
        conc_mode (Optional[Literal["thread", "process"]]): Concurrency mode.
        max_workers (Optional[int]): Maximum number of concurrent jobs.
        executor (Optional[ProcessPoolExecutor | ThreadPoolExecutor]): Executor for parallel jobs.

    Abstract Methods:
        step(*args, **kwargs): Increment iteration count and optionally perform profiling.
        infer(*args, **kwargs): Perform synchronous inference.
        infer_async(*args, **kwargs): Perform asynchronous inference.
        run(*args, **kwargs): Start synchronous execution loop.
        run_async(*args, **kwargs): Start asynchronous execution loop.
        dispatch(): Dispatch synchronous operations.
        dispatch_async(): Dispatch asynchronous operations.
        load(*args, **kwargs): Define model loading procedure.
    """

    def __init__(
        self,
        a_call_mode: Literal["sync", "async"] = 'sync',
        a_io_mode: Literal["args", "queue", "ipc"] = "args",
        a_proc_mode: Literal["batch", "online"] = "online",
        a_backend: Literal["ovms", "openvino", "sys", "opencv", "decord", "pyarrow", "ultralytics"] = "sys",
        a_conc_mode: Optional[Literal["thread", "process"]] = None,
        a_max_workers: Optional[int] = None,
        a_io: Optional[IOT] = None,
        a_stop_event: Optional[StopEvent] = None,
        a_id: Optional[int] = None,
        a_name: str = 'BaseModel',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the BaseModel.

        Args:
            a_call_mode (Literal["sync", "async"], optional): Execution mode for the model.
                Defaults to "sync".
            a_io_mode (Literal["args", "queue", "ipc"], optional): Input/output handling mode.
                Defaults to "args".
            a_proc_mode (Literal["batch", "online"], optional): Processing mode.
                Defaults to "online".
            a_backend (Literal["ovms", "openvino", "sys", "opencv", "decord", "pyarrow", "ultralytics"], optional):
                Backend engine for execution. Defaults to "sys".
            a_conc_mode (Optional[Literal["thread", "process"]], optional): Concurrency mode.
                Defaults to None.
            a_max_workers (Optional[int], optional): Maximum number of concurrent jobs.
                Defaults to None.
            a_io (Optional[IOT], optional): Input/output interface object for feeding or retrieving data.
                Defaults to None.
            a_stop_event (Optional[StopEvent], optional): Event object used to signal termination.
                Defaults to None.
            a_id (Optional[int], optional): Unique identifier for the model instance.
                Defaults to None.
            a_name (str, optional): Human-readable name of the model instance.
                Defaults to "BaseModel".
            a_use_prof (bool, optional): Enable profiling support (e.g., iteration timing, FPS tracking).
                Defaults to False.
            a_use_cfg (bool, optional): Enable configuration management features.
                Defaults to True.
            a_use_log (bool, optional): Enable logging for debugging and runtime events.
                Defaults to True.
            **kwargs (Any): Additional keyword arguments passed to BaseObject.
        """
        super().__init__(
            a_id=a_id,
            a_name=a_name,
            a_use_prof=a_use_prof,
            a_use_cfg=a_use_cfg,
            a_use_log=a_use_log,
            **kwargs,
        )
        self._iter: int = 0
        self._call_mode: Literal["sync", "async"] = a_call_mode
        self._io_mode: Literal["args", "queue", "ipc"] = a_io_mode
        self._proc_mode: Literal["batch", "online"] = a_proc_mode
        self._backend: Literal['ovms', 'openvino', 'sys', 'opencv', 'decord', 'pyarrow', 'ultralytics'] = a_backend
        self._conc_mode: Optional[Literal["thread", "process"]] = a_conc_mode
        self._max_workers: Optional[int] = a_max_workers
        self._io: Optional[IOT] = a_io
        self._stop_event: Optional[StopEvent] = a_stop_event
        self._executor: Optional[ProcessPoolExecutor | ThreadPoolExecutor] = None
        if a_conc_mode == "process":
            if self._max_workers is None:
                self._max_workers = os.process_cpu_count() or 1
            self._executor = ProcessPoolExecutor(max_workers=self._max_workers)
        elif a_conc_mode == "thread":
            if self._max_workers is None:
                self._max_workers = min(32, (os.process_cpu_count() or 1) + 4)
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

    @property
    def conc_mode(self) -> Optional[str]:
        """Concurrency mode used by the model.

        Returns:
            Optional[str]: The concurrency mode ("thread" or "process").
        """
        return self._conc_mode

    @property
    def executor(self) -> Optional[ProcessPoolExecutor | ThreadPoolExecutor]:
        """Accessor for the jobs pool executor.

        Returns:
            Optional[ProcessPoolExecutor | ThreadPoolExecutor]:
                The executor handling concurrent job execution.
        """
        return self._executor

    @property
    def iter(self) -> int:
        """Current iteration count of the model.

        Returns:
            int: The number of iterations completed.
        """
        return self._iter

    @property
    def io(self) -> Optional[IOT]:
        """Accessor for the input/output interface.

        Returns:
            Optional[IOT]: The input/output interface object.
        """
        return self._io

    @property
    def stop_event(self) -> Optional[StopEvent]:
        """Accessor for the stop event.

        Returns:
            Optional[StopEvent]: The event object used to signal termination.
        """
        return self._stop_event

    @property
    def max_workers(self) -> Optional[int]:
        """Number of concurrent jobs.

        Returns:
            Optional[int]: Maximum number of workers allowed for concurrency.
        """
        return self._max_workers

    @property
    def backend(self) -> str:
        """Backend engine used by the model.

        Returns:
            str: The backend engine (e.g., "ovms", "openvino", "sys").
        """
        return self._backend

    @property
    def call_mode(self) -> str:
        """Execution call mode.

        Returns:
            str: Either "sync" for synchronous mode or "async" for asynchronous mode.
        """
        return self._call_mode

    @property
    def io_mode(self) -> str:
        """Input feeding mode.

        Returns:
            str: The IO feeding mode ("args", "queue", or "ipc").
        """
        return self._io_mode

    @property
    def proc_mode(self) -> str:
        """Processing mode.

        Returns:
            str: Either "batch" for batch processing or "online" for online processing.
        """
        return self._proc_mode

    def step(self, *args: Any, **kwargs: Any) -> None:
        """Increment iteration count and perform optional profiling.

        Can be extended in derived classes for logging FPS or additional metrics.
        """
        self._iter += 1

    def infer(self, *args: Any, **kwargs: Any) -> Any:
        """Perform synchronous inference.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement `infer` method.")

    async def infer_async(self, *args: Any, **kwargs: Any) -> Any:
        """Perform asynchronous inference.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement `infer_async` method.")

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Start synchronous execution loop.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement `run` method.")

    async def run_async(self, *args: Any, **kwargs: Any) -> Any:
        """Start asynchronous execution loop.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement `run_async` method.")

    def dispatch(self) -> None:
        """Dispatch method for synchronous operations.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement `dispatch` method.")

    async def dispatch_async(self) -> None:
        """Dispatch method for asynchronous operations.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement `dispatch_async` method.")

    def reset(self, a_force: bool = False) -> None:
        """Reset the model state.

        Resets the internal state of the model to its initial conditions, including
        clearing iteration counters, shutting down or reinitializing executors, and
        resetting IO queues (if applicable). This is useful when restarting execution
        or reusing the model instance without reconstructing it.

        Args:
            a_force (bool, optional): If True, forcefully resets all components
                (including IO and executor), even if they still contain pending
                tasks or data. If False, a warning may be raised if the model or IO
                is not in a clean state. Defaults to False.

        Raises:
            NotImplementedError: This base method must be implemented by subclasses.
            RuntimeWarning: If queues, workers, or executors still contain pending
                tasks and `a_force=False`.
        """
        raise NotImplementedError("Subclasses must implement `reset` method.")

    def load(self, *args: Any, **kwargs: Any) -> None:
        """Load Method

        This method is intended to be overridden by subclasses to implement specific loading logic.
        It should handle loading the model's state or configuration.

        Args:
            *args (Tuple[Any, ...]): Positional arguments for loading.
            **kwargs (Dict[str, Any]): Keyword arguments for loading.

        Returns:
            None: This method does not return any value.
        """
        raise NotImplementedError(f"`{self.name}` must implement `load`")
