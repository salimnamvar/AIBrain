"""System - Base Abstract Synchronous Job

This module defines the base abstract job classes, serving as a foundation
for all job implementations in the system.

Classes:
    BaseSyncJob: An abstract base class for jobs that extends BaseModel.

Type Variables:
    - IOT: Type variable bound to BaseIO for input/output handling.

Type Aliases:
    - StopEvent: Union type for different event objects used in stopping execution.
"""

import logging
import multiprocessing.queues as mpq
from abc import ABC
from typing import Any, Literal, Optional, TypeVar

from aib.cnt.io import BaseIO
from aib.misc.common_types import StopEvent
from aib.sys.b_mdl import BaseModel

IOT = TypeVar("IOT", bound=BaseIO, default=BaseIO[Any, Any])


class BaseJob(BaseModel[IOT], ABC):
    """Base Abstract Synchronous Job.

    This class provides the foundation for implementing jobs,
    extending the BaseModel with job-specific attributes such as logging support.

    Attributes:
        log_queue (mpq.Queue[logging.LogRecord]): Multiprocessing-safe queue for log messages.
    """

    def __init__(
        self,
        a_log_queue: mpq.Queue[logging.LogRecord],
        a_call_mode: Literal["sync", "async"] = 'sync',
        a_io_mode: Literal["args", "queue", "ipc"] = "args",
        a_proc_mode: Literal["batch", "online"] = "online",
        a_backend: Literal["ovms", "openvino", "sys", "opencv", "decord"] = "sys",
        a_conc_mode: Optional[Literal["thread", "process"]] = None,
        a_max_workers: Optional[int] = None,
        a_io: Optional[IOT] = None,
        a_stop_event: Optional[StopEvent] = None,
        a_id: Optional[int] = None,
        a_name: str = 'BaseJob',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the base synchronous job.

        Args:
            a_log_queue (mpq.Queue[logging.LogRecord]): Queue for collecting log messages.
            a_call_mode (Literal["sync", "async"], optional): Execution mode for the model.
                Defaults to "sync".
            a_io_mode (Literal["args", "queue", "ipc"], optional): Input/output handling mode.
                Defaults to "args".
            a_proc_mode (Literal["batch", "online"], optional): Processing mode.
                Defaults to "online".
            a_backend (Literal["ovms", "openvino", "sys", "opencv"], optional): Backend engine for execution.
                Defaults to "sys".
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
            **kwargs (Any): Additional keyword arguments passed to BaseModel.
        """
        super().__init__(
            a_call_mode=a_call_mode,
            a_proc_mode=a_proc_mode,
            a_io_mode=a_io_mode,
            a_backend=a_backend,
            a_conc_mode=a_conc_mode,
            a_io=a_io,
            a_stop_event=a_stop_event,
            a_max_workers=a_max_workers,
            a_id=a_id,
            a_name=a_name,
            a_use_prof=a_use_prof,
            a_use_cfg=a_use_cfg,
            a_use_log=a_use_log,
            **kwargs,
        )
        self._log_queue: mpq.Queue[logging.LogRecord] = a_log_queue

    @property
    def log_queue(self) -> mpq.Queue[logging.LogRecord]:
        """Return the multiprocessing logging queue associated with this job."""
        return self._log_queue
