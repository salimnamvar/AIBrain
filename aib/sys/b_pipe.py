"""System - Base Abstract Pipeline

This module defines the base abstract pipeline classes, serving as foundational
classes for all pipelines within the system. Pipelines are modular components
responsible for sequential or parallel processing of data through multiple stages.

Classes:
    BasePipe: Base abstract pipeline class with common functionality.

Type Variables:
    - IOT: Type variable bound to BaseIO for input/output handling.

Type Aliases:
    - StopEvent: Union type for different event objects used in stopping execution.
"""

from abc import ABC
from typing import Any, Literal, Optional, TypeVar

from src.utils.cnt.io import BaseIO
from src.utils.misc.common_types import StopEvent
from src.utils.sys.b_mdl import BaseModel

IOT = TypeVar("IOT", bound=BaseIO, default=BaseIO[Any, Any])


class BasePipe(BaseModel[IOT], ABC):
    """Base Abstract Pipeline Class.

    Provides a foundational class for pipeline implementations, inheriting from BaseModel.
    Pipelines represent sequences of processing stages that can handle data synchronously
    or asynchronously, with optional input/output interfaces, logging, profiling, and
    concurrency control.

    Attributes:
        Inherited from BaseModel.
    """

    def __init__(
        self,
        a_call_mode: Literal["sync", "async"] = 'sync',
        a_io_mode: Literal["args", "queue", "ipc"] = "args",
        a_proc_mode: Literal["batch", "online"] = "online",
        a_backend: Literal["ovms", "openvino", "sys", "opencv", "decord"] = "sys",
        a_conc_mode: Optional[Literal["thread", "process"]] = None,
        a_max_workers: Optional[int] = None,
        a_io: Optional[IOT] = None,
        a_stop_event: Optional[StopEvent] = None,
        a_id: Optional[int] = None,
        a_name: str = 'BasePipe',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the base pipeline.

        Args:
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
            **kwargs (Any): Additional keyword arguments passed to BasePipe.
        """
        super().__init__(
            a_call_mode=a_call_mode,
            a_io_mode=a_io_mode,
            a_backend=a_backend,
            a_conc_mode=a_conc_mode,
            a_io=a_io,
            a_stop_event=a_stop_event,
            a_max_workers=a_max_workers,
            a_id=a_id,
            a_proc_mode=a_proc_mode,
            a_name=a_name,
            a_use_prof=a_use_prof,
            a_use_cfg=a_use_cfg,
            a_use_log=a_use_log,
            **kwargs,
        )
