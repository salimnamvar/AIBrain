"""System - Base Object Class

This module defines the BaseObject class, which serves as a foundational class for all objects in the system.

Classes:
    BaseObject:
        Base abstract object class with common functionality such as logging, configuration access, and profiling.
"""

import logging
import time
from abc import ABC
from typing import Any, Optional

from src.utils.cfg import Configuration
from src.utils.perf.profile import Profiler


class BaseObject(ABC):
    """Base Object Class
    This class serves as a base for all objects in the system, providing common functionality
    such as logging, configuration access, and profiling.

    Attributes:
        id (Optional[int]): Unique identifier for the object.
        name (str): Name of the object.
        cfg (Optional[Configuration]): Configuration instance if enabled.
        logger (Optional[logging.Logger]): Logger instance if logging is enabled.
        use_prof (bool): Flag to enable profiling.
        use_cfg (bool): Flag to enable configuration access.
        use_log (bool): Flag to enable logging.
        init_time (float): Time when the object was initialized.
    """

    def __init__(
        self,
        a_id: Optional[int] = None,
        a_name: str = 'BaseObject',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the BaseObject with optional parameters.

        Args:
            a_id (Optional[int]): Unique identifier for the object.
            a_name (str): Name of the object.
            a_version (int): Version number of the object.
            a_use_prof (bool): Enable profiling for the object.
            a_use_cfg (bool): Enable configuration access.
            a_use_log (bool): Enable logging for the object.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        super().__init__(**kwargs)
        self._id: Optional[int] = a_id
        self._init_time = time.time()
        self._name: str = a_name
        self._use_prof: bool = a_use_prof
        self._use_cfg: bool = a_use_cfg
        self._use_log: bool = a_use_log

        if self._use_cfg:
            self._cfg: Optional[Configuration] = Configuration.get_instance()
        else:
            self._cfg = None

        if self._use_log:
            self._logger: Optional[logging.Logger] = logging.getLogger(self._name)
        else:
            self._logger = None

        if self._use_prof:
            self._profiler: Optional[Profiler] = Profiler.get_instance()

    @property
    def use_prof(self) -> bool:
        """Check if profiling is enabled.

        Returns:
            bool: True if profiling is enabled, False otherwise.
        """
        return self._use_prof

    @property
    def use_log(self) -> bool:
        """Check if logging is enabled.

        Returns:
            bool: True if logging is enabled, False otherwise.
        """
        return self._use_log

    @property
    def use_cfg(self) -> bool:
        """Check if configuration access is enabled.

        Returns:
            bool: True if configuration access is enabled, False otherwise.
        """
        return self._use_cfg

    @property
    def id(self) -> Optional[int]:
        """Get the ID of the object.

        Returns:
            Optional[int]: The unique identifier of the object.
        """
        return self._id

    @property
    def logger(self) -> Optional[logging.Logger]:
        """Get the logger for the object.

        Returns:
            Optional[logging.Logger]: The logger instance for the object.
        """
        return self._logger

    @property
    def cfg(self) -> Optional[Configuration]:
        """Get the configuration for the object.

        Returns:
            Optional[Configuration]: The configuration instance for the object.
        """
        return self._cfg

    @property
    def name(self) -> str:
        """Get the name of the object.

        Returns:
            str: The name of the object.
        """
        return self._name

    @property
    def profiler(self) -> Optional[Profiler]:
        """Get the profiler for the object.

        Returns:
            Optional[Profiler]: The profiler instance for the object.
        """
        return self._profiler
