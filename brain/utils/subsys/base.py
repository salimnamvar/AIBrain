""" Base Subsystem

    This file defines a base subsystem class that serves as a template for other subsystems. Subsystems based on this
    template are expected to provide implementations for train, test, and inference.
"""

# region Imported Dependencies
import logging
from abc import ABC, abstractmethod

from brain.utils.cfg import BrainConfig
from brain.utils.misc import BaseHealthStatus
from brain.utils.ml.util import BaseModel


# endregion Imported Dependencies


class HealthStatus(BaseHealthStatus):
    """Health status class for a base subsystem.

    This class inherits from :class:`BaseHealthStatus` and is designed to manage the health status of a
    base subsystem. It includes a constant NAME attribute indicating the name of the subsystem.

    Attributes:
        NAME (str):
            A string representing the name of the base subsystem.
    """

    NAME: str = "Base_Subsystem"


class BaseSubsystem(BaseModel, ABC):
    """Base Subsystem

    This abstract base class defines the structure for subsystems in the AIBrain framework.

    Subsystems are integral parts of the AIBrain framework responsible for specific tasks or functionality.
    They include components for inference, training, testing, and are used to organize and modularize
    the functionality of the framework.

    Attributes:
        cfg (:class:`BrainConfig`):
            An instance of the :class:`BrainConfig` class, providing access to configuration settings.
        name (str):
            The name of the subsystem.
        logger (:class:`logging.Logger`):
            The logger for recording log messages related to the subsystem.
        health_status (:class:`HealthStatus`):
            The health status object that tracks the health and status of the subsystem.
    """

    def __init__(self, *args, a_name: str = "SUBSYS", **kwargs):
        """Subsystem Constructor

        Initializes a new instance of the base subsystem class.

        Args:
            a_name (str, optional):
                The name of the subsystem. Defaults to 'SUBSYS'.
            *args:
                Variable-length arguments.
            **kwargs:
                Keyword arguments.
        """
        super().__init__(a_name)
        self.health_status: HealthStatus = HealthStatus(a_name)

    @abstractmethod
    def infer(self, *args, **kwargs):
        """Inference Method (Abstract)

        An abstract method to be implemented by subclasses. It defines the process of making inferences
        based on the subsystem's functionality.

        Args:
            *args: Variable-length arguments.
            **kwargs: Keyword arguments.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        NotImplementedError("Subclasses must implement `inference`")
