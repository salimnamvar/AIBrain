""" BasePipeline Module

This module defines the abstract base class `BasePipeline` for creating pipelines that consist of a chain of processing
subsystems. Pipelines can run independently by having an input generator and passing the inputs through processing units
(subsystems) with customizable behavior.

"""

# region Imported Dependencies
import logging
from abc import ABC, abstractmethod

from aib.misc import BaseHealthStatus
from aib.ml.util import BaseModel


# endregion Imported Dependencies


class HealthStatus(BaseHealthStatus):
    """Health status class for a base pipeline.

    This class inherits from :class:`BaseHealthStatus` and is designed to manage the health status of a
    base pipeline. It includes a constant NAME attribute indicating the name of the pipeline.

    Attributes:
        NAME (str):
            A string representing the name of the base pipeline.
    """

    NAME: str = "Base_Pipeline"


class BasePipeline(BaseModel, ABC):
    """Base Pipeline

    This class serves as a foundation for creating pipelines. A pipeline is a series of processing
    subsystems arranged to process inputs and produce outputs. The base pipeline provides common
    functionality for initializing pipeline attributes and managing its health status.

    Attributes:
        cfg (BrainConfig):
            An instance of the BrainConfig class containing configuration settings for the pipeline.
        name (str):
            A string specifying the name of the pipeline.
        logger (logging.Logger):
            A logger object for logging pipeline-related messages.
        health_status (HealthStatus):
            An instance of the HealthStatus class to manage the health status of the pipeline.
    """

    def __init__(self, *args, a_name: str = "BasePipeline", **kwargs):
        """
        Constructor for the BasePipeline class.

        Args:
            *args:
                Variable length argument list.
            a_name (str, optional):
                A string specifying the name of the BasePipeline instance (default is "BasePipeline").
            **kwargs:
                Arbitrary keyword arguments.

        Returns:
            None: The constructor does not return any values.
        """
        super().__init__(a_name, a_use_cfg=True)
        self.health_status: HealthStatus = HealthStatus(a_name)

    @abstractmethod
    def infer(self, *args, **kwargs):
        """Carry out the inference process.

        Subclasses must implement this method to define the behavior of the inference process.

        Args:
            *args:
                Variable length argument list.
            **kwargs:
                Arbitrary keyword arguments.

        Returns:
            None: This method does not return any values.
        """
        NotImplementedError("Subclasses must implement `inference`")
