"""Base Job

    This file defines an abstract base brain job. A brain job represents a signature for jobs in the AIBrain
    framework. The Brain core can take one job and run it as a pipeline that may consist of running other subsystems
    and modules.
"""

# region Imported Dependencies
import logging
from abc import ABC, abstractmethod

from brain.util.cfg import BrainConfig
from brain.util.misc import BaseHealthStatus
from brain.util.ml.util import BaseModel


# endregion Imported Dependencies


class HealthStatus(BaseHealthStatus):
    """Health Status for Base Job

    Represents the health status of the Base Job.

    Attributes:
        NAME (str):
            The name of the health status, which is set to 'Base_Job'.
    """

    NAME: str = "Base_Job"


class BaseJob(BaseModel, ABC):
    """Base Job

    This class defines an abstract base job for the AIBrain framework.

    Attributes:
        cfg (:class:`BrainConfig`):
            The :class:`BrainConfig` instance for the AIBrain framework.
        name (str):
            The name of the job.
        logger (:class:`logging.Logger`):
            The :class:`logging.Logger` instance for logging messages related to the job.
        health_status (HealthStatus):
            The health status of the job.
    """

    def __init__(self, *args, a_name: str = "BaseJob", **kwargs):
        """Initialize Base Job

        Initialize an instance of the BaseJob.

        Args:
            a_name (str, optional):
                The name of the job (default is 'BaseJob').
            *args:
                Additional positional arguments.
            **kwargs:
                Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(a_name)
        self.health_status: HealthStatus = HealthStatus(a_name)

    @abstractmethod
    def infer(self, *args, **kwargs):
        """Inference Job

        This abstract method defines the behavior of an inference job.

        Args:
            *args:
                Additional positional arguments.
            **kwargs:
                Additional keyword arguments.

        Returns:
            None
        """
        NotImplementedError("Subclasses must implement `inference`")
