""" Health Status Module.

    This module defines classes for monitoring and managing the health status of components in an application.
"""

# region Imported Dependencies
import json
import threading
import time
from typing import List

from brain.obj import ExtBaseObject, BaseObjectList


# endregion Imported Dependencies


class BaseHealthStatus(ExtBaseObject):
    """Base Health Status

    Represents the base class for health status of a component.

    Attributes:
        name (str):
            A string that specifies the name of the health status component.
    """

    def __init__(self, a_name: str = "BaseHealthStatus") -> None:
        super().__init__(a_name)

    @property
    def is_healthy(self) -> bool:
        """Check if the component is healthy.

        Returns:
            bool:
                True if the component is healthy, False otherwise.
        """
        errors = []
        for name, sub_obj in self.__dict__.items():
            if isinstance(sub_obj, bool):
                errors.append(sub_obj)
            elif isinstance(sub_obj, (BaseHealthStatus, HealthStatusList)):
                errors.append(not sub_obj.is_healthy)
        healthy = not (any(errors) if len(errors) else False)
        return healthy

    def to_dict(self) -> dict:
        """Convert the health status to a dictionary.

        Returns:
            dict:
                A dictionary representation of the health status.
        """
        dic = {self.name.upper(): self.is_healthy}
        return dic

    def to_str(self) -> str:
        """Convert the health status to a string.

        Returns:
            str:
                A string representation of the health status.
        """
        return f"{self.name.upper()}: {self.is_healthy}"


class HealthStatusList(BaseObjectList[BaseHealthStatus]):
    """Health Status List.

    Represents a list of health status objects.

    Attributes:
        name (str): A string that specifies the name of the health status list.
        max_size (int): An integer that represents the maximum size of the list.
        items (List[BaseHealthStatus]): A list of BaseHealthStatus objects.
    """

    def __init__(
        self,
        a_name: str = "HealthStatusList",
        a_max_size: int = -1,
        a_items: List[BaseHealthStatus] = None,
    ):
        """Initialize a HealthStatusList object.

        Args:
            a_name (str): A string that specifies the name of the health status list.
            a_max_size (int): An integer that represents the maximum size of the list.
            a_items (List[BaseHealthStatus]): A list of BaseHealthStatus objects.
        """
        super().__init__(a_name, a_max_size, a_items)

    @property
    def is_healthy(self) -> bool:
        """Check if all items in the list are healthy.

        Returns:
            bool: True if all items are healthy, False otherwise.
        """
        return all([item.is_healthy for item in self.items])

    def to_dict(self) -> dict:
        """Convert the health status list to a dictionary.

        Returns:
            dict: A dictionary representation of the health status list.
        """
        sub = {}
        for item in self.items:
            sub.update(item.to_dict())
        dic = {self.name.upper(): sub}
        return dic


class HealthHub(BaseObjectList[BaseHealthStatus]):
    """Health Check Hub

    Represents a collection of health statuses for various components. It is inherited from BaseObjectList
    and contains BaseHealthStatus objects. This hub periodically performs health checks and writes the status data to a
    file.

    Attributes:
        file (str):
            The file to write health status data to.
        heartbeat_interval (int):
            The heartbeat interval, which defines how often health checks are performed.
    """

    def __init__(
        self,
        a_heartbeat_interval: int,
        a_file: str,
        a_name: str = "Health_Statuses",
        a_max_size: int = -1,
    ):
        """Initialize HealthCheckHub.

        Initialize a new instance of HealthCheckHub.

        Args:
            a_heartbeat_interval (int):
                The interval in minutes for heartbeats.
            a_file (str):
                The file to write health status data to.
            a_name (str, optional):
                The name of the collection (default is 'Health_Checks').
            a_max_size (int, optional):
                The maximum size of the collection (default is -1).
        """
        super().__init__(a_name=a_name, a_max_size=a_max_size)
        self.file: str = a_file
        self.heartbeat_interval: int = a_heartbeat_interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._health_check, daemon=True)
        self._thread.start()

    @property
    def is_healthy(self) -> bool:
        """Check if all components in the collection are healthy.

        Returns:
            bool:
                True if all components are healthy, False otherwise.
        """
        return all([item.is_healthy for item in self.items])

    def _health_check(self) -> None:
        """Health Check Thread

        Periodically write health status data to a file.
        """
        while not self._stop_event.is_set():
            with open(self.file, "w") as file:
                json.dump(self.to_dict(), file)
            time.sleep(self.heartbeat_interval * 60)

    def to_dict(self) -> dict:
        """Convert health statuses to a dictionary.

        Returns:
            dict:
                A dictionary representation of the collection.
        """
        sub = {}
        for item in self.items:
            sub.update(item.to_dict())
        dic = {self.name.upper(): sub}
        return dic
