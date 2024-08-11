"""Disk Monitoring Utilities

This module provides utilities for monitoring disk space usage and warnings.
"""

# region Imported Dependencies
import json
import logging
import threading
import time

import psutil

from brain.util.cfg import BrainConfig
from brain.util.disk.stats import DiskStats
from brain.util.misc import BaseHealthStatus


# endregion Imported Dependencies


class DiskHealthStatus(BaseHealthStatus):
    """Health status class for monitoring disk space warnings.

    Attributes:
        SPACE_WARNING (bool): A boolean indicating whether there is a disk space warning.
    """
    SPACE_WARNING: bool = False


class DiskMonitor:
    """Disk Monitoring Utility.

    This class provides disk monitoring functionality, periodically checking and saving disk usage statistics.
    It runs as a background thread and saves the disk statistics to a specified file at regular intervals.

    Attributes:
        name (str): The name of the disk monitor instance.
        path (str): The path of the disk directory to monitor.
        file (str): The file path to save the disk usage statistics.
        usage_hc_threshold (float): The usage threshold for generating a high-usage warning.
        heartbeat_interval (int): The interval (in minutes) at which the disk usage is monitored and saved.
        logger (Logger): The logger instance for logging disk monitoring information.
        health_status (DiskHealthStatus): An instance of `DiskHealthStatus` to track the health status of the disk.
    """

    def __init__(self, a_heartbeat_interval: int, a_file: str, a_usage_hc_threshold: float,
                 a_name: str = 'DISK', a_path: str = '/') -> None:
        """Initialize a DiskMonitor instance.

        Args:
            a_heartbeat_interval (int): The interval (in minutes) at which the disk usage is monitored and saved.
            a_file (str): The file path to save the disk usage statistics.
            a_usage_hc_threshold (float): The usage threshold for generating a high-usage warning.
            a_name (str, optional): The name of the disk monitor instance. Defaults to 'DISK'.
            a_path (str, optional): The path of the disk directory to monitor. Defaults to '/'.
        """
        self.cfg: BrainConfig = BrainConfig.get_instance()
        self.name: str = a_name
        self.path: str = a_path
        self.file: str = a_file
        self.usage_hc_threshold: float = a_usage_hc_threshold
        self.heartbeat_interval: int = a_heartbeat_interval
        self.logger = logging.getLogger(self.cfg.log.name + "." + self.name)
        self.health_status: DiskHealthStatus = DiskHealthStatus(a_name)
        self._thread = threading.Thread(target=self.__monitor_disk, daemon=True)

    def __monitor_disk(self) -> None:
        """Periodically monitor disk usage and save statistics.

        This private method runs in a loop, periodically checking the disk usage statistics using the
        `__space_usage` method and saving the results to the specified file using the `__save` method. The interval
        between checks is determined by the `heartbeat_interval` attribute.

        Note:
            This method is intended to run as a background thread, continuously monitoring and updating disk usage
            statistics.
        """
        while True:
            stats: DiskStats = self.__space_usage()
            self.__save(a_stats=stats)
            time.sleep(self.heartbeat_interval * 60)

    def __save(self, a_stats: DiskStats) -> None:
        """Save disk usage statistics to a file.

        This private method takes the provided `DiskStats` instance, converts it to a dictionary using the `to_dict`
        method, and then saves the dictionary to the specified file in JSON format.

        Args:
            a_stats (DiskStats): The disk usage statistics to be saved.
        """
        with open(self.file, 'w') as file:
            json.dump(a_stats.to_dict(), file)

    def __space_usage(self) -> DiskStats:
        """Retrieve and calculate disk space usage statistics.

        This private method uses the `psutil` library to retrieve disk usage statistics, calculates the percentage of
        used and free space, and updates the `SPACE_WARNING` flag in the `health_status` instance based on the
        specified threshold.

        Returns:
            DiskStats: An instance of `DiskStats` containing the calculated disk usage statistics.
        """
        total, used, free, percent = psutil.disk_usage(self.path)
        free_percentage = (free / total) * 100
        used_percentage = (used / total) * 100

        if used_percentage > self.usage_hc_threshold:
            self.health_status.SPACE_WARNING = True
        else:
            self.health_status.SPACE_WARNING = False

        return DiskStats(a_total=total, a_used=used, a_free=free, a_used_percentage=used_percentage,
                         a_free_percentage=free_percentage)

    def monitor(self) -> None:
        """Start monitoring the disk.

        This method starts the background thread for monitoring disk usage. The thread periodically checks the disk
        usage statistics and saves them to the specified file at intervals defined by the `heartbeat_interval`.

        Note:
            This method should be called after initializing the `DiskMonitor` instance to start the monitoring process.
        """
        self._thread.start()
