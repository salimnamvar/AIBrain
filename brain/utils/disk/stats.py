"""Disk Monitoring Statistics Utilities

This module provides a utility class for representing disk usage statistics.
"""

# region Imported Dependencies
from brain.utils.obj import BaseObject


# endregion Imported Dependencies


class DiskStats(BaseObject):
    """
    Represents disk usage statistics.

    Attributes:
        total (float): Total disk space.
        used (float): Used disk space.
        used_percentage (float): Percentage of disk space used.
        free (float): Free disk space.
        free_percentage (float): Percentage of free disk space.
    """

    def __init__(self, a_total: float, a_used: float, a_used_percentage: float,
                 a_free: float, a_free_percentage: float, a_name: str = 'DiskStats') -> None:
        """
        Initialize DiskStats object.

        Args:
            a_total (float): Total disk space.
            a_used (float): Used disk space.
            a_used_percentage (float): Percentage of disk space used.
            a_free (float): Free disk space.
            a_free_percentage (float): Percentage of free disk space.
            a_name (str, optional): Name of the disk statistics object. Defaults to 'DiskStats'.
        """
        super().__init__(a_name)
        self.total: float = a_total
        self.used: float = a_used
        self.used_percentage: float = a_used_percentage
        self.free: float = a_free
        self.free_percentage: float = a_free_percentage

    def to_dict(self) -> dict:
        """
        Convert the disk statistics to a dictionary.

        Returns:
            dict: A dictionary representation of the disk statistics.
        """
        dic = {'Disk_Usage_Statistics': {'Total': self.total,
                                         'Used': self.used,
                                         'Used_Percentage': self.used_percentage,
                                         'Free': self.free,
                                         'Free_Percentage': self.free_percentage
                                         }}
        return dic
