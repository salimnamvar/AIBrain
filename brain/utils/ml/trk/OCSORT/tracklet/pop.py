"""Population Target Container Modules

This module contains classes related to storing populations of target lists.

"""

# region Imported Dependencies
import uuid
from typing import Union, List
from .tgt import KFTargetList
from brain.utils.obj import BaseObjectDict

# endregion Imported Dependencies


class PopulationDict(BaseObjectDict[uuid.UUID, KFTargetList]):
    """Dictionary for storing populations of KFTargetLists with UUID keys.

    This class extends the BaseObjectDict and is specifically designed for storing populations of KFTargetList objects
    with UUID keys. It provides additional functionality for managing a dictionary of KFTargetList instances, where each
    instance represents a population of targets related to a specific camera or scenario.

    Attributes:
        name (str): Name of the PopulationDict.
        max_size (int): Maximum size limit for the dictionary. If set to -1, there is no limit.
        items (Dict[uuid.UUID, KFTargetList]): Dictionary containing KFTargetList objects with UUID keys.
    """

    def __init__(
        self,
        a_name: str = "PopulationDict",
        a_max_size: int = -1,
        a_key: Union[uuid.UUID, List[uuid.UUID]] = None,
        a_value: Union[KFTargetList, List[KFTargetList]] = None,
    ):
        """Initialize a PopulationDict instance.

        Args:
            a_name (str, optional): Name of the PopulationDict (default is "PopulationDict").
            a_max_size (int, optional):
                Maximum size limit for the dictionary. If set to -1, there is no limit (default is -1).
            a_key (Union[uuid.UUID, List[uuid.UUID]], optional):
                Initial key or list of keys for the dictionary (default is None).
            a_value (Union[KFTargetList, List[KFTargetList]], optional):
                Initial value or list of values for the dictionary (default is None).

        Returns:
            None
        """
        super().__init__(a_name, a_max_size, a_key, a_value)
