"""Configuration Parameters

    This file contains the struct classes of defining the configuration parameters.
"""

# region Imported Dependencies
import pprint
from brain.util.cfg.type_parse import parser

# endregion Imported Dependencies


class CFG:
    """CFG Base Class

    This is a base class for configuration parameters. It is used to define the structure of configuration settings.

    Attributes:
        This class has no specific attributes at the initial state.
    """

    def __init__(self):
        """CFG Constructor

        This constructor initializes an instance of the CFG class.

        Args:
            This constructor does not accept any arguments.

        Returns:
            None
        """

    def to_dict(self) -> dict:
        """To Dictionary
        This method represent the object as a dictionary.

        Returns
            dic:
                A dictionary that contains the object elements.
        """
        return self.__dict__

    def to_str(self) -> str:
        """To String

        This method represent the object as a string.

        Returns
            message:
                A :type:`string` as the object representative.
        """
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """Represent Instance

        This method represents the object of the class as a string.

        Returns
            message:
                This method returns a :type:`string` as the representation of the class object.
        """
        return self.to_str()


class BrainConfig:
    """Brain Configuration Manager

    This class represents the configuration manager for the application's brain. It reads and manages the
    configuration settings from a configuration file.

    Attributes:
        cfg_path (str):
            The path to the configuration file.
    """

    __instance = None

    def __init__(self, a_cfg: str) -> None:
        """Initialize BrainConfig

        Initialize the BrainConfig instance.

        Args:
            a_cfg (str):
                The path to the configuration file.

        Raises:
            Exception: If there is an attempt to create multiple instances of BrainConfig.

        Returns:
            None
        """

        if BrainConfig.__instance is not None:
            raise Exception("The `Config` class is allowed to have one instance.")
        else:
            self.cfg_path: str = a_cfg

            # Parse Configuration file
            self.__parse_file()
            # Sub Configurations

            BrainConfig.__instance = self

    def __parse_file(self):
        """Parse Configuration File

        This method parses the configuration file and populates the configuration settings accordingly.

        Returns:
            None
        """
        with open(self.cfg_path, "r") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                key, value = line.split("=")
                keys = key.split(".")

                current_level = self

                for k in keys[:-1]:
                    if k not in current_level.__dict__:
                        obj = CFG()
                        current_level.__dict__[k] = obj
                    current_level = current_level.__dict__[k]
                current_level.__dict__[keys[-1]] = parser(value)

    @staticmethod
    def get_instance(a_cfg: str = None) -> "BrainConfig":
        """Get BrainConfig Instance

        This method retrieves the instance of BrainConfig or creates one if it doesn't exist.

        Args:
            a_cfg (str, optional):
                The path to the configuration file. It's only used when creating the instance.

        Returns:
            'BrainConfig':
                The BrainConfig instance.
        """
        if BrainConfig.__instance is None:
            BrainConfig(a_cfg=a_cfg)
        return BrainConfig.__instance
