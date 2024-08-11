""" BaseOutput Module

This module defines the :class:`BaseOutput` class, which serves as a base class for defining output objects of
processing units such as subsystems.

"""


# region Imported Dependencies
from brain.util.obj import BaseObject

# endregion Imported Dependencies


class BaseOutput(BaseObject):
    """
    Base class for defining output objects of processing units.

    Attributes:
        name (str): Name of the output object.
        producer (str): Name of the producer.

    Methods:
        to_dict() -> dict:
            Converts the output object to a dictionary.

    Usage:
        This class is intended to be used as a base for creating specific output classes. Subclasses of `BaseOutput` can
        include additional attributes and methods as needed.
    """

    def __init__(self, a_producer: str, a_name: str = "Base_Output"):
        """
        Constructor for the `BaseOutput` class.

        Args:
            a_name (str):
                A string specifying the name of the `BaseOutput` instance.
            a_producer (str):
                A string specifying the name of the producer which created this output object.

        Returns:
            None: The constructor does not return any values.
        """
        super().__init__(a_name=a_name)
        self.producer: str = a_producer

    def to_dict(self) -> dict:
        """
        Converts the output object to a dictionary.

        Returns:
            dict: A dictionary representation of the output object.
        """
        dic = {"name": self.name, "producer": self.producer}
        return dic
