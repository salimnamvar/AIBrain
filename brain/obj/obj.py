"""Base Object Module

This module defines the `BaseObject` class, which serves as the fundamental building block for handling objects.

The `BaseObject` class provides common features and functionalities for managing objects, including the ability
to represent an object as a dictionary, convert it to a string, and create deep copies.

Classes:
    BaseObject (ABC): A principle basic object class.
    ExtBaseObject (ABC): An extended basic object class.

Variables:
    None

Functions:
    None
"""

# region Imported Dependencies
import pprint
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TypeVar

# endregion Imported Dependencies

# TODO(doc): Complete the document of following type
TypeBaseObject = TypeVar("TypeBaseObject", bound="BaseObject")


class BaseObject(ABC):
    """Base Object

    The Base Object is a principle basic object class that has the common features and functionalities in handling
    an object.

    Attributes
        name:
            A :type:`string` that specifies the class object name.
    """

    def __init__(
        self,
        a_name: str = "Object",
    ) -> None:
        """Base Object

        This is a constructor that create an instance of the BaseObject object.

        Args
            a_name:
                A :type:`string` that specifies the name of the object.

        Returns
                The constructor does not return any values.
        """
        self.name: str = a_name

    @property
    def name(self) -> str:
        """Instance Name Getter

        This property specifies the name of the class object.

        Returns
            str: This property returns a :type:`string` as the name of the class object.
        """
        return self._name

    @name.setter
    def name(self, a_name: str = "BASE_OBJECT") -> None:
        """Instance Name Setter

        This setter is used to set the name of the class object.

        Args:
            a_name (str): A :type:`string` that specifies the class object's name.

        Returns:
            None
        """
        self._name = a_name.upper().replace(" ", "_")

    @abstractmethod
    def to_dict(self) -> dict:
        """To Dictionary

        This method represent the object as a dictionary. The method should be overridden.

        Returns
            dic:
                A dictionary that contains the object elements.
        """
        NotImplementedError("Subclasses must implement `to_dict`")

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

    def copy(self) -> TypeBaseObject:
        """Copy Instance

        This method copies the object deeply.

        Returns
            The method returns the duplicated object of the class.
        """
        return deepcopy(self)
