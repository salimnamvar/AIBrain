"""Miscellaneous - Singleton Meta Class Utilities

This module provides a SingletonMeta class that ensures only one instance of a class exists.

Classes:
    SingletonMeta:
        A metaclass that implements the Singleton design pattern.

Type Variables:
    T: Type variable for singleton instances.
"""

from abc import ABCMeta
from typing import ClassVar, Dict, Optional, Type, TypeVar, cast

T = TypeVar("T")


class SingletonMeta(ABCMeta):
    """Singleton Meta Class.

    A metaclass that enforces the Singleton design pattern, ensuring that a class
    using this metaclass can only have one instance throughout the application.

    Attributes:
        __instances (Dict[type, object]): A dictionary to hold singleton instances
            of classes, keyed by the class type.
    """

    __instances: ClassVar[Dict[type, object]] = {}

    def __call__(cls: Type[T], *args: object, **kwargs: object) -> T:
        """Creates or retrieves the singleton instance of the class.

        Args:
            *args (object): Positional arguments to initialize the instance (only
                used the first time).
            **kwargs (object): Keyword arguments to initialize the instance (only
                used the first time).

        Returns:
            T: The singleton instance of the class.
        """
        if cls not in SingletonMeta.__instances:
            instance = super().__call__(*args, **kwargs)
            SingletonMeta.__instances[cls] = instance
        return cast(T, SingletonMeta.__instances[cls])

    def get_instance(cls: Type[T]) -> Optional[T]:
        """Retrieves the singleton instance of the class if it exists.

        Returns:
            Optional[T]: The existing singleton instance, or None if it has not
            been created yet.
        """
        return cast(Optional[T], SingletonMeta.__instances.get(cls, None))

    def release(cls: Type[T]) -> None:
        """Removes the singleton instance of the class.

        This allows creating a new instance the next time the class is instantiated.
        """
        if cls in SingletonMeta.__instances:
            del SingletonMeta.__instances[cls]
