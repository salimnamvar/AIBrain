""" Data Type Check Utilities

This module includes the utility functions for checking the data-type of values.
"""

# region Import Dependencies
import uuid

import numpy as np


# endregion Import Dependencies


def is_bool(a_value):
    """
    Check if the given value is a bool, an array of bools, or a list of bools.

    Args:
        a_value: The value to be checked. It can be a single value, a NumPy array, or a list.

    Returns:
        bool: True if the value is a bool, an array of bools, or a list of bools, False otherwise.
    """
    if isinstance(a_value, (list, tuple)):
        return all(is_bool(v) for v in a_value)
    if isinstance(a_value, np.ndarray):
        return np.issubdtype(a_value.dtype, np.bool_)
    else:
        return np.issubdtype(type(a_value), np.bool_)


def is_float(a_value):
    """
    Check if the given value is a float, an array of floats, or a list of floats.

    Args:
        a_value: The value to be checked. It can be a single value, a NumPy array, or a list.

    Returns:
        bool: True if the value is a float, an array of floats, or a list of floats, False otherwise.
    """
    if isinstance(a_value, (list, tuple)):
        return all(is_float(v) for v in a_value)
    if isinstance(a_value, np.ndarray):
        return np.issubdtype(a_value.dtype, np.floating)
    else:
        return np.issubdtype(type(a_value), np.floating)


def is_int(a_value):
    """
    Check if the given value is an integer, an array of integers, or a list of integers.

    Args:
        a_value: The value to be checked. It can be a single value, a NumPy array, or a list.

    Returns:
        bool: True if the value is an integer, an array of integers, or a list of integers, False otherwise.
    """
    if isinstance(a_value, (list, tuple)):
        return all(is_int(v) for v in a_value)
    if isinstance(a_value, np.ndarray):
        return np.issubdtype(a_value.dtype, np.integer)
    else:
        return np.issubdtype(type(a_value), np.integer)


def is_list_of_uuids(a_obj):
    """Check if an object is a List[uuid.UUID].

    Args:
        a_obj: The object to check.

    Returns:
        bool: True if the object is a List[uuid.UUID], False otherwise.
    """
    check = False
    # Check if obj is a list
    if isinstance(a_obj, list):
        # Check if all elements in the list are instances of uuid.UUID
        check = all(isinstance(item, uuid.UUID) for item in a_obj)
    return check
