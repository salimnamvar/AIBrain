"""Miscellaneous - Data Type Checker Utilities

This module provides functions to check if a value is of a specific data type,
including boolean, float, and integer. It supports nested structures like lists and tuples.

Functions:
    is_bool(a_value: Any) -> bool:
        Checks if the value is a boolean or a boolean-like type.
    is_float(a_value: Any) -> bool:
        Checks if the value is a float or a float-like type.
    is_int(a_value: Any) -> bool:
        Checks if the value is an integer or an integer-like type.
"""

from typing import Any, List, Tuple, Union

import numpy as np


def is_bool(a_value: Union[Any, List[Any], Tuple[Any, ...]]):
    """Check if the value is a boolean or a boolean-like type.

    Args:
        a_value (Union[Any, List[Any], Tuple[Any, ...]]): The value to check.

    Returns:
        bool: True if the value is a boolean or a boolean-like type, False otherwise.
    """
    if isinstance(a_value, (list, tuple)):
        return all(is_bool(v) for v in a_value)
    if isinstance(a_value, np.ndarray):
        return np.issubdtype(a_value.dtype, np.bool_)
    return np.issubdtype(type(a_value), np.bool_)


def is_float(a_value: Union[Any, List[Any], Tuple[Any, ...]]):
    """Check if the value is a float or a float-like type.

    Args:
        a_value (Union[Any, List[Any], Tuple[Any, ...]]): The value to check.

    Returns:
        bool: True if the value is a float or a float-like type, False otherwise.
    """
    if isinstance(a_value, (list, tuple)):
        return all(is_float(v) for v in a_value)
    if isinstance(a_value, np.ndarray):
        return np.issubdtype(a_value.dtype, np.floating)
    return np.issubdtype(type(a_value), np.floating)


def is_int(a_value: Union[Any, List[Any], Tuple[Any, ...]]):
    """Check if the value is an integer or an integer-like type.

    Args:
        a_value (Union[Any, List[Any], Tuple[Any, ...]]): The value to check.

    Returns:
        bool: True if the value is an integer or an integer-like type, False otherwise.
    """
    if isinstance(a_value, (list, tuple)):
        return all(is_int(v) for v in a_value)
    if isinstance(a_value, np.ndarray):
        return np.issubdtype(a_value.dtype, np.integer)
    return np.issubdtype(type(a_value), np.integer)
