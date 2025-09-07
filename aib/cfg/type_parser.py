"""Configuration - Configuration Data Type Parser

This module provides a Configuration class that reads and parses a configuration file,
allowing for nested keys and type conversion of values.

Functions:
    type_parser(a_node: str) -> Any:
        Parses a string representation of a Python data structure into its corresponding type.
"""

import ast
from typing import Any


def type_parser(a_node: str) -> Any:
    """Type Parser

    Parses a string representation of a Python data structure into its corresponding type.

    Args:
        a_node (str): A string representation of a Python data structure.

    Returns:
        Any: The parsed data structure, which can be a list, tuple, dict, str, int, float, or bool.

    Raises:
        ValueError: If the string cannot be parsed into a valid Python data structure.
    """
    try:
        try:
            node = ast.literal_eval(a_node)
        except (ValueError, SyntaxError):
            pass
        try:
            node = eval(a_node)
        except (NameError, SyntaxError):
            pass
        if isinstance(node, (list, tuple)):
            # If it's a list, recursively parse its elements
            parsed = [type_parser(item) for item in node]
            return tuple(parsed) if isinstance(node, tuple) else parsed
        if isinstance(node, dict):
            # If it's a dictionary, recursively parse its values
            return {key: type_parser(value) for key, value in node.items()}
        if isinstance(node, str):
            return node
        if isinstance(node, (int, float, bool)):
            # Return numbers and booleans as is
            return node

        # Handle unsupported data types or custom types
        raise ValueError(f"Unsupported data type: {type(node).__name__}")
    except Exception:
        return a_node
