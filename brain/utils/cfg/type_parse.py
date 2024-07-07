""" Python Data Type Parser

    This module provides a function for safely parsing Python data types from string representations using the
    ast.literal_eval function. It can handle various data types, including strings, numbers, booleans, lists,
    and dictionaries, while ensuring safe evaluation of input data.
"""

# region Imported Dependencies
import ast

# endregion Imported Dependencies


def parser(a_node):
    """
    Safely parse a string representation of a Python data type into its corresponding data type.

    Args:
        a_node (str): The string representation of the Python data type to parse.

    Returns:
        object: The parsed Python data type. If parsing is unsuccessful, the original string is returned.

    Raises:
        ValueError: If the input string represents an unsupported or custom data type.
    """
    try:
        try:
            node = ast.literal_eval(a_node)
        except:
            pass
        try:
            node = eval(a_node)
        except:
            pass
        if isinstance(node, (list, tuple)):
            # If it's a list, recursively parse its elements
            return [parser(item) for item in node]
        elif isinstance(node, dict):
            # If it's a dictionary, recursively parse its values
            return {key: parser(value) for key, value in node.items()}
        elif isinstance(node, str):
            return node
        elif isinstance(node, (int, float, bool)):
            # Return numbers and booleans as is
            return node
        else:
            # Handle unsupported data types or custom types
            raise ValueError(f"Unsupported data type: {type(node).__name}")
    except:
        return a_node
