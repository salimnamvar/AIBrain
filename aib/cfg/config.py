"""Configuration - Configuration Utilities

This module provides a Configuration class that reads and parses a configuration file,
allowing for nested keys and type conversion of values.

Classes:
    Configuration:
        A singleton class that reads a configuration file and provides access to its parsed contents.
"""

from dataclasses import dataclass
from os import PathLike
from typing import Optional

from src.utils.cfg.type_parser import type_parser
from src.utils.misc.single import SingletonMeta


@dataclass
class CFG:
    """Base class for configuration objects."""


class Configuration(metaclass=SingletonMeta):
    """Configuration Class

    This class is responsible for parsing a configuration file and storing the configuration
    in a structured way. It uses the SingletonMeta to ensure that only one instance of this class
    exists throughout the application.

    Class Attributes:
        _cfg_path (Optional[str | PathLike[str]]): Class-level path to the configuration file.

    Instance Attributes:
        _name (str): Name of the configuration context.
    """

    _cfg_path: Optional[str | PathLike[str]] = None

    def __init__(self, a_cfg_path: Optional[str | PathLike[str]] = None, a_name: str = 'Configuration') -> None:
        """Initialize the Configuration instance.

        Args:
            a_cfg_path (Optional[str | PathLike[str]]): Path to the configuration file.
                If not provided, it must be set later when loading the configuration.
            a_name (str): Identifier for the configuration context. Defaults to "Configuration".
        """
        self._name: str = a_name
        type(self)._cfg_path = a_cfg_path

    @classmethod
    def load1(cls, a_cfg_path: Optional[str | PathLike[str]] = None):
        """Load a configuration file and populate a nested CFG object.

        Reads the configuration file line by line, ignoring comments and empty lines.
        Each valid line is split into a key and value. Keys may use dot notation to
        represent nested configuration structures. Values are parsed with `type_parser`
        to convert them to the appropriate Python types.

        Args:
            a_cfg_path (Optional[str | PathLike[str]]): Path to the configuration file.
                If not provided, the class-level `_cfg_path` is used.

        Raises:
            AssertionError: If neither `a_cfg_path` nor `_cfg_path` is provided.
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the configuration file contains invalid lines.
        """
        instance = cls(a_cfg_path)

        cfg_path = a_cfg_path or cls._cfg_path
        cls._cfg_path = cfg_path
        assert cfg_path is not None, "Configuration file path must be provided."

        try:
            with open(cfg_path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    try:
                        key, value = line.split("=")
                        keys = key.split(".")
                    except ValueError as exc:
                        raise ValueError(f"Invalid line in configuration file: {line}") from exc

                    current_level = instance

                    for k in keys[:-1]:
                        if k not in current_level.__dict__:
                            obj = CFG()
                            current_level.__dict__[k] = obj
                        current_level = current_level.__dict__[k]
                    current_level.__dict__[keys[-1]] = type_parser(value)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file {cfg_path} not found.") from e

    @classmethod
    def load2(cls, a_cfg_path: Optional[str | PathLike[str]] = None) -> "Configuration":
        """Load configuration from file with support for multi-line values."""

        instance = cls(a_cfg_path or getattr(cls, "_cfg_path", None))
        cfg_path = instance._cfg_path
        assert cfg_path is not None, "Configuration file path must be provided."

        buffer = None  # temporary buffer for multi-line values
        braces = 0  # brace counter

        try:
            with open(cfg_path, "r", encoding="utf-8") as file:
                for lineno, raw_line in enumerate(file, 1):
                    # Strip inline comments
                    line = raw_line.split("#", 1)[0].strip()
                    if not line:
                        continue

                    # If accumulating multi-line value
                    if buffer is not None:
                        buffer += " " + line
                        braces += line.count("{") + line.count("[") + line.count("(")
                        braces -= line.count("}") + line.count("]") + line.count(")")
                        if braces == 0:
                            # finished accumulation
                            key, value = buffer.split("=", 1)
                            cls._assign(instance, key.strip(), value.strip())
                            buffer = None
                        continue

                    # Normal case: start of new key=value
                    if "=" in line:
                        key, value = line.split("=", 1)
                        value_stripped = value.strip()
                        if any(value_stripped.startswith(c) for c in ("{", "[", "(")):
                            # begin multiline block
                            buffer = line
                            braces = value_stripped.count("{") + value_stripped.count("[") + value_stripped.count("(")
                            braces -= value_stripped.count("}") + value_stripped.count("]") + value_stripped.count(")")
                        else:
                            cls._assign(instance, key.strip(), value_stripped)
                    else:
                        raise ValueError(f"Invalid line {lineno}: {raw_line.rstrip()}")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file {cfg_path} not found.") from e

    @classmethod
    def load(cls, a_cfg_path: Optional[str | PathLike[str]] = None) -> "Configuration":
        """Load configuration from file with full multi-line support."""

        instance = cls(a_cfg_path or getattr(cls, "_cfg_path", None))
        cfg_path = instance._cfg_path
        assert cfg_path is not None, "Configuration file path must be provided."

        buffer = None  # temporary buffer for multi-line values

        def _is_value_complete(value: str) -> bool:
            """Check if all braces, brackets, and parentheses are balanced."""
            braces = value.count("{") - value.count("}")
            brackets = value.count("[") - value.count("]")
            parens = value.count("(") - value.count(")")
            return braces == 0 and brackets == 0 and parens == 0

        try:
            with open(cfg_path, "r", encoding="utf-8") as file:
                for lineno, raw_line in enumerate(file, 1):
                    # Remove inline comments
                    line = raw_line.split("#", 1)[0].strip()
                    if not line:
                        continue

                    # Accumulate multi-line if buffer exists
                    if buffer is not None:
                        buffer += " " + line
                        if _is_value_complete(buffer.split("=", 1)[1]):
                            key, value = buffer.split("=", 1)
                            cls._assign(instance, key.strip(), value.strip())
                            buffer = None
                        continue

                    # Normal line
                    if "=" in line:
                        key, value = line.split("=", 1)
                        if not _is_value_complete(value.strip()):
                            buffer = line  # start multi-line accumulation
                        else:
                            cls._assign(instance, key.strip(), value.strip())
                    else:
                        raise ValueError(f"Invalid line {lineno}: {raw_line.rstrip()}")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file {cfg_path} not found.") from e

        return instance

    @classmethod
    def get_cfg_path(cls) -> str | PathLike[str]:
        """Return the class-level configuration file path.

        Returns:
            str | PathLike[str]: The path to the configuration file.

        Raises:
            ValueError: If the configuration path has not been set yet.
        """
        if cls._cfg_path is None:
            raise ValueError("Configuration path has not been set yet.")
        return cls._cfg_path

    @classmethod
    def _assign(cls, instance: "Configuration", key: str, value: str) -> None:
        """Assign parsed value into nested CFG objects."""
        keys = key.split(".")
        current_level = instance
        for k in keys[:-1]:
            if k not in current_level.__dict__:
                current_level.__dict__[k] = CFG()
            current_level = current_level.__dict__[k]
        current_level.__dict__[keys[-1]] = type_parser(value)
