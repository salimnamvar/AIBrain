"""Base Error Handler

This module provides a base error handler for OpenCV errors, re-raising them
to a specified logger. It captures OpenCV errors and logs them with detailed
information including the function name, error message, file name, line number,
and status code.

Classes:
    CVErrorHandler: Handles OpenCV errors and logs them using a specified logger.
    InvalidConfigurationError: Exception for invalid configuration arguments.
"""

import logging
import traceback
from typing import Any, List, Optional, Sequence, cast

import cv2


class CVErrorHandler:
    """OpenCV Error Handler

    This class reargss OpenCV errors to a specified logger.
    It captures OpenCV errors and logs them with detailed information including the function name,
    error message, file name, line number, and status code.

    Attributes:
        _logger (logging.Logger): Logger instance to log errors.
        _name (str): Name of the error handler.
    """

    def __init__(self, a_logger: logging.Logger, a_name: str = 'CVErrorHandler', **kwargs: Any) -> None:
        """Initialize the OpenCV error handler.

        Args:
            a_logger (logging.Logger): Logger instance to log errors.
            a_name (str): Name of the error handler.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._name: str = a_name
        self._logger: logging.Logger = a_logger
        cv2.redirectError(self._opencv_error_handler)

    @staticmethod
    def _opencv_error_handler(status: int, func_name: str, err_msg: str, file_name: str, line: int) -> None:
        """Handle OpenCV errors by logging them.

        Args:
            status (int): The OpenCV error status code.
            func_name (str): The name of the OpenCV function where the error occurred.
            err_msg (str): The error message.
            file_name (str): The name of the file where the error occurred.
            line (int): The line number in the file where the error occurred.
        """
        logger = logging.getLogger('CVErrorHandler')
        msg = (
            f"CVErrorHandler Got OpenCV Exception: OpenCV Error in {func_name}: {err_msg}. "
            f"File: {file_name}, Line: {line}, Status: {status}"
        )
        logger.error(msg)
        error = traceback.format_exc()
        logger.error("CVErrorHandler Traceback: %s", error)
        raise RuntimeError(msg)


class InvalidConfigurationError(Exception):
    """Exception raised for invalid configuration arguments.

    Provides detailed information about the argument, its current value,
    allowed values, and a custom message if provided.

    Attributes:
        arg (str): Name of the invalid argument.
        value (Optional[Any]): Current value of the argument.
        valid_values (Optional[List[Any]]): List of allowed values.
        msg (Optional[str]): Full error message.
    """

    def __init__(
        self,
        a_arg: str,
        a_value: Optional[Any] = None,
        a_valid_value: Optional[Any | Sequence[Any]] = None,
        a_msg: Optional[str] = None,
    ):
        """Initialize the InvalidConfigurationError.

        Args:
            a_arg (str): Name of the invalid argument.
            a_value (Optional[Any]): Current value of the argument.
            a_valid_value (Optional[Any | Sequence[Any]]): Allowed value(s) for the argument.
            a_msg (Optional[str]): Additional custom error message.
        """
        self._arg = a_arg
        self._value = a_value
        self._msg = a_msg

        if a_valid_value is None:
            self._valid_value: Optional[List[Any]] = None
        elif isinstance(a_valid_value, (list, tuple, set)):
            self._valid_value = list(cast(Sequence[Any], a_valid_value))
        else:
            self._valid_value = [a_valid_value]

        parts = [f"Configuration is not supported for `{a_arg}`."]

        if a_value is not None:
            parts.append(f"Current value: {repr(a_value)}.")

        if self._valid_value:
            valid_values_str = ", ".join(repr(v) for v in self._valid_value)
            parts.append(f"Valid values: {valid_values_str}.")

        if a_msg:
            parts.append(a_msg)

        self._msg = " ".join(parts)

        super().__init__(self._msg)

    @property
    def arg(self) -> str:
        """str: Name of the invalid argument."""
        return self._arg

    @property
    def value(self) -> Optional[Any]:
        """Optional[Any]: Current value of the argument."""
        return self._value

    @property
    def valid_values(self) -> Optional[List[Any]]:
        """Optional[List[Any]]: List of valid values for the argument."""
        return self._valid_value

    @property
    def msg(self) -> Optional[str]:
        """Optional[str]: Full error message."""
        return self._msg
