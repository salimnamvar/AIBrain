"""Logging Setup

    This module defines the :class:`LogHandler` class, which configures the logging for the application.

"""

# region Imported Dependencies
import logging
import os
from logging.handlers import RotatingFileHandler

# endregion Imported Dependencies


class LogHandler:
    """Logger Configuration

    The `LogHandler` class configures the logging for a project based on the provided configuration.

    Attributes:
        filename (str):
            The name of the log file to write log messages to.
        name (str):
            The name of the logger.
        level_name (str):
            The logging level name (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'FATAL').
        format (str):
            The log message format.
        max_bytes (int):
            The maximum file size in bytes before the log file is rolled over.
        backup_count (int):
            The number of backup log files to keep.
    """

    def __init__(
        self, a_filename: str, a_name: str, a_level_name: str, a_format: str, a_max_byte: int, a_backup_count: int
    ):
        """Initialize Logger Configuration

        Initialize an instance of the `LogHandler` class.

        Args:
            a_filename (str):
                A str as the filename of the log file to write the messages into. It can include the full-path to
                the file, without concern about making new directories.
            a_name (str):
                A str as the name of the base logger.
            a_level_name (str):
                The logging level name (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'FATAL').
            a_format (str):
                The log message format.
            a_max_bytes (int):
                The maximum file size in bytes before the log file is rolled over.
            a_backup_count (int):
                The number of backup log files to keep.

        Returns:
            None
        """
        self.filename: str = a_filename
        self.name: str = a_name
        self.level_name: str = a_level_name
        self.format: str = a_format
        self.max_bytes: int = a_max_byte
        self.backup_count: int = a_backup_count

        self.levels: dict = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
            "FATAL": logging.FATAL,
        }
        self.logger: logging.Logger = None

    def setup(self) -> None:
        """Logger Configuration Setup

        Private method to configure the logger based on the provided settings.

        Returns:
            None
        """
        try:
            self.logger = logging.getLogger(self.name)
            level = self.levels.get(self.level_name.upper(), logging.INFO)
            self.logger.setLevel(level)

            if not os.path.exists(os.path.dirname(self.filename)):
                os.makedirs(os.path.dirname(self.filename))

            file_handler = RotatingFileHandler(self.filename, maxBytes=self.max_bytes, backupCount=self.backup_count)
            file_handler.setFormatter(logging.Formatter(self.format))
            self.logger.addHandler(file_handler)
        except Exception as e:
            raise Exception("Error setting up logger: %s", e)

    @property
    def filename(self) -> str:
        """Get the filename for log messages.

        Returns:
            str: The filename for log messages.

        """
        return self._filename

    @filename.setter
    def filename(self, a_filename: str):
        """Set the filename for log messages.

        Args:
            a_filename (str): The filename for log messages.

        Raises:
            TypeError: If the input filename is not a str.

        """
        if a_filename is None or not isinstance(a_filename, str):
            raise TypeError(f"The `a_filename` must be a `str` but it is given as a `{type(a_filename)}`.")
        self._filename: str = a_filename

    @property
    def name(self) -> str:
        """Get the name of the logger.

        Returns:
            str: The name of the logger.

        """
        return self._name

    @name.setter
    def name(self, a_name: str):
        """Set the name of the logger.

        Args:
            a_name (str): The name of the logger.

        Raises:
            TypeError: If the input name is not a str.

        """
        if a_name is None or not isinstance(a_name, str):
            raise TypeError(f"The `a_name` must be a `str` but it is given as a `{type(a_name)}`.")
        self._name: str = a_name

    @property
    def level_name(self) -> str:
        """Get the logging level name.

        Returns:
            str: The logging level name.

        """
        return self._level_name

    @level_name.setter
    def level_name(self, a_level_name: str):
        """Set the logging level name.

        Args:
            a_level_name (str): The logging level name.

        Raises:
            TypeError: If the input level_name is not a str.

        """
        if a_level_name is None or not isinstance(a_level_name, str):
            raise TypeError(f"The `a_level_name` must be a `str` but it is given as a `{type(a_level_name)}`.")
        self._level_name: str = a_level_name

    @property
    def format(self) -> str:
        """Get the log message format.

        Returns:
            str: The log message format.

        """
        return self._format

    @format.setter
    def format(self, a_format: str):
        """Set the log message format.

        Args:
            a_format (str): The log message format.

        Raises:
            TypeError: If the input format is not a str.

        """
        if a_format is None or not isinstance(a_format, str):
            raise TypeError(f"The `a_format` must be a `str` but it is given as a `{type(a_format)}`.")
        self._format: str = a_format

    @property
    def max_bytes(self) -> int:
        """Get the maximum file size in bytes before log file rollover.

        Returns:
            int: The maximum file size in bytes.

        """
        return self._max_bytes

    @max_bytes.setter
    def max_bytes(self, a_max_bytes: int):
        """Set the maximum file size in bytes before log file rollover.

        Args:
            a_max_bytes (int): The maximum file size in bytes.

        Raises:
            TypeError: If the input max_bytes is not an int.

        """
        if a_max_bytes is None or not isinstance(a_max_bytes, int):
            raise TypeError(f"The `a_max_bytes` must be a `int` but it is given as a `{type(a_max_bytes)}`.")
        self._max_bytes: int = a_max_bytes

    @property
    def backup_count(self) -> int:
        """Get the number of backup log files to keep.

        Returns:
            int: The number of backup log files to keep.

        """
        return self._backup_count

    @backup_count.setter
    def backup_count(self, a_backup_count: int):
        """Set the number of backup log files to keep.

        Args:
            a_backup_count (int): The number of backup log files to keep.

        Raises:
            TypeError: If the input backup_count is not an int.

        """
        if a_backup_count is None or not isinstance(a_backup_count, int):
            raise TypeError(f"The `a_backup_count` must be a `int` but it is given as a `{type(a_backup_count)}`.")
        self._backup_count: int = a_backup_count
