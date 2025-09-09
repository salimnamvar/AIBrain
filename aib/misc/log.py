"""Logging Handler

This module provides a logging handler that supports multiprocessing and
configurable logging settings. It allows for logging messages to a file with
rotation and backup capabilities, and it can be used across different processes
using a multiprocessing queue.

Classes:
    LogHandler: Manages logging configuration, file handling, and multiprocessing support.
"""

import atexit
import logging
import logging.handlers
import multiprocessing as mp
import multiprocessing.queues as mpq
import os
import threading
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    QueueType = mpq.Queue[logging.LogRecord]
else:
    QueueType = mp.Queue


class LogHandler:
    """Log Handler

    This class manages logging for the application, allowing for configuration
    of log files, levels, and formats. It supports multiprocessing by using
    a queue to handle log messages across different processes. It can optionally
    capture Python warnings and redirect them to the logger.

    Attributes:
        _filename (str): The name of the log file.
        _name (str): The name of the logger.
        _level_name (str): The logging level as a string.
        _format (str): The format for log messages.
        _max_bytes (int): Maximum size of the log file before rotation.
        _backup_count (int): Number of backup files to keep.
        _log_queue (QueueType): The queue for log messages.
        _logger (Optional[logging.Logger]): The logger instance.
        _listener (Optional[logging.handlers.QueueListener]): The queue listener.
        _levels (Dict[str, int]): Mapping of level names to logging levels.
        _configured (bool): Whether the handler has been configured.
        _listening (bool): Whether the listener is active.
        _lock (threading.Lock): Thread lock for synchronization.
        _capture_warnings (bool): Whether Python warnings should be captured and redirected to the logger.
    """

    def __init__(
        self,
        a_filename: str = "./app.log",
        a_name: str = "",
        a_level_name: str = "INFO",
        a_format: str = "[%(levelname)s] - %(asctime)s - [%(name)s - %(module)s - %(funcName)s() - %(lineno)d] -- %(message)s",
        a_max_bytes: int = 524288000,
        a_backup_count: int = 20,
        a_capture_warnings: bool = True,
    ) -> None:
        """Initialize the LogHandler.

        Args:
            a_filename (str): The log file name.
            a_name (str): The name of the logger.
            a_level_name (str): The logging level as a string.
            a_format (str): The format for log messages.
            a_max_bytes (int): Maximum size of the log file before rotation.
            a_backup_count (int): Number of backup files to keep.
            a_capture_warnings (bool): If True, Python warnings will be redirected to the logger.
        """
        self._filename: str = a_filename
        self._name: str = a_name
        self._level_name: str = a_level_name
        self._format: str = a_format
        self._max_bytes: int = a_max_bytes
        self._backup_count: int = a_backup_count
        self._capture_warnings: bool = a_capture_warnings
        self._log_queue: QueueType = mp.Queue()
        self._logger: Optional[logging.Logger] = None
        self._listener: Optional[logging.handlers.QueueListener] = None
        self._configured: bool = False
        self._listening: bool = False
        self._lock: threading.Lock = threading.Lock()
        self._levels: Dict[str, int] = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
            "FATAL": logging.FATAL,
        }
        atexit.register(self.stop)

    def configure(self) -> None:
        """Configure the logging handler.

        This method sets up the logging configuration, including the file handler
        and the logging format. It creates necessary argsories if they do not exist.
        """
        with self._lock:
            if self._configured:
                return

            try:
                # Create argsory if it doesn't exist
                log_dir = os.path.dirname(self._filename)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

                # Create and configure the root logger for this handler
                self._logger = logging.getLogger(self._name)

                # Clear existing handlers to avoid duplicates
                self._logger.handlers.clear()

                # Create file handler with rotation
                file_handler = RotatingFileHandler(
                    self._filename, maxBytes=self._max_bytes, backupCount=self._backup_count
                )
                file_handler.setFormatter(logging.Formatter(self._format))

                # Set levels
                level = self._levels.get(self._level_name.upper(), logging.INFO)
                file_handler.setLevel(level)
                self._logger.setLevel(level)
                self._logger.addHandler(file_handler)

                # Prevent propagation to avoid duplicate logs
                self._logger.propagate = False

                # Capture warnings into logging
                if self._capture_warnings:
                    logging.captureWarnings(True)
                    warnings_logger = logging.getLogger("py.warnings")
                    warnings_logger.setLevel(level)
                    # Attach same handlers (so warnings also go to file/queue)
                    if not warnings_logger.handlers:
                        warnings_logger.addHandler(file_handler)

                self._configured = True

            except Exception as e:
                msg = f"{self.__class__.__name__} setup failed: {e}"
                raise RuntimeError(msg) from e

    def start(self) -> None:
        """Start the logging listener.

        This method sets up a QueueListener that will handle log messages
        sent to the multiprocessing queue.
        """
        with self._lock:
            if self._listening or not self._configured:
                return

            try:
                if not self._logger:
                    raise RuntimeError("Logger not configured. Call configure() first.")

                self._listener = logging.handlers.QueueListener(
                    self._log_queue, *self._logger.handlers, respect_handler_level=True
                )
                self._listener.start()
                self._listening = True

            except Exception as e:
                msg = f"Failed to start listener: {e}"
                raise RuntimeError(msg) from e

    def stop(self) -> None:
        """Stop the logging handler and clean up resources.

        This method ensures that the logging listener is terminated
        and the log queue is properly cleaned up.
        """
        with self._lock:
            if self._listening and self._listener:
                try:
                    self._listener.stop()
                    self._listening = False
                except Exception as e:
                    print(f"Warning: Error stopping listener: {e}")
                finally:
                    self._listener = None

    def is_configured(self) -> bool:
        """Check if the handler is configured.

        Returns:
            bool: True if configured, False otherwise.
        """
        return self._configured

    def is_listening(self) -> bool:
        """Check if the listener is active.

        Returns:
            bool: True if listening, False otherwise.
        """
        return self._listening

    @property
    def queue(self) -> QueueType:
        """Get the log queue.

        Returns:
            mp.Queue: The multiprocessing queue used for logging.
        """
        return self._log_queue

    @property
    def logger(self) -> Optional[logging.Logger]:
        """Get the main logger instance.

        Returns:
            Optional[logging.Logger]: The configured logger or None if not configured.
        """
        return self._logger

    @staticmethod
    def get_logger(
        a_name: str, a_queue: QueueType, a_level: int = logging.INFO, a_capture_warnings: bool = True
    ) -> logging.Logger:
        """Get a logger instance configured with a queue handler.

        This static method creates a logger that sends log messages to the specified queue.
        This is useful for child processes in multiprocessing, where all log messages
        can be collected centrally in the main process. Optionally, Python warnings can
        also be captured and redirected to this logger.

        Args:
            a_name (str): Name of the logger.
            a_queue (QueueType): Queue to which log messages will be sent.
            a_level (int, optional): Logging level (e.g., `logging.INFO`, `logging.DEBUG`). Defaults to `logging.INFO`.
            a_capture_warnings (bool, optional): If True, captures Python warnings (`warnings.warn`) 
                and sends them to the logger. Defaults to True.

        Returns:
            logging.Logger: Configured logger instance with a `QueueHandler` attached.
        """
        logger = logging.getLogger(a_name)
        logger.setLevel(a_level)

        has_qh = any(isinstance(h, logging.handlers.QueueHandler) and h.queue is a_queue for h in logger.handlers)

        if not has_qh:
            qh = logging.handlers.QueueHandler(a_queue)
            qh.setLevel(a_level)
            logger.addHandler(qh)
            logger.propagate = False

            if a_capture_warnings:
                # Capture warnings.warn() in this process
                logging.captureWarnings(True)
                warnings_logger = logging.getLogger("py.warnings")
                warnings_logger.setLevel(a_level)
                # Attach the same QueueHandler if not already attached
                has_qh_warn = any(
                    isinstance(h, logging.handlers.QueueHandler) and h.queue is a_queue
                    for h in warnings_logger.handlers
                )
                if not has_qh_warn:
                    warnings_logger.addHandler(qh)
        return logger
