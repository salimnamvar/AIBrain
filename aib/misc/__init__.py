"""Miscellaneous Utilities

Submodules:
    - error: Custom error handling
    - log: Logging utilities
    - single: Singleton pattern utilities
    - type: Type-checking helpers
"""

# Submodules
from . import common_errors, common_types, log, single, type_check

# Main exports
from .common_errors import CVErrorHandler
from .log import LogHandler
from .single import SingletonMeta
from .type_check import is_bool, is_float, is_int

# Public API
__all__ = [
    # Core utilities
    "CVErrorHandler",
    "LogHandler",
    "SingletonMeta",
    "is_bool",
    "is_float",
    "is_int",
    # Submodules
    "common_errors",
    "log",
    "single",
    "type_check",
    "common_types",
]
