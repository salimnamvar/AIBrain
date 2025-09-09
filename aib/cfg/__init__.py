"""Configuration Utilities

Submodules:
    - config: Main configuration management
    - type_parser: Type parsing helpers
"""

# Submodules
from . import type_parser

# Main exports
from .config import Configuration

# Public API
__all__ = [
    "Configuration",
    "type_parser",
]
