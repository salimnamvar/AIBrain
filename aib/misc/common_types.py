"""Miscellaneous - Common Types Utilities

This module defines reusable type aliases that are shared across the codebase.

Type Aliases:
    StopEvent:
        A general-purpose event type used for signaling stop/termination
        conditions across different concurrency models in Python. It unifies:
            - threading.Event  (thread-based synchronization)
            - asyncio.Event    (async/await-based synchronization)
            - multiprocessing.synchronize.Event (process-based synchronization)
"""

import asyncio
import multiprocessing.synchronize as smp
import threading
from typing import TYPE_CHECKING, TypeAlias, Union

if TYPE_CHECKING:
    StopEvent: TypeAlias = Union[threading.Event, asyncio.Event, smp.Event]
else:
    StopEvent = Union[threading.Event, asyncio.Event, smp.Event]
