"""Data Container Utilities

Submodules:
    - b_data: Base data container
    - b_dict: Dictionary container utilities
    - b_list: List container utilities
    - b_seq_data: Sequential data container
    - arr_buf: Array Shared memory ring buffers and slots
    - seq_queue: Asynchronous sequence queue
"""

# Submodules
from . import arr_buf, io, seq_queue
from .arr_buf import ShmPacket, ShmRBuffer, ShmSlot

# Main exports
from .b_data import BaseData
from .b_dict import BaseDict
from .b_list import BaseList
from .b_seq_data import BaseSeqData
from .io import BaseIO, QueueIO
from .seq_queue import AsyncSeqQueue

# Public API
__all__ = [
    # Core containers
    "BaseData",
    "BaseDict",
    "BaseList",
    "BaseSeqData",
    # Array Buffer utilities
    "ShmPacket",
    "ShmRBuffer",
    "ShmSlot",
    # IO utilities
    "BaseIO",
    "QueueIO",
    # Asynchronous queue
    "AsyncSeqQueue",
    # Submodules
    "arr_buf",
    "seq_queue",
    "io",
]
