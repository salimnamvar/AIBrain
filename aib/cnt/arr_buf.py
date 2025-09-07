"""Array Shared Memory Buffer Utilities

This module provides classes for managing array-based shared memory buffers for inter-process communication,
with support for concurrent producer-consumer scenarios. It includes:

Classes:
    ShmPacket:
        Represents metadata for a shared memory packet, including producer identification and slot indexing.

    ShmSlot:
        Represents a shared memory slot capable of storing a single data packet and its metadata.
        Manages concurrent access, lifetime, release status for multiple consumers, and synchronization.

    ShmRBuffer:
        Implements a shared memory ring buffer with multiple slots for concurrent producers and consumers.
        Manages slot allocation, synchronization, and resource cleanup.
"""

import atexit
import time
from ctypes import c_bool
from dataclasses import dataclass, field
from multiprocessing import Array, Value
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt

from src.utils.cnt.b_seq_data import BaseSeqData
from src.utils.sys.b_obj import BaseObject


@dataclass(frozen=True)
class ShmPacket(BaseSeqData):
    """Shared Memory Packet

    Represents a shared memory packet with producer identification and slot indexing.

    Attributes:
        producer_id (int): Identifier for the producer of the packet. Not used for comparison.
        slot_index (int): Index of the slot in shared memory. Not used for comparison.
    """

    producer_id: int = field(compare=False)
    slot_index: int = field(compare=False)


class ShmSlot(BaseObject):
    """Shared Memory Slot

    ShmSlot is a shared memory slot for inter-process communication, designed to hold a single data packet and
    its metadata. It manages concurrent access, lifetime, and release status for multiple consumers.

    Attributes:
        index (int): Index of the slot.
        _data_shape (Tuple[int, ...]): Shape of the data stored in the slot.
        _data_type (np.dtype[Any]): Data type of the data stored in the slot.
        _num_consumers (int): Number of consumers that can access the slot.
        _lifetime (float): Lifetime of the slot in seconds.
        _name_prefix (str): Prefix for the shared memory name.
        _data_size (int): Size of the data stored in the slot in bytes.
        _shm (SharedMemory): Shared memory object for the slot.
        _timestamp (float): Timestamp of the last access to the slot.
        _id (int): Unique identifier for the slot.
        _producer_id (int): Identifier for the producer of the data in the slot.
        _released (bool): Flag indicating whether the slot has been released.
        _ttl (float): Time-to-live for the slot.
        _wip (bool): Flag indicating whether the slot is a work-in-progress.
    """

    def __init__(
        self,
        a_index: int,
        a_data_shape: Tuple[int, ...],
        a_data_type: np.dtype[Any],
        a_num_consumers: int,
        a_lifetime: float = 10.0,
        a_name_prefix: str = 'Slot',
        a_id: Optional[int] = None,
        a_name: str = 'ShmSlot',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the shared memory slot.

        Args:
            a_index (int): Index of the slot.
            a_data_shape (Tuple[int, ...]): Shape of the data array to be stored.
            a_data_type (np.dtype[Any]): Data type of the array.
            a_num_consumers (int): Number of consumers that will access the slot.
            a_lifetime (float, optional): Lifetime of the slot in seconds. Defaults to 10.0.
            a_name_prefix (str, optional): Prefix for the shared memory name. Defaults to 'Slot'.
            a_id (Optional[int], optional): Unique identifier for the slot. Defaults to None.
            a_name (str, optional): Name of the slot object. Defaults to 'ShmSlot'.
            a_use_prof (bool, optional): Enable profiling. Defaults to False.
            a_use_cfg (bool, optional): Enable configuration. Defaults to True.
            a_use_log (bool, optional): Enable logging. Defaults to True.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(
            a_id=a_id,
            a_name=a_name,
            a_use_prof=a_use_prof,
            a_use_cfg=a_use_cfg,
            a_use_log=a_use_log,
            **kwargs,
        )
        self._index: int = a_index
        self._data_shape: Tuple[int, ...] = a_data_shape
        self._data_type: np.dtype[Any] = a_data_type
        self._num_consumers: int = a_num_consumers
        self._lifetime: float = a_lifetime
        self._name_prefix: str = a_name_prefix
        self._data_size: int = int(np.prod(self._data_shape)) * np.dtype(self._data_type).itemsize

        shm_name = f"{self._name_prefix}_{self._index}"
        try:
            old = SharedMemory(name=shm_name, create=False)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass
        except FileExistsError:
            try:
                old.unlink()
            except FileNotFoundError:
                pass
        self._shm: SharedMemory = SharedMemory(create=True, size=self._data_size, name=shm_name)

        self._timestamp: Synchronized[float] = Value('d', 0.0)
        self._id: Synchronized[int] = Value('i', -1)
        self._producer_id: Synchronized[int] = Value('i', -1)
        self._released: SynchronizedArray[Any] = Array(c_bool, [True] * self._num_consumers)
        self._ttl: Synchronized[float] = Value('d', time.time() + self._lifetime)
        self._wip: Synchronized[bool] = Value('b', 0)
        atexit.register(self.close)

    @property
    def index(self) -> int:
        """Get the index of the slot.

        Returns:
            int: Index of the slot.
        """
        return self._index

    def put(self, a_data: npt.NDArray[Any], a_producer_id: int, a_id: int) -> ShmPacket:
        """Put Data

        Stores the provided data array into the shared memory buffer slot if it is available.

        Args:
            a_data (npt.NDArray[Any]): The data array to be stored. Must match the buffer's shape and dtype.
            a_producer_id (int): Identifier for the producer writing the data.
            a_id (int): Unique identifier for the data packet.

        Returns:
            ShmPacket: Metadata about the stored packet, including id, producer_id, timestamp, and slot index.

        Raises:
            ValueError: If the shape or dtype of `a_data` does not match the buffer's expected shape or type.

        Notes:
            - The method waits until the buffer slot is released or expired before writing.
            - Uses locks to ensure thread/process safety when accessing shared memory and metadata.
            - If the slot is not available, the method sleeps briefly and retries.
        """
        if a_data.shape != self._data_shape:
            raise ValueError(f"Data shape {a_data.shape} doesn't match slot shape {self._data_shape}")
        if a_data.dtype != self._data_type:
            raise ValueError(f"Data type {a_data.dtype} doesn't match slot type {self._data_type}")

        while True:
            acquired = False
            with self._wip.get_lock():
                if not self._wip.value:
                    with self._released.get_lock():
                        is_released = all(self._released[:])
                    with self._ttl.get_lock():
                        is_expired = time.time() > self._ttl.value
                    if is_released or is_expired:
                        self._wip.value = True
                        acquired = True

            if acquired:
                try:
                    np_buffer: npt.NDArray[Any] = np.ndarray(
                        self._data_shape, dtype=self._data_type, buffer=self._shm.buf
                    )
                    np.copyto(np_buffer, a_data)
                    timestamp = time.time()
                    with self._timestamp.get_lock():
                        self._timestamp.value = timestamp
                    with self._id.get_lock():
                        self._id.value = a_id
                    with self._producer_id.get_lock():
                        self._producer_id.value = a_producer_id
                    with self._ttl.get_lock():
                        self._ttl.value = timestamp + self._lifetime
                    self.unrelease()
                    meta = ShmPacket(id=a_id, producer_id=a_producer_id, timestamp=timestamp, slot_index=self._index)
                    return meta
                finally:
                    with self._wip.get_lock():
                        self._wip.value = False
            else:
                time.sleep(0.001)

    def get(self) -> Tuple[npt.NDArray[Any], ShmPacket]:
        """Get Data

        Retrieves the data array and metadata from the shared memory buffer slot.

        Returns:
            Tuple[npt.NDArray[Any], ShmPacket]: The data array and its metadata.

        Notes:
            - The method waits until the buffer slot is available before reading.
            - Uses locks to ensure thread/process safety when accessing shared memory and metadata.
        """
        while True:
            with self._wip.get_lock():
                if not self._wip.value:
                    break
            time.sleep(0.001)

        data: npt.NDArray[Any] = np.ndarray(self._data_shape, dtype=self._data_type, buffer=self._shm.buf)
        with self._id.get_lock():
            id_val = self._id.value
        with self._producer_id.get_lock():
            producer_id_val = self._producer_id.value
        with self._timestamp.get_lock():
            timestamp_val = self._timestamp.value
        meta = ShmPacket(
            id=id_val,
            producer_id=producer_id_val,
            timestamp=timestamp_val,
            slot_index=self._index,
        )
        return data, meta

    def release(self, a_consumer_id: Optional[int] = None) -> None:
        """Release Slot

        Marks the buffer slot as released, allowing new data to be written.

        Args:
            a_consumer_id (Optional[int]):
                Identifier for the consumer releasing the slot. If None, all consumers are released.

        Notes:
            - This method should be called when the consumer is done processing the data.
            - Uses locks to ensure thread/process safety when accessing shared memory and metadata.
        """
        with self._released.get_lock():
            if a_consumer_id is not None:
                self._released[a_consumer_id] = True
            else:
                for i in range(self._num_consumers):
                    self._released[i] = True

    def unrelease(self) -> None:
        """Unrelease Slot

        Marks all resources in the buffer as unreleased.

        This method acquires a lock to ensure thread-safe access to the `_released` array,
        then sets each element to `False`, indicating that none of the resources are released.
        """
        with self._released.get_lock():
            for i in range(self._num_consumers):
                self._released[i] = False

    def is_released(self) -> bool:
        """Is Released

        Checks if all resources in the buffer are released.

        Returns:
            bool: True if all resources are released, False otherwise.
        """
        with self._released.get_lock():
            return all(self._released[:])

    def is_expired(self) -> bool:
        """Is Expired

        Checks if the buffer slot has expired.

        Returns:
            bool: True if the slot is expired, False otherwise.
        """
        with self._ttl.get_lock():
            return time.time() > self._ttl.value

    def close(self) -> None:
        """Close Slot

        Closes the shared memory slot and releases any resources.
        """
        try:
            self._shm.close()
            self._shm.unlink()
        except FileNotFoundError:
            pass


class ShmRBuffer(BaseObject):
    """
    ShmRBuffer is a shared memory ring buffer for concurrent producer-consumer scenarios.

    This class manages a fixed number of slots, each capable of storing a numpy array and associated metadata.
    Multiple producers and consumers can safely put and get data from the buffer. The buffer uses shared memory
    primitives for synchronization and supports context management for automatic resource cleanup.

    Attributes:
        _data_shape (Tuple[int, ...]): Shape of the data arrays to be stored in each slot.
        _data_type (np.dtype[Any]): Data type of the arrays to be stored.
        _num_slots (int): Number of slots in the ring buffer.
        _num_consumers (int): Number of consumers accessing the buffer.
        _num_producers (int): Number of producers writing to the buffer.
        _slot_name_prefix (str): Prefix for slot names.
        _ttl (float): Time-to-live for each slot in seconds.
        _slots (List[ShmSlot]): List of shared memory slots.
        _slot_index (Value): Current index for the next slot to use.
    """

    def __init__(
        self,
        a_data_shape: Tuple[int, ...],
        a_data_type: np.dtype[Any],
        a_num_slots: int,
        a_num_consumers: int,
        a_num_producers: int,
        a_lifetime: float = 10.0,
        a_name_prefix: str = 'Slot',
        a_id: Optional[int] = None,
        a_name: str = 'ShmRB',
        a_use_prof: bool = False,
        a_use_cfg: bool = True,
        a_use_log: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize ShmRBuffer.

        Args:
            a_data_shape (Tuple[int, ...]): Shape of the data arrays to be stored in each slot.
            a_data_type (np.dtype[Any]): Data type of the arrays to be stored.
            a_num_slots (int): Number of slots in the ring buffer.
            a_num_consumers (int): Number of consumers accessing the buffer.
            a_num_producers (int): Number of producers writing to the buffer.
            a_lifetime (float, optional): Time-to-live for each slot in seconds. Defaults to 10.0.
            a_name_prefix (str, optional): Prefix for slot names. Defaults to 'Slot'.
            a_id (Optional[int], optional): Unique identifier for the buffer. Defaults to None.
            a_name (str, optional): Name of the buffer. Defaults to 'ShmRB'.
            a_use_prof (bool, optional): Enable profiling. Defaults to False.
            a_use_cfg (bool, optional): Enable configuration. Defaults to True.
            a_use_log (bool, optional): Enable logging. Defaults to True.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(
            a_id=a_id,
            a_name=a_name,
            a_use_prof=a_use_prof,
            a_use_cfg=a_use_cfg,
            a_use_log=a_use_log,
            **kwargs,
        )
        self._data_shape: Tuple[int, ...] = a_data_shape
        self._data_type: np.dtype[Any] = a_data_type
        self._num_slots: int = a_num_slots
        self._num_consumers: int = a_num_consumers
        self._num_producers: int = a_num_producers
        self._slot_name_prefix: str = a_name_prefix
        self._ttl: float = a_lifetime
        self._slots: List[ShmSlot] = [
            ShmSlot(
                a_index=i,
                a_data_shape=self._data_shape,
                a_data_type=self._data_type,
                a_num_consumers=self._num_consumers,
                a_lifetime=self._ttl,
                a_name_prefix=self._slot_name_prefix,
            )
            for i in range(self._num_slots)
        ]
        self._slot_index = Value('i', 0)
        atexit.register(self.close)

    def put(self, a_data: npt.NDArray[Any], a_producer_id: int, a_id: int) -> ShmPacket:
        """Put Data

        Stores the provided data into a shared memory slot.

        Args:
            a_data (npt.NDArray[Any]): The data to be stored in the buffer slot.
            a_producer_id (int): Identifier for the producer of the data.
            a_id (int): Unique identifier for the data packet.

        Returns:
            ShmPacket: Metadata object representing the stored packet.
        """
        with self._slot_index.get_lock():
            index = self._slot_index.value % self._num_slots
            self._slot_index.value = (self._slot_index.value + 1) % self._num_slots
        slot = cast(ShmSlot, self._slots[index])
        meta = slot.put(a_data=a_data, a_producer_id=a_producer_id, a_id=a_id)
        return meta

    def get(self, a_index: int) -> Tuple[npt.NDArray[Any], ShmPacket]:
        """Get Data

        Retrieves the data and metadata from the specified buffer slot.

        Args:
            a_index (int): The index of the buffer slot to retrieve data from.

        Returns:
            Tuple[npt.NDArray[Any], ShmPacket]: The data and metadata from the buffer slot.
        """
        slot = self._slots[a_index % self._num_slots]
        data, meta = slot.get()
        return data, meta

    def release(self, a_index: int, a_consumer_id: Optional[int] = None) -> None:
        """Release Data

        Releases the specified buffer slot, making it available for new data.

        Args:
            a_index (int): The index of the buffer slot to release.
            a_consumer_id (Optional[int]): Identifier for the consumer releasing the slot.
        """
        slot = self._slots[a_index % self._num_slots]
        slot.release(a_consumer_id)

    def close(self) -> None:
        """Close the buffer and release all resources."""
        for slot in self._slots:
            slot.close()

    def __enter__(self):
        """Context manager enter."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit."""
        self.close()
