"""Performance Profiler

This module provides a comprehensive profiling utility for Python applications,
capable of monitoring function execution times, CPU usage, memory usage, and
I/O wait times. It supports synchronous and asynchronous code, as well as
decorators and context managers for profiling functions, methods, and classes.

Classes:
    Record: Stores individual profiling data.
    Records: Collection of Record objects with optional size limit.
    Entities: Dictionary of Records objects with optional size limit.
    Profiler: Singleton class providing profiling functionality.
"""

import asyncio
import csv
import inspect
import os
import statistics
import time
import tracemalloc
import warnings
from contextlib import ContextDecorator, asynccontextmanager, contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Union

from aib.cnt.b_dict import BaseDict
from aib.cnt.b_list import BaseList
from aib.misc.single import SingletonMeta


@dataclass
class Record:
    """Stores profiling information for a single execution.

    Attributes:
        name (str): The name of the profiled entity.
        start_time (float): Start time in seconds.
        end_time (Optional[float]): End time in seconds.
        cpu_start (float): CPU time at start.
        cpu_end (Optional[float]): CPU time at end.
        memory_start (float): Memory usage at start (MB).
        memory_end (Optional[float]): Memory usage at end (MB).
        io_wait (Optional[float]): Estimated I/O wait time.
        success (bool): Whether the profiled execution succeeded.
    """

    name: str
    start_time: float
    end_time: Optional[float] = None
    cpu_start: float = 0.0
    cpu_end: Optional[float] = None
    memory_start: float = 0.0
    memory_end: Optional[float] = None
    io_wait: Optional[float] = None
    success: bool = True


class Records(BaseList[Record]):
    """A collection of Record objects with optional maximum size.

    Args:
        a_iterable (Iterable[Record] | None): Initial records to populate the collection.
        a_max_size (Optional[int]): Maximum number of records to keep.
        a_name (str): Name of the collection.
    """

    def __init__(
        self,
        a_iterable: Iterable[Record] | None = None,
        a_max_size: Optional[int] = None,
        a_name: str = "Records",
    ):
        super().__init__(a_iterable=a_iterable, a_max_size=a_max_size, a_name=a_name)


class Entities(BaseDict[str, Records]):
    """A dictionary mapping entity names to their Records collections.

    Args:
        a_dict (Dict[str, Records] | None): Initial dictionary of entities.
        a_max_size (Optional[int]): Maximum number of entities allowed.
        a_name (str): Name of the dictionary.
    """

    def __init__(
        self,
        a_dict: Dict[str, Records] | None = None,
        a_max_size: Optional[int] = None,
        a_name: str = "Entities",
    ):
        super().__init__(a_dict=a_dict, a_max_size=a_max_size, a_name=a_name)


class Profiler(metaclass=SingletonMeta):
    """Singleton class for monitoring performance of Python code.

    Provides decorators, context managers, and methods for profiling
    synchronous and asynchronous code execution.

    Attributes:
        _max_entities (Optional[int]): Maximum number of entities that can be recorded.
            If None, there is no limit.
        _max_records_per_entity (Optional[int]): Maximum number of records per entity.
            If None, there is no limit.
        _entities (Entities): Dictionary mapping entity names to their corresponding
            Records object, storing profiling data for each entity.
    """

    def __init__(
        self,
        a_max_entities: Optional[int] = None,
        a_max_records_per_entity: Optional[int] = None,
        a_name: str = 'Profiler',
    ) -> None:
        """
        Initialize the Profiler instance.

        Args:
            a_max_entities (Optional[int]): Maximum number of entities to track.
            a_max_records_per_entity (Optional[int]): Maximum records per entity.
            a_name (str): Profiler name.
        """
        self._name: str = a_name
        self._max_entities: Optional[int] = a_max_entities
        self._max_records_per_entity: Optional[int] = a_max_records_per_entity
        self._entities: Entities = Entities(a_max_size=a_max_entities)

        if not tracemalloc.is_tracing():
            tracemalloc.start()

    @property
    def name(self) -> str:
        """Get the name of the object.

        Returns:
            str: The name of the object.
        """
        return self._name

    @staticmethod
    def get_entity_name(a_name: Optional[str] = None, a_stack_offset: int = 2) -> str:
        """Generate a unique entity name from the call stack.

        Args:
            a_name (Optional[str]): Optional provided name.
            a_stack_offset (int): Stack offset for naming.

        Returns:
            str: Generated entity name.
        """
        if a_name:
            return a_name

        stack = inspect.stack()
        for frame_idx in range(a_stack_offset, len(stack)):
            frame = stack[frame_idx]
            filename = os.path.basename(frame.filename)
            if "metric.py" in filename:
                continue

            func_name = frame.function
            self_obj = frame.frame.f_locals.get("self", None)
            cls_name = self_obj.__class__.__name__ if self_obj else None

            if cls_name:
                a_name = f"{filename}::{cls_name}.{func_name}::line_{frame.lineno}"
            else:
                a_name = f"{filename}::{func_name}::line_{frame.lineno}"
            break
        else:
            a_name = "unknown_entity"

        return a_name

    @classmethod
    def start(cls, a_name: Optional[str] = None) -> Record:
        """Start profiling an entity.

        Args:
            a_name (Optional[str]): Optional entity name.

        Returns:
            Record: Newly created Record object.
        """
        profiler = cls()
        a_name = Profiler.get_entity_name(a_name)

        mem_before = tracemalloc.get_traced_memory()[1]

        start_time = time.perf_counter()
        cpu_start = time.process_time()

        metrics = Record(
            name=a_name, start_time=start_time, cpu_start=cpu_start, memory_start=mem_before / (1024 * 1024)
        )

        if a_name not in profiler._entities:
            profiler._entities[a_name] = Records(a_max_size=profiler._max_records_per_entity, a_name=a_name)
        profiler._entities[a_name].append(metrics)
        return metrics

    @classmethod
    def end(
        cls,
        a_record: Record,
        a_success: bool = True,
        a_io_wait: Optional[float] = None,
    ) -> Record:
        """Stop profiling and update the Record object.

        Args:
            a_record (Record): Record to stop profiling.
            a_success (bool): Whether the execution was successful.
            a_io_wait (Optional[float]): Optional I/O wait time.

        Returns:
            Record: Updated Record object.

        Raises:
            ValueError: If the entity does not exist.
        """
        profiler = cls()

        if a_record.name not in profiler._entities:
            raise ValueError(f"PerfRecord Entity '{a_record.name}' not registered")

        mem_after = tracemalloc.get_traced_memory()[1]

        a_record.end_time = time.perf_counter()
        a_record.cpu_end = time.process_time()
        a_record.memory_end = mem_after / (1024 * 1024)
        a_record.success = a_success

        if a_record.cpu_end and a_record.cpu_start and a_record.end_time and a_record.start_time:
            cpu_time = a_record.cpu_end - a_record.cpu_start
            wall_time = a_record.end_time - a_record.start_time
            a_record.io_wait = a_io_wait if a_io_wait is not None else max(0.0, wall_time - cpu_time)
        else:
            a_record.io_wait = a_io_wait or 0.0

        return a_record

    @classmethod
    def report(
        cls, a_csv_path: str | os.PathLike[str], a_mode: Literal["summary", "all"] = "summary"
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Generate a profiling report for all recorded entities.

        This method can generate two types of reports:

        1. "summary" mode: Aggregates statistics per entity including call counts,
        runtime, CPU usage, memory usage, latency percentiles, and throughput.
        2. "all" mode: Returns detailed metrics for each individual record.

        Optionally, the report can be saved as a CSV file.

        Args:
            a_mode (Literal["summary", "all"], optional): Determines the type of report.
                Defaults to "summary".
            a_csv_path (str | os.PathLike[str]): Path to save the report as a CSV file.

        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]:
                - In "summary" mode: A dictionary mapping entity names to aggregated statistics.
                - In "all" mode: A list of dictionaries, each representing a single Record.
        """
        profiler = cls()
        report: Union[Dict[str, Any], List[Dict[str, Any]]] = {}

        csv_path = Path(a_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        if a_mode == "summary":
            for entity, records in profiler._entities.items():
                durations = [r.end_time - r.start_time for r in records if r.end_time is not None]
                cpu_times = [r.cpu_end - r.cpu_start for r in records if r.cpu_end is not None]
                memory_peaks = [r.memory_end - r.memory_start for r in records if r.memory_end is not None]
                successes = [r.success for r in records]
                io_waits = [r.io_wait for r in records if r.io_wait is not None]

                total_calls = len(records)
                success_count = sum(successes)
                failure_count = total_calls - success_count
                cumulative_time = sum(durations)

                def percentile(lst: List[float], p: int) -> float:
                    if not lst:
                        return 0.0
                    return (
                        statistics.quantiles(lst, n=100)[p - 1]
                        if len(lst) >= 100
                        else sorted(lst)[min(p - 1, len(lst) - 1)]
                    )

                report[entity] = {
                    "Entity": entity,
                    "Call": total_calls,
                    "Call Success": success_count,
                    "Call Failure": failure_count,
                    "Runtime": cumulative_time,
                    "Latency Min": min(durations) if durations else 0,
                    "Latency Avg": sum(durations) / len(durations) if durations else 0,
                    "Latency Max": max(durations) if durations else 0,
                    "Latency P50": percentile(durations, 50),
                    "Latency P95": percentile(durations, 95),
                    "Latency P99": percentile(durations, 99),
                    "Throughput": len(durations) / cumulative_time if cumulative_time > 0 else 0,
                    "CPU Min": min(cpu_times) if cpu_times else 0,
                    "CPU Avg": sum(cpu_times) / len(cpu_times) if cpu_times else 0,
                    "CPU Max": max(cpu_times) if cpu_times else 0,
                    "Memory Min": min(memory_peaks) if memory_peaks else 0,
                    "Memory Avg": sum(memory_peaks) / len(memory_peaks) if memory_peaks else 0,
                    "Memory Max": max(memory_peaks) if memory_peaks else 0,
                    "IO Wait Min": min(io_waits) if io_waits else 0,
                    "IO Wait Avg": sum(io_waits) / len(io_waits) if io_waits else 0,
                    "IO Wait Max": max(io_waits) if io_waits else 0,
                }

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(next(iter(report.values())).keys()))
                writer.writeheader()
                for entity, data in report.items():
                    writer.writerow(data)

        elif a_mode == "all":
            all_rows: List[Dict[str, Any]] = []
            for entity, records in profiler._entities.items():
                for r in records:
                    all_rows.append(
                        {
                            "Entity": entity,
                            "Start Time": r.start_time,
                            "End Time": r.end_time,
                            "Latency": (r.end_time - r.start_time) if r.end_time else None,
                            "Cpu Time": (r.cpu_end - r.cpu_start) if r.cpu_end else None,
                            "Memory Usage": (r.memory_end - r.memory_start) if r.memory_end else None,
                            "IO Wait": r.io_wait,
                            "Success": r.success,
                        }
                    )

            report = all_rows

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else [])
                writer.writeheader()
                for row in all_rows:
                    writer.writerow(row)

        return report

    def _profile_internal(self, a_name: Optional[str] = None) -> Record:
        """Start profiling an internal entity with automatic name resolution.

        This method is intended for internal use by the Profiler to start
        profiling a code block or function. It automatically generates an
        entity name based on the call stack if no name is provided.

        Args:
            a_name (Optional[str]): Optional custom name for the entity. If None,
                the entity name is generated from the caller's filename, class,
                function, and line number.

        Returns:
            Record: A new Record object tracking the start time, CPU, and memory usage.
        """
        entity_name = Profiler.get_entity_name(a_name, a_stack_offset=3)
        return self.start(entity_name)

    def _wrap_callable(self, a_obj: Callable[..., Any], a_name: str) -> Callable[..., Any]:
        """Wrap a callable (function or coroutine) with profiling.

        This method returns a new function that profiles execution time, CPU usage,
        memory usage, and I/O wait of the original callable. It works for both
        synchronous and asynchronous functions.

        Args:
            a_obj (Callable[..., Any]): The function or coroutine to be profiled.
            a_name (str): The entity name used in profiling records.

        Returns:
            Callable[..., Any]: A wrapped callable that records performance metrics
            each time it is invoked.
        """
        if asyncio.iscoroutinefunction(a_obj):

            @wraps(a_obj)
            async def wrapper_async(*args: Any, **kwargs: Any) -> Any:
                metrics = self._profile_internal(a_name)
                success = True
                try:
                    return await a_obj(*args, **kwargs)
                except Exception:
                    success = False
                    raise
                finally:
                    self.end(metrics, a_success=success)

            return wrapper_async

        else:

            @wraps(a_obj)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                metrics = self._profile_internal(a_name)
                success = True
                try:
                    return a_obj(*args, **kwargs)
                except Exception:
                    success = False
                    raise
                finally:
                    self.end(metrics, a_success=success)

            return wrapper

    @classmethod
    def profile_function(cls, a_name: Optional[str] = None):
        """Decorator to profile a function or coroutine.

        This method returns a decorator that wraps a function or coroutine
        to automatically record performance metrics (execution time, CPU usage,
        memory usage, and I/O wait) each time it is called.

        Args:
            a_name (Optional[str]): Optional custom name for the profiled entity.
                If not provided, the function's qualified name (`__qualname__`)
                is used.

        Returns:
            Callable[[Callable[..., Any]], Callable[..., Any]]: A decorator that
            wraps the target function or coroutine with profiling.
        """
        profiler = cls()

        def decorator(a_obj: Callable[..., Any]):
            name = a_name or a_obj.__qualname__
            return profiler._wrap_callable(a_obj, name)

        return decorator

    @classmethod
    def profile_class(cls, a_class: type, a_method_names: Optional[list[str]] = None) -> type:
        """Profile selected or all methods of a class by wrapping them with performance monitoring.

        This method modifies the given class in-place, wrapping its methods
        (excluding dunder methods) so that each call is automatically profiled.
        It can optionally restrict profiling to a specific list of method names.

        Args:
            a_class (type): The class whose methods should be profiled.
            a_method_names (Optional[list[str]]): Optional list of method names to profile.
                If None, all non-dunder methods are profiled.

        Returns:
            type: The same class with profiled methods.
        """
        profiler = cls()

        for attr_name, attr_value in a_class.__dict__.items():
            if attr_name.startswith("__"):
                continue

            if a_method_names and attr_name not in a_method_names:
                continue

            if isinstance(attr_value, classmethod):
                original_func = attr_value.__func__
                wrapped_func = profiler._wrap_callable(original_func, f"{a_class.__name__}.{attr_name}")
                setattr(a_class, attr_name, classmethod(wrapped_func))

            elif isinstance(attr_value, staticmethod):
                original_func = attr_value.__func__
                wrapped_func = profiler._wrap_callable(original_func, f"{a_class.__name__}.{attr_name}")
                setattr(a_class, attr_name, staticmethod(wrapped_func))

            elif callable(attr_value):
                wrapped_func = profiler._wrap_callable(attr_value, f"{a_class.__name__}.{attr_name}")
                setattr(a_class, attr_name, wrapped_func)
        return a_class

    @contextmanager
    @classmethod
    def profile_sync_context(cls, a_name: Optional[str] = None):
        """Context manager to profile a synchronous code block.

        This context manager starts profiling when entering the block and
        records execution metrics such as runtime, CPU usage, memory usage,
        and I/O wait. It ensures that metrics are finalized even if an
        exception occurs within the block.

        Args:
            a_name (Optional[str]): Optional name for the profiled entity.
                If not provided, the entity name is automatically generated
                from the caller's filename, class, function, and line number.

        Yields:
            Record: The profiling Record object capturing the start metrics.
        """
        profiler = cls()

        metrics = profiler._profile_internal(a_name)
        success = True
        try:
            yield metrics
        except Exception:
            success = False
            raise
        finally:
            profiler.end(metrics, a_success=success)

    @asynccontextmanager
    @classmethod
    async def profile_async_cotext(cls, a_name: Optional[str] = None):
        """Asynchronous context manager to profile an async code block.

        This async context manager starts profiling when entering the block and
        records execution metrics such as runtime, CPU usage, memory usage, and
        I/O wait. Metrics are finalized automatically even if an exception occurs
        within the async block.

        Args:
            a_name (Optional[str]): Optional name for the profiled entity. If not
                provided, the entity name is automatically generated from the
                caller's filename, class, function, and line number.

        Yields:
            Record: The profiling Record object capturing the start metrics.
        """
        profiler = cls()
        metrics = profiler._profile_internal(a_name)
        success = True
        try:
            yield metrics
        except Exception:
            success = False
            raise
        finally:
            profiler.end(metrics, a_success=success)

    @classmethod
    def profile_context(cls, a_name: Optional[str] = None):
        """Context manager that automatically selects sync or async profiling.

        This method detects whether there is a running asyncio event loop.
        If so, it returns an asynchronous context manager suitable for
        `async with` blocks. Otherwise, it returns a synchronous context
        manager for normal `with` blocks.

        Args:
            a_name (Optional[str]): Optional name for the profiled entity. If not
                provided, the entity name is automatically generated from the
                caller's filename, class, function, and line number.

        Returns:
            Union[contextmanager, asynccontextmanager]: A context manager (sync or async)
            that records performance metrics for the enclosed code block.
        """
        profiler = cls()
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return profiler.profile_async_cotext(a_name)
        except RuntimeError:
            pass
        return profiler.profile_sync_context(a_name)

    @classmethod
    def profile(cls, a_name: Optional[str] = None):
        """Unified profiler interface for functions, methods, classes, and context blocks.

        This class method returns a `ProfileContext` object, which can be used in
        multiple ways:

        1. As a decorator for functions or coroutines.
        2. As a decorator for an entire class to profile its methods.
        3. As a synchronous or asynchronous context manager to profile a code block.

        Args:
            a_name (Optional[str]): Optional custom name for the profiling entity.
                If not provided, the entity name is automatically generated for
                context blocks or derived from the function/class name for decorators.

        Returns:
            ProfileContext: A versatile profiler object that can be used as a decorator,
            context manager, or class method profiler.
        """

        class ProfileContext(ContextDecorator):
            """ContextDecorator for profiling functions, classes, or code blocks.

            Attributes:
                _profiler (Profiler): The Profiler instance used for recording metrics.
                _name (Optional[str]): Optional name of the profiled entity.
                _record (Optional[Record]): Active profiling record.
            """

            def __init__(self, a_monitor: Profiler, a_name: Optional[str] = None):
                self._profiler: Profiler = a_monitor
                self._name: Optional[str] = a_name
                self._record: Optional[Record] = None

            # Sync context manager
            def __enter__(self):
                self._record = self._profiler._profile_internal(self._name)
                return self._record

            def __exit__(
                self,
                exc_type: Optional[type[BaseException]],
                exc_val: Optional[BaseException],
                exc_tb: Optional[Any],
            ) -> None:
                success = exc_type is None
                if self._record is None:
                    raise ValueError(f"PerfRecord Entity '{self._name}' not registered")
                self._profiler.end(self._record, a_success=success)

            # Async context manager
            async def __aenter__(self):
                self._record = self._profiler._profile_internal(self._name)
                return self._record

            async def __aexit__(
                self,
                exc_type: Optional[type[BaseException]],
                exc_val: Optional[BaseException],
                exc_tb: Optional[Any],
            ):
                success = exc_type is None
                if self._record is None:
                    raise ValueError(f"PerfRecord Entity '{self._name}' not registered")
                self._profiler.end(self._record, a_success=success)

            def __call__(self, obj: Any):
                if isinstance(obj, type):
                    return cls.profile_class(obj)
                elif callable(obj):
                    entity_name = self._name or obj.__qualname__
                    return cls.profile_function(entity_name)(obj)
                else:
                    return obj

        return ProfileContext(cls(), a_name)

    @classmethod
    def reset(cls, a_force: bool = False) -> None:
        """Reset all collected profiling records and entities.

        Args:
            a_force (bool, optional): If False (default), raises a warning if there
                are still profiling records present. If True, clears everything
                unconditionally.
        """
        profiler = cls()

        if not a_force and any(len(records) > 0 for records in profiler._entities.values()):
            warnings.warn(
                "Profiler.reset() called while there are existing profiling records. "
                "Use force=True to clear anyway.",
                UserWarning,
                stacklevel=2,
            )
            return

        profiler._entities = Entities(a_max_size=profiler._max_entities)
