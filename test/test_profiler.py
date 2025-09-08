import asyncio
import time
import tracemalloc
from random import randint

from aib.perf.profile import Profiler

# Assuming your updated Profiler is already imported

# Create singleton instance
profiler = Profiler()


async def async_task(n: int):
    """Simulate async work with CPU, memory allocation, and sleep (I/O wait)."""
    with profiler.profile():
        # Simulate memory allocation
        data = [i for i in range(10_000 * n)]

        # Simulate CPU work
        s = sum(data)

        # Simulate async I/O
        await asyncio.sleep(randint(1, 3) * 0.1)

        return s


async def main():
    tasks = [async_task(i) for i in range(1, 6)]
    results = await asyncio.gather(*tasks)

    # Generate report (all mode)
    report = profiler.report("async_profiler_report.csv", a_mode="all")

    # Print results
    for r in report:
        print(r)


if __name__ == "__main__":
    asyncio.run(main())
