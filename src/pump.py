"""Runs generations on a worker thread so the window never stops responding.

Measured cause of the freezing (tools/profile_live_loop.py, snake, pop 100):

    generation 0-7      0.28 - 0.92s per generation
    generation 8+       1.6 - 1.7s, every single generation
    H-key burst         16.9s

Once cars stop dying early, every generation simulates the full 2400 ticks, and
main.py did that inline between playbacks with no event pumping at all. Windows
marks a process that has not pumped its message queue for a couple of seconds as
not responding, which is exactly what "it freezes sometimes" was -- it starts as
soon as the cars get good enough to survive.

The same measurement shows the fix has room to work: playback of a full
generation is 10s at 4x speed, so a 1.7s simulation hides behind it completely.
Generation N+1 is submitted the moment generation N starts playing back.

Threading is safe here because run_generation uses no RNG and touches no shared
state -- it is a pure function of (genomes, track, config). Evolution stays on
the main thread in the same order it always was, so results are bit-identical
to the synchronous path. test_pumped_run_is_identical_to_synchronous pins that.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from src.simulation import run_generation


class GenerationPump:
    def __init__(self, track, cfg):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="neuroracer-sim")
        self._track = track
        self._cfg = cfg
        self._future = None

    def submit(self, genomes, record: bool = True) -> None:
        """Start simulating a generation. Returns immediately.

        The genomes are copied because the caller evolves its population as soon
        as this returns, and the worker must keep the array it was given.
        """
        self._future = self._pool.submit(
            run_generation, genomes.copy(), self._track, self._cfg, record)

    def ready(self) -> bool:
        return self._future is not None and self._future.done()

    def wait(self, on_wait=None, poll: float = 0.01):
        """Block until the pending generation finishes.

        `on_wait` is called repeatedly while waiting -- main.py pumps pygame
        events and redraws through it, which is what actually keeps the window
        alive. Without it this is just a blocking call with extra steps.
        """
        if self._future is None:
            raise RuntimeError("wait() with nothing submitted")
        while not self._future.done():
            if on_wait is not None:
                on_wait()
            # Always yield, even when there is a callback. Spinning on done()
            # without sleeping starves the worker of the GIL -- the first
            # version of this loop did exactly that and produced a 1.3s stall
            # inside the callback, which is the very freeze it was meant to fix.
            time.sleep(poll)
        future, self._future = self._future, None
        return future.result()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
