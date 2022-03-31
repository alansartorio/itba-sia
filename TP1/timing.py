
from contextlib import contextmanager
import signal
import time
from typing import Any, Callable


def thread_time():
    return time.clock_gettime(time.CLOCK_REALTIME)


def raise_timeout(signum, frame):
    raise TimeoutError


@contextmanager
def timeout_block(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGPROF, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.setitimer(signal.ITIMER_PROF, time)
    # signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGPROF, signal.SIG_IGN)


def timeout(func, timeout: float):
    with timeout_block(timeout):
        return func()
    return TimeoutError


def time_method(method: Callable[[], Any], timeout_seconds: float = None):
    start = thread_time()
    # output = timeout(method, 60 * 5)
    if timeout_seconds is not None:
        output = timeout(method, timeout_seconds)
    else:
        output = method()
    end = thread_time()
    return end - start, output
