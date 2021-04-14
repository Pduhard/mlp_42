import time
import functools
import atexit
from Protodeep.utils.env import Env

timers = []


def class_timer(obj):
    if Env.timer is False:
        return obj
    if isinstance(obj, type) is False:
        return timer_f(obj)
    for name, method in obj.__dict__.items():
        if callable(method):
            setattr(obj, name, timer_f(getattr(obj, name), obj.__name__))
    return obj


def print_timers_logs():
    if Env.timer:
        for t in timers:
            t.print_logs()


atexit.register(print_timers_logs)


class Timer:

    run_time = 0
    call_count = 0
    running = False

    def __init__(self, func, class_name, *args, **kwargs):
        self.func = func
        self.class_name = class_name
        timers.append(self)

    def print_logs(self):
        if self.call_count > 0:
            avg_time = self.run_time / self.call_count
            full_name = ''
            if self.class_name is not None:
                full_name += self.class_name + '.'
            full_name += self.func.__name__
            print(
                f"{full_name.ljust(30)}: "
                f"total run time {self.run_time:.6f} s - "
                f"Called {str(self.call_count).ljust(8)} times - "
                f"Average time {avg_time:.6f} s"
            )


def timer_f(func, class_name=None):

    T = Timer(func, class_name)

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        if not T.running:
            T.call_count += 1
            start_time = time.perf_counter()
            T.running = True
            value = func(*args, **kwargs)
            T.running = False
            end_time = time.perf_counter()
            T.run_time += end_time - start_time
            return value
        else:
            return func(*args, **kwargs)
    return wrapper_timer
