from typing import Callable


def memoize(func: Callable) -> Callable:
    """Memoize function"""
    memo = {}

    def wrapped_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in memo:
            memo[key] = func(*args, **kwargs)
        return memo[key]

    return wrapped_func
