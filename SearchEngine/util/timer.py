import time

TIME_DURATION_UNITS = (
    ('week', 60*60*24*7),
    ('day', 60*60*24),
    ('hour', 60*60),
    ('min', 60),
    ('sec', 1)
)


def format_seconds(seconds):
    """
    Formats the given number of seconds into a human-readable string.

    :param float seconds: the number of seconds to format
    :return: the formatted string
    :rtype: str
    """
    if seconds == 0:
        return 'inf'
    parts = []
    for unit, div in TIME_DURATION_UNITS:
        amount, seconds = divmod(int(seconds), div)
        if amount > 0:
            parts.append('{} {}{}'.format(amount, unit, "" if amount == 1 else "s"))
    if not parts:
        return '0 sec'
    return ', '.join(parts)


def measure(func):
    """
    Creates a decorator.
    Times the given function and returns the time it took to execute.

    :param function func: the function to time
    :return: the function decorator
    :rtype: function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"\tTime Elapsed: {format_seconds(end_time - start_time)}")
        return result
    return wrapper

