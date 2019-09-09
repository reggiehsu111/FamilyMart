import json
from datetime import datetime
from itertools import repeat
from collections import OrderedDict


def read_json(fname):
    with open(fname) as file:
        return json.load(file, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """Wrapper function for endless repeated data loader."""
    for loader in repeat(data_loader):
        yield from loader


class Timer:
    def __init__(self):
        self._prev_time = datetime.now()

    def check(self):
        """Return duration since last reset in seconds."""
        now = datetime.now()
        duration = now - self._prev_time
        self._prev_time = now
        return duration.total_seconds()

    def reset(self):
        """Reset timer."""
        self._prev_time = datetime.now()
