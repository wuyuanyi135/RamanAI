import os
import sys
from functools import reduce


def str_to_obj(descriptor: str):
    tokens = descriptor.split(".")
    assert len(tokens) > 0
    if len(tokens) == 1:
        return sys.modules[tokens[0]]
    else:
        return reduce(getattr, tokens[1:], sys.modules[tokens[0]])

def abs_or_offset_from(tested_path: str, start_path = "."):
    if os.path.isabs(tested_path):
        return tested_path

    return os.path.abspath(os.path.join(start_path, tested_path))