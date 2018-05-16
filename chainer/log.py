from __future__ import print_function
import sys

_enabled = False

def enable():
    global _enabled
    _enabled = True


def disable():
    global _enabled
    _enabled = False


def show(msg):
    if _enabled:
        print(msg, file=sys.stderr)

