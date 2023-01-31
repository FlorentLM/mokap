import subprocess
from typing import NoReturn
import numpy as np


def randframe(w=1440, h=1080):
    return np.random.randint(0, 255, (w, h), dtype='<u1')


def to_ticks(frequency: float, duration=1.0) -> (float, float):
    """ Converts frequency, duration to tick length and ticks count """
    count = frequency * duration
    interval = 1/frequency
    return interval, count


def to_freq(interval: float, count: float) -> (float, float):
    """ Converts tick length and number of ticks into frequency, total duration """
    duration = interval * count
    frequency = 1/interval
    return frequency, duration


def USB_on() -> NoReturn:
    subprocess.Popen(["uhubctl", "-l", "4-2", "-a", "1"], stdout=subprocess.PIPE)


def USB_off() -> NoReturn:
    subprocess.Popen(["uhubctl", "-l", "4-2", "-a", "0"], stdout=subprocess.PIPE)


def USB_status() -> int:
    ret = subprocess.Popen(["uhubctl", "-l", "4-2"], stdout=subprocess.PIPE)
    out, error = ret.communicate()
    if 'off' in str(out):
        return 1
    elif 'power' in str(out):
        return 0