import subprocess
from typing import List, Optional, Set, Tuple, Union
import numpy as np
from scipy import ndimage


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


def focus_zone(img):
    # LoG
    fz = ndimage.gaussian_laplace(img, sigma=2).astype(np.uint8)
    fz = ndimage.gaussian_filter(fz, 5).astype(np.uint8)

    # Zero-crossing approximation
    # fz = fz / fz.max() * 255
    return fz.astype(np.uint8)


def compute_focus_plan(img, axis):
    indices = np.arange(img.shape[axis])

    summed = img.sum(axis=axis).astype(float)
    summed[summed == 0] = np.nan

    terms = [img, img]
    terms[axis] = indices

    plan = np.dot(*terms) / summed

    return plan


def USB_on() -> None:
    subprocess.Popen(["uhubctl", "-l", "4-2", "-a", "1"], stdout=subprocess.PIPE)


def USB_off() -> None:
    subprocess.Popen(["uhubctl", "-l", "4-2", "-a", "0"], stdout=subprocess.PIPE)


def USB_status() -> int:
    ret = subprocess.Popen(["uhubctl", "-l", "4-2"], stdout=subprocess.PIPE)
    out, error = ret.communicate()
    if 'off' in str(out):
        return 1
    elif 'power' in str(out):
        return 0


def ensure_list(s: Optional[Union[str, List[str], Tuple[str], Set[str]]]) -> List[str]:
    # Ref: https://stackoverflow.com/a/56641168/
    return s if isinstance(s, list) else list(s) if isinstance(s, (tuple, set)) else [] if s is None else [s]


def pretty_size(value: int, verbose=False, decimal=True) -> str:
    """ Get sizes in strings in human-readable format """

    prefixes_dec = ['Yotta', 'Zetta', 'Exa', 'Peta', 'Tera', 'Giga', 'Mega', 'kilo', '']
    prefixes_bin = ['Yobi', 'Zebi', 'Exbi', 'Pebi', 'Tebi', 'Gibi', 'Mebi', 'Kibi', '']

    prefixes, _i = (prefixes_dec, '') if decimal else (prefixes_bin, 'i')

    suffix = 'Byte'
    for p, prefix in enumerate(prefixes, start=-len(prefixes) + 1):
        div = 1000 ** -p if decimal else 1 << -p * 10
        if value >= div:
            break

    amount = value / div
    if amount > 1:
        suffix += 's'

    s, e, _b = (1, None, 'b') if verbose else (None, 1, '')
    unit = f"{prefix[:e]}{_b+_i[:bool(len(prefix[:e]))]}{suffix[s:e]}"

    print(f"s: {(s)}, e: {e}, _b: {_b}, _i: {_i}, len(prefix[:e]): {len(prefix[:e])}, verbose: {verbose}, decimal: {decimal}")
    return f"{int(amount)} {unit}" if amount.is_integer() else f"{amount:.2f} {unit}"
