import subprocess
from typing import List, Optional, Set, Tuple, Union
import numpy as np
from scipy import ndimage
import colorsys
from PIL import Image
import cv2
from pathlib import Path

def hex_to_rgb(hex_str: str):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 3:
        hex_str = ''.join([c + c for c in hex_str])

    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(*rgb):
    if len(rgb) == 1:
        r, g, b = rgb[0]
    elif len(rgb) != 3:
        raise TypeError('Either pass three separate values or a tuple')
    else:
        r, g, b = rgb

    new_hex = f'#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}'
    return new_hex


def hex_to_hls(hex_str: str):
    r_i, g_i, b_i = hex_to_rgb(hex_str)
    r_f, g_f, b_f = colorsys.rgb_to_hls(r_i / 255.0, g_i / 255.0, b_i / 255.0)
    return round(r_f * 360), round(g_f * 100), round(b_f * 100)


def hls_to_hex(*hls):
    if len(hls) == 1:
        h, l, s = hls[0]
    elif len(hls) != 3:
        raise TypeError('Either pass three separate values or a tuple')
    else:
        h, l, s = hls

    if not ((h <= 1 and l <= 1 and s <= 1) and (type(h) == float and type(l) == float and type(s) == float)):
        h = h / 360
        l = l / 100
        s = s / 100
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    new_hex = f'#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}'
    return new_hex


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


def pretty_size(value: int, verbose=False, decimal=False) -> str:
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
    unit = f"{prefix[:e]}{_b + _i[:bool(len(prefix[:e]))]}{suffix[s:e]}"

    return f"{int(amount)} {unit}" if amount.is_integer() else f"{amount:.2f} {unit}"


def probe_video(video_path):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path.resolve())
    cap = cv2.VideoCapture(video_path.as_posix())
    r, frame = cap.read()
    if not r:
        raise IOError(f"Can't read video {video_path.resolve()}")
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame.shape, nb_frames

