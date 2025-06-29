import errno
import platform
import sys
import os
import colorsys
from pathlib import Path
from typing import Union, Sequence, Tuple, Optional

import numpy as np
from numpy.typing import ArrayLike

# TODO: Functions in this file need to be moved


class CallbackOutputStream:
    """
    Simple class to capture stdout and use it with alive_progress
    """
    def __init__(self, callback, keep_stdout=False, keep_stderr=False):
        self.callback = callback
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.keep_stdout = keep_stdout
        self.keep_stderr = keep_stderr

    def __enter__(self):
        sys.stdout = self
        if not self.keep_stderr:
            sys.stderr = self
            sys.stderr = open(os.devnull, 'w')  # Redirect stderr to null device
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        if not self.keep_stderr:
            sys.stderr.close()  # Close the null device
            sys.stderr = self.original_stderr

    def write(self, data):
        if '\n' in data:
            self.callback()
        if self.keep_stdout:
            self.original_stdout.write(data)

    def flush(self):
        self.original_stdout.flush()


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


def pretty_size(value: int, verbose=False, decimal=False) -> str:
    """ Get sizes in strings in human-readable format """

    prefixes_dec = ['Yotta', 'Zetta', 'Exa', 'Peta', 'Tera', 'Giga', 'Mega', 'kilo', '']
    prefixes_bin = ['Yobi', 'Zebi', 'Exbi', 'Pebi', 'Tebi', 'Gibi', 'Mebi', 'Kibi', '']

    prefixes, _i = (prefixes_dec, '') if decimal else (prefixes_bin, 'i')

    suffix = 'Byte'
    div = 1
    prefix = ''
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


def pretty_microseconds(microsecons_value):
    """ Formats microseconds into a human-readable string (µs, ms, s) """
    if microsecons_value < 1000:
        return f"{microsecons_value:.0f} µs"
    elif microsecons_value < 1_000_000:
        return f"{microsecons_value / 1000:.1f} ms"
    else:
        return f"{microsecons_value / 1_000_000:.2f} s"


def is_locked(path: Union[Path, str]):
    if 'Windows' in platform.platform():
        try:
            import msvcrt

            # open existing file without truncate
            fd = os.open(path, os.O_RDWR | os.O_EXCL)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, os.path.getsize(path) or 1)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, os.path.getsize(path) or 1)
            os.close(fd)
            return False
        except OSError:
            return True
    else:
        try:
            import fcntl

            with open(path, 'a+') as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
            return False  # lock acquired so probably not in use
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                return True  # already locked by another process
            raise


# This can be used to estimate thepretical camera matrices
# Values from https://www.digicamdb.com/sensor-sizes/
SENSOR_SIZES = {'1/4"': [3.20, 2.40],
                '1/3.6"': [4, 3],
                '1/3.4"': [4.23, 3.17],
                '1/3.2"': [4.5, 3.37],
                '1/3"': [4.8, 3.6],
                '1/2.9"': [4.96, 3.72],
                '1/2.7"': [5.33, 4],
                '1/2.5"': [5.75, 4.32],
                '1/2.4"': [5.90, 4.43],
                '1/2.35"': [6.03, 4.52],
                '1/2.33"': [6.08, 4.56],
                '1/2.3"': [6.16, 4.62],
                '1/2"': [6.4, 4.8],
                '1/1.9"': [6.74, 5.05],
                '1/1.8"': [7.11, 5.33],
                '1/1.76"': [7.27, 5.46],
                '1/1.75"': [7.31, 5.49],
                '1/1.72"': [7.44, 5.58],
                '1/1.7"': [7.53, 5.64],
                '1/1.65"': [7.76, 5.81],
                '1/1.63"': [7.85, 5.89],
                '1/1.6"': [8, 6],
                '8.64 x 6 mm': [8.64, 6],
                '2/3"': [8.8, 6.6],
                '10.82 x 7.52 mm': [10.82, 7.52],
                '1"': [13.2, 8.8],
                '14 x 9.3 mm': [14, 9.3],
                'Four Thirds': [17.3, 13],
                '18.1 x 13.5 mm': [18.1, 13.5],
                '1.5"': [18.7, 14],
                '20.7 x 13.8 mm': [20.7, 13.8],
                '21.5 x 14.4 mm': [21.5, 14.4],
                '22.2 x 14.8 mm': [22.2, 14.8],
                '22.3 x 14.9 mm': [22.3, 14.9],
                '22.4 x 15 mm': [22.4, 15],
                '22.5 x 15 mm': [22.5, 15],
                '22.7 x 15.1 mm': [22.7, 15.1],
                '22.8 x 15.5 mm': [22.8, 15.5],
                '23.1 x 15.4 mm': [23.1, 15.4],
                '23 x 15.5 mm': [23, 15.5],
                '23.2 x 15.4 mm': [23.2, 15.4],
                '23.3 x 15.5 mm': [23.3, 15.5],
                '23.4 x 15.6 mm': [23.4, 15.6],
                '23.5 x 15.6 mm': [23.5, 15.6],
                '23.7 x 15.5 mm': [23.7, 15.5],
                '23.6 x 15.6 mm': [23.6, 15.6],
                '23.5 x 15.7 mm': [23.5, 15.7],
                '23.7 x 15.6 mm': [23.7, 15.6],
                '23.6 x 15.7 mm': [23.6, 15.7],
                '23.7 x 15.7 mm': [23.7, 15.7],
                '23.6 x 15.8 mm': [23.6, 15.8],
                '24 x 16 mm': [24, 16],
                '27 x 18 mm': [27, 18],
                '27.65 x 18.43 mm': [27.65, 18.43],
                '27.9 x 18.6 mm': [27.9, 18.6],
                '28.7 x 18.7 mm': [28.7, 18.7],
                '28.7 x 19.1 mm': [28.7, 19.1],
                '35.6 x 23.8 mm': [35.6, 23.8],
                '35.7 x 23.8 mm': [35.7, 23.8],
                '35.8 x 23.8 mm': [35.8, 23.8],
                '35.8 x 23.9 mm': [35.8, 23.9],
                '35.9 x 23.9 mm': [35.9, 23.9],
                '36 x 23.9 mm': [36, 23.9],
                '35.9 x 24 mm': [35.9, 24],
                '36 x 24 mm': [36, 24],
                '45 x 30 mm': [45, 30],
                '44 x 33 mm': [44, 33]
                }


def estimate_camera_matrix(
    f_mm:               float,
    image_wh_px:        Sequence,
    sensor_wh_mm:       Optional[Sequence[float]] = None,
    pixel_pitch_um:     Optional[float] = None,
    roi:                Optional[Sequence[float]] = None,
    skew:               float = 0.0
) -> np.ndarray:
    """
    Estimate the camera intrinsic matrix K

    Args:
        f_mm: focal length in millimeters
        image_wh_px: (width, height) of the full image in pixels
        sensor_wh_mm: optional (sensor_width, sensor_height) in millimeters
        pixel_pitch_um: optional pixel pitch in micrometers
            If provided, this overrides sensor_wh_mm
        roi: optional region of interest as (x_offset, y_offset, width, height) in pixels
        skew: optional skew coefficient (default 0)

    Returns:
        K: 3×3 camera intrinsic matrix
    """
    image_w, image_h = image_wh_px[:2]

    # Compute fx, fy in pixels
    if pixel_pitch_um is not None:
        p_mm = pixel_pitch_um * 1e-3
        fx = f_mm / p_mm
        fy = f_mm / p_mm
    elif sensor_wh_mm is not None:
        sensor_w_mm, sensor_h_mm = sensor_wh_mm
        pixel_size_x = sensor_w_mm / image_w
        pixel_size_y = sensor_h_mm / image_h
        fx = f_mm / pixel_size_x
        fy = f_mm / pixel_size_y
    else:
        raise ValueError("Provide either sensor_wh_mm or pixel_pitch_um")

    # Principal point (center of full image)
    cx_full = image_w / 2.0
    cy_full = image_h / 2.0

    # Adjust for ROI if given
    if roi is not None:
        x0, y0 = roi[:2]
        cx = cx_full - x0
        cy = cy_full - y0
    else:
        cx, cy = cx_full, cy_full

    K = np.array([
        [fx,    skew, cx],
        [0.0,   fy,   cy],
        [0.0,   0.0,  1.0]
    ], dtype=float)

    return K


def pol_to_hsv(quad_0:   ArrayLike,
               quad_45:  ArrayLike,
               quad_90:  ArrayLike,
               quad_135: ArrayLike
) -> np.ndarray:
    """
    Packs 4 polarisation quadrants into a HSV colour image
          - Hue encodes polarisation angle (0–180°)
          - Saturation encodes degree of linear polarisation (0–1)
          - Value encodes relative total intensity
    """
    # Stokes
    S0 = quad_0 + quad_90
    S1 = quad_0 - quad_90
    S2 = quad_45 - quad_135

    # Degree of linear polarisation and angle
    dolp = np.sqrt(S1 ** 2 + S2 ** 2) / (S0 + 1e-8)
    theta = 0.5 * np.arctan2(S2, S1)  # radians
    theta = np.where(theta < 0, theta + np.pi, theta)  # wrap to [0, pi]

    # Normalize channels
    H = theta / np.pi
    S = np.clip(dolp, 0, 1)
    V = (S0 - S0.min()) / np.ptp(S0)

    return np.dstack((H, S, V)).astype(np.float32)
