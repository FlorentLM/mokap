import errno
import platform
import os
import colorsys
import re
from pathlib import Path
from typing import Union, Tuple
import numpy as np
from numpy.typing import ArrayLike


## General stuff

def common_prefix_suffix(s1: str, s2: str) -> Tuple[str, str]:
    """Finds the longest common prefix and suffix between two strings."""

    # common prefix
    prefix = ''
    for char1, char2 in zip(s1, s2):
        if char1 == char2:
            prefix += char1
        else:
            break

    # common suffix
    suffix = ''
    for char1, char2 in zip(s1[::-1], s2[::-1]):
        if char1 == char2:
            suffix += char1
        else:
            break
    suffix = suffix[::-1]

    return prefix, suffix


def natural_sort_key(s):
    _nsre = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]


## Colour conversions

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


## Pretty printing

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