import subprocess
from typing import List, Optional, Set, Tuple, Union
import numpy as np
from scipy import ndimage
import colorsys
from PIL import Image
import cv2


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


def get_random_colors(count):
    # TODO - improve ths thing

    rng = np.random.default_rng()
    hues = rng.integers(low=0, high=359, size=count)
    luminance = rng.integers(low=45, high=60, size=count)

    while any(hues[1:] - hues[:-1].copy() < 65):
        hues = rng.integers(low=0, high=359, size=count)

    saturation = [85] * count

    return hues, saturation, luminance


def randframe(w=1440, h=1080):
    return np.random.randint(0, 255, (w, h), dtype='<u1')


def to_ticks(frequency: float, duration=1.0) -> (float, float):
    """ Converts frequency, duration to tick length and ticks count """
    count = frequency * duration
    interval = 1 / frequency
    return interval, count


def to_freq(interval: float, count: float) -> (float, float):
    """ Converts tick length and number of ticks into frequency, total duration """
    duration = interval * count
    frequency = 1 / interval
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


def generate_charuco(board_rows, board_cols, square_length=None, marker_length=None, marker_bits=4, dict_size=100, save_svg=True):

    if square_length is None:
        square_length = 8
    if marker_length is None:
        marker_length = 6

    dict_name = f'DICT_{marker_bits}X{marker_bits}_{dict_size}'

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.CharucoBoard((board_cols, board_rows),   # number of chessboard squares in x and y directions
                                   square_length,                     # chessboard square side length (normally in meters)
                                   marker_length,                     # marker side length (same unit than squareLength)
                                   aruco_dict)

    if save_svg:
        image_size = (board_cols * square_length, board_rows * square_length)
        array = ~board.generateImage(image_size, None, 0, 1).astype(bool)

        chessboard_arrays = array[::square_length, ::square_length]
        chessboard_mask = chessboard_arrays.repeat(square_length, axis=0).repeat(square_length, axis=1)

        svg_lines = [f'<svg version="1.1" width="100%" height="100%" viewBox="0 0 {image_size[0]} {image_size[1]}" xmlns="http://www.w3.org/2000/svg">']

        svg_lines.append('  <g id="charuco">')
        svg_lines.append('    <g id="chessboard">')

        for r in range(board_rows):
            for c in range(board_cols):
                if chessboard_arrays[r, c]:
                    svg_lines.append(f'      <rect x="{c * square_length}" y="{r * square_length}" width="{square_length}" height="{square_length}" fill="#000000"/>')

        svg_lines.append('    </g>')
        svg_lines.append('    <g id="aruco_markers">')

        array[chessboard_mask] = False
        for r in range(image_size[1]):
            for c in range(image_size[0]):
                if array[r, c]:
                    svg_lines.append(f'      <rect x="{c}" y="{r}" width="1" height="1" fill="#000000"/>')

        svg_lines.append('    </g>')
        svg_lines.append('  </g>')
        svg_lines.append('</svg>')

        with open(f'Charuco_{board_rows}x{board_cols}_{dict_name}.svg', 'w') as f:
            f.write('\n'.join(svg_lines))

    return aruco_dict, board


##

