import subprocess
from typing import List, Optional, Set, Tuple, Union
import numpy as np
from scipy import ndimage
import colorsys
from PIL import Image
import cv2


def hex_to_hls(hex_str: str):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 3:
        hex_str = ''.join([c + c for c in hex_str])

    r_i, g_i, b_i = tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))
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


def generate_board(board_rows, board_cols, aruco_sq_l, physical_l, dpi_out=1200, save=True):
    A4 = 210, 297

    ratio = 2  # chessboard squares are twice as big as aruco markers
    aruco_padding = 1  # There is a 1 aruco-grid wide margin around the aruco symbol

    marker_sq_px = (aruco_padding + aruco_sq_l + aruco_padding)
    chessboard_sq_px = marker_sq_px * ratio

    chessboard_sq_mm = physical_l / board_cols
    marker_sq_mm = chessboard_sq_mm / ratio

    board_w_px = chessboard_sq_px * board_cols
    board_h_px = chessboard_sq_px * board_rows

    board_w_mm = chessboard_sq_mm * board_cols
    board_h_mm = chessboard_sq_mm * board_rows

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, f'DICT_{aruco_sq_l}X{aruco_sq_l}_50'))
    board = cv2.aruco.CharucoBoard((board_cols, board_rows),  # number of chessboard squares in x and y directions
                                   chessboard_sq_mm,  # chessboard square side length (normally in meters)
                                   marker_sq_mm,  # marker side length (same unit than squareLength)
                                   aruco_dict)

    ppm = board_w_px / board_w_mm

    array = board.generateImage((board_w_px, board_h_px), None, 0, aruco_padding)
    img = Image.fromarray(array, 'L')

    doc_size_px = (int(round(A4[0] * ppm)), int(round(A4[1] * ppm)))

    padded = Image.new('RGB', doc_size_px, (255, 255, 255))
    padded.paste(img, ((doc_size_px[0] - board_w_px) // 2, (doc_size_px[1] - board_h_px) // 2))

    doc_size_inches = (A4[0] / 25.4, A4[1] / 25.4)

    doc_size_px_final = int(round(doc_size_inches[0] * dpi_out)), int(round(doc_size_inches[1] * dpi_out))
    out = padded.resize(doc_size_px_final, resample=Image.Resampling.NEAREST)

    if save:
        out.save(f'{board_rows}X{board_cols}_{board_w_mm}mm_{dpi_out}dpi.bmp', dpi=(dpi_out, dpi_out))
    return aruco_dict, board
