import sys
import os
import subprocess
from typing import List, Optional, Set, Tuple, Union
import colorsys
import cv2
from pathlib import Path
import numpy as np
from alive_progress import alive_bar
from tempfile import mkdtemp

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mokap.utils.datatypes import ChessBoard, CharucoBoard


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


def USB_on() -> None:
    subprocess.Popen(["uhubctl", "-l", "4-2", "-a", "1"], stdout=subprocess.PIPE)


def USB_off() -> None:
    subprocess.Popen(["uhubctl", "-l", "4-2", "-a", "0"], stdout=subprocess.PIPE)


def USB_status() -> int:
    ret = subprocess.Popen(["uhubctl", "-l", "4-2"], stdout=subprocess.PIPE)
    out, error = ret.communicate()
    if 'power' in str(out):
        return 0
    else:
        return 1


def ensure_list(s: Optional[str | List[str] | Tuple[str] | Set[str]]) -> List[str]:
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


def probe_video(video_path: Path | str):
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


def pad_dist_coeffs(dist_coeffs, max_nb=8):
    dist_coeffs = np.asarray(dist_coeffs)
    dist_coeffs_pad = np.zeros(max_nb, dtype=np.float32)
    dist_coeffs_pad[:len(dist_coeffs[:max_nb])] = dist_coeffs[:max_nb]
    return dist_coeffs_pad


def load_full_video(video_path: Path | str):
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(video_path.resolve())

    cap = cv2.VideoCapture(video_path.as_posix())
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    r, frame = cap.read()
    if not r:
        raise IOError(f"Can't read video {video_path.resolve()}")
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    try:
        full_video = np.zeros((nb_frames, *frame.shape), dtype=np.uint8)
    except np.core._exceptions._ArrayMemoryError:
        temp_file = Path(mkdtemp()) / f'{video_path.stem}.dat'
        full_video = np.memmap(temp_file, dtype=np.uint8, mode='w+', shape=(nb_frames, *frame.shape))
        print(f"Allocated disk-mapped file: {temp_file}")

    with alive_bar(nb_frames, force_tty=True) as bar:
        for i in range(nb_frames):
            r, frame = cap.read()
            if r:
                np.copyto(full_video[i, ...], frame)
            bar()

    cap.release()

    if isinstance(full_video, np.memmap):
        # Flush and reopen in read-only
        full_video.flush()
        full_video = np.memmap(temp_file, dtype=np.uint8, mode='r', shape=(nb_frames, *frame.shape))
    return full_video, temp_file


def generate_board_svg(board_params:    Union["ChessBoard", "CharucoBoard"],
                       file_path:       Union[Path, str],
                       multi_size:      bool = False,
                       factor:          float = 2.0,
                       dpi:             int = 1200):

    board_rows = board_params.rows
    board_cols = board_params.cols
    square_length_mm = board_params.square_length

    # We check like this to avoid circular imports
    if board_params.kind == 'charuco':
        is_charuco = True
    else:
        is_charuco = False

    if is_charuco:
        marker_bits = board_params.markers_size
        marker_length_mm = board_params.marker_length
        markers_dictionary = board_params.aruco_dict

        mk_l_bits = marker_bits + 2
        sq_l_bits = square_length_mm / marker_length_mm * mk_l_bits
        if not int(sq_l_bits) == sq_l_bits:
            raise AssertionError('Error creating board svg :(')         # TODO make sure this never happens
        else:
            sq_l_bits = int(sq_l_bits)

        # Ratios to convert between mm and 'bits' (i.e. pixels ...but not really, since we work with svg here)
        mm_to_bits = sq_l_bits / square_length_mm
        margin = int((sq_l_bits - mk_l_bits) / 2)

    else:
        # For classic chessboard, no markers so 1 'bit' = 1 entire square
        # 1 bit = 1 square = square_length_mm mm
        sq_l_bits = 1
        mm_to_bits = 1.0 / square_length_mm     # 1 mm = 1 / square_length_mm bits
        margin = 0

    chessboard_arr = (~np.indices((board_rows, board_cols)).sum(axis=0) % 2).astype(bool)

    A4_mm = np.array([210, 297])
    A4_size_bits = A4_mm * mm_to_bits

    text_h_bits = 3.0 * mm_to_bits
    board_size_bits = np.array([sq_l_bits * board_cols, sq_l_bits * board_rows])

    # If only one size, use the one that comes with the board object and position in the centre of the page
    if is_charuco:
        filename = f'Charuco{board_rows}x{board_cols}_markers{marker_bits}x{marker_bits}-margin{margin}.svg'
    else:
        filename = f'Chessboard{board_rows}x{board_cols}.svg'

    if multi_size:
        filename = 'Multi_' + filename

        bleed = 15.0 * mm_to_bits
        spacing = 25.0 * mm_to_bits
        text_width = 30.0 * mm_to_bits

        # Compute the scales to generate, from 1/4 of the page width, to the theoretical smallest with current dpi
        min_scale = 1 / dpi * 25.4 * mm_to_bits   # Theoretical smallest marker size with visible bits
        max_scale = A4_size_bits[1] / 8 / board_size_bits[1]

        nb_scales = int(np.ceil(np.log(max_scale / min_scale) / np.log(factor)) + 1)

        scales = np.array([max_scale / (factor ** i) for i in range(nb_scales)])

        # Position the boards on the page
        positions = [np.array([bleed, bleed])]

        # These are used when one page width is full
        x_ref, y_ref = positions[0]
        scale_ref = scales[0]

        for i in range(1, nb_scales):
            x, y = positions[i-1]
            next_x = max(x + text_width + spacing, x + board_size_bits[0] * scales[i-1] + spacing)
            if max(next_x + text_width, next_x + board_size_bits[0] * scales[i]) < A4_size_bits[0] - bleed:
                next_pos = np.array([next_x, y_ref])
            else:
                next_y = y_ref + board_size_bits[1] * scale_ref + text_h_bits * 4 + spacing
                next_pos = np.array([x_ref, next_y])
                x_ref, y_ref = next_pos
                scale_ref = scales[i]
            positions.append(next_pos)
    else:
        scales = [1.0]
        positions = [A4_size_bits / 2.0 - board_size_bits / 2.0]    # page centre

    # Start svg content
    svg_lines = [
        f'<svg version="1.1" width="100%" height="100%" viewBox="0 0 {A4_size_bits[0]} {A4_size_bits[1]}" xmlns="http://www.w3.org/2000/svg">',
        f' <rect id="background" x="0" y="0" width="{A4_size_bits[0]}" height="{A4_size_bits[1]}" fill="none" stroke="#000000" stroke-width="0.1"/>'
    ]

    for board_pos, board_scale in zip(positions, scales):

        # Container group with charuco board, cutting guides and info text
        svg_lines.append(f'  <g id="container" transform="translate({board_pos[0]}, {board_pos[1]})">')

        # Add cutting guides
        svg_lines.append(f'    <line x1="{-2 * mm_to_bits}" y1="{-2 * mm_to_bits}" x2="{-2 * mm_to_bits}" y2="{-7 * mm_to_bits}" stroke="black" stroke-width="{0.1 * mm_to_bits}" />')
        svg_lines.append(f'    <line x1="{-7 * mm_to_bits}" y1="{-2 * mm_to_bits}" x2="{-2 * mm_to_bits}" y2="{-2 * mm_to_bits}" stroke="black" stroke-width="{0.1 * mm_to_bits}" />')
        svg_lines.append(f'    <line x1="{board_size_bits[0] * board_scale + 2 * mm_to_bits}" y1="{-2 * mm_to_bits}" x2="{board_size_bits[0] * board_scale + 2 * mm_to_bits}" y2="{-7 * mm_to_bits}" stroke="black" stroke-width="{0.1 * mm_to_bits}" />')
        svg_lines.append(f'    <line x1="{board_size_bits[0] * board_scale + 2 * mm_to_bits}" y1="{-2 * mm_to_bits}" x2="{board_size_bits[0] * board_scale + 7 * mm_to_bits}" y2="{-2 * mm_to_bits}" stroke="black" stroke-width="{0.1 * mm_to_bits}" />')
        svg_lines.append(f'    <line x1="{-2 * mm_to_bits}" y1="{board_size_bits[1] * board_scale + 2 * mm_to_bits}" x2="{-2 * mm_to_bits}" y2="{board_size_bits[1] * board_scale + 7 * mm_to_bits}" stroke="black" stroke-width="{0.1 * mm_to_bits}" />')
        svg_lines.append(f'    <line x1="{-7 * mm_to_bits}" y1="{board_size_bits[1] * board_scale + 2 * mm_to_bits}" x2="{-2 * mm_to_bits}" y2="{board_size_bits[1] * board_scale + 2 * mm_to_bits}" stroke="black" stroke-width="{0.1 * mm_to_bits}" />')
        svg_lines.append(f'    <line x1="{board_size_bits[0] * board_scale + 2 * mm_to_bits}" y1="{board_size_bits[1] * board_scale + 2 * mm_to_bits}" x2="{board_size_bits[0] * board_scale + 2 * mm_to_bits}" y2="{board_size_bits[1] * board_scale + 7 * mm_to_bits}" stroke="black" stroke-width="{0.1 * mm_to_bits}" />')
        svg_lines.append(f'    <line x1="{board_size_bits[0] * board_scale + 2 * mm_to_bits}" y1="{board_size_bits[1] * board_scale + 2 * mm_to_bits}" x2="{board_size_bits[0] * board_scale + 7 * mm_to_bits}" y2="{board_size_bits[1] * board_scale + 2 * mm_to_bits}" stroke="black" stroke-width="{0.1 * mm_to_bits}" />')

        # subcontainer group with white background
        svg_lines.append(f'    <g id="subcontainer" transform="scale({board_scale})">')
        svg_lines.append(f'      <rect id="background" x="0" y="0" width="{board_size_bits[0]}" height="{board_size_bits[1]}" fill="#ffffff"/>')

        # Chessboard group
        svg_lines.append('      <g id="chessboard">')
        cc, rr = np.where(chessboard_arr)
        for i, rc in enumerate(zip(rr, cc)):
            svg_lines.append(f'        <rect id="{i}" x="{rc[0] * sq_l_bits}" y="{rc[1] * sq_l_bits}" width="{sq_l_bits}" height="{sq_l_bits}" fill="#000000"/>')
        svg_lines.append('      </g>')

        if is_charuco:
            # Aruco markers group
            svg_lines.append('      <g id="aruco_markers">')
            cc, rr = np.where(~chessboard_arr)
            for i, rc in enumerate(zip(rr, cc)):
                marker = markers_dictionary.generateImageMarker(i, mk_l_bits, mk_l_bits).astype(bool)
                py, px = np.where(marker)
                svg_lines.append(f'        <g id="{i}">')
                svg_lines.append(
                    f'          <rect x="{rc[0] * sq_l_bits + margin}" y="{rc[1] * sq_l_bits + margin}" width="{mk_l_bits}" height="{mk_l_bits}" fill="#000000"/>')
                for x, y in zip(px, py):
                    svg_lines.append(f'          <rect x="{rc[0] * sq_l_bits + x + margin}" y="{rc[1] * sq_l_bits + y + margin}" width="1" height="1" fill="#ffffff"/>')
                svg_lines.append('        </g>')

            svg_lines.append('      </g>')
        svg_lines.append('    </g>')

        # Add text with sizes
        bsize_text = f'{board_scale * square_length_mm * board_rows:.1f} x {board_scale * square_length_mm * board_cols:.1f} mm'
        sqsize_text = f'(squares: {board_scale * square_length_mm:.3f} mm)'
        if is_charuco:
            msize_text = f'(markers: {board_scale * marker_length_mm:.3f} mm)'
        svg_lines.append(f'    <text x="0" y="{board_size_bits[1] * board_scale + text_h_bits * 4}" font-family="monospace" font-size="{text_h_bits}" font-weight="bold">{bsize_text}</text>')
        svg_lines.append(f'    <text x="0" y="{board_size_bits[1] * board_scale + text_h_bits * 5}" font-family="monospace" font-size="{text_h_bits}" font-weight="bold">{sqsize_text}</text>')
        if is_charuco:
            svg_lines.append(f'    <text x="0" y="{board_size_bits[1] * board_scale + text_h_bits * 6}" font-family="monospace" font-size="{text_h_bits}" font-weight="bold">{msize_text}</text>')
        svg_lines.append('  </g>')

    svg_lines.append('</svg>')

    file_path.mkdir(parents=True, exist_ok=True)    # TODO - check if user passed the file name in there...
    with open(file_path / filename, 'w') as f:
        f.write('\n'.join(svg_lines))
        print(f'Saved calibration board as {file_path / filename}')