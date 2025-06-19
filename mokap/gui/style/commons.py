import platform
from pathlib import Path
from mokap.utils import hex_to_rgb


# Height (in pixels) of the OS taskbar
# TODO: this probably should be improved
if platform.system() == 'Windows':
    TASKBAR_H = 48
    TOPBAR_H = 23
else:
    TASKBAR_H = 48
    TOPBAR_H = 23

SPACING = 5

INFO_PANEL_MINSIZE_H = 200
VIDEO_PANEL_MINSIZE_H = 50
WINDOW_MIN_W = 630

# Icons

cwd = Path(__file__).parent

icon_capture = (cwd / 'icons' / 'capture.png').as_posix()
icon_capture_bw = (cwd / 'icons' / 'capture_bw.png').as_posix()
icon_snapshot = (cwd / 'icons' / 'snapshot.png').as_posix()
icon_snapshot_bw = (cwd / 'icons' / 'snapshot_bw.png').as_posix()
icon_rec_on = (cwd / 'icons' / 'rec.png').as_posix()
icon_rec_bw = (cwd / 'icons' / 'srec_bw.png').as_posix()
icon_move_bw =(cwd / 'icons' / 'move.png').as_posix()     # TODO make an icon - this is a temp one

# Colours
col_white = "#ffffff"
col_white_rgb = hex_to_rgb(col_white)
col_black = "#000000"
col_black_rgb = hex_to_rgb(col_black)
col_lightgray = "#e3e3e3"
col_lightgray_rgb = hex_to_rgb(col_lightgray)
col_midgray = "#c0c0c0"
col_midgray_rgb = hex_to_rgb(col_midgray)
col_darkgray = "#515151"
col_darkgray_rgb = hex_to_rgb(col_darkgray)
col_red = "#FF3C3C"
col_red_rgb = hex_to_rgb(col_red)
col_darkred = "#bc2020"
col_darkred_rgb = hex_to_rgb(col_darkred)
col_orange = "#FF9B32"
col_orange_rgb = hex_to_rgb(col_orange)
col_darkorange = "#cb782d"
col_darkorange_rgb = hex_to_rgb(col_darkorange)
col_yellow = "#FFEB1E"
col_yellow_rgb = hex_to_rgb(col_yellow)
col_yelgreen = "#A5EB14"
col_yelgreen_rgb = hex_to_rgb(col_yelgreen)
col_green = "#00E655"
col_green_rgb = hex_to_rgb(col_green)
col_darkgreen = "#39bd50"
col_darkgreen_rgb = hex_to_rgb(col_green)
col_blue = "#5ac3f5"
col_blue_rgb = hex_to_rgb(col_blue)
col_purple = "#c887ff"
col_purple_rgb = hex_to_rgb(col_purple)