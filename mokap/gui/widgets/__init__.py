from mokap.utils.datatypes import ChessBoard, CharucoBoard

BOARD_TYPES = {
    "ChArUco": CharucoBoard,
    "Chessboard": ChessBoard
}

# TODO: Board should be loaded from config file
DEFAULT_BOARD = CharucoBoard(rows=6, cols=5, square_length=1.5, markers_size=4)


MAX_PLOT_X = 50
VERBOSE = True

def do_nothing():
    print('Nothing')


SLOW_UPDATE = 5.0
SLOW_UPDATE_INTERVAL = 1.0 / SLOW_UPDATE

DISPLAY_FRAMERATE = 30.0        # 60 is fine when not recording, but not when recording so 30 it is
DISPLAY_INTERVAL = 1.0 / DISPLAY_FRAMERATE

PROCESSING_FRAMERATE = 15.0     # let's process at... idk, 15 Hz?
PROCESSING_INTERVAL = 1.0 / PROCESSING_FRAMERATE