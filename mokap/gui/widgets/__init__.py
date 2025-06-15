from mokap.utils.datatypes import ChessBoard, CharucoBoard

GUI_LOGGER = False
MAX_PLOT_X = 50
VERBOSE = True

# TODO: Board should be loaded from config file
DEFAULT_BOARD = CharucoBoard(rows=6, cols=5, square_length=1.5, markers_size=4)
# DEFAULT_BOARD = ChessBoard(rows=6, cols=5, square_length=1.5)
# DEFAULT_BOARD.to_file(Path.home())
# TODO: Add a "print board" button to GUI

def do_nothing():
    print('Nothing')

SLOW_UPDATE = 15.0
SLOW_UPDATE_INTERVAL = 1.0 / SLOW_UPDATE

DISPLAY_FRAMERATE = 50.0
DISPLAY_INTERVAL = 1.0 / DISPLAY_FRAMERATE

PROCESSING_FRAMERATE = 15.0     # let's process at... idk, 15 Hz?
PROCESSING_INTERVAL = 1.0 / PROCESSING_FRAMERATE