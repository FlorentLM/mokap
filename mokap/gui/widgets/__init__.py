from mokap.utils.datatypes import ChessBoard, CharucoBoard

FAST_UPDATE = 16
SLOW_UPDATE = 200
GUI_LOGGER = False
MAX_PLOT_X = 50
VERBOSE = True

# TODO: Board should be loaded from config file
DEFAULT_BOARD = CharucoBoard(rows=6, cols=5, square_length=1.5, markers_size=4)
# DEFAULT_BOARD = ChessBoard(rows=6, cols=5, square_length=1.5)
# DEFAULT_BOARD.to_file(Path.home())

def do_nothing():
    print('Nothing')
