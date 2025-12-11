# Text, histograms, sliders updates can run slowly
UI_UPDATE_FPS = 15.0

# Display runs at 30 FPS for reasonable smoothness, regardless of camera speed
DISPLAY_FPS = 30.0

# Limits how often processing runs during calibration
CALIB_PROCESSING_FPS = 15.0

# Cap camera acquisition during calibration
CALIB_HARDWARE_FPS_MAX = 30.0

from .windows_base import SharedBase, VideoWindowBase, FastImageItem
from .window_viewer3d import Viewer3D
from .windows_video import CalibrationVideoWindow, RecordingVideoWindow