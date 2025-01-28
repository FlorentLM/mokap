import os
import subprocess
import sys
import platform
import psutil
import screeninfo
import cv2
from functools import partial
from collections import deque
from datetime import datetime
from pathlib import Path
import numpy as np
from mokap import utils, calibration_utils
from mokap.calibration import DetectionTool, MonocularCalibrationTool
from PIL import Image
from PySide6.QtCore import Qt, QTimer, QEvent, QDir, QObject, Signal, Slot, QThread
from PySide6.QtGui import QIcon, QImage, QPixmap, QCursor, QBrush, QPen, QColor, QFont
from PySide6.QtWidgets import (QApplication, QMainWindow, QStatusBar, QSlider, QGraphicsView, QGraphicsScene,
                               QGraphicsRectItem, QComboBox, QLineEdit, QProgressBar, QCheckBox, QScrollArea,
                               QWidget, QLabel, QFrame, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
                               QPushButton, QSizePolicy, QGraphicsTextItem, QTextEdit, QFileDialog, QSpinBox,
                               QDialog, QDoubleSpinBox, QDialogButtonBox)

##

class GUILogger:
    def __init__(self):
        self.text_area = None
        self._temp_output = ''
        sys.stdout = self
        sys.stderr = self

    def register_text_area(self, text_area):
        self.text_area = text_area
        self.text_area.insertPlainText(self._temp_output)

    def write(self, text):
        if self.text_area is None:
            # Temporarily capture console output to display later in the log widget
            self._temp_output += f'{text}'
        else:
            self.text_area.insertPlainText(text)

    def flush(self):
        pass


class BoardParamsDialog(QDialog):

    def __init__(self, rows=6, cols=5, square_length=1.5, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Board Settings")
        self.setModal(True)

        self._rows = rows
        self._cols = cols
        self._square_length = square_length

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # Rows
        row_layout = QHBoxLayout()
        row_label = QLabel("Rows:")
        self.row_spin = QSpinBox()
        self.row_spin.setMinimum(2)
        self.row_spin.setMaximum(30)
        self.row_spin.setValue(self._rows)
        row_layout.addWidget(row_label)
        row_layout.addWidget(self.row_spin)
        main_layout.addLayout(row_layout)

        # Columns
        col_layout = QHBoxLayout()
        col_label = QLabel("Columns:")
        self.col_spin = QSpinBox()
        self.col_spin.setMinimum(2)
        self.col_spin.setMaximum(30)
        self.col_spin.setValue(self._cols)
        col_layout.addWidget(col_label)
        col_layout.addWidget(self.col_spin)
        main_layout.addLayout(col_layout)

        # Square size
        sq_layout = QHBoxLayout()
        sq_label = QLabel("Square length (mm):")
        self.sq_spin = QDoubleSpinBox()
        self.sq_spin.setMinimum(0.01)
        self.sq_spin.setMaximum(1000.0)
        self.sq_spin.setDecimals(2)
        self.sq_spin.setValue(self._square_length)
        sq_layout.addWidget(sq_label)
        sq_layout.addWidget(self.sq_spin)
        main_layout.addLayout(sq_layout)

        # 'Apply to all' checkbox
        self.apply_all_checkbox = QCheckBox("Apply to all cameras")
        self.apply_all_checkbox.setChecked(True)
        main_layout.addWidget(self.apply_all_checkbox)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        main_layout.addWidget(btns)

        self.setLayout(main_layout)

    def get_values(self):
        return self.row_spin.value(), self.col_spin.value(), self.sq_spin.value(), self.apply_all_checkbox.isChecked()

##

class MainWorker(QObject):
    """
        This worker lives in its own thread and does stuff on the full resolution image
    """
    # testing stuff for now, let's send bounding boxes of a fake detection:
    result_ready = Signal(list)
    finished_processing = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True

    @Slot(np.ndarray)
    def process_frame(self, frame):
        if not self._running:
            return

        # 1- TODO: motiuon detection
        bboxes = self._do_motion_detection(frame)

        # 2- Emit the results (metadata) back to the main thread, and emit 'done' signal
        self.result_ready.emit(bboxes)
        self.finished_processing.emit()

    def _do_motion_detection(self, frame):
        # TODO. for now let's just assume we found one bounding box around (100,100) of size (50,40)
        return [(100, 100, 50, 40)]

    def stop(self):
        self._running = False


class CalibWorker(QObject):
    """
        This worker lives in its own thread and does detection/calibration
    """
    result_ready = Signal(np.ndarray)   # carry back the annotated image back to the main thread (to didplay it)
    finished_processing = Signal()      # signals when this worker is busy / free

    def __init__(self, calib_tool, auto_sample=True, auto_compute=True, parent=None):
        super().__init__(parent)
        self.calib_tool = calib_tool
        self.auto_sample = auto_sample
        self.auto_compute = auto_compute
        self._running = True

    @Slot(np.ndarray)
    def process_frame(self, frame):
        # Called for each new frame received from main thread to process

        if not self._running:
            return

        # 1- Detect
        self.calib_tool.detect(frame)

        # 2- Auto-register samples
        if self.auto_sample:
            self.calib_tool.auto_register(area_threshold=0.2, nb_points_threshold=4)

        # 3- Auto-calibrate
        if self.auto_compute:
            self.calib_tool.auto_compute_intrinsics(
                coverage_threshold=60,
                stack_length_threshold=15,
                simple_focal=True,
                complex_distortion=True
            )

        # 4- Compute extrinsics (if intrinsics exist)
        self.calib_tool.compute_extrinsics()

        # 5- Visualize (this returns the annotated image in full resolution) - TODO: maybe this should be scaled...
        annotated = self.calib_tool.visualise(errors_mm=True)

        # 6- Emit the annotated frame to the main thread, and emit the 'done' signal
        self.result_ready.emit(annotated)
        self.finished_processing.emit()

    def stop(self):
        self._running = False

##

# Create this immediately to capture everything
# gui_logger = GUILogger()
gui_logger = False


##


class VideoWindowBase(QWidget):
    BOTTOM_PANEL_H = 300
    WINDOW_MIN_W = 550
    if 'Windows' in platform.system():
        TASKBAR_H = 48
        TOPBAR_H = 23
    else:
        # TODO
        TASKBAR_H = 48
        TOPBAR_H = 23
    SPACING = 10

    def __init__(self, main_window_ref, idx):
        super().__init__()

        self._main_window = main_window_ref
        self.idx = idx

        self._camera = self._main_window.mc.cameras[self.idx]
        self._source_shape = self._camera.shape

        self._bg_colour = f'#{self._main_window.mc.colours[self._camera.name].lstrip("#")}'
        self._fg_colour = self._main_window.col_white if utils.hex_to_hls(self._bg_colour)[
                                                             1] < 60 else self._main_window.col_black

        # Where the frame data will be stored
        self._frame_buffer = np.zeros((*self._source_shape[:2], 3), dtype=np.uint8)
        self._display_buffer = np.zeros((*self._source_shape[:2], 3), dtype=np.uint8)

        # Init clock and counter
        self._clock = datetime.now()
        self._capture_fps = deque(maxlen=10)
        self._last_capture_count = 0

        # Init states
        self._warning = False
        self._warning_text = '[WARNING]'

        # Some other stuff
        self._wanted_fps = self._camera.framerate

        self.setWindowTitle(f'{self._camera.name.title()} camera')

        self.positions = np.array([['nw', 'n', 'ne'],
                                   ['w', 'c', 'e'],
                                   ['sw', 's', 'se']])

        # This updater function foes not need to run super frequently
        self.timer_update = QTimer(self)
        self.timer_update.timeout.connect(self._update_vars)
        self.timer_update.start(100)

    #  ============= UI constructors =============
    def _init_common_ui(self):

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.VIDEO_FEED = QLabel()
        self.VIDEO_FEED.setStyleSheet('background-color: black;')
        self.VIDEO_FEED.setMinimumSize(1, 1)  # Important! Otherwise it crashes when reducing the size of the window
        self.VIDEO_FEED.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.VIDEO_FEED, 1)

        self._blit_image()     # Call this once to initialise it

        main_layout.addWidget(self.VIDEO_FEED, 1)

        self.BOTTOM_PANEL = QWidget()
        bottom_panel_v_layout = QVBoxLayout(self.BOTTOM_PANEL)
        bottom_panel_v_layout.setContentsMargins(0, 0, 0, 0)
        bottom_panel_v_layout.setSpacing(0)
        bottom_panel_h = QWidget()
        bottom_panel_h_layout = QHBoxLayout(bottom_panel_h)

        # Camera name bar
        camera_name_bar = QLabel(f'{self._camera.name.title()} camera')
        camera_name_bar.setFixedHeight(25)
        camera_name_bar.setAlignment(Qt.AlignCenter)
        camera_name_bar.setStyleSheet(f"color: {self.colour_2}; background-color: {self.colour}; font: bold;")

        bottom_panel_v_layout.addWidget(camera_name_bar)
        bottom_panel_v_layout.addWidget(bottom_panel_h)

        self.LEFT_GROUP = QGroupBox("Information")
        bottom_panel_h_layout.addWidget(self.LEFT_GROUP)

        self.CENTRE_GROUP = QGroupBox("Control")
        bottom_panel_h_layout.addWidget(self.CENTRE_GROUP, 1)     # Expand the centre group only

        self.RIGHT_GROUP = QGroupBox("View")
        bottom_panel_h_layout.addWidget(self.RIGHT_GROUP)

        main_layout.addWidget(self.BOTTOM_PANEL)

        # LEFT GROUP
        left_group_layout = QVBoxLayout(self.LEFT_GROUP)

        self.triggered_value = QLabel()
        self.resolution_value = QLabel()
        self.capturefps_value = QLabel()
        self.exposure_value = QLabel()
        self.brightness_value = QLabel()
        self.temperature_value = QLabel()

        self.triggered_value.setText("Yes" if self._camera.triggered else "No")
        self.resolution_value.setText(f"{self.source_shape[1]}×{self.source_shape[0]} px")
        self.capturefps_value.setText(f"Off")
        self.exposure_value.setText(f"{self._camera.exposure} µs")
        self.brightness_value.setText(f"-")
        self.temperature_value.setText(f"{self._camera.temperature}°C" if self._camera.temperature is not None else '-')

        labels_and_values = [
            ('Triggered', self.triggered_value),
            ('Resolution', self.resolution_value),
            ('Capture', self.capturefps_value),
            ('Exposure', self.exposure_value),
            ('Brightness', self.brightness_value),
            ('Temperature', self.temperature_value),
        ]

        for label, value in labels_and_values:
            line = QWidget()
            line_layout = QHBoxLayout(line)
            line_layout.setContentsMargins(1, 1, 1, 1)
            line_layout.setSpacing(5)

            label = QLabel(f"{label} :")
            label.setAlignment(Qt.AlignRight)
            label.setStyleSheet(f"color: {self._main_window.col_darkgray}; font: bold;")
            label.setMinimumWidth(88)
            line_layout.addWidget(label)

            value.setStyleSheet("font: regular;")
            value.setAlignment(Qt.AlignLeft)
            value.setMinimumWidth(90)
            line_layout.addWidget(value)

            left_group_layout.addWidget(line)

        # RIGHT GROUP
        right_group_layout = QVBoxLayout(self.RIGHT_GROUP)

        line = QWidget()
        line_layout = QHBoxLayout(line)
        line_layout.setContentsMargins(1, 1, 1, 1)
        line_layout.setSpacing(5)

        line_layout.addStretch(1)

        l_windowsnap = QLabel("Window snap: ")
        l_windowsnap.setAlignment(Qt.AlignVCenter)
        l_windowsnap.setStyleSheet(f"color: {self._main_window.col_darkgray};")
        line_layout.addWidget(l_windowsnap)

        buttons_windowsnap = QWidget()
        buttons_windowsnap.setFixedSize(55, 55)
        buttons_windowsnap_layout = QGridLayout(buttons_windowsnap)
        buttons_windowsnap_layout.setContentsMargins(0, 0, 0, 0)
        buttons_windowsnap_layout.setSpacing(0)
        buttons_windowsnap_layout.setColumnStretch(3, 1)
        buttons_windowsnap_layout.setColumnStretch(3, 1)

        for r in range(3):
            for c in range(3):
                button = QPushButton()
                buttons_windowsnap_layout.setContentsMargins(0, 0, 0, 0)
                button.clicked.connect(partial(self.move_to, self.positions[r, c]))
                buttons_windowsnap_layout.addWidget(button, r, c)

        line_layout.addWidget(buttons_windowsnap, 1)

        right_group_layout.addWidget(line)

    def _init_specific_ui(self):
        pass

    #  ============= Some useful attributes =============
    @property
    def name(self) -> str:
        return self._camera.name

    @property
    def colour(self) -> str:
        return f'#{self._bg_colour.lstrip("#")}'

    color = colour

    @property
    def colour_2(self) -> str:
        return f'#{self._fg_colour.lstrip("#")}'

    color_2 = colour_2

    @property
    def source_shape(self):
        return self._source_shape

    @property
    def aspect_ratio(self):
        return self._source_shape[1] / self._source_shape[0]

    #  ============= Some common window-related methods =============
    def auto_size(self):

        # If landscape screen
        if self._main_window.selected_monitor.height < self._main_window.selected_monitor.width:
            available_h = (self._main_window.selected_monitor.height - VideoWindowBase.TASKBAR_H) // 2 - VideoWindowBase.SPACING * 3
            video_max_h = available_h - self.BOTTOM_PANEL.height() - VideoWindowBase.TOPBAR_H
            video_max_w = video_max_h * self.aspect_ratio

            h = int(video_max_h + self.BOTTOM_PANEL.height())
            w = int(video_max_w)

        # If portrait screen
        else:
            video_max_w = self._main_window.selected_monitor.width // 2 - VideoWindowBase.SPACING * 3
            video_max_h = video_max_w / self.aspect_ratio

            h = int(video_max_h + self.BOTTOM_PANEL.height())
            w = int(video_max_w)

        self.resize(w, h)

    def auto_move(self):
        if self._main_window.selected_monitor.height < self._main_window.selected_monitor.width:
            # First corners, then left right, then top and bottom,  and finally centre
            positions = ['nw', 'sw', 'ne', 'se', 'n', 's', 'w', 'e', 'c']
        else:
            # First corners, then top and bottom, then left right, and finally centre
            positions = ['nw', 'sw', 'ne', 'se', 'w', 'e', 'n', 's', 'c']

        nb_positions = len(positions)

        if self.idx <= nb_positions:
            pos = positions[self.idx]
        else:  # Start over to first position
            pos = positions[self.idx % nb_positions]

        self.move_to(pos)

    def move_to(self, pos):

        monitor = self._main_window.selected_monitor
        w = self.geometry().width()
        h = self.geometry().height()

        sp = VideoWindowBase.SPACING

        match pos:
            case 'nw':
                self.move(monitor.x + sp, monitor.y + sp)
            case 'n':
                self.move(monitor.x + monitor.width // 2 - w // 2, monitor.y + sp)
            case 'ne':
                self.move(monitor.x - sp + monitor.width - w - 1, monitor.y + sp)
            case 'w':
                self.move(monitor.x + sp, monitor.y + monitor.height // 2 - h // 2)
            case 'c':
                self.move(monitor.x + monitor.width // 2 - w // 2, monitor.y + monitor.height // 2 - h // 2)
            case 'e':
                self.move(monitor.x - sp + monitor.width - w - 1, monitor.y + monitor.height // 2 - h // 2)
            case 'sw':
                self.move(monitor.x + sp, monitor.y + monitor.height - h - VideoWindowBase.TASKBAR_H - sp)
            case 's':
                self.move(monitor.x + monitor.width // 2 - w // 2, monitor.y + monitor.height - h - VideoWindowBase.TASKBAR_H - sp)
            case 'se':
                self.move(monitor.x - sp + monitor.width - w - 1, monitor.y + monitor.height - h - VideoWindowBase.TASKBAR_H - sp)

    def toggle_visibility(self, override=None):

        if override is None:
            override = not self.isVisible()

        if self.isVisible() and override is False:
            self._main_window.secondary_windows_visibility_buttons[self.idx].setChecked(False)
            self.hide()

        elif not self.isVisible() and override is True:
            self._main_window.secondary_windows_visibility_buttons[self.idx].setChecked(True)
            self.show()

    #  ============= Display-related common methods =============
    def _refresh_framebuffer(self):
        """ Grabs a new frame from the cameras and stores it in the frame buffer """
        if self._main_window.mc.acquiring:
            arr = self._main_window.mc.get_current_framebuffer(self.idx)
            if arr is not None:
                if len(self.source_shape) == 2:
                    # Using cv for this is faster than any way using numpy (?)
                    self._frame_buffer = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB, dst=self._frame_buffer)
                else:
                    self._frame_buffer = arr
        else:
            self._frame_buffer.fill(0)

    def _resize_to_display(self):
        """ Fills and resizes the display buffer to the current window size """
        scale = min(self.VIDEO_FEED.width() / self._frame_buffer.shape[1], self.VIDEO_FEED.height() / self._frame_buffer.shape[0])
        self._display_buffer = cv2.resize(self._frame_buffer, (0, 0), dst=self._display_buffer, fx=scale, fy=scale)

    def _blit_image(self):
        """ Applies the content of display buffers to the GUI """
        h, w = self._display_buffer.shape[:2]
        q_img = QImage(self._display_buffer.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.VIDEO_FEED.setPixmap(pixmap)

    #  ============= Common update method for texts and stuff =============
    def _update_vars(self):

        if self.isVisible():

            now = datetime.now()

            if self._main_window.mc.acquiring:

                cap_fps = sum(list(self._capture_fps)) / len(self._capture_fps) if self._capture_fps else 0

                if 0 < cap_fps < 1000:
                    if abs(cap_fps - self._wanted_fps) > 10:
                        self._warning_text = '[WARNING] Framerate'
                        self._warning = True
                    else:
                        self._warning = False
                    self.capturefps_value.setText(f"{cap_fps:.2f} fps")
                else:
                    self.capturefps_value.setText("-")

                brightness = np.round(self._frame_buffer.mean() / 255 * 100, decimals=2)
                self.brightness_value.setText(f"{brightness:.2f}%")
            else:
                self.capturefps_value.setText("Off")
                self.brightness_value.setText("-")

            # Update the temperature label colour
            if self._camera.temperature is not None:
                self.temperature_value.setText(f'{self._camera.temperature:.1f}°C')
            if self._camera.temperature_state == 'Ok':
                self.temperature_value.setStyleSheet(f"color: {self._main_window.col_green}; font: bold;")
            elif self._camera.temperature_state == 'Critical':
                self.temperature_value.setStyleSheet(f"color: {self._main_window.col_orange}; font: bold;")
            elif self._camera.temperature_state == 'Error':
                self.temperature_value.setStyleSheet(f"color: {self._main_window.col_red}; font: bold;")
            else:
                self.temperature_value.setStyleSheet(f"color: {self._main_window.col_yellow}; font: bold;")

            # Update display fps
            dt = (now - self._clock).total_seconds()
            ind = int(self._main_window.mc.indices[self.idx])
            if dt > 0:
                self._capture_fps.append((ind - self._last_capture_count) / dt)

            self._clock = now
            self._last_capture_count = ind


class VideoWindowMain(VideoWindowBase):
    newFrameSignal = Signal(np.ndarray)

    def __init__(self, main_window_ref, idx):
        super().__init__(main_window_ref, idx)

        self._n_enabled = False
        self._magnifier_enabled = False

        # Magnification parameters
        self.magn_window_w = 100
        self.magn_window_h = 100
        self.magn_window_x = 10
        self.magn_window_y = 10

        # Target area for the magnification (initialise at the centre)
        self.magn_target_cx = 0.5
        self.magn_target_cy = 0.5

        # Mouse states
        self.left_mouse_btn = False
        self.right_mouse_btn = False

        # Focus view parameters
        # Kernel to use for focus detection
        self._kernel = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]], dtype=np.uint8)

        ##

        # Setup worker
        self.worker_thread = QThread(self)
        self.worker = MainWorker()
        self.worker.moveToThread(self.worker_thread)

        # Setup signals
        self.newFrameSignal.connect(self.worker.process_frame, type=Qt.QueuedConnection)
        self.worker.result_ready.connect(self.on_worker_result)
        self.worker.finished_processing.connect(self.on_worker_finished)
        self.worker_thread.start()

        # Store worker results and its current state
        self._bboxes = []
        self._worker_busy = False
        self._latest_frame = None

        # This updater function should only run at 60 fps
        self.timer_video = QTimer(self)
        self.timer_video.timeout.connect(self._update_images)
        self.timer_video.start(16)  # 60 FPS

        # Finish building the UI by calling the other constructors
        self._init_common_ui()
        self._init_specific_ui()
        self.auto_size()

    #  ============= UI constructors =============
    def _init_specific_ui(self):

        # Add mouse click detection to video feed (for the magnifier)
        self.VIDEO_FEED.installEventFilter(self)

        # CENTRE GROUP
        centre_group_layout = QHBoxLayout(self.CENTRE_GROUP)
        centre_group_layout.setContentsMargins(5, 5, 5, 5)

        self.camera_controls_sliders = {}
        self.camera_controls_sliders_labels = {}
        self.camera_controls_sliders_scales = {}
        self._val_in_sync = {}

        slider_params = [
            ('framerate', (int, 1, int(self._main_window.mc.cameras[self.idx].max_framerate), 1, 1)),
            ('exposure', (int, 21, 100000, 5, 1)),  # in microseconds - 100000 microseconds ~ 10 fps
            ('blacks', (float, 0.0, 32.0, 0.5, 3)),
            ('gain', (float, 0.0, 36.0, 0.5, 3)),
            ('gamma', (float, 0.0, 3.99, 0.05, 3))
        ]

        centre_group_sliders = QWidget()
        centre_group_sliders_layout = QVBoxLayout(centre_group_sliders)
        centre_group_sliders_layout.setContentsMargins(0, 20, 0, 5)
        centre_group_sliders_layout.setSpacing(0)

        sync_groupbox = QGroupBox("Sync")
        sync_groupbox.setContentsMargins(5, 20, 0, 5)
        sync_groupbox_layout = QVBoxLayout(sync_groupbox)

        sync_groupbox_layout.setSpacing(12)

        for label, params in slider_params:
            type_, min_val, max_val, step, digits = params

            line = QWidget()
            line_layout = QHBoxLayout(line)
            line_layout.setContentsMargins(1, 1, 1, 1)
            line_layout.setSpacing(2)

            param_value = getattr(self._main_window.mc.cameras[self.idx], label)

            slider_label = QLabel(f'{label.title()}:')
            slider_label.setFixedWidth(70)
            slider_label.setContentsMargins(0, 5, 5, 0)
            slider_label.setAlignment(Qt.AlignRight)

            line_layout.addWidget(slider_label)

            if type_ == int:
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(min_val)
                slider.setMaximum(max_val)
                slider.setSingleStep(step)
                slider.setValue(param_value)

                self.camera_controls_sliders_scales[label] = 1
            else:
                # For floats, map to an integer range
                scale = 10 ** digits
                scaled_min = int(min_val * scale)
                scaled_max = int(max_val * scale)
                scaled_step = int(step * scale)
                scaled_initial = int(param_value * scale)

                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(scaled_min)
                slider.setMaximum(scaled_max)
                slider.setSingleStep(scaled_step)
                slider.setValue(scaled_initial)

                self.camera_controls_sliders_scales[label] = scale

            slider.setMinimumWidth(100)
            slider.valueChanged.connect(lambda value, lbl=label: self._slider_changed(lbl, value))
            slider.sliderReleased.connect(lambda lbl=label: self._slider_released(lbl))
            line_layout.addWidget(slider, 1)

            value_label = QLabel(f"{param_value}")
            value_label.setFixedWidth(40)
            value_label.setAlignment(Qt.AlignVCenter)
            self.camera_controls_sliders_labels[label] = value_label
            line_layout.addWidget(value_label)

            vis_checkbox = QCheckBox()
            vis_checkbox.setMaximumWidth(16)
            vis_checkbox.setChecked(True)
            sync_groupbox_layout.addWidget(vis_checkbox)

            self.camera_controls_sliders[label] = slider
            self._val_in_sync[label] = vis_checkbox

            centre_group_sliders_layout.addWidget(line)

        centre_group_layout.addWidget(centre_group_sliders)
        centre_group_layout.addWidget(sync_groupbox)

        # RIGHT GROUP - Additional elements
        right_group_layout = self.RIGHT_GROUP.layout()

        line = QWidget()
        line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        line.setMaximumHeight(80)
        line_layout = QHBoxLayout(line)

        # line_layout.addStretch(1)

        self.n_button = QPushButton('Nothing')
        # self.n_button.setCheckable(True)
        self.n_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.n_button.clicked.connect(self._toggle_n_display)
        line_layout.addWidget(self.n_button)

        self.magn_button = QPushButton('Magnification')
        # self.magn_button.setCheckable(True)
        self.magn_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.magn_button.clicked.connect(self._toggle_mag_display)
        line_layout.addWidget(self.magn_button)

        self.magn_slider = QSlider(Qt.Vertical)
        self.magn_slider.setMinimum(1)
        self.magn_slider.setMaximum(5)
        self.magn_slider.setSingleStep(1)
        self.magn_slider.setValue(2)
        line_layout.addWidget(self.magn_slider)

        right_group_layout.addWidget(line)

    #  ============= Qt method overrides =============
    def closeEvent(self, event):
        self.worker.stop()
        self.worker_thread.quit()
        self.worker_thread.wait()
        # super().closeEvent(event)
        event.ignore()
        self.toggle_visibility(False)

    def eventFilter(self, obj, event):

        if event.type() in (QEvent.MouseButtonPress, QEvent.MouseMove):

            # Get mouse position relative to displayed image
            mouse_x = int(event.pos().x() - (self.VIDEO_FEED.width() - self._display_buffer.shape[1]) / 2)
            mouse_y = int(event.pos().y() - (self.VIDEO_FEED.height() - self._display_buffer.shape[0]) / 2)

            if event.button() == Qt.LeftButton:
                self.left_mouse_btn = True
            if event.button() == Qt.RightButton:
                self.right_mouse_btn = True

            if self.left_mouse_btn:
                self.magn_target_cx = mouse_x / self._display_buffer.shape[0]
                self.magn_target_cy = mouse_y / self._display_buffer.shape[1]

            if self.right_mouse_btn:
                self.magn_window_x = mouse_y
                self.magn_window_y = mouse_x

        elif event.type() == QEvent.MouseButtonRelease:
            if event.button() == Qt.LeftButton:
                self.left_mouse_btn = False
            if event.button() == Qt.RightButton:
                self.right_mouse_btn = False

        return super().eventFilter(obj, event)

    #  ============= Update frame, and worker communication =============
    def _annotate(self):

        # Get new coordinates
        h, w = self._display_buffer.shape[:2]

        x_centre, y_centre = w // 2, h // 2
        x_north, y_north = w // 2, 0
        x_south, y_south = w // 2, h
        x_east, y_east = w, h // 2
        x_west, y_west = 0, h // 2

        # Draw crosshair
        cv2.line(self._display_buffer, (x_west, y_west), (x_east, y_east), self._main_window.col_white_rgb, 1)
        cv2.line(self._display_buffer, (x_north, y_north), (x_south, y_south), self._main_window.col_white_rgb, 1)

        if self._magnifier_enabled:

            target_cx_fb = self.magn_target_cx * self._frame_buffer.shape[0]
            target_cy_fb = self.magn_target_cy * self._frame_buffer.shape[1]

            # Position of the slice (in frame_buffer coordinates)
            slice_x1 = max(0, int(target_cx_fb - self.magn_window_w // 2))
            slice_y1 = max(0, int(target_cy_fb - self.magn_window_h // 2))
            slice_x2 = slice_x1 + self.magn_window_w
            slice_y2 = slice_y1 + self.magn_window_h

            if slice_x2 > self._source_shape[1]:
                slice_x1 = self._source_shape[1] - self.magn_window_w
                slice_x2 = self._source_shape[1]

            if slice_y2 > self._source_shape[0]:
                slice_y1 = self._source_shape[0] - self.magn_window_h
                slice_y2 = self._source_shape[0]

            # Slice directly from the frame_buffer and make the small, zoomed window image
            ratio_w = w / self._frame_buffer.shape[1]
            ratio_h = h / self._frame_buffer.shape[0]
            magn_img = cv2.resize(self._frame_buffer[slice_y1:slice_y2, slice_x1:slice_x2], (0, 0),
                                  fx=float(self.magn_slider.value() * ratio_w),
                                  fy=float(self.magn_slider.value() * ratio_h))

            # Add frame around the magnified area
            target_x1 = int((target_cx_fb - self.magn_window_w / 2) * ratio_w)
            target_x2 = int((target_cx_fb + self.magn_window_w / 2) * ratio_w)
            target_y1 = int((target_cy_fb - self.magn_window_h / 2) * ratio_h)
            target_y2 = int((target_cy_fb + self.magn_window_h / 2) * ratio_h)
            self._display_buffer = cv2.rectangle(self._display_buffer,
                                                 (target_x1, target_y1), (target_x2, target_y2),
                                                 self._main_window.col_yellow_rgb, 1)

            # Paste the zoom window into the display buffer
            magn_x1 = min(self._display_buffer.shape[0], max(0, self.magn_window_x))
            magn_y1 = min(self._display_buffer.shape[1], max(0, self.magn_window_y))
            magn_x2 = min(self._display_buffer.shape[0], magn_x1 + magn_img.shape[0])
            magn_y2 = min(self._display_buffer.shape[1], magn_y1 + magn_img.shape[1])
            self._display_buffer[magn_x1:magn_x2, magn_y1:magn_y2, :] = magn_img[:magn_x2 - magn_x1, :magn_y2 - magn_y1, :]

            # Add frame around the magnification
            self._display_buffer = cv2.rectangle(self._display_buffer,
                                                 (magn_y1, magn_x1), (magn_y2, magn_x2),
                                                 self._main_window.col_yellow_rgb, 1)

        # Position the 'Recording' indicator
        font, txtsiz, txtth = cv2.FONT_HERSHEY_DUPLEX, 1.0, 2
        textsize = cv2.getTextSize(self._main_window._recording_text, font, txtsiz, txtth)[0]
        self._display_buffer = cv2.putText(self._display_buffer, self._main_window._recording_text,
                                           (int(x_centre - textsize[0] / 2), int(1.5 * y_centre - textsize[1])),
                                           font, txtsiz, self._main_window.col_red_rgb, txtth, cv2.LINE_AA)

        # Position the 'Warning' indicator
        if self._warning:
            textsize = cv2.getTextSize(self._warning_text, font, txtsiz, txtth)[0]
            self._display_buffer = cv2.putText(self._display_buffer, self._warning_text,
                                               (int(x_north - textsize[0] / 2), int(y_centre / 2 - textsize[1])),
                                               font, txtsiz, self._main_window.col_orange_rgb, txtth, cv2.LINE_AA)


    def _update_images(self):
        # 1- Grab camera frame
        self._refresh_framebuffer()
        frame = self._frame_buffer.copy()

        # 2- if worker is free, send frame
        if not self._worker_busy:
            self._send_frame_to_worker(frame)
        else:
            self._latest_frame = frame

        # 3- resize to current window
        self._resize_to_display()

        disp_h, disp_w = self._display_buffer.shape[:2]
        scale = min(disp_w / frame.shape[1], disp_h / frame.shape[0])
        display_img = cv2.resize(frame, (0,0), fx=scale, fy=scale)

        # 4- annotate resized image - TODO: this will be moved to _annotate()
        h, w = display_img.shape[:2]
        self._display_buffer[:h, :w] = display_img
        if h < disp_h:
            self._display_buffer[h:, :] = 0
        if w < disp_w:
            self._display_buffer[:, w:] = 0

        # TODO - this is a test fake bounding box
        for (x, y, bw, bh) in self._bboxes:
            sx, sy = int(x*scale), int(y*scale)
            sw, sh = int(bw*scale), int(bh*scale)
            cv2.rectangle(self._display_buffer, (sx, sy), (sx+sw, sy+sh),
                          (0,255,255), 2)

        self._annotate()

        self._blit_image()

    def _send_frame_to_worker(self, frame):
        self.newFrameSignal.emit(frame)
        self._worker_busy = True

    def on_worker_finished(self):
        self._worker_busy = False
        if self._latest_frame is not None:
            frame = self._latest_frame
            self._latest_frame = None
            self._send_frame_to_worker(frame)

    def on_worker_result(self, bboxes):
        # called in the main thread when worker finishes processing and emits 'result_ready'
        self._bboxes = bboxes

    #  ============= Custom functions =============
    def _toggle_n_display(self):
        if self._n_enabled:
            self.n_button.setStyleSheet('')
            self._n_enabled = False
        else:
            self.n_button.setStyleSheet(f'background-color: #80{self._main_window.col_green.lstrip("#")};')
            self._n_enabled = True

    def _toggle_mag_display(self):
        if self._magnifier_enabled:
            self.magn_button.setStyleSheet('')
            # self.magn_slider.setDisabled(True)
            self._magnifier_enabled = False
        else:
            self.magn_button.setStyleSheet(f'background-color: #80{self._main_window.col_yellow.lstrip("#")};')
            # self.magn_slider.setDisabled(False)
            self._magnifier_enabled = True

    def update_param(self, label):
        if label == 'framerate' and self._main_window.mc.triggered and self._main_window.mc.acquiring:
            return

        slider = self.camera_controls_sliders[label]

        new_val_float = slider.value() / self.camera_controls_sliders_scales[label]

        setattr(self._main_window.mc.cameras[self.idx], label, new_val_float)

        # And update the slider to the actual new value (can be different from the one requested)
        read_back = getattr(self._main_window.mc.cameras[self.idx], label)

        actual_new_val = int(read_back * self.camera_controls_sliders_scales[label])
        slider.setValue(actual_new_val)

        if label == 'exposure':
            # Refresh exposure value for UI display
            self.exposure_value.setText(f"{self._main_window.mc.cameras[self.idx].exposure} µs")

            # We also need to update the framerate slider to current resulting fps after exposure change
            self.update_param('framerate')

        elif label == 'framerate':
            # Keep a local copy to warn user if actual framerate is too different from requested fps
            wanted_fps_val = slider.value() / self.camera_controls_sliders_scales[label]
            self._wanted_fps = wanted_fps_val

            if self._main_window.mc.triggered:
                self._main_window.mc.framerate = self._wanted_fps
            else:
                self._main_window.mc.cameras[self.idx].framerate = self._wanted_fps

            new_max = int(self._main_window.mc.cameras[self.idx].max_framerate * self.camera_controls_sliders_scales[label])
            self.camera_controls_sliders['framerate'].setMaximum(new_max)

    def _slider_changed(self, label, int_value):
        value_float = self.camera_controls_sliders[label].value() / self.camera_controls_sliders_scales[label]
        self.camera_controls_sliders_labels[label].setText(f'{int(value_float)}' if value_float.is_integer() else f'{value_float:.2f}')

    def _slider_released(self, label):

        self.update_param(label)
        should_apply = bool(self._val_in_sync[label].isChecked())

        if should_apply:
            # This should not be needed, the scale is supposed to be the same anyway but... just in case
            new_val_float = self.camera_controls_sliders[label].value() / self.camera_controls_sliders_scales[label]

            for window in self._main_window.secondary_windows:
                if window is not self and bool(window._val_in_sync[label].isChecked()):
                    # Apply the window's scale (which should be the same anyway)
                    w_new_val = int(new_val_float * window.camera_controls_sliders_scales[label])
                    window.camera_controls_sliders[label].setValue(w_new_val)
                    window.update_param(label)


class VideoWindowCalib(VideoWindowBase):

    newFrameSignal = Signal(np.ndarray)

    def __init__(self, main_window_ref, idx):
        super().__init__(main_window_ref, idx)

        # Default board params - TODO: Needs to be loaded from config file
        self.BOARD_ROWS = 6
        self.BOARD_COLS = 5
        self.SQUARE_LENGTH_MM = 1.5
        self.MARKER_BITS = 4

        # Setup Detection classes
        self._update_board()
        self.detection_tool = DetectionTool(self.charuco_board)
        self.calib_tool = MonocularCalibrationTool(
            detectiontool=self.detection_tool,
            imsize=self._source_shape[:2]   # pass frame size so it can track coverage
        )

        ##

        # Setup worker
        self.calib_thread = QThread(self)
        self.calib_worker = CalibWorker(self.calib_tool, auto_sample=True, auto_compute=True)
        self.calib_worker.moveToThread(self.calib_thread)

        # Setup signals
        self.newFrameSignal.connect(self.calib_worker.process_frame, type=Qt.QueuedConnection)
        self.calib_worker.result_ready.connect(self.on_worker_result)
        self.calib_worker.finished_processing.connect(self.on_worker_finished)
        self.calib_thread.start()

        # Store worker results and its current state
        self.annotated_frame = None
        self._worker_busy = False
        self._latest_frame = None

        # This updater function should only run at 60 fps
        self.timer_calib = QTimer(self)
        self.timer_calib.timeout.connect(self._update_images)
        self.timer_calib.start(16)  # 60 FPS

        # Finish building the UI by calling the other constructors
        self._init_common_ui()
        self._init_specific_ui()
        self.auto_size()

        self._update_board_preview()

    #  ============= UI constructors =============
    def _init_specific_ui(self):

        layout = QHBoxLayout(self.CENTRE_GROUP)
        layout.setContentsMargins(5, 5, 5, 5)

        detection_group = QGroupBox("Detection")
        detection_layout = QHBoxLayout(detection_group)

        board_group = QWidget()
        board_layout = QVBoxLayout(board_group)

        # A label to show the board preview
        self.board_preview_label = QLabel()
        self.board_preview_label.setAlignment(Qt.AlignCenter)
        board_layout.addWidget(self.board_preview_label)

        board_settings_button = QPushButton("Board Settings...")
        board_settings_button.clicked.connect(self.show_board_params_dialog)
        board_layout.addWidget(board_settings_button)

        detection_layout.addWidget(board_group)

        # Detection and sampling

        sampling_group = QWidget()
        sampling_layout = QVBoxLayout(sampling_group)

        self.auto_sample_check = QCheckBox("Sample automatically")
        self.auto_sample_check.setChecked(True)
        sampling_layout.addWidget(self.auto_sample_check)

        sampling_btns_group = QWidget()
        sampling_btns_layout = QHBoxLayout(sampling_btns_group)

        self.sample_button = QPushButton("Add sample")
        self.sample_button.clicked.connect(self.calib_tool.register_sample)
        self.sample_button.setStyleSheet(f"background-color: {self._main_window.col_darkgreen}; color: {self._main_window.col_white};")
        sampling_btns_layout.addWidget(self.sample_button)

        self.clear_samples_button = QPushButton("Clear samples")
        self.clear_samples_button.clicked.connect(self.calib_tool.clear_stacks)
        sampling_btns_layout.addWidget(self.clear_samples_button)

        sampling_layout.addWidget(sampling_btns_group)

        self.auto_compute_check = QCheckBox("Compute intrinsics automatically")
        self.auto_compute_check.setChecked(True)
        sampling_layout.addWidget(self.auto_compute_check)

        detection_layout.addWidget(sampling_group)
        layout.addWidget(detection_group)

        ##

        calib_io_group = QGroupBox("Load/Save")
        calib_io_layout = QVBoxLayout(calib_io_group)

        self.load_calib_button = QPushButton("Load intrinsics")
        self.load_calib_button.clicked.connect(self.on_load_calib)
        calib_io_layout.addWidget(self.load_calib_button)

        self.save_calib_button = QPushButton("Save intrinsics")
        self.save_calib_button.clicked.connect(self.on_save_calib)
        calib_io_layout.addWidget(self.save_calib_button)

        self.load_save_message = QLabel("")
        self.load_save_message.setMaximumWidth(200)
        self.load_save_message.setWordWrap(True)
        calib_io_layout.addWidget(self.load_save_message)

        layout.addWidget(calib_io_group)

    #  ============= Qt method overrides =============
    def closeEvent(self, event):
        self.calib_worker.stop()
        self.calib_thread.quit()
        self.calib_thread.wait()
        # super().closeEvent(event)
        event.ignore()
        self.toggle_visibility(False)

    #  ============= Update frame, and worker communication =============
    def _update_images(self):
        self._refresh_framebuffer()
        frame = self._frame_buffer.copy()

        if not self._worker_busy:
            self._send_frame_to_worker(frame)
        else:
            self._latest_frame = frame

        self._resize_to_display()

        # Now scale + display the last annotated frame we got from the worker
        disp_h, disp_w = self._display_buffer.shape[:2]
        if self.annotated_frame is not None and self.annotated_frame.size > 0:
            scale = min(disp_w / self.annotated_frame.shape[1],
                        disp_h / self.annotated_frame.shape[0])
            out = cv2.resize(self.annotated_frame, (0, 0), fx=scale, fy=scale)

            h, w = out.shape[:2]
            self._display_buffer[:h, :w] = out
            if h < disp_h:
                self._display_buffer[h:, :] = 0
            if w < disp_w:
                self._display_buffer[:, w:] = 0
        else:
            self._display_buffer.fill(0)

        self._blit_image()

    def _send_frame_to_worker(self, frame):
        self.newFrameSignal.emit(frame)
        self._worker_busy = True

    def on_worker_finished(self):
        self._worker_busy = False
        if self._latest_frame is not None:
            f = self._latest_frame
            self._latest_frame = None
            self._send_frame_to_worker(f)

    def on_worker_result(self, annotated):
        # called in the main thread when worker finishes processing and emits 'result_ready'
        self.annotated_frame = annotated

    #  ============= Custom functions =============
    def _update_board(self):
        self.charuco_board = calibration_utils.generate_charuco(
            board_rows=self.BOARD_ROWS,
            board_cols=self.BOARD_COLS,
            square_length_mm=self.SQUARE_LENGTH_MM,
            marker_bits=self.MARKER_BITS
        )

    def _reset_detector(self):
        self.detection_tool = DetectionTool(self.charuco_board)
        self.calib_tool.dt = self.detection_tool
        self.calib_tool.clear_stacks()

    def show_board_params_dialog(self):
        """
            Opens the small BoardParamsDialog to let the user set board parameters
        """
        dlg = BoardParamsDialog(
            rows=self.BOARD_ROWS,
            cols=self.BOARD_COLS,
            square_length=self.SQUARE_LENGTH_MM,
            parent=self
        )
        ret = dlg.exec_()
        if ret == QDialog.Accepted:
            # retrieve updated parameters
            rows, cols, sq, all = dlg.get_values()

            if all:
                # Loop over all secondary windows in the main app
                for w in self._main_window.secondary_windows:
                    if isinstance(w, VideoWindowCalib):
                        w.BOARD_ROWS = rows
                        w.BOARD_COLS = cols
                        w.SQUARE_LENGTH_MM = sq
                        w._update_board()
                        w._reset_detector()
                        w._update_board_preview()
            else:
                self.BOARD_ROWS = rows
                self.BOARD_COLS = cols
                self.SQUARE_LENGTH_MM = sq

                self._update_board()
                self._reset_detector()
                self._update_board_preview()

    def _update_board_preview(self):
        MAX_W, MAX_H = 100, 100

        r = self.BOARD_ROWS / self.BOARD_COLS
        h, w = 100, int(r * 100)
        board_arr = self.charuco_board.generateImage((h, w))
        q_img = QImage(board_arr, h, w, h, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        bounded_pixmap = pixmap.scaled(MAX_W, MAX_H,
                                       Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation)
        self.board_preview_label.setPixmap(bounded_pixmap)

    def on_load_calib(self):
        file_path = self.file_dialog(self._main_window.mc.full_path.parent)
        if file_path is not None:
            self.calib_tool.readfile(file_path)
            self.load_save_message.setText(f"<b>Intrinsics loaded:</b>\n{file_path}")

    def on_save_calib(self):
        file_path = self._main_window.mc.full_path / f"{self._camera.name}_intrinsics.toml"
        self.calib_tool.writefile(file_path)
        self.load_save_message.setText(f"<b>Intrinsics saved:</b>\n{file_path}")

    def file_dialog(self, startpath):
        dial = QFileDialog(self)
        dial.setWindowTitle("Choose folder")
        dial.setFileMode(QFileDialog.FileMode.Directory)
        dial.setOption(QFileDialog.Option.ShowDirsOnly, False)
        dial.setViewMode(QFileDialog.ViewMode.Detail)
        dial.setDirectory(QDir(startpath.resolve()))

        selected_path = None
        if dial.exec():
            selected = dial.selectedFiles()
            if selected:
                folder = Path(selected[0])
                file_path = folder / f"{self._camera.name}_intrinsics.toml"
                if file_path.exists():
                    selected_path = file_path
        return selected_path

##


class MainWindow(QMainWindow):
    INFO_PANEL_MINSIZE_H = 200
    VIDEO_PANEL_MINSIZE_H = 50  # haha
    WINDOW_MIN_W = 630

    # Colours
    col_white = "#ffffff"
    col_white_rgb = utils.hex_to_rgb(col_white)
    col_black = "#000000"
    col_black_rgb = utils.hex_to_rgb(col_black)
    col_lightgray = "#e3e3e3"
    col_lightgray_rgb = utils.hex_to_rgb(col_lightgray)
    col_midgray = "#c0c0c0"
    col_midgray_rgb = utils.hex_to_rgb(col_midgray)
    col_darkgray = "#515151"
    col_darkgray_rgb = utils.hex_to_rgb(col_darkgray)
    col_red = "#FF3C3C"
    col_red_rgb = utils.hex_to_rgb(col_red)
    col_darkred = "#bc2020"
    col_darkred_rgb = utils.hex_to_rgb(col_darkred)
    col_orange = "#FF9B32"
    col_orange_rgb = utils.hex_to_rgb(col_orange)
    col_darkorange = "#cb782d"
    col_darkorange_rgb = utils.hex_to_rgb(col_darkorange)
    col_yellow = "#FFEB1E"
    col_yellow_rgb = utils.hex_to_rgb(col_yellow)
    col_yelgreen = "#A5EB14"
    col_yelgreen_rgb = utils.hex_to_rgb(col_yelgreen)
    col_green = "#00E655"
    col_green_rgb = utils.hex_to_rgb(col_green)
    col_darkgreen = "#39bd50"
    col_darkgreen_rgb = utils.hex_to_rgb(col_green)
    col_blue = "#5ac3f5"
    col_blue_rgb = utils.hex_to_rgb(col_blue)
    col_purple = "#c887ff"
    col_purple_rgb = utils.hex_to_rgb(col_purple)

    def __init__(self, mc):
        super().__init__()

        self.setWindowTitle('Controls')
        self.gui_logger = gui_logger

        self.mc = mc

        # Identify monitors
        self.selected_monitor = None
        self._monitors = screeninfo.get_monitors()
        self.set_monitor()

        # Icons
        resources_path = [p for p in Path().cwd().glob('./**/*') if p.is_dir() and p.name == 'icons'][0]

        self.icon_capture = QIcon((resources_path / 'capture.png').as_posix())
        self.icon_capture_bw = QIcon((resources_path / 'capture_bw.png').as_posix())
        self.icon_snapshot = QIcon((resources_path / 'snapshot.png').as_posix())
        self.icon_snapshot_bw = QIcon((resources_path / 'snapshot_bw.png').as_posix())
        self.icon_rec_on = QIcon((resources_path / 'rec.png').as_posix())
        self.icon_rec_bw = QIcon((resources_path / 'rec_bw.png').as_posix())

        # States
        self.editing_disabled = True
        self._is_calibrating = False

        self._recording_text = ''

        # Refs for the secondary windows
        self.secondary_windows = []

        # Other things to init
        self._current_buffers = None
        self._mem_pressure = 0.0

        # Build the gui
        self.init_gui()

        self.update_monitors_buttons()

        # Start the secondary windows
        self._start_secondary_windows()

        # Setup MainWindow secondary update
        self.timer_update = QTimer(self)
        self.timer_update.timeout.connect(self._update_main)
        self.timer_update.start(100)

        self._mem_baseline = psutil.virtual_memory().percent

    def init_gui(self):
        self.MAIN_LAYOUT = QVBoxLayout()
        self.MAIN_LAYOUT.setContentsMargins(5, 5, 5, 5)
        self.MAIN_LAYOUT.setSpacing(5)
        # self.setStyleSheet('QGroupBox { border: 1px solid #807f7f7f; border-radius: 5px; margin-top: 0.5em;} '
                           # 'QGroupBox::title { subcontrol-origin: margin; left: 3px; padding: 0 3 3 3;}')

        central_widget = QWidget()
        central_widget.setLayout(self.MAIN_LAYOUT)
        self.setCentralWidget(central_widget)

        toolbar = QFrame()
        toolbar.setFixedHeight(38)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(3, 0, 3, 0)

        # Mode switch
        mode_label = QLabel('Mode: ')
        toolbar_layout.addWidget(mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Recording', 'Calibration'])
        self.mode_combo.currentIndexChanged.connect(self._toggle_calibrate)
        toolbar_layout.addWidget(self.mode_combo, 1)    # 1 unit

        toolbar_layout.addStretch(2)    # spacing of 2 units

        # Exit button
        self.button_exit = QPushButton("Exit (Esc)")
        self.button_exit.clicked.connect(self.quit)
        self.button_exit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.button_exit.setStyleSheet(f"background-color: {self.col_red}; color: {self.col_white};")
        toolbar_layout.addWidget(self.button_exit)

        self.MAIN_LAYOUT.addWidget(toolbar)  # End toolbar

        # Main content
        maincontent = QWidget()
        maincontent_layout = QHBoxLayout(maincontent)

        left_pane = QGroupBox("Acquisition")
        left_pane.setMinimumWidth(400)
        left_pane_layout = QVBoxLayout(left_pane)
        maincontent_layout.addWidget(left_pane, 4)

        right_pane = QGroupBox("Display")
        right_pane.setMinimumWidth(300)
        right_pane_layout = QVBoxLayout(right_pane)
        maincontent_layout.addWidget(right_pane, 3)

        self.MAIN_LAYOUT.addWidget(maincontent)

        # LEFT HALF
        f_name_and_path = QWidget()
        f_name_and_path_layout = QVBoxLayout(f_name_and_path)
        f_name_and_path_layout.setContentsMargins(0, 0, 0, 0)
        f_name_and_path_layout.setSpacing(0)

        line_1 = QWidget()
        line_1_layout = QHBoxLayout(line_1)

        acquisition_label = QLabel('Name: ')
        line_1_layout.addWidget(acquisition_label)

        self.acq_name_textbox = QLineEdit()
        self.acq_name_textbox.setDisabled(True)
        self.acq_name_textbox.setPlaceholderText("yymmdd-hhmm")
        line_1_layout.addWidget(self.acq_name_textbox, 1)

        self.acq_name_edit_btn = QPushButton("Edit")
        self.acq_name_edit_btn.setCheckable(True)
        self.acq_name_edit_btn.clicked.connect(self._toggle_text_editing)
        line_1_layout.addWidget(self.acq_name_edit_btn)

        f_name_and_path_layout.addWidget(line_1)

        line_2 = QWidget()
        line_2_layout = QHBoxLayout(line_2)

        self.save_dir_current = QLabel()
        self.save_dir_current.setStyleSheet(f"color: {self.col_darkgray};")
        self.save_dir_current.setWordWrap(True)
        folderpath_label_font = QFont()
        folderpath_label_font.setPointSize(10)
        self.save_dir_current.setFont(folderpath_label_font)
        line_2_layout.addWidget(self.save_dir_current, 1)

        self.save_dir_current.setText(f'{self.mc.full_path.resolve()}')

        f_name_and_path_layout.addWidget(line_2)

        line_3 = QWidget()
        line_3_layout = QHBoxLayout(line_3)

        self.button_open_folder = QPushButton("Open folder")
        self.button_open_folder.clicked.connect(self.open_session_folder)
        line_3_layout.addStretch(2)
        line_3_layout.addWidget(self.button_open_folder)

        f_name_and_path_layout.addWidget(line_3)

        left_pane_layout.addWidget(f_name_and_path, 1)

        # Buttons
        f_buttons = QWidget()
        f_buttons_layout = QVBoxLayout(f_buttons)
        f_buttons_layout.setContentsMargins(3, 0, 3, 0)

        self.button_acquisition = QPushButton("Acquisition off")
        self.button_acquisition.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.button_acquisition.setCheckable(True)
        self.button_acquisition.clicked.connect(self._toggle_acquisition)
        f_buttons_layout.addWidget(self.button_acquisition, 1)

        self.button_snapshot = QPushButton("Snapshot")
        self.button_snapshot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.button_snapshot.clicked.connect(self._take_snapshot)
        self.button_snapshot.setIcon(self.icon_snapshot_bw)
        self.button_snapshot.setDisabled(True)
        f_buttons_layout.addWidget(self.button_snapshot, 1)

        self.button_recpause = QPushButton("Not recording (Space to toggle)")
        self.button_recpause.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.button_recpause.setCheckable(True)
        self.button_recpause.clicked.connect(self._toggle_recording)
        self.button_recpause.setIcon(self.icon_rec_bw)
        self.button_recpause.setDisabled(True)
        f_buttons_layout.addWidget(self.button_recpause, 1)

        left_pane_layout.addWidget(f_buttons, 2)

        # RIGHT HALF
        live_previews = QGroupBox('Live previews')
        live_previews_layout = QVBoxLayout(live_previews)

        windows_list_frame = QScrollArea()
        windows_list_layout = QVBoxLayout()
        windows_list_widget = QWidget()
        windows_list_layout.setContentsMargins(0, 0, 0, 0)
        windows_list_layout.setSpacing(5)
        windows_list_widget.setLayout(windows_list_layout)
        windows_list_frame.setStyleSheet('border: none; background-color: #00000000;')
        windows_list_frame.setWidget(windows_list_widget)
        windows_list_frame.setWidgetResizable(True)
        live_previews_layout.addWidget(windows_list_frame)

        right_pane_layout.addWidget(live_previews)

        self.secondary_windows_visibility_buttons = []

        for i in range(self.mc.nb_cameras):
            vis_checkbox = QCheckBox(f"Camera {i}")
            vis_checkbox.setChecked(True)
            vis_checkbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            vis_checkbox.setMinimumHeight(25)
            windows_list_layout.addWidget(vis_checkbox)
            self.secondary_windows_visibility_buttons.append(vis_checkbox)

        monitors_frame = QGroupBox('Active monitor')
        monitors_frame_layout = QVBoxLayout(monitors_frame)

        right_pane_layout.addWidget(monitors_frame)

        self.monitors_buttons = QGraphicsView()
        self.monitors_buttons_scene = QGraphicsScene()
        monitors_frame_layout.setContentsMargins(3, 3, 3, 3)
        monitors_frame_layout.setSpacing(5)

        self.monitors_buttons.setStyleSheet("border: none; background-color: #00000000")
        self.monitors_buttons.setScene(self.monitors_buttons_scene)
        if 'Darwin' in platform.system():
            self.monitors_buttons.viewport().setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, False)

        monitors_frame_layout.addWidget(self.monitors_buttons)

        self.autotile_button = QPushButton("Auto-tile windows")
        self.autotile_button.clicked.connect(self.autotile_windows)
        monitors_frame_layout.addWidget(self.autotile_button)

        # LOG PANEL
        if self.gui_logger:
            log_button = QPushButton('Show log')
            log_button.setCheckable(True)
            log_button.setChecked(False)
            log_button.setMaximumWidth(80)
            self.MAIN_LAYOUT.addWidget(log_button)

            log_text_area = QTextEdit()
            log_text_area.setFont(QFont('consolas', 9))
            log_text_area.setDisabled(True)
            log_text_area.setVisible(False)
            self.MAIN_LAYOUT.addWidget(log_text_area)

            log_button.clicked.connect(log_text_area.setVisible)

            self.gui_logger.register_text_area(log_text_area)

        # Status bar
        statusbar = QStatusBar()
        statusbar.setStyleSheet(f"background-color: {'#157f7f7f'}; color: {'#ff7f7f7f'};")
        self.setStatusBar(statusbar)

        mem_pressure_label = QLabel('Memory pressure: ')
        mem_pressure_label.setStyleSheet(f"background-color: {'#00000000'}")
        statusbar.addWidget(mem_pressure_label)

        self._mem_pressure_bar = QProgressBar()
        self._mem_pressure_bar.setMaximum(100)
        statusbar.addWidget(self._mem_pressure_bar)

        self.frames_saved_label = QLabel()
        self.frames_saved_label.setText(f'Saved frames: {self.mc.saved} (0 bytes)')
        self.frames_saved_label.setStyleSheet(f"background-color: {'#00000000'}")
        statusbar.addPermanentWidget(self.frames_saved_label)

    def closeEvent(self, event):
        event.ignore()
        self.quit()

    def quit(self):
        # Close the secondary windows and stop their threads
        self._stop_secondary_windows()

        # Stop camera acquisition
        self.mc.off()

        self.mc.disconnect()

        # Close the main window
        QWidget.close(self)
        QApplication.instance().quit()
        sys.exit()

    def _toggle_calibrate(self):

        if self._is_calibrating and self.mode_combo.currentIndex() == 0:
            self._is_calibrating = False

            self._stop_secondary_windows()

            if self.mc.acquiring:
                self.button_snapshot.setDisabled(False)
                self.button_recpause.setDisabled(False)

            self._start_secondary_windows()

        elif not self._is_calibrating and self.mode_combo.currentIndex() == 1:
            self._is_calibrating = True

            self._stop_secondary_windows()

            self.button_recpause.setDisabled(True)

            self._start_secondary_windows()

        else:
            pass

    def _toggle_text_editing(self, override=None):

        if override is None:
            override = not self.editing_disabled

        if self.editing_disabled and override is True:
            self.acq_name_textbox.setDisabled(False)
            self.acq_name_edit_btn.setText('Set')
            self.editing_disabled = False

        elif not self.editing_disabled and override is False:
            self.acq_name_textbox.setDisabled(True)
            self.acq_name_edit_btn.setText('Edit')
            self.mc.session_name = self.acq_name_textbox.text()
            self.editing_disabled = True

            self.save_dir_current.setText(f'{self.mc.full_path.resolve()}')

    def open_session_folder(self):

        path = self.mc.full_path.resolve()

        try:
            if 'Linux' in platform.system():
                subprocess.Popen(['xdg-open', path])
            elif 'Windows' in platform.system():
                os.startfile(path)
            elif 'Darwin' in platform.system():
                subprocess.Popen(['open', path])
        except:
            pass

    def _toggle_acquisition(self, override=None):

        if override is None:
            override = not self.mc.acquiring

        # If we're currently acquiring, then we should stop
        if self.mc.acquiring and override is False:

            self._toggle_recording(False)
            self.mc.off()

            # Reset Acquisition folder name
            self.acq_name_textbox.setText('')
            self.save_dir_current.setText('')

            self.button_acquisition.setText("Acquisition off")
            self.button_acquisition.setIcon(self.icon_capture_bw)
            self.button_snapshot.setDisabled(True)
            self.button_recpause.setDisabled(True)

            # Re-enable the framerate sliders (only in case of hardware-triggered cameras)
            if not self._is_calibrating and self.mc.triggered:
                for w in self.secondary_windows:
                    w.camera_controls_sliders['framerate'].setDisabled(True)

        elif not self.mc.acquiring and override is True:
            self.mc.on()

            if not self._is_calibrating and self.mc.triggered:
                for w in self.secondary_windows:
                    w.camera_controls_sliders['framerate'].setDisabled(True)

            self.save_dir_current.setText(f'{self.mc.full_path.resolve()}')

            self.button_acquisition.setText("Acquiring")
            self.button_acquisition.setIcon(self.icon_capture)
            self.button_snapshot.setDisabled(False)

            if not self._is_calibrating:
                self.button_recpause.setDisabled(False)

    def _take_snapshot(self):
        """
            Takes an instantaneous snapshot from all cameras
        """

        now = datetime.now().strftime('%y%m%d-%H%M%S')

        if self.mc.acquiring:

            arrays = self.mc.get_current_framebuffer()

            for i, arr in enumerate(arrays):
                if len(arr.shape) == 3:
                    img = Image.fromarray(arr, mode='RGB')
                else:
                    img = Image.fromarray(arr, mode='L')
                img.save(self.mc.full_path.resolve() / f"snapshot_{now}_{self.mc.cameras[i].name}.bmp")

    def _toggle_recording(self, override=None):

        if override is None:
            override = not self.mc.recording

        # If we're currently recording, then we should stop
        if self.mc.acquiring:

            if self.mc.recording and override is False:
                self.mc.pause()
                self._recording_text = ''
                self.button_recpause.setText("Not recording (Space to toggle)")
                self.button_recpause.setIcon(self.icon_rec_bw)
            elif not self.mc.recording and override is True:
                self.mc.record()
                self._recording_text = '[Recording]'
                self.button_recpause.setText("Recording... (Space to toggle)")
                self.button_recpause.setIcon(self.icon_rec_on)

    def nothing(self):
        print('Nothing')
        pass

    def screen_update(self, val, event):

        # Get current monitor coordinates
        prev_monitor = self.selected_monitor

        # Get current mouse cursor position in relation to window origin
        prev_mouse_pos = QCursor.pos() - self.geometry().topLeft()

        # Set new monitor
        self.set_monitor(val)
        self.update_monitors_buttons()

        new_monitor = self.selected_monitor

        # Move windows by the difference
        for win in self.visible_windows(include_main=True):
            geo = win.geometry()
            w = geo.width()
            h = geo.height()
            x = geo.x()
            y = geo.y()

            d_x = x - prev_monitor.x
            d_y = y - prev_monitor.y

            # minmax to make sure the window stays inside the monitor
            new_x = min(max(new_monitor.x, new_monitor.x + d_x), new_monitor.width + new_monitor.x - w)
            new_y = min(max(new_monitor.y, new_monitor.y + d_y), new_monitor.height + new_monitor.y - h)

            win.move(new_x, new_y)

            # Also move cursor with main window
            if w is self:
                try:
                    QCursor.setPos(new_x + prev_mouse_pos.x(), new_y + prev_mouse_pos.y())
                except:
                    pass

    def autotile_windows(self):
        """
            Automatically arranges and resizes the windows
        """
        for w in self.visible_windows(include_main=True):
            w.auto_size()
            w.auto_move()

    def cascade_windows(self):

        monitor = self.selected_monitor

        for win in self.visible_windows(include_main=True):
            w = win.geometry().width()
            h = win.geometry().height()

            if win is self:
                d_x = 30 * len(self.visible_windows(include_main=False)) + 30
                d_y = 30 * len(self.visible_windows(include_main=False)) + 30
            else:
                d_x = 30 * win.idx + 30
                d_y = 30 * win.idx + 30

            # minmax to make sure the window stays inside the monitor
            new_x = min(max(monitor.x, monitor.x + d_x), self.selected_monitor.width + monitor.x - w)
            new_y = min(max(monitor.y, monitor.y + d_y), monitor.height + monitor.y - h)

            win.move(new_x, new_y)

    def auto_size(self):
        # Do nothing on the main window
        pass

    def auto_move(self):
        if self.selected_monitor.height < self.selected_monitor.width:
            # First corners, then left right, then top and bottom,  and finally centre
            positions = ['nw', 'sw', 'ne', 'se', 'n', 's', 'w', 'e', 'c']
        else:
            # First corners, then top and bottom, then left right, and finally centre
            positions = ['nw', 'sw', 'ne', 'se', 'w', 'e', 'n', 's', 'c']

        nb_positions = len(positions)

        idx = len(self.secondary_windows)
        if idx <= nb_positions:
            pos = positions[idx]
        else:  # Start over to first position
            pos = positions[idx % nb_positions]

        self.move_to(pos)

    # TODO - The functions auto_move and mote_to are almost the same in the Main window and the secondary windows
    # this needs to be refactored

    def move_to(self, pos):

        monitor = self.selected_monitor
        w = self.geometry().width()
        h = self.geometry().height()

        match pos:
            case 'nw':
                self.move(monitor.x, monitor.y)
            case 'n':
                self.move(monitor.x + monitor.width // 2 - w // 2, monitor.y)
            case 'ne':
                self.move(monitor.x + monitor.width - w - 1, monitor.y)
            case 'w':
                self.move(monitor.x, monitor.y + monitor.height // 2 - h // 2)
            case 'c':
                self.move(monitor.x + monitor.width // 2 - w // 2, monitor.y + monitor.height // 2 - h // 2)
            case 'e':
                self.move(monitor.x + monitor.width - w - 1, monitor.y + monitor.height // 2 - h // 2)
            case 'sw':
                self.move(monitor.x, monitor.y + monitor.height - h - VideoWindowBase.TASKBAR_H)
            case 's':
                self.move(monitor.x + monitor.width // 2 - w // 2,
                          monitor.y + monitor.height - h - VideoWindowBase.TASKBAR_H)
            case 'se':
                self.move(monitor.x + monitor.width - w - 1, monitor.y + monitor.height - h - VideoWindowBase.TASKBAR_H)

    def update_monitors_buttons(self):
        self.monitors_buttons_scene.clear()

        for i, m in enumerate(self._monitors):
            w, h, x, y = m.width // 40, m.height // 40, m.x // 40, m.y // 40
            col = '#7f7f7f' if m == self.selected_monitor else '#807f7f7f'

            rect = QGraphicsRectItem(x, y, w - 2, h - 2)
            rect.setBrush(QBrush(QColor(col)))      # Fill colour
            rect.setPen(QPen(Qt.PenStyle.NoPen))    # No outline
            rect.mousePressEvent = partial(self.screen_update, i)  # Bind the function

            text_item = QGraphicsTextItem(f"{i}")
            text_item.setDefaultTextColor(QColor.fromString('#99ffffff'))
            text_item.setFont(QFont("Monospace", 9))

            text_rect = text_item.boundingRect()
            text_item.setPos(x, y + h - text_rect.height())
            text_item.setZValue(1)
            rect.setZValue(0)

            self.monitors_buttons_scene.addItem(text_item)
            self.monitors_buttons_scene.addItem(rect)

    def set_monitor(self, idx=None):
        if len(self._monitors) > 1 and idx is None:
            self.selected_monitor = next(m for m in self._monitors if m.is_primary)
        elif len(self._monitors) > 1 and idx is not None:
            self.selected_monitor = self._monitors[idx]
        else:
            self.selected_monitor = self._monitors[0]

    def visible_windows(self, include_main=False):
        windows = [w for w in self.secondary_windows if w.isVisible()]
        if include_main:
            windows += [self]
        return windows

    def _start_secondary_windows(self):

        for i, cam in enumerate(self.mc.cameras):
            if self._is_calibrating:
                w = VideoWindowCalib(main_window_ref=self, idx=cam.idx)
            else:
                w = VideoWindowMain(main_window_ref=self, idx=cam.idx)

            self.secondary_windows.append(w)
            self.secondary_windows_visibility_buttons[i].setText(f" {w.name.title()} camera")
            self.secondary_windows_visibility_buttons[i].setStyleSheet(f"border-radius: 5px; padding: 0 10 0 10; color: {w.colour_2}; background-color: {w.colour};")
            self.secondary_windows_visibility_buttons[i].clicked.connect(w.toggle_visibility)
            self.secondary_windows_visibility_buttons[i].setChecked(True)

            w.show()

        self.cascade_windows()

    def _stop_secondary_windows(self):
        for w in self.secondary_windows:
            QWidget.close(w)
        self.secondary_windows = []

    def _update_main(self):

        # Get an estimation of the saved data size
        if self.mc._estim_file_size is None:
            self.frames_saved_label.setText(f'Saved frames: {self.mc.saved} (0 bytes)')

        elif self.mc._estim_file_size == -1:
            size = sum(sum(os.path.getsize(os.path.join(res[0], element)) for element in res[2]) for res in
                       os.walk(self.mc.full_path))
            self.frames_saved_label.setText(f'Saved frames: {self.mc.saved} ({utils.pretty_size(size)})')
        else:
            saved = self.mc.saved
            size = sum(self.mc._estim_file_size * saved)
            self.frames_saved_label.setText(f'Saved frames: {saved} ({utils.pretty_size(size)})')

        # Update memory pressure estimation
        self._mem_pressure += (psutil.virtual_memory().percent - self._mem_baseline) / self._mem_baseline * 100
        self._mem_pressure_bar.setValue(int(round(self._mem_pressure)))
        self._mem_baseline = psutil.virtual_memory().percent
