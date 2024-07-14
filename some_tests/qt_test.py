import os
import subprocess
import sys
import platform
import psutil
import screeninfo

import time
import cv2

from functools import partial
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from mokap import utils

from PIL import Image, ImageQt
from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtGui import QIcon, QImage, QPixmap, QCursor, QBrush, QPen, QColor, QPixmapCache, QFont, QPalette
from PyQt6.QtWidgets import (QApplication, QMainWindow, QSplitter, QStatusBar, QSlider, QGraphicsView, QGraphicsScene,
                             QGraphicsRectItem, QComboBox, QLineEdit, QProgressBar, QCheckBox, QScrollArea, QWidget,
                             QLabel, QFrame, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QPushButton, QSizePolicy,
                             QGraphicsTextItem)
from PyQt6.QtOpenGL import QOpenGLVersionProfile, QOpenGLTexture, QOpenGLVersionFunctionsFactory
from PyQt6.QtOpenGLWidgets import QOpenGLWidget


class DoubleSlider(QSlider):
    """
        A slider widget that accepts floats (from https://gist.github.com/dennis-tra/994a65d6165a328d4eabaadbaedac2cc)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decimals = 5
        self._max_int = 10 ** self.decimals

        super().setMinimum(0)
        super().setMaximum(self._max_int)

        self._min_value = 0.0
        self._max_value = 1.0

    @property
    def _value_range(self):
        return self._max_value - self._min_value

    def value(self):
        return float(super().value()) / self._max_int * self._value_range + self._min_value

    def setValue(self, value):
        super().setValue(int((value - self._min_value) / self._value_range * self._max_int))

    def setMinimum(self, value):
        if value > self._max_value:
            raise ValueError("Minimum limit cannot be higher than maximum")
        self._min_value = value
        self.setValue(self.value())

    def setMaximum(self, value):
        if value < self._min_value:
            raise ValueError("Minimum limit cannot be higher than maximum")
        self._max_value = value
        self.setValue(self.value())

    def minimum(self):
        return self._min_value

    def maximum(self):
        return self._max_value


class VideoGLWidget(QOpenGLWidget):
    TEX_SLOTS = None

    def __init__(self, height, width, idx, parent=None, *args, **kwargs):
        super(VideoGLWidget, self).__init__(parent, *args, **kwargs)
        self.idx = idx
        self.gl = None
        self.texture = None
        self.ratio = width/height
        self.arraybuffer = np.zeros((height, width, 3), dtype=np.uint8)

    def initializeGL(self):
        version_profile = QOpenGLVersionProfile()
        version_profile.setVersion(2, 0)
        self.gl = QOpenGLVersionFunctionsFactory.get(version_profile, self.context())
        self.gl.initializeOpenGLFunctions()
        # self.gl.glClearColor(0.5, 0.8, 0.7, 1.0)

        self.gl.glEnable(self.gl.GL_TEXTURE_2D)

        if VideoGLWidget.TEX_SLOTS is None:
            VideoGLWidget.TEX_SLOTS = self.gl.glGenTextures(5)

        self.texture = VideoGLWidget.TEX_SLOTS[self.idx]

        self._gen_texture()

    def resizeGL(self, width, height):
        side = min(width, height)
        x = int((width - side) / 2)
        y = int((height - side) / 2)
        self.gl.glViewport(x, y, side, side)

    def paintGL(self):
        self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT)
        if self.texture:

            self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.texture)

            self.gl.glBegin(self.gl.GL_QUADS)
            self.gl.glTexCoord2f(0, 0)
            self.gl.glVertex2f(-1, -1)
            self.gl.glTexCoord2f(1, 0)
            self.gl.glVertex2f(1, -1)
            self.gl.glTexCoord2f(1, 1)
            self.gl.glVertex2f(1, 1)
            self.gl.glTexCoord2f(0, 1)
            self.gl.glVertex2f(-1, 1)
            self.gl.glEnd()

            self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, 0)

    def _gen_texture(self):

        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.texture)

        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MIN_FILTER, self.gl.GL_LINEAR)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MAG_FILTER, self.gl.GL_LINEAR)

        self.gl.glTexImage2D(self.gl.GL_TEXTURE_2D,
                     0,
                     self.gl.GL_RGB,
                     self.arraybuffer.shape[1],
                     self.arraybuffer.shape[0],
                     0,
                     self.gl.GL_RGB,
                     self.gl.GL_UNSIGNED_BYTE,
                     self.arraybuffer.tobytes())

        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, 0)

    def updatedata(self, imagedata):
        flipped = np.flip(imagedata, axis=0)
        self.arraybuffer[:, :, 0] = flipped
        self.arraybuffer[:, :, 1] = flipped
        self.arraybuffer[:, :, 2] = flipped
        self._gen_texture()
        self.update()


class VideoWindowBase(QWidget):
    INFO_PANEL_H = 300
    WINDOW_MIN_W = 650
    TASKBAR_H = 80

    def __init__(self, main_window_ref, idx):
        super().__init__()

        self._main_window = main_window_ref
        self.idx = idx

        self._camera = self._main_window.mgr.cameras[self.idx]
        self._source_shape = self._camera.shape

        self._bg_colour = f'#{self._main_window.mgr.colours[self._camera.name].lstrip("#")}'
        self._fg_colour = self._main_window.col_white if utils.hex_to_hls(self._bg_colour)[
                                                             1] < 60 else self._main_window.col_black

        # Where the frame data will be stored
        self._frame_buffer = np.zeros((*self._source_shape[:2], 3), dtype=np.uint8)

        # Init clock and counter
        self._clock = datetime.now()
        self._capture_fps = deque(maxlen=10)
        self._last_capture_count = 0

        # Init states
        self._warning = False

        # Some other stuff
        self._wanted_fps = self._camera.framerate

        self.setWindowTitle(f'{self._camera.name.title()} camera')

        self.positions = np.array([['nw', 'n', 'ne'],
                                   ['w', 'c', 'e'],
                                   ['sw', 's', 'se']])

        self.auto_size()

        # Setup MainWindow update
        self.timer_update = QTimer(self)
        self.timer_update.timeout.connect(self._update_secondary)
        self.timer_update.start(100)

        # Setup VideoWindow video update
        self.timer_video = QTimer(self)
        self.timer_video.timeout.connect(self.update_image)
        self.timer_video.start(30)

    def init_common_ui(self):

        v_layout = QVBoxLayout(self)

        self.VIDEO_FEED = QLabel()
        self.VIDEO_FEED.setMinimumSize(1, 1)  # Important! Otherwise it crashes when reducing the size of the window
        self.VIDEO_FEED.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v_layout.addWidget(self.VIDEO_FEED)

        self.update_image() # Call this once to initialise it

        # self.VIDEO_FEED = VideoGLWidget(self._camera.shape[0], self._camera.shape[1], self.idx, parent=self)

        # Camera name bar
        camera_name_bar = QLabel()
        camera_name_bar.setMinimumHeight(20)
        camera_name_bar.setMaximumHeight(20)
        camera_name_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        camera_name_bar.setStyleSheet(f"color: {self.colour_2}; background-color: {self.colour}; font: bold;")
        v_layout.addWidget(camera_name_bar)

        controlsRestrictorWidget = QWidget()
        layoutHControls = QHBoxLayout()
        controlsRestrictorWidget.setLayout(layoutHControls)
        controlsRestrictorWidget.setMinimumHeight(VideoWindowBase.INFO_PANEL_H)
        controlsRestrictorWidget.setMaximumHeight(VideoWindowBase.INFO_PANEL_H)

        h_layout = layoutHControls
        v_layout.addWidget(controlsRestrictorWidget)

        self.LEFT_FRAME = QGroupBox("Information")
        self.LEFT_FRAME.setStyleSheet("font: bold;")
        h_layout.addWidget(self.LEFT_FRAME)

        self.CENTRE_FRAME = QGroupBox("Centre")
        self.CENTRE_FRAME.setStyleSheet("font: bold;")
        h_layout.addWidget(self.CENTRE_FRAME)

        self.RIGHT_FRAME = QGroupBox("View")
        self.RIGHT_FRAME.setStyleSheet("font: bold;")
        h_layout.addWidget(self.RIGHT_FRAME)

        # Left frame: Information block
        left_layout = QHBoxLayout(self.LEFT_FRAME)

        # This layout is for the immutable labels
        f_labels = QVBoxLayout()
        left_layout.addLayout(f_labels)

        # This layout is for the variable values
        f_values = QVBoxLayout()
        left_layout.addLayout(f_values)

        self.resolution_value = QLabel()
        self.capturefps_value = QLabel()
        self.exposure_value = QLabel()
        self.brightness_value = QLabel()
        self.temperature_value = QLabel()

        self.resolution_value.setText(f"{self.source_shape[1]}×{self.source_shape[0]} px")
        self.capturefps_value.setText(f"Off")
        self.exposure_value.setText(f"{self._camera.exposure} µs")
        self.brightness_value.setText(f"-")
        self.temperature_value.setText(f"{self._camera.temperature}°C" if self._camera.temperature is not None else '-')

        labels_and_values = [
            ('Resolution', self.resolution_value),
            ('Capture', self.capturefps_value),
            ('Exposure', self.exposure_value),
            ('Brightness', self.brightness_value),
            ('Temperature', self.temperature_value),
        ]

        for label, value in labels_and_values:
            label = QLabel(f"{label} :")
            label.setAlignment(Qt.AlignmentFlag.AlignRight)
            label.setStyleSheet(f"color: {self._main_window.col_darkgray}; font: bold;")
            f_labels.addWidget(label)

            value.setStyleSheet("font: regular;")
            f_values.addWidget(value)

        # Right frame: View controls block
        right_layout = QVBoxLayout(self.RIGHT_FRAME)

        f_windowsnap = QFrame()
        f_windowsnap_layout = QHBoxLayout(f_windowsnap)
        right_layout.addWidget(f_windowsnap)

        l_windowsnap = QLabel("Window snap : ")
        l_windowsnap.setAlignment(Qt.AlignmentFlag.AlignRight)
        l_windowsnap.setStyleSheet(f"color: {self._main_window.col_darkgray}; font: bold;")
        f_windowsnap_layout.addWidget(l_windowsnap)

        f_buttons_windowsnap = QWidget()
        f_buttons_windowsnap_layout = QGridLayout(f_buttons_windowsnap)
        f_windowsnap_layout.addWidget(f_buttons_windowsnap)

        for r in range(3):
            for c in range(3):
                button = QPushButton()
                button.setFixedSize(26, 26)
                button.clicked.connect(partial(self.move_to, self.positions[r, c]))
                f_buttons_windowsnap_layout.addWidget(button, r, c)

    def init_specific_ui(self):
        pass

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

    def auto_size(self):

        # If landscape screen
        if self._main_window.selected_monitor.height < self._main_window.selected_monitor.width:
            h = self._main_window.selected_monitor.height // 2 - VideoWindowBase.TASKBAR_H
            w = int(self.aspect_ratio * (h - VideoWindowBase.INFO_PANEL_H))

        # If portrait screen
        else:
            w = self._main_window.selected_monitor.width // 2
            h = int(w / self.aspect_ratio) + VideoWindowBase.INFO_PANEL_H

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

    def closeEvent(self, event):
        event.ignore()
        self.toggle_visibility(False)

    def toggle_visibility(self, override=None):

        if override is None:
            override = not self.isVisible()

        if self.isVisible() and override is False:
            self._main_window.secondary_windows_visibility_buttons[self.idx].setChecked(False)
            self.hide()

        elif not self.isVisible() and override is True:
            self._main_window.secondary_windows_visibility_buttons[self.idx].setChecked(True)
            self.show()

    def _refresh_framebuffer(self):
        if self._main_window.mgr.acquiring:
            arr = self._main_window.mgr.get_current_framebuffer(self.idx)
            if arr is not None:
                if len(self.source_shape) == 2:
                    np.copyto(self._frame_buffer[:, :, 0], arr)
                    np.copyto(self._frame_buffer[:, :, 1], arr)
                    np.copyto(self._frame_buffer[:, :, 2], arr)
                else:
                    np.copyto(self._frame_buffer, arr)
        else:
            self._frame_buffer.fill(0)

        # Debug
        self._frame_buffer = cv2.putText(
            img=self._frame_buffer,
            text=f"{self.idx}",
            org=(200, 200),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=3.0,
            color=(125, 246, 55),
            thickness=3
        )

    def update_image(self):
        if self.isVisible():
            self._refresh_framebuffer()

            h, w = self._frame_buffer.shape[:2]
            q_img = QImage(self._frame_buffer.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.VIDEO_FEED.setPixmap(pixmap.scaled(self.VIDEO_FEED.width(), self.VIDEO_FEED.height(), Qt.AspectRatioMode.KeepAspectRatio))

            # self.VIDEO_FEED_GL.updatedata(self._main_window.mgr.get_current_framebuffer(self.idx))

    def _update_secondary(self):

        if self.isVisible():

            now = datetime.now()

            if self._main_window.mgr.acquiring:

                cap_fps = sum(list(self._capture_fps)) / 10 if self._capture_fps else 0

                if 0 < cap_fps < 1000:
                    if abs(cap_fps - self._wanted_fps) > 10:
                        # self.txtvar_warning.set('[ WARNING: Framerate ]')
                        self._warning = True
                    else:
                        self._warning = False
                    self.capturefps_value.setText(f"{cap_fps:.2f} fps")
                else:
                    self.capturefps_value.setText("-")

                # brightness = np.round(self._frame_buffer.mean() / 255 * 100, decimals=2)
                # self.brightness_value.setText(f"{brightness:.2f}%")
            else:
                self.capturefps_value.setText("Off")
                self.brightness_value.setText("-")

            # # Update the temperature label colour
            # if self._camera.temperature is not None:
            #     self.temperature_value.setText(f'{self._camera.temperature:.1f}°C')
            # if self._camera.temperature_state == 'Ok':
            #     self.temperature_value.setStyleSheet(f"color: {self._main_window.col_green}; font: bold;")
            # elif self._camera.temperature_state == 'Critical':
            #     self.temperature_value.setStyleSheet(f"color: {self._main_window.col_orange}; font: bold;")
            # elif self._camera.temperature_state == 'Error':
            #     self.temperature_value.setStyleSheet(f"color: {self._main_window.col_red}; font: bold;")
            # else:
            #     self.temperature_value.setStyleSheet(f"color: {self._main_window.col_yellow}; font: bold;")

            # Update display fps
            dt = (now - self._clock).total_seconds()
            ind = int(self._main_window.mgr.indices[self.idx])
            if dt > 0:
                self._capture_fps.append((ind - self._last_capture_count) / dt)

            self._clock = now
            self._last_capture_count = ind

class VideoWindowMain(VideoWindowBase):
    def __init__(self, main_window_ref, idx):
        super().__init__(main_window_ref, idx)

        self._show_focus = False
        self._magnification = False

        # Magnification parameters
        self.magn_zoom = 1.0
        self.magn_window_w = 100
        self.magn_window_h = 100
        self.magn_window_x = 10 + self.magn_window_w // 2  # Initialise in the corner
        self.magn_window_y = 10 + self.magn_window_h // 2

        self.magn_target_cx = self.source_shape[1] // 2
        self.magn_target_cy = self.source_shape[0] // 2

        # Focus view parameters
        # Kernel to use for focus detection
        self._kernel = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]], dtype=np.uint8)

        self.col_default = None

        self.init_common_ui()
        self.init_specific_ui()

    def init_specific_ui(self):
        # Centre frame: Information block
        centre_layout = QVBoxLayout(self.CENTRE_FRAME)

        f_labels = QVBoxLayout()
        centre_layout.addLayout(f_labels)
        self.camera_controls_sliders = {}

        slider_params = [
            ('framerate', 1.0, self._main_window.mgr.cameras[self.idx].max_framerate, 1, 1),
            ('exposure', 21, 10000, 5, 1),  # in microseconds - 100000 microseconds ~ 10 fps
            ('blacks', 0.0, 32.0, 0.5, 3),
            ('gain', 0.0, 36.0, 0.5, 3),
            ('gamma', 0.0, 3.99, 0.05, 3)
        ]

        for label, from_, to, resolution, digits in slider_params:
            param_value = getattr(self._main_window.mgr.cameras[self.idx], label)

            label_widget = QLabel(f'{label.title()} :', self)
            centre_layout.addWidget(label_widget)

            slider = DoubleSlider(Qt.Orientation.Horizontal, self)

            slider.setMaximum(to)
            slider.setMinimum(from_)
            # slider.setSingleStep(resolution)
            slider.setValue(param_value)
            slider.valueChanged.connect(lambda value, l=label: self._update_param_all(l, value))
            centre_layout.addWidget(slider)

            scale_val_label = QLabel('', self)
            self.camera_controls_sliders[label] = slider
            centre_layout.addWidget(scale_val_label)

        # # Right Frame: Specific buttons
        # right_layout = QHBoxLayout(self.RIGHT_FRAME)
        #
        # f_buttons_controls = QFrame()
        # f_buttons_controls_layout = QVBoxLayout(f_buttons_controls)
        #
        # f = QFrame()
        # f_buttons_controls_layout.addWidget(f)
        #
        # self.show_focus_button = QPushButton("Focus zone", f)
        # self.show_focus_button.clicked.connect(self._toggle_focus_display)
        # f_buttons_controls_layout.addWidget(self.show_focus_button)
        #
        # f = QFrame()
        # f_buttons_controls_layout.addWidget(f)
        #
        # self.show_mag_button = QPushButton("Magnifier", f)
        # self.show_mag_button.clicked.connect(self._toggle_mag_display)
        # f_buttons_controls_layout.addWidget(self.show_mag_button)
        #
        # self.slider_magn = QSlider(Qt.Orientation.Horizontal, f)
        # self.slider_magn.setMinimum(1)
        # self.slider_magn.setMaximum(5)
        # self.slider_magn.setSingleStep(1)
        # self.slider_magn.setValue(1)
        # f_buttons_controls_layout.addWidget(self.slider_magn)

    def _toggle_focus_display(self):
        pass

    def _toggle_mag_display(self):
        pass


# class VideoWindowCalib(VideoWindowBase):
#     def __init__(self, rootwindow, idx):
#         super().__init__(rootwindow, idx)
#         self.init_specific_ui()
#
#     def init_specific_ui(self):
#         vbox = QVBoxLayout(self)
#
#         # Centre Frame: Calibration controls
#         self.CENTRE_FRAME = QFrame()
#         vbox.addWidget(self.CENTRE_FRAME)
#
#         f_snapshots = QFrame()
#         vbox.addWidget(f_snapshots)
#
#         self.snap_button = QPushButton("Take Snapshot", f_snapshots)
#         self.snap_button.clicked.connect(self._toggle_snapshot)
#
#         rf = QFrame()
#         vbox.addWidget(rf)
#
#         self.autosnap_var = QCheckBox("Auto snapshot", rf)
#         self.autosnap_var.setChecked(False)
#
#         self.reset_coverage_button = QPushButton("Clear snapshots", rf)
#         self.reset_coverage_button.clicked.connect(self._reset_coverage)
#
#         f_calibrate = QFrame()
#         vbox.addWidget(f_calibrate)
#
#         separator = QFrame()
#         separator.setFrameShape(QFrame.Shape.HLine)
#         vbox.addWidget(separator)
#
#         self.calibrate_button = QPushButton("Calibrate", f_calibrate)
#         self.calibrate_button.clicked.connect(self._perform_calibration)
#
#         f_saveload = QFrame()
#         vbox.addWidget(f_saveload)
#
#         f_saveload_buttons = QFrame()
#         vbox.addWidget(f_saveload_buttons)
#
#         self.load_button = QPushButton("Load", f_saveload_buttons)
#         self.load_button.clicked.connect(self.load_calibration)
#
#         self.save_button = QPushButton("Save", f_saveload_buttons)
#         self.save_button.clicked.connect(self.save_calibration)
#
#         self.saved_label = QLabel('', f_saveload)


class MainWindow(QMainWindow):
    INFO_PANEL_MINSIZE_H = 200
    VIDEO_PANEL_MINSIZE_H = 50  # haha
    WINDOW_MIN_W = 630

    def __init__(self, mgr):
        super().__init__()

        self.setWindowTitle('Controls')
        # self.setGeometry(100, 100, MainWindow.WINDOW_MIN_W, MainWindow.INFO_PANEL_MINSIZE_H)  # todo

        self.mgr = mgr

        # Identify monitors
        self.selected_monitor = None
        self._monitors = screeninfo.get_monitors()
        self.set_monitor()

        self.col_white = "#ffffff"
        self.col_black = "#000000"
        self.col_lightgray = "#e3e3e3"
        self.col_midgray = "#c0c0c0"
        self.col_darkgray = "#515151"
        self.col_red = "#FF3C3C"
        self.col_orange = "#FF9B32"
        self.col_yellow = "#FFEB1E"
        self.col_yelgreen = "#A5EB14"
        self.col_green = "#00E655"
        self.col_blue = "#5ac3f5"
        self.col_purple = "#c887ff"

        # Icons
        resources_path = [p for p in Path().cwd().glob('../**/*') if p.is_dir() and p.name == 'icons'][0]

        self.icon_capture = QIcon((resources_path / 'capture.png').as_posix())
        self.icon_capture_bw = QIcon((resources_path / 'capture_bw.png').as_posix())
        self.icon_snapshot = QIcon((resources_path / 'snapshot.png').as_posix())
        self.icon_snapshot_bw = QIcon((resources_path / 'snapshot_bw.png').as_posix())
        self.icon_rec_on = QIcon((resources_path / 'rec.png').as_posix())
        self.icon_rec_bw = QIcon((resources_path / 'rec_bw.png').as_posix())

        # States
        self.editing_disabled = True
        self._is_calibrating = False

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

        self.cascade_windows()

        # Setup MainWindow secondary update
        self.timer_update = QTimer(self)
        self.timer_update.timeout.connect(self._update_main)
        self.timer_update.start(200)

        self._mem_baseline = psutil.virtual_memory().percent

    def init_gui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        # self.setStyleSheet('QGroupBox { border: 1px solid #807f7f7f; border-radius: 5px; margin-top: 0.5em;} '
                           # 'QGroupBox::title { subcontrol-origin: margin; left: 3px; padding: 0 3 3 3;}')

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
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
        toolbar_layout.addWidget(self.mode_combo, 1)    # 1 unit

        toolbar_layout.addStretch(2)    # spacing of 2 units

        # Exit button
        self.button_exit = QPushButton("Exit (Esc)")
        self.button_exit.clicked.connect(self.quit)
        self.button_exit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.button_exit.setStyleSheet(f"background-color: {self.col_red}; color: {self.col_white};")
        toolbar_layout.addWidget(self.button_exit)

        main_layout.addWidget(toolbar)  # End toolbar

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

        main_layout.addWidget(maincontent)

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

        self.save_dir_current.setText(f'{self.mgr.full_path.resolve()}')

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

        for i in range(self.mgr.nb_cameras):
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

        # # LOG PANEL
        # if self.gui_logger:
        #     log_label_frame = QWidget()
        #     log_label_frame_layout = QVBoxLayout(log_label_frame)
        #     log_label = QLabel('↓ pull for log ↓')
        #     log_label.setFont(QFont('Arial', 6))
        #     log_label_frame_layout.addWidget(log_label)
        #     content_panels.addWidget(log_label_frame)
        #
        #     log_frame = QWidget()
        #     log_frame_layout = QVBoxLayout(log_frame)
        #     log_text_area = QTextEdit()
        #     log_text_area.setFont(QFont('consolas', 9))
        #     log_frame_layout.addWidget(log_text_area)
        #     content_panels.addWidget(log_frame)
        #
        #     self.gui_logger.register_text_area(log_text_area)

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
        self.frames_saved_label.setText(f'Saved frames: {self.mgr.saved} (0 bytes)')
        self.frames_saved_label.setStyleSheet(f"background-color: {'#00000000'}")
        statusbar.addPermanentWidget(self.frames_saved_label)

    def closeEvent(self, event):
        event.ignore()
        self.quit()

    def quit(self):
        # Close the secondary windows and stop their threads
        self._stop_secondary_windows()

        # Stop camera acquisition
        if self.mgr.acquiring:
            self.mgr.off()

        self.mgr.disconnect()

        # Close the main window
        QWidget.close(self)
        QApplication.instance().quit()
        sys.exit()

    def _toggle_calibrate(self):
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
            self.mgr.session_name = self.acq_name_textbox.text()
            self.editing_disabled = True

            self.save_dir_current.setText(f'{self.mgr.full_path.resolve()}')

    def open_session_folder(self):
        path = Path(self.acq_name_textbox.text()).resolve()

        if self.acq_name_textbox.text() == "":
            path = self.mgr.full_path

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
            override = not self.mgr.acquiring

        # If we're currently acquiring, then we should stop
        if self.mgr.acquiring and override is False:

            self._toggle_recording(False)
            self.mgr.off()

            # Reset Acquisition folder name
            self.acq_name_textbox.setText('')
            self.save_dir_current.setText('')

            self.button_acquisition.setText("Acquisition off")
            self.button_acquisition.setIcon(self.icon_capture_bw)
            self.button_snapshot.setDisabled(True)
            self.button_recpause.setDisabled(True)

            # Re-enable the framerate sliders (only in case of hardware-triggered cameras)
            if self.mgr.triggered:
                for w in self.secondary_windows:
                    w.camera_controls_sliders['framerate'].config(state='normal', troughcolor=w.col_default)

        elif not self.mgr.acquiring and override is True:
            self.mgr.on()

            if self.mgr.triggered:
                for w in self.secondary_windows:
                    w.camera_controls_sliders['framerate'].config(state='disabled', troughcolor=self.col_lightgray)

            self.save_dir_current.setText(f'{self.mgr.full_path.resolve()}')

            self.button_acquisition.setText("Acquiring")
            self.button_acquisition.setIcon(self.icon_capture)
            self.button_snapshot.setDisabled(False)
            if not self._is_calibrating:
                self.button_recpause.setDisabled(False)

    def _take_snapshot(self):
        """
            Takes an instantaneous snapshot from all cameras
        """

        dims = np.array([(cam.height, cam.width) for cam in self.mgr.cameras], dtype=np.uint32)
        ext = self.mgr.saving_ext
        now = datetime.now().strftime('%y%m%d-%H%M')

        if self.mgr.acquiring and self._current_buffers is not None:
            arrays = [np.frombuffer(c, dtype=np.uint8) for c in self._current_buffers]

            for a, arr in enumerate(arrays):
                img = Image.fromarray(arr.reshape(dims[a]))
                img.save(self.mgr.full_path.resolve() / f"snapshot_{now}_{self.mgr.cameras[a].name}.{ext}")

    def _toggle_recording(self, override=None):

        if override is None:
            override = not self.mgr.recording

        # If we're currently recording, then we should stop
        if self.mgr.acquiring:

            if self.mgr.recording and override is False:
                # self.mgr.pause()
                self.txt_recording = ''
                self.button_recpause.setIcon(self.icon_rec_bw)
            elif not self.mgr.recording and override is True:
                # self.mgr.record()
                self.txt_recording = '[ Recording... ]'
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
        for i, cam in enumerate(self.mgr.cameras):

            if self._is_calibrating:
                # w = VideoWindowCalib(rootwindow=self, idx=cam.idx)
                # self.secondary_windows.append(w)
                # self.secondary_windows_visibility_buttons[i].setText(f" {w.name.title()} camera")
                # self.secondary_windows_visibility_buttons[i].setStyleSheet(f"border-radius: 5px; padding: 0 10 0 10; color: {w.colour_2}; background-color: {w.colour};")
                # self.secondary_windows_visibility_buttons[i].clicked.connect(w.toggle_visibility)
                # self.secondary_windows_visibility_buttons[i].setChecked(True)

                # For now, do nothing
                continue

            else:
                w = VideoWindowMain(main_window_ref=self, idx=cam.idx)
                self.secondary_windows.append(w)
                self.secondary_windows_visibility_buttons[i].setText(f" {w.name.title()} camera")
                self.secondary_windows_visibility_buttons[i].setStyleSheet(f"border-radius: 5px; padding: 0 10 0 10; color: {w.colour_2}; background-color: {w.colour};")
                self.secondary_windows_visibility_buttons[i].clicked.connect(w.toggle_visibility)
                self.secondary_windows_visibility_buttons[i].setChecked(True)

                w.show()

    def _stop_secondary_windows(self):
        for w in self.secondary_windows:
            QWidget.close(w)
        self.secondary_windows = []

    def _update_main(self):

        # Get an estimation of the saved data size
        if self.mgr._estim_file_size is None:
            self.frames_saved_label.setText(f'Saved frames: {self.mgr.saved} (0 bytes)')

        elif self.mgr._estim_file_size == -1:
            size = sum(sum(os.path.getsize(os.path.join(res[0], element)) for element in res[2]) for res in
                       os.walk(self.mgr.full_path))
            self.frames_saved_label.setText(f'Saved frames: {self.mgr.saved} ({utils.pretty_size(size)})')
        else:
            saved = self.mgr.saved
            size = sum(self.mgr._estim_file_size * saved)
            self.frames_saved_label.setText(f'Saved frames: {saved} ({utils.pretty_size(size)})')

        # Update memory pressure estimation
        self._mem_pressure += (psutil.virtual_memory().percent - self._mem_baseline) / self._mem_baseline * 100
        self._mem_pressure_bar.setValue(int(round(self._mem_pressure)))
        self._mem_baseline = psutil.virtual_memory().percent

##

from mokap.core import Manager

QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
app = QApplication(sys.argv)

mgr = Manager(config='../config.yaml', triggered=False, silent=False)

# Set exposure for all cameras (in µs)
mgr.exposure = 4800

# Enable binning
mgr.binning = 1
mgr.binning_mode = 'avg'

# Set framerate in images per second for all cameras at once
mgr.framerate = 100

mgr.gamma = 1.0
mgr.blacks = 1.0
mgr.gain = 0.0

main_window = MainWindow(mgr)
main_window.show()

sys.exit(app.exec())