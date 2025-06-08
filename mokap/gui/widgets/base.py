from collections import deque
from datetime import datetime
from typing import Optional, Tuple

import cv2
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize, Slot, QPoint
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QStatusBar, QToolButton

from mokap.gui.widgets import FAST_UPDATE, SLOW_UPDATE
from mokap.gui.widgets.dialogs import SnapPopup


class SnapMixin:
    """
    Mixin class for window movement helpers

    Methods:
      snap(Optional[pos]): snap into the passed position on the current monitor, or auto decide based on index

    Any class that uses this mixin should have:
        self.idx:              the window's index
        self.selected_monitor:  object with .x, .y, .width, .height as in screeninfo)
        self.frameGeometry():   full QRect including title bar/borders)
        self.move(x, y):        Qt's function to reposition the window on screen
    """

    def snap(self, pos: Optional[str] = None):
        """
        Snap this window’s frame so that it touches one of the 9 zones:
        'nw', 'n', 'ne', 'w', 'c', 'e', 'sw', 's', 'se'
        """

        monitor = self.selected_monitor

        if pos is None:
            if monitor.height < monitor.width:
                # landscape: corners first, then L/R, then T/B, then center
                positions = ['nw', 'sw', 'ne', 'se', 'n', 's', 'w', 'e', 'c']
            else:
                # portrait: corners, then T/B, then L/R, then center
                positions = ['nw', 'sw', 'ne', 'se', 'w', 'e', 'n', 's', 'c']

            pos = positions[self.idx % len(positions)]

        frame = self.frameGeometry()
        w, h = frame.width(), frame.height()
        sp = SPACING
        tb = TASKBAR_H

        left_x = monitor.x + sp
        right_x = monitor.x + monitor.width - w - sp
        center_x = monitor.x + (monitor.width // 2) - (w // 2)

        top_y = monitor.y + sp
        center_y = monitor.y + (monitor.height // 2) - (h // 2)
        bottom_y = monitor.y + monitor.height - h - tb - sp

        match pos:
            case 'nw':
                self.move(left_x, top_y)
            case 'n':
                self.move(center_x, top_y)
            case 'ne':
                self.move(right_x, top_y)
            case 'w':
                self.move(left_x, center_y)
            case 'c':
                self.move(center_x, center_y)
            case 'e':
                self.move(right_x, center_y)
            case 'sw':
                self.move(left_x, bottom_y)
            case 's':
                self.move(center_x, bottom_y)
            case 'se':
                self.move(right_x, bottom_y)
            case _:
                return


class Base(QWidget, SnapMixin):
    """
    Base for any camera‐preview or 3D window
      - Worker/thread/timer setup
      - Window snap via the mixin class
    """

    def __init__(self, main_window_ref):
        super().__init__()
        self._force_destroy = False  # This is used to defined whether we only hide or destroy the window
        self.setAttribute(Qt.WA_DeleteOnClose, True)  # force PySide to destroy the windows on mode change

        # References for easier access
        self._mainwindow = main_window_ref
        self._coordinator = self._mainwindow.coordinator

        self.worker = None
        self.worker_thread = None

        # This updater function does not need to run super frequently
        self.timer_slow = QTimer(self)
        self.timer_slow.timeout.connect(self._update_slow)

        # This updater function should only run at 60 fps
        self.timer_fast = QTimer(self)
        self.timer_fast.timeout.connect(self._update_fast)

    @property
    def selected_monitor(self):
      return self._mainwindow.selected_monitor

    def _setup_worker(self, worker_object):
        self.worker_thread = QThread(self)
        self.worker = worker_object
        self.worker.moveToThread(self.worker_thread)

    def _update_slow(self):
        """ Subclasses override if they need a slow update """
        pass

    def _update_fast(self):
        """ Subclasses override if they need a fast (60fps) update """
        pass

    def _start_timers(self, fast=FAST_UPDATE, slow=SLOW_UPDATE):
        self.timer_fast.start(fast)
        self.timer_slow.start(slow)


class PreviewBase(Base):

    send_frame = Signal(np.ndarray, int)

    def __init__(self, camera, main_window_ref):
        super().__init__(main_window_ref)

        # Refs to anything preview window -related
        self._camera = camera
        self._cam_name = self._camera.name
        self._cam_idx = self._mainwindow.cameras_names.index(self._cam_name)
        self._source_shape = self._camera.shape
        self._source_framerate = self._camera.framerate
        self._cam_colour = self._mainwindow.cams_colours[self._cam_name]
        self._secondary_colour = self._mainwindow.secondary_colours[self._cam_name]
        self._fmt = self._camera.pixel_format

        # Qt things
        self.setWindowTitle(f'{self._camera.name.title()} camera')

        # Init where the frame data will be stored
        self._frame_buffer = np.zeros((*self._source_shape[:2], 3), dtype=np.uint8)
        self._display_buffer = np.zeros((*self._source_shape[:2], 3), dtype=np.uint8)

        # Init states
        self._worker_busy = False
        self._worker_blocking = False
        self._warning = False

        self._warning_text = '[WARNING]'

        self._latest_frame = None

        # Init clock and counter
        self._clock = datetime.now()
        self._capture_fps = deque(maxlen=10)
        self._last_capture_count = 0

    def _setup_worker(self, worker_object):
        super()._setup_worker(worker_object)

        # Setup preview-windows-specific signals (direct Main thread - Worker connections)
        # Emit frames to worker
        self.send_frame.connect(self.worker.handle_frame, type=Qt.QueuedConnection)
        # Receive results and state from worker
        self.worker.annotations.connect(self.on_worker_result)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.blocking.connect(self.blocking_toggle)

    def _init_common_ui(self):
        """
        This constructor creates all the UI elements that are common to all modes
        """
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.video_container = QWidget()
        self.video_container.setStyleSheet('background-color: black;')
        self.video_container_layout = QHBoxLayout(self.video_container)
        self.video_container_layout.setContentsMargins(0, 0, 0, 0)

        self.VIDEO_FEED = QLabel()
        self.VIDEO_FEED.setStyleSheet('background-color: black;')
        self.VIDEO_FEED.setMinimumSize(1, 1)  # Important! Otherwise it crashes when reducing the size of the window
        self.VIDEO_FEED.setAlignment(Qt.AlignCenter)
        self.video_container_layout.addWidget(self.VIDEO_FEED, 1)
        main_layout.addWidget(self.video_container, 1)

        self._blit_image()     # Call this once to initialise it

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
        camera_name_bar.setStyleSheet(f"color: {self.secondary_colour}; background-color: {self.colour}; font: bold;")

        bottom_panel_v_layout.addWidget(camera_name_bar)
        bottom_panel_v_layout.addWidget(bottom_panel_h)

        self.LEFT_GROUP = QGroupBox("Information")
        bottom_panel_h_layout.addWidget(self.LEFT_GROUP)

        self.RIGHT_GROUP = QGroupBox("Control")
        bottom_panel_h_layout.addWidget(self.RIGHT_GROUP, 1)     # Expand the right group only

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
        temp = self._camera.temperature
        self.temperature_value.setText(f"{temp}°C" if temp else '-')

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
            label.setStyleSheet(f"color: {self._mainwindow.col_darkgray}; font: bold;")
            label.setMinimumWidth(88)
            line_layout.addWidget(label)

            value.setStyleSheet("font: regular;")
            value.setAlignment(Qt.AlignLeft)
            value.setMinimumWidth(90)
            line_layout.addWidget(value)

            left_group_layout.addWidget(line)

        # Status bar
        statusbar = QStatusBar()

        self.snap_button = QToolButton()
        # self.snap_button.setText("Snap to:")
        self.snap_button.setIcon(self._mainwindow.icon_move_bw)
        self.snap_button.setIconSize(QSize(16, 16))
        self.snap_button.setToolTip("Move current window to a position")
        self.snap_button.setPopupMode(QToolButton.InstantPopup)

        self.snap_popup = SnapPopup(parent=self, move_callback=self.snap)
        self.snap_button.clicked.connect(self.show_snap_popup)
        statusbar.addPermanentWidget(self.snap_button)
        main_layout.addWidget(statusbar)

    def _init_specific_ui(self):
        """
        This does nothing in the base class, each VideoWindow implements its own specific UI elements
        """
        pass

    #  ============= Qt method overrides =============
    def closeEvent(self, event):
        if self._force_destroy:
            # stop the worker and allow Qt event to actually destroy the window
            self._stop_worker()
            super().closeEvent(event)
        else:
            # pause worker and only hide window
            event.ignore()
            self.hide()
            self._pause_worker()
            self._mainwindow.secondary_windows_visibility_buttons[self._cam_idx].setChecked(False)

    #  ============= Thread control =============
    def _pause_worker(self):
        self.worker.set_paused(True)

    def _resume_worker(self):
        self.worker.set_paused(False)

    def _stop_worker(self):
        self.worker_thread.quit()
        self.worker_thread.wait()

    #  ============= Some common signals =============
    def _send_frame_to_worker(self, frame):
        self.send_frame.emit(frame, int(self._mainwindow.mc.indices[self._cam_idx]))
        self._worker_busy = True

    @Slot()
    def on_worker_result(self, bboxes):
        # called in the main thread when worker finishes processing and emits its 'annotation'
        # Needs to be defined in each subclass because the result is not necessarily the same thing
        pass

    @Slot()
    def on_worker_finished(self):
        self._worker_busy = False
        if self._latest_frame is not None:
            f = self._latest_frame
            self._latest_frame = None
            self._send_frame_to_worker(f)

    @Slot(bool)
    def blocking_toggle(self, state):
        self._worker_blocking = state

    #  ============= Some useful properties =============
    @property
    def name(self) -> str:
        return self._cam_name

    @property
    def idx(self) -> int:
        return self._cam_idx

    @property
    def colour(self) -> str:
        return f'#{self._cam_colour.lstrip("#")}'

    color = colour

    @property
    def secondary_colour(self) -> str:
        return f'#{self._secondary_colour.lstrip("#")}'

    secondary_color = secondary_colour

    @property
    def source_shape(self) -> Tuple[int, int]:
        return self._source_shape

    @property
    def aspect_ratio(self) -> float:
        return self._source_shape[1] / self._source_shape[0]

    #  ============= Some common video window-related methods =============

    def show_snap_popup(self):
        button_pos = self.snap_button.mapToGlobal(QPoint(0, self.snap_button.height()))
        self.snap_popup.show_popup(button_pos)

    def auto_size(self):

        # If landscape screen
        if self._mainwindow.selected_monitor.height < self._mainwindow.selected_monitor.width:
            available_h = (self._mainwindow.selected_monitor.height - TASKBAR_H) // 2 - SPACING * 3
            video_max_h = available_h - self.BOTTOM_PANEL.height() - TOPBAR_H
            video_max_w = video_max_h * self.aspect_ratio

            h = int(video_max_h + self.BOTTOM_PANEL.height())
            w = int(video_max_w)

        # If portrait screen
        else:
            video_max_w = self._mainwindow.selected_monitor.width // 2 - SPACING * 3
            video_max_h = video_max_w / self.aspect_ratio

            h = int(video_max_h + self.BOTTOM_PANEL.height())
            w = int(video_max_w)

        self.resize(w, h)

    def toggle_visibility(self, override=None):

        if override is None:
            override = not self.isVisible()

        if self.isVisible() and override is False:
            self._mainwindow.secondary_windows_visibility_buttons[self.idx].setChecked(False)
            self.hide()
            self._pause_worker()

        elif not self.isVisible() and override is True:
            self._mainwindow.secondary_windows_visibility_buttons[self.idx].setChecked(True)
            self.show()
            self._resume_worker()

    #  ============= Display-related common methods =============
    def _refresh_framebuffer(self):
        """
        Grabs a new frame from the cameras and stores it in the frame buffer
        """
        if self._mainwindow.mc.acquiring:
            arr = self._mainwindow.mc.get_current_framebuffer(self.idx)
            if arr is not None:
                if self._fmt == "BayerBG8":
                    self._frame_buffer = cv2.cvtColor(arr, cv2.COLOR_BayerBG2BGR, dst=self._frame_buffer)
                elif self._fmt == "Mono8":
                    self._frame_buffer = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR, dst=self._frame_buffer)
                else:
                    self._frame_buffer = arr
        else:
            self._frame_buffer.fill(0)

    def _resize_to_display(self):
        """
        Fills and resizes the display buffer to the current window size
        """
        scale = min(self.VIDEO_FEED.width() / self._frame_buffer.shape[1], self.VIDEO_FEED.height() / self._frame_buffer.shape[0])
        self._display_buffer = cv2.resize(self._frame_buffer, (0, 0), dst=self._display_buffer, fx=scale, fy=scale)

    def _blit_image(self):
        """
        Applies the content of display buffers to the GUI
        """
        h, w = self._display_buffer.shape[:2]
        q_img = QImage(self._display_buffer.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.VIDEO_FEED.setPixmap(pixmap)

    #  ============= Common update method for texts and stuff =============
    def _update_slow(self):
        if self.isVisible():
            now = datetime.now()
            if self._mainwindow.mc.acquiring:
                cap_fps = sum(list(self._capture_fps)) / len(self._capture_fps) if self._capture_fps else 0
                if 0 < cap_fps < 1000:
                    if abs(cap_fps - self._source_framerate) > 10:
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

            temp = self._camera.temperature
            temp_state = self._camera.temperature_state

            # Update the temperature label colour
            if temp is not None:
                self.temperature_value.setText(f'{temp:.1f}°C')
            if temp_state == 'Ok':
                self.temperature_value.setStyleSheet(f"color: {self._mainwindow.col_green}; font: bold;")
            elif temp_state == 'Critical':
                self.temperature_value.setStyleSheet(f"color: {self._mainwindow.col_orange}; font: bold;")
            elif temp_state == 'Error':
                self.temperature_value.setStyleSheet(f"color: {self._mainwindow.col_red}; font: bold;")
            else:
                self.temperature_value.setStyleSheet(f"color: {self._mainwindow.col_yellow}; font: bold;")

            # Update display fps
            dt = (now - self._clock).total_seconds()
            ind = int(self._mainwindow.mc.indices[self.idx])
            if dt > 0:
                self._capture_fps.append((ind - self._last_capture_count) / dt)

            self._clock = now
            self._last_capture_count = ind
