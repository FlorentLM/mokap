import queue
import time
from collections import deque
from datetime import datetime
from threading import Thread
from typing import Optional, Tuple
import numpy as np
import cv2
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize, Slot, QPoint, QRectF
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QStatusBar, QToolButton, QGraphicsObject
import pyqtgraph as pg
from numpy.typing import ArrayLike

from mokap.gui.style.commons import *
from mokap.gui.widgets import FAST_UPDATE, SLOW_UPDATE
from mokap.gui.widgets.dialogs import SnapPopup


DISPLAY_FRAMERATE = 60.0
DISPLAY_INTERVAL = 1.0 / DISPLAY_FRAMERATE


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

class FastImageItem(QGraphicsObject):
    """ A minimal, fast QGraphicsObject for displaying a QImage
    This bypasses all of the complex machinery of pyqtgraph.ImageItem
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image = QImage()
        self.height = 0
        self.width = 0
        self.channels = 0
        self.bytes_per_line = 0

    def setImageData(self, data: ArrayLike):
        """ Set the image using raw data """

        # on first call, cache the image properties
        if self.height == 0:
            self.height, self.width, self.channels = data.shape
            self.bytes_per_line = self.channels * self.width

        # This is important. It tells the scene that the item's
        # content is about to change and needs a repaint
        self.prepareGeometryChange()

        # this here is why it's fast: we create a zero-copy QImage view of the numpy data
        # (note that QImage needs a C-contiguous array - the full display buffer is already contiguous,
        # but the magnifier data is a slice so it is not)
        contiguous_data = np.ascontiguousarray(data)    # this is a no-op for an already contiguous array so it's free
        self._image = QImage(contiguous_data.data, self.width, self.height, self.bytes_per_line, QImage.Format.Format_BGR888)

        # this needs to be called after changing the geometry or content
        self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget=None):
        if not self._image.isNull():
            painter.drawImage(0, 0, self._image)


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

    frame_received = Signal(np.ndarray, dict)   # frame and metadata
    send_frame = Signal(np.ndarray, int)

    def __init__(self, camera, main_window_ref):
        super().__init__(main_window_ref)

        self._camera = camera
        self._cam_name = self._camera.name
        self._cam_idx = self._mainwindow.get_camera_index(self._camera.unique_id)

        self.setWindowTitle(f'{self._camera.name.title()} camera')

        # All these properties come directly from the camera object
        self._source_framerate = self._camera.framerate
        self._cam_colour = self._mainwindow.cams_colours[self._cam_name]
        self._secondary_colour = self._mainwindow.secondary_colours[self._cam_name]
        self._fmt = self._camera.pixel_format

        img_x, img_y, img_w, img_h = self._camera.roi
        self._source_shape = (img_h, img_w)

        # This holds the *latest frame received* from the consumer thread
        # Access to this should be quick, also it will be None if no new frame has arrived
        self._latest_frame: Optional[np.ndarray] = None

        # This is the 'safe' buffer for display: we copy the _latest_frame into this
        # at the start of the update cycle, so we can annotate it without worrying about the consumer thread overwriting it
        self._latest_display_frame = np.zeros((self._source_shape[0], self._source_shape[1], 3), dtype=np.uint8)

        self._video_initialised = False

        self._current_frame_metadata = {}

        # states
        self._worker_busy = False
        self._worker_blocking = False
        self._warning = False

        self._last_polled_values = {}

        # clock and counter
        self._clock = datetime.now()
        self._capture_fps = deque(maxlen=30)
        self._frame_count_for_fps = 0

        # Connect the new frame signal to a slot that updates the UI
        self.frame_received.connect(self.on_frame_received)

        # Start a dedicated thread to consume frames from the manager's queue
        # a Thread (and not a QThread) is better for such a simple op
        self._consumer_thread_active = True
        self._frame_consumer = Thread(target=self._consume_frames_loop, daemon=True)
        self._frame_consumer.start()

    def _setup_worker(self, worker_object):
        """ Setup preview-windows-specific signals (direct Main thread - Worker connections) """
        super()._setup_worker(worker_object)

        # emit frames to worker
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

        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.video_container_layout.addWidget(self.graphics_widget, 1)

        # Add a ViewBox to hold the image and disable its native mouse interaction/menus
        self.view_box = self.graphics_widget.addViewBox(row=0, col=0)
        self.view_box.setAspectLocked(True)              # for correct aspect ratio
        self.view_box.setMouseEnabled(x=False, y=False)  # no pan/zoom
        self.view_box.setMenuEnabled(False)
        self.view_box.disableAutoRange()

        self.image_item = FastImageItem()
        self.view_box.addItem(self.image_item)

        main_layout.addWidget(self.video_container)

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

        self.triggered_value.setText("Yes" if self._camera.hardware_triggered else "No")
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
            label.setStyleSheet(f"color: {col_darkgray}; font: bold;")
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
        self.snap_button.setIcon(icon_move_bw)
        self.snap_button.setIconSize(QSize(16, 16))
        self.snap_button.setToolTip("Move current window to a position")
        self.snap_button.setPopupMode(QToolButton.InstantPopup)

        self.snap_popup = SnapPopup(parent=self, move_callback=self.snap)
        self.snap_button.clicked.connect(self.show_snap_popup)
        statusbar.addPermanentWidget(self.snap_button)
        main_layout.addWidget(statusbar)

    def _init_specific_ui(self):
        """  This does nothing in the base class, each VideoWindow implements its own specific UI elements """
        pass

    def _consume_frames_loop(self):
        """ This runs in a background thread and grabs frames from the camera manager, and handles color conversion """

        display_queue = self._mainwindow.manager._display_queues[self._cam_idx]

        # This buffer lives in the background thread, we initialise it once and then update its content
        bgr_frame = np.empty((self._source_shape[0], self._source_shape[1], 3), dtype=np.uint8)

        last_emit_time = 0  # this is a bit of a hack...
        # ...but otherwise this function emits waaaay too many Signals and cripples everything

        while self._consumer_thread_active:
            try:
                raw_frame, metadata = display_queue.get(timeout=1)
                current_time = time.time()
                should_emit = (current_time - last_emit_time) > DISPLAY_INTERVAL

                if not should_emit:
                    continue  # skip processing and emitting if it's too soon

                last_emit_time = current_time

                if raw_frame is None:
                    self.frame_received.emit(None, metadata)
                    continue

                pixel_format = metadata.get('pixel_format_effective') or self._fmt

                try:
                    match pixel_format:
                        # ... (match case is the same)
                        case 'BayerBG8':
                            cv2.cvtColor(raw_frame, cv2.COLOR_BAYER_BG2BGR, dst=bgr_frame)
                        case 'BayerRG8':
                            cv2.cvtColor(raw_frame, cv2.COLOR_BAYER_RG2BGR, dst=bgr_frame)
                        case 'Mono8':
                            cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR, dst=bgr_frame)
                        case 'RGB8':
                            cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR, dst=bgr_frame)
                        case _:
                            np.copyto(bgr_frame, raw_frame)
                except cv2.error as e:
                    print(f"[{self.name}] OpenCV Error in _consume_frames_loop: {e}")
                    bgr_frame.fill(255)

                self.frame_received.emit(bgr_frame, metadata)

            except queue.Empty:
                continue

    @Slot(np.ndarray, dict)
    def on_frame_received(self, frame, metadata):
        """ this runs in the main GUI thread. We just update the references and counter """
        self._current_frame_metadata = metadata
        self._frame_count_for_fps += 1
        self._latest_frame = frame

    #  ============= Common update method for texts and stuff =============
    def _update_slow(self):
        if self.isVisible():
            now = datetime.now()

            dt = (now - self._clock).total_seconds()
            if dt > 0:
                # Calculate FPS based on frames received by the GUI thread
                gui_fps = self._frame_count_for_fps / dt
                self._capture_fps.append(gui_fps)

            # Reset counters for the next interval
            self._clock = now
            self._frame_count_for_fps = 0

            params_to_poll = ['exposure', 'framerate', 'gain', 'blacks', 'gamma']

            for param in params_to_poll:
                current_value = getattr(self._camera, param)
                last_value = self._last_polled_values.get(param)

                if current_value != last_value:
                    self.update_slider_ui(param, current_value)
                    self._last_polled_values[param] = current_value

            if self._mainwindow.manager.acquiring:
                # Calculate the smoothed captured FPS from the deque
                cap_fps = sum(list(self._capture_fps)) / len(self._capture_fps) if self._capture_fps else 0

                # Get the current target framerate from the camera object itself
                target_framerate = self._camera.framerate

                # Check if the calculated FPS is within a reasonable range to display
                if 0 < cap_fps < 1000:
                    # Check for significant deviation from the target framerate to set a warning
                    if abs(cap_fps - target_framerate) > 10:
                        self._warning = True
                    else:
                        self._warning = False

                    # Update the UI label with the measured framerate
                    self.capturefps_value.setText(f"{cap_fps:.2f} fps")
                else:
                    # If FPS is 0 or an absurd value, don't display it and don't warn
                    self.capturefps_value.setText("-")
                    self._warning = False

                # Also update the brightness value
                brightness = np.round(self._latest_display_frame.mean() / 255 * 100, decimals=2)
                self.brightness_value.setText(f"{brightness:.2f}%")
            else:
                # Reset UI elements when not acquiring
                self.capturefps_value.setText("Off")
                self.brightness_value.setText("-")
                self._warning = False

            # temp = self._camera.temperature
            # temp_state = self._camera.temperature_state
            #
            # # Update the temperature label colour
            # if temp is not None:
            #     self.temperature_value.setText(f'{temp:.1f}°C')
            # if temp_state == 'Ok':
            #     self.temperature_value.setStyleSheet(f"color: {col_green}; font: bold;")
            # elif temp_state == 'Critical':
            #     self.temperature_value.setStyleSheet(f"color: {col_orange}; font: bold;")
            # elif temp_state == 'Error':
            #     self.temperature_value.setStyleSheet(f"color: {col_red}; font: bold;")
            # else:
            #     self.temperature_value.setStyleSheet(f"color: {col_yellow}; font: bold;")

            # # Update display fps
            # dt = (now - self._clock).total_seconds()
            # ind = int(self._mainwindow.manager.indices[self.idx])
            # if dt > 0:
            #     self._capture_fps.append((ind - self._last_capture_count) / dt)
            #
            # self._clock = now
            # self._last_capture_count = ind

    #  ============= Fast update methods for image refresh =============

    def _annotate_frame(self):
        """ Subclasses must implement this method to draw their specific annotations """
        pass

    def _update_fast(self):
        """ This is now the main display updater. It runs at a controlled rate via its QTimer """

        # Check if there's a new frame from the consumer thread
        if self._latest_frame is None:
            return # No new frame, do nothing.

        # Copy the new frame into the safe display buffer
        # This is the *only* full-frame copy, and it happens at the display rate
        np.copyto(self._latest_display_frame, self._latest_frame)
        self._latest_frame = None   # mark as consumed

        # Let the worker process the frame if it's free. We send it the safe display buffer.
        if not self._worker_busy:
            self._send_frame_to_worker(self._latest_display_frame)

        # subclasses do their own thing
        self._annotate_frame()

        # Update the image on the screen
        self.image_item.setImageData(self._latest_display_frame)

        # One-time setup to fit the image to the viewbox
        if not self._video_initialised:
            self.view_box.autoRange()
            self._video_initialised = True

    def _send_frame_to_worker(self, frame):
        # Retrieve the synchronized frame number from our stored metadata
        frame_number = self._current_frame_metadata.get('frame_number', -1)  # Use -1 as a default/error value

        # Emit the frame and its correct, safe frame number
        self.send_frame.emit(frame, frame_number)
        self._worker_busy = True

    #  ============= Qt method overrides =============
    def closeEvent(self, event):
        if self._force_destroy:
            # Make sure to stop the consumer thread
            self._consumer_thread_active = False
            self._frame_consumer.join(timeout=1.0)

            # stop the worker and allow Qt event to actually destroy the window
            self._stop_worker()
            super().closeEvent(event)
        else:
            # pause worker and only hide window
            event.ignore()
            self.hide()
            self._pause_worker()
            self._mainwindow.secondary_windows_visibility_buttons[self._cam_idx].setChecked(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if self.view_box:
            # After the normal window resize we just tell our ViewBox to fit the item
            self.view_box.autoRange()

    #  ============= Thread control =============
    def _pause_worker(self):
        self.worker.set_paused(True)

    def _resume_worker(self):
        self.worker.set_paused(False)

    def _stop_worker(self):
        self.worker_thread.quit()
        self.worker_thread.wait()

    #  ============= Some signals =============

    @Slot(str, object)
    def update_slider_ui(self, label, value):
        # Implemented in the concrete class
        pass

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
        return float(self._source_shape[1] / self._source_shape[0])

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