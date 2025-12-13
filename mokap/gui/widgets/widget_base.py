"""
Base classes for video windows.

SharedBase: Common functionality for all secondary windows
VideoWindowBase: Base for camera video windows with frame consumption and display
"""
import logging
import time
from collections import deque
from threading import Thread
from typing import Optional, Tuple
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QGraphicsObject, QSizePolicy
import pyqtgraph as pg
from mokap.gui.style import *
from mokap.gui.widgets import UI_UPDATE_FPS, DISPLAY_FPS, CALIB_PROCESSING_FPS

logger = logging.getLogger(__name__)


class FastImageItem(QGraphicsObject):
    """A minimal and fast QGraphicsObject for displaying a QImage."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image = QImage()
        self._height = 0
        self._width = 0
        self._channels = 0
        self._bytes_per_line = 0

    def setImageData(self, data: np.ndarray):
        """Set the image using raw data."""

        if self._height == 0:
            self._height, self._width = data.shape[:2]
            self._channels = data.shape[2] if data.ndim == 3 else 1
            self._bytes_per_line = self._channels * self._width

        self.prepareGeometryChange()
        contiguous_arr = np.ascontiguousarray(data)
        self._image = QImage(contiguous_arr.data, self._width, self._height, self._bytes_per_line,
                             QImage.Format.Format_BGR888)
        self.update()

    def boundingRect(self) -> QRectF:
        # The bounding rectangle is defined in the item's *local* coordinates before any transforms are applied
        return QRectF(0, 0, self._width, self._height)

    def paint(self, painter, option, widget=None):
        if not self._image.isNull():
            painter.drawImage(0, 0, self._image)


class SharedBase(QWidget):
    """
    Shared base for any secondary window (video or 3D view).

    Provides:
    - Reference to main window
    - Slow update timer for UI elements
    - Force destroy flag for window lifecycle
    """

    def __init__(self, main_window_ref):
        super().__init__()
        self._force_destroy = False  # used to defined whether we only hide or destroy the window
        self.setAttribute(Qt.WA_DeleteOnClose, True)  # force PySide to destroy the windows on mode change

        # References for easier access
        self._mainwindow = main_window_ref

        # This updater function does not need to run super frequently
        self.timer_slow = QTimer(self)
        self.timer_slow.timeout.connect(self._update_slow)

    @property
    def selected_monitor(self):
        return self._mainwindow.selected_monitor

    def _update_slow(self):
        """Subclasses override if they need a slow update."""
        pass

    def _start_timers(self, ui_frequency=UI_UPDATE_FPS, **kwargs):
        """Subclasses can override if they need more timers."""
        self.timer_slow.start(int(1 / UI_UPDATE_FPS * 1000))

    def _stop_timers(self):
        """Subclasses can override if they need more timers."""
        self.timer_slow.stop()


class VideoWindowBase(SharedBase):
    """
    Base class for camera video windows.

    Provides:
    - Frame consumption from hardware camera (background thread)
    - Display updates (timer-driven)
    - Processing hook for subclasses (timer-driven)
    - Common UI elements (info panel, control panel)
    - Pause/resume functionality

    Subclasses must implement:
    - _init_specific_ui(): Create mode-specific UI elements
    - _send_frame_for_processing(): Send frames to processing pipeline (optional)
    - _pause_worker() / _resume_worker(): Pause/resume processing (if has workers)
    """

    def __init__(self, hw_cam, main_window_ref):
        super().__init__(main_window_ref)

        self._hw_camera = hw_cam
        self._hw_cam_name = self._hw_camera.name
        self._hw_cam_idx = self._mainwindow.get_camera_index(self._hw_camera.unique_id)

        self.setWindowTitle(f'{self._hw_camera.name.title()} camera')

        # All these properties come directly from the camera object
        self._main_colour = self._mainwindow.main_colours[self._hw_cam_name]
        self._secondary_colour = self._mainwindow.secondary_colours[self._hw_cam_name]
        self._fmt = self._hw_camera.pixel_format

        _, _, img_w, img_h = self._hw_camera.roi
        self._source_height = img_h
        self._source_width = img_w

        # This holds the *latest frame received* from the consumer thread
        # Access to this should be quick, also it will be None if no new frame has arrived
        self._latest_frame: Optional[np.ndarray] = None

        # This is the 'safe' buffer for display: we copy the _latest_frame into this at the start of the update cycle
        # -> we can annotate it without worrying about the consumer thread overwriting it
        self._latest_display_frame = np.zeros((self._source_height, self._source_width, 3), dtype=np.uint8)

        self._video_initialised = False

        self._current_frame_data = {}

        # States
        self._warning = False

        self._last_polled_values = {}
        self._last_polled_ranges = {}

        # Clock and counter
        self._fps_clock = time.monotonic()
        self._last_frame_number_for_fps = 0
        self._capture_fps_deque = deque(maxlen=5)

        # This timer is for video display only (updating the QImage)
        self.timer_display = QTimer(self)
        self.timer_display.timeout.connect(self._update_display)

        # This timer is for processing only (calibration stuff, etc)
        self.timer_processing = QTimer(self)
        self.timer_processing.timeout.connect(self._send_frame_for_processing)

        # Start a dedicated thread to consume frames from the manager's queue
        # (a regular Thread (i.e. not a QThread) is better for this)
        self._consumer_thread_active = True
        self._frame_consumer = Thread(target=self._consume_frames_loop)
        self._frame_consumer.start()

    def _start_timers(self,
                      video_display_frequency=DISPLAY_FPS,
                      processing_frequency=CALIB_PROCESSING_FPS,
                      ui_frequency=UI_UPDATE_FPS):

        super()._start_timers(ui_frequency=ui_frequency)

        self.timer_display.start(int(1.0 / video_display_frequency * 1000))
        self.timer_processing.start(int(1.0 / processing_frequency * 1000))

    def _stop_timers(self):
        self.timer_display.stop()
        self.timer_processing.stop()
        self.timer_slow.stop()

    # ──────────────────────────────── UI setup ────────────────────────────────

    def _init_common_ui(self):
        """
        Creates all the UI elements that are common to all video window modes.
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
        self.view_box.setAspectLocked(True)  # for correct aspect ratio
        self.view_box.setMouseEnabled(x=False, y=False)  # no pan/zoom
        self.view_box.setMenuEnabled(False)
        self.view_box.disableAutoRange()
        self.view_box.invertY(True)  # set origin to match image coordinates

        self.image_item = FastImageItem()
        self.view_box.addItem(self.image_item)

        main_layout.addWidget(self.video_container)

        self.BOTTOM_PANEL = QWidget()
        self.BOTTOM_PANEL.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        bottom_panel_v_layout = QVBoxLayout(self.BOTTOM_PANEL)
        bottom_panel_v_layout.setContentsMargins(0, 0, 0, 0)
        bottom_panel_v_layout.setSpacing(0)
        bottom_panel_h = QWidget()
        bottom_panel_h_layout = QHBoxLayout(bottom_panel_h)

        # Camera name bar
        camera_name_bar = QLabel(f'{self._hw_camera.name.title()} camera')
        camera_name_bar.setFixedHeight(25)
        camera_name_bar.setAlignment(Qt.AlignCenter)
        camera_name_bar.setStyleSheet(f"color: {self.secondary_colour}; background-color: {self.colour}; font: bold;")

        bottom_panel_v_layout.addWidget(camera_name_bar)
        bottom_panel_v_layout.addWidget(bottom_panel_h)

        self.LEFT_GROUP = QGroupBox("Information")
        bottom_panel_h_layout.addWidget(self.LEFT_GROUP)

        self.RIGHT_GROUP = QGroupBox("Control")
        bottom_panel_h_layout.addWidget(self.RIGHT_GROUP, 1)  # Expand the right group only

        main_layout.addWidget(self.BOTTOM_PANEL)

        # LEFT GROUP
        left_group_layout = QVBoxLayout(self.LEFT_GROUP)

        self.triggered_value = QLabel()
        self.resolution_value = QLabel()
        self.capturefps_value = QLabel()
        self.exposure_value = QLabel()
        self.brightness_value = QLabel()
        self.temperature_value = QLabel()

        self.triggered_value.setText("Yes" if self._hw_camera.hardware_triggered else "No")
        self.resolution_value.setText(f"{self.source_shape_hw[1]}×{self.source_shape_hw[0]} px")
        self.capturefps_value.setText(f"Off")
        self.exposure_value.setText(f"{self._hw_camera.exposure} µs")
        self.brightness_value.setText(f"-")
        temp = self._hw_camera.temperature
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

    def _init_specific_ui(self):
        """
        Subclasses implement this to create mode-specific UI elements.
        """
        pass

    # ──────────────────────────────── Frame consumption ────────────────────────────────

    def _consume_frames_loop(self):
        """
        This runs in a background thread. It polls the manager's latest frame
        buffer at a controlled rate, converts the frame to a displayable format,
        and updates the reference used by the GUI's display timer.
        """

        manager = self._mainwindow.manager

        lock = manager._latest_frame_locks[self._hw_cam_idx]
        last_frame_id = -1

        # Pre-allocate the destination buffer to avoid creating new arrays in the loop
        # bgr_frame = np.empty((self._source_height, self._source_width, 3), dtype=np.uint8)

        while self._consumer_thread_active:

            bgr_frame = np.empty((self._source_height, self._source_width, 3), dtype=np.uint8)

            # This sleep controls the display framerate and prevents this thread from consuming 100% CPU
            time.sleep(1.0 / DISPLAY_FPS)

            raw_frame = None
            frame_data = None
            with lock:
                # check if a new frame has arrived since last check
                latest_data = manager._latest_frames[self._hw_cam_idx]

                if latest_data:
                    current_frame_id = latest_data[1].get('frame_number', -1)

                    if current_frame_id != last_frame_id:
                        raw_frame, frame_data = latest_data
                        last_frame_id = current_frame_id

            # if a new frame arrived, process it for display
            if raw_frame is not None and frame_data is not None:

                pixel_format = frame_data.get('pixel_format') or self._fmt

                try:
                    match pixel_format:
                        case 'Mono16':
                            gray_8bit = (raw_frame >> 8).astype(np.uint8)
                            cv2.cvtColor(gray_8bit, cv2.COLOR_GRAY2BGR, dst=bgr_frame)
                        case 'Mono12':
                            gray_8bit = (raw_frame >> 4).astype(np.uint8)
                            cv2.cvtColor(gray_8bit, cv2.COLOR_GRAY2BGR, dst=bgr_frame)
                        case 'Mono10':
                            gray_8bit = (raw_frame >> 2).astype(np.uint8)
                            cv2.cvtColor(gray_8bit, cv2.COLOR_GRAY2BGR, dst=bgr_frame)
                        case 'Mono8':
                            cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR, dst=bgr_frame)
                        case 'BayerRG8':
                            cv2.cvtColor(raw_frame, cv2.COLOR_BAYER_RG2BGR, dst=bgr_frame)
                        case 'BayerGR8':
                            cv2.cvtColor(raw_frame, cv2.COLOR_BAYER_GR2BGR, dst=bgr_frame)
                        case 'BayerGB8':
                            cv2.cvtColor(raw_frame, cv2.COLOR_BAYER_GB2BGR, dst=bgr_frame)
                        case 'BayerBG8':
                            cv2.cvtColor(raw_frame, cv2.COLOR_BAYER_BG2BGR, dst=bgr_frame)
                        case 'RGB8':
                            cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR, dst=bgr_frame)
                        case 'RGBA8':
                            cv2.cvtColor(raw_frame, cv2.COLOR_RGBA2BGR, dst=bgr_frame)
                        case 'HSV':
                            if raw_frame.dtype != np.uint8:
                                h = (raw_frame[..., 0] * 180).astype(np.uint8)
                                s = (raw_frame[..., 1] * 255).astype(np.uint8)
                                v = (raw_frame[..., 2] * 255).astype(np.uint8)
                                cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR, dst=bgr_frame)
                            else:
                                cv2.cvtColor(raw_frame, cv2.COLOR_HSV2BGR, dst=bgr_frame)
                        case _:
                            if raw_frame.shape == bgr_frame.shape and raw_frame.dtype == bgr_frame.dtype:
                                np.copyto(bgr_frame, raw_frame)
                            else:
                                logger.error(f"[{self.name}] Unsupported pixel format for display: {pixel_format}")
                                bgr_frame[:] = (255, 0, 255)

                except cv2.error as e:
                    logger.error(f"[{self.name}] OpenCV Error during color conversion: {e}")
                    bgr_frame[:] = (0, 0, 255)

                # Directly update the shared variables used by the main GUI thread
                self._latest_frame = bgr_frame
                self._current_frame_data = frame_data

    # ──────────────────────────────── Processing hook ────────────────────────────────

    def _send_frame_for_processing(self):
        """
        Hook for subclasses to send frames to their processing pipeline.

        Called by timer_processing at CALIB_PROCESSING_FPS rate.

        Subclasses should:
        - Check if their worker(s) are busy/blocking before sending
        - Emit their own signal to send the frame
        - Set appropriate busy flags

        Default implementation does nothing (no processing).
        """
        pass

    # ──────────────────────────────── Display update ────────────────────────────────

    def _annotate_frame(self):
        """
        Hook for subclasses to annotate the display frame.

        Called at the display rate before updating the image on screen.
        Default behavior: just copy the latest raw frame to the display buffer.
        """
        if self._latest_frame is not None:
            np.copyto(self._latest_display_frame, self._latest_frame)
            self._latest_frame = None  # mark as consumed for display

    def _update_display(self):
        """
        Main display updater. Runs at a controlled rate for smooth video.
        """
        if self._latest_frame is None:
            return

        # Subclasses do their own thing
        self._annotate_frame()

        # Update the image on the screen
        self.image_item.setImageData(self._latest_display_frame)

        # One-time setup to fit the image to the viewbox
        if not self._video_initialised:
            self.view_box.autoRange()
            self._video_initialised = True

    def _clear_display(self):
        """Clears the video display to black and resets the initialisation flag."""
        self._latest_display_frame.fill(0)
        self.image_item.setImageData(self._latest_display_frame)
        self._video_initialised = False

    # ──────────────────────────────── Slow update (text, etc) ────────────────────────────────

    def _update_slow(self):

        if not self.isVisible():
            return

        now = time.monotonic()
        dt = now - self._fps_clock

        if dt > 0 and self._mainwindow.manager.acquiring:
            current_frame_number = self._current_frame_data.get('frame_number', 0)

            frames_acquired = current_frame_number - self._last_frame_number_for_fps

            if frames_acquired > 0:
                current_acquisition_fps = frames_acquired / dt
                self._capture_fps_deque.append(current_acquisition_fps)
                avg_fps = sum(self._capture_fps_deque) / len(self._capture_fps_deque)

                target_framerate = self._hw_camera.framerate

                if abs(avg_fps - target_framerate) > (target_framerate * 0.1):  # 10% tolerance
                    self._warning = True
                else:
                    self._warning = False

                self.capturefps_value.setText(f"{avg_fps:.2f} fps")

            self._fps_clock = now
            self._last_frame_number_for_fps = current_frame_number

        params_to_poll = ['exposure', 'framerate', 'gain', 'black_level', 'gamma']

        for param in params_to_poll:

            # Poll for parameter *value* changes
            current_value = getattr(self._hw_camera, param)
            last_value = self._last_polled_values.get(param)

            if current_value != last_value:
                self._on_slider_value_changed(param, current_value)
                self._last_polled_values[param] = current_value

            # Poll for parameter *range* changes
            current_range = getattr(self._hw_camera, f"{param}_range")
            last_range = self._last_polled_ranges.get(param)

            if current_range != last_range:
                self._on_slider_range_changed(param, current_range)
                self._last_polled_ranges[param] = current_range

        if self._mainwindow.manager.acquiring:
            h, w, _ = self._latest_display_frame.shape
            if w > 0:
                scale = 100 / w
                thumbnail_h = int(h * scale)
                thumbnail = cv2.resize(self._latest_display_frame, (100, thumbnail_h), interpolation=cv2.INTER_AREA)
                brightness = np.round(thumbnail.mean() / 255 * 100, decimals=2)
                self.brightness_value.setText(f"{brightness:.2f}%")
        else:
            self.capturefps_value.setText("Off")
            self.brightness_value.setText("-")
            self._warning = False
            self._capture_fps_deque.clear()

            if self._video_initialised:
                self._clear_display()

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

    def _on_slider_value_changed(self, label, value):
        """
        Hook for subclasses to handle camera parameter value changes.
        Called when a polled camera parameter changes.
        """
        pass

    def _on_slider_range_changed(self, label, value):
        """
        Hook for subclasses to handle camera parameter range changes.
        Called when a polled camera parameter range changes.
        """
        pass

    # ──────────────────────────────── Worker control ────────────────────────────────

    def _pause_worker(self):
        """
        Pause processing workers. Subclasses with workers must override this.
        """
        pass

    def _resume_worker(self):
        """
        Resume processing workers. Subclasses with workers must override this.
        """
        pass

    # ──────────────────────────────── Qt method overrides ────────────────────────────────

    def closeEvent(self, event):
        """This is an important part of the graceful shutdown."""

        if self._force_destroy:
            # Stop the consumer thread first
            # (otherwise it crashes on quitting on macOS)
            if hasattr(self, '_frame_consumer') and self._frame_consumer.is_alive():
                self._consumer_thread_active = False
                self._frame_consumer.join(timeout=2.0)
                if self._frame_consumer.is_alive():
                    logger.warning(f"{self.name} consumer thread did not shut down cleanly.")

            # Stop local timers
            self._stop_timers()

            # Accept the close event to allow Qt to destroy the window
            event.accept()
        else:
            # This is for hiding the window only
            event.ignore()
            self.hide()
            self._pause_worker()

        self._mainwindow.secondary_windows_visibility_buttons[self._hw_cam_idx].setChecked(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        # This forces the image to always fill the view correctly.
        if self.view_box and self._video_initialised:
            self.view_box.setRange(rect=self.image_item.boundingRect(), padding=0)

    # ──────────────────────────────── Properties ────────────────────────────────

    @property
    def name(self) -> str:
        return self._hw_cam_name

    @property
    def idx(self) -> str:
        return self._hw_cam_idx

    @property
    def colour(self) -> str:
        return f'#{self._main_colour.lstrip("#")}'

    color = colour

    @property
    def secondary_colour(self) -> str:
        return f'#{self._secondary_colour.lstrip("#")}'

    secondary_color = secondary_colour

    @property
    def source_shape_hw(self) -> Tuple[int, int]:
        return (self._source_height, self._source_width)

    @property
    def aspect_ratio(self) -> float:
        return float(self._source_width / self._source_height)

    # ──────────────────────────────── Other methods ────────────────────────────────

    def auto_size(self):
        width_multiplier = 1.0
        layout = self.video_container_layout
        if layout and layout.count() > 1:
            total_stretch = 0
            video_widget_stretch = 1

            for i in range(layout.count()):
                stretch = layout.stretch(i)
                total_stretch += stretch

                # Find the graphics_widget to get its specific stretch factor
                item = layout.itemAt(i)
                if item and item.widget() is self.graphics_widget:
                    video_widget_stretch = stretch

            if video_widget_stretch > 0:
                width_multiplier = total_stretch / video_widget_stretch

        monitor = self._mainwindow.selected_monitor

        if monitor.height < monitor.width:  # landscape
            available_h = (monitor.height - TASKBAR_H) // 2 - SPACING * 3
            video_max_h = available_h - self.BOTTOM_PANEL.height() - TOPBAR_H
            video_max_w = video_max_h * self.aspect_ratio

            h = int(video_max_h + self.BOTTOM_PANEL.height())
            w = int(video_max_w * width_multiplier)

        else:  # portrait
            video_max_w = monitor.width // 2 - SPACING * 3
            video_max_h = video_max_w / self.aspect_ratio

            h = int(video_max_h + self.BOTTOM_PANEL.height())
            w = int(video_max_w * width_multiplier)

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