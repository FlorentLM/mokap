"""
Main control window for the application.

Manages:
- Mode switching (Recording/Calibration)
- Secondary windows (CalibrationLiveView, RecordingLiveView, CentralCalibrationWindow)
- Camera rig initialisation
- Coordinator and multiview worker setup
"""
import logging
import os
import platform
import subprocess
import sys
import time
from functools import partial
import numpy as np
import screeninfo
from PySide6.QtCore import QTimer, Qt, QThread
from PySide6.QtGui import QFont, QBrush, QColor, QPen, QGuiApplication, QIcon
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QWidget, QFrame, QHBoxLayout, QLabel, QComboBox,
                               QPushButton, QGroupBox, QLineEdit, QCheckBox, QGraphicsView, QGraphicsScene,
                               QProgressBar, QFileDialog, QApplication, QGraphicsRectItem, QGraphicsTextItem)
from mokap.utils import hex_to_hls, pretty_size, get_size
from mokap.gui import GUI_LOGGER
from mokap.gui.style import *
from mokap.gui.widgets import (CalibrationVideoWindow, RecordingVideoWindow, Viewer3D,
                               UI_UPDATE_FPS, CALIB_HARDWARE_FPS_MAX)
from mokap.gui.workers import CalibrationCoordinator, MultiviewWorker
from lucida import CameraRig, CameraModel, Intrinsics, Extrinsics
from lucida.calibration import CharucoBoard

logger = logging.getLogger(__name__)


# TODO: Board should be loaded from config file
DEFAULT_BOARD = CharucoBoard(rows=6, cols=5, square_length=1.5, marker_size=4)


def create_camera_rig(hardware_cameras):
    """Create CameraRig from hardware camera sources."""

    # TODO: Interface between hardware camera class and CameraModel should be better

    camera_models = []

    for hw_cam in hardware_cameras:
        width, height = hw_cam.roi[2], hw_cam.roi[3]

        # TODO: Hardware camera interface should expose more sensor info
        # TODO: This should otherwise load values from the config file
        intrinsics = Intrinsics.from_specs(
            image_size=(width, height),
            focal=65.0,
            sensor=hw_cam.resolution or [4.968, 3.726],
            distortion_model='standard'
        )
        camera = CameraModel(intrinsics, Extrinsics(), name=hw_cam.name)
        camera_models.append(camera)

    return CameraRig(camera_models)



class MainControls(QMainWindow):
    """Main control window."""

    def __init__(self, camera_controller):
        super().__init__()

        self.setWindowTitle('Controls')

        # References
        self.gui_logger = GUI_LOGGER
        self.controller = camera_controller

        # Create camera rig
        self.rig = create_camera_rig(self.controller.cameras)
        logger.info(f"Created CameraRig with cameras: {[c.name for c in self.rig]}")

        # Calibration board parameters
        self.board_params = DEFAULT_BOARD

        # Coordinator (created once, reused)
        self.coordinator = CalibrationCoordinator(self.rig)

        # Multiview worker (created when entering calibration mode)
        self.multiview_worker = None
        self.multiview_thread = None

        # Camera info
        self.nb_cams = self.controller.nb_cameras
        self._cameras_names = tuple(hwcam.name for hwcam in self.controller.cameras)

        self.sources_shapes_hw = {
            hwcam.name: (hwcam.roi[3], hwcam.roi[2])
            for hwcam in self.controller.cameras
        }

        # Camera colours
        self.main_colours = {
            hwcam.name: self.controller.colours[hwcam.unique_id]
            for hwcam in self.controller.cameras
        }
        self.secondary_colours = {
            k: col_white if hex_to_hls(v)[1] < 60 else col_black
            for k, v in self.main_colours.items()
        }

        # Monitor setup
        self.selected_monitor = None
        self._monitors = screeninfo.get_monitors()
        self._set_monitor()

        # State
        self.is_calibrating = False
        self.is_recording = False

        # Window references
        self.viewer_3d = None
        self.video_windows = []
        self.calibration_views = {}

        # Timing
        self._tick = time.monotonic()

        # Build UI
        self._init_gui()

        # Keep track of fps settings to restore them after calib
        self._recording_framerates = {}

        # Start secondary windows
        self._start_secondary_windows()

        # Slow update timer
        self.timer_slow = QTimer(self)
        self.timer_slow.timeout.connect(self._update_main)
        self.timer_slow.start(int(1.0 / UI_UPDATE_FPS * 1000))

    @property
    def cameras_names(self):
        return list(self._cameras_names)

    def get_camera_index(self, unique_id: str) -> int:
        """Find camera index by unique ID."""
        for i, cam in enumerate(self.controller.cameras):
            if cam.unique_id == unique_id:
                return i
        raise ValueError(f"Camera not found: {unique_id}")

    # ──────────────────────────────── GUI setup ────────────────────────────────

    def _init_gui(self):
        """Build the main control GUI."""
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)

        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        # Toolbar
        toolbar = QFrame()
        toolbar.setFixedHeight(38)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(3, 0, 3, 0)

        toolbar_layout.addWidget(QLabel('Mode: '))

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Recording', 'Calibration'])
        self.mode_combo.currentIndexChanged.connect(self._toggle_calib_record)
        toolbar_layout.addWidget(self.mode_combo, 1)

        toolbar_layout.addStretch(2)

        self.button_exit = QPushButton("Exit (Esc)")
        self.button_exit.clicked.connect(self.quit)
        self.button_exit.setStyleSheet(f"background-color: {col_red}; color: {col_white};")
        toolbar_layout.addWidget(self.button_exit)

        self.main_layout.addWidget(toolbar)

        # Main content
        content = QWidget()
        content_layout = QHBoxLayout(content)

        # Left pane: Acquisition controls
        left_pane = QGroupBox("Acquisition")
        left_pane.setMinimumWidth(400)
        left_layout = QVBoxLayout(left_pane)

        # Session name
        name_widget = QWidget()
        name_layout = QHBoxLayout(name_widget)
        name_layout.addWidget(QLabel('Name: '))

        self.acq_name_textbox = QLineEdit()
        self.acq_name_textbox.setDisabled(True)
        self.acq_name_textbox.setText(self.controller.session_name)
        self.acq_name_textbox.setPlaceholderText("Default name: yymmdd-hhmm")
        # Connect returnPressed to save the name when Enter is pressed
        self.acq_name_textbox.returnPressed.connect(self._apply_session_name)
        name_layout.addWidget(self.acq_name_textbox, 1)

        self.acq_name_edit_btn = QPushButton("Edit")
        self.acq_name_edit_btn.setCheckable(True)
        self.acq_name_edit_btn.clicked.connect(self._toggle_text_editing)
        name_layout.addWidget(self.acq_name_edit_btn)

        left_layout.addWidget(name_widget)

        # Path display
        path_label_widget = QWidget()
        path_label_layout = QHBoxLayout(path_label_widget)
        self.current_dir_label = QLabel()
        self.current_dir_label.setStyleSheet(f"color: {col_darkgray};")
        self.current_dir_label.setWordWrap(True)
        self.current_dir_label.setText(f'{self.controller.full_path.resolve()}')
        path_label_layout.addWidget(self.current_dir_label)

        self.button_open_folder = QPushButton("Open")
        self.button_open_folder.clicked.connect(self._open_session_folder)
        path_label_layout.addWidget(self.button_open_folder)

        left_layout.addWidget(path_label_widget)

        # Buttons
        btns_h = 60

        self.button_acquisition = QPushButton("Acquisition off")
        self.button_acquisition.setMinimumHeight(btns_h)
        self.button_acquisition.setCheckable(True)
        self.button_acquisition.setIcon(QIcon(icon_capture_bw))
        self.button_acquisition.clicked.connect(self._toggle_acquisition)
        left_layout.addWidget(self.button_acquisition)

        self.button_snapshot = QPushButton("Snapshot")
        self.button_snapshot.setMinimumHeight(btns_h)
        self.button_snapshot.setIcon(QIcon(icon_snapshot_bw))
        self.button_snapshot.clicked.connect(self._take_snapshot)
        self.button_snapshot.setDisabled(True)
        left_layout.addWidget(self.button_snapshot)

        self.button_recpause = QPushButton("Not recording (Space)")
        self.button_recpause.setMinimumHeight(btns_h)
        self.button_recpause.setCheckable(True)
        self.button_recpause.setIcon(QIcon(icon_rec_bw))
        self.button_recpause.clicked.connect(self._toggle_recording)
        self.button_recpause.setDisabled(True)
        left_layout.addWidget(self.button_recpause)

        content_layout.addWidget(left_pane, 4)

        # Right pane: Secondary windows
        right_pane = QWidget()
        right_pane.setMinimumWidth(300)
        right_layout = QVBoxLayout(right_pane)

        windows_group = QGroupBox('Secondary windows')
        windows_layout = QVBoxLayout(windows_group)

        # Window visibility checkboxes
        self.secondary_windows_visibility_buttons = []
        for i in range(self.nb_cams):
            checkbox = QCheckBox(f"Camera {i}")
            checkbox.setChecked(True)
            checkbox.setMinimumHeight(25)
            windows_layout.addWidget(checkbox)
            self.secondary_windows_visibility_buttons.append(checkbox)

        right_layout.addWidget(windows_group)

        # Monitor selector
        monitor_group = QGroupBox('Active monitor')
        monitor_layout = QVBoxLayout(monitor_group)

        self.monitors_buttons = QGraphicsView()
        self.monitors_buttons_scene = QGraphicsScene()
        self.monitors_buttons.setScene(self.monitors_buttons_scene)
        self.monitors_buttons.setStyleSheet("border: none; background-color: transparent;")

        if 'Darwin' in sys.platform:
            self.monitors_buttons.viewport().setAttribute(
                Qt.WidgetAttribute.WA_AcceptTouchEvents, False
            )

        monitor_layout.addWidget(self.monitors_buttons)
        right_layout.addWidget(monitor_group)

        # Memory pressure
        mem_group = QGroupBox('Memory')
        mem_layout = QVBoxLayout(mem_group)

        self._mem_pressure_bar = QProgressBar()
        self._mem_pressure_bar.setRange(0, 100)
        self._mem_pressure_bar.setValue(0)
        mem_layout.addWidget(self._mem_pressure_bar)

        self.frames_saved_label = QLabel("Saved: (0 B)")
        mem_layout.addWidget(self.frames_saved_label)

        right_layout.addWidget(mem_group)
        right_layout.addStretch()

        content_layout.addWidget(right_pane, 3)
        self.main_layout.addWidget(content)

    # ──────────────────────────────── Mode switching ────────────────────────────────

    def _toggle_calib_record(self, idx: int):
        """Switch between Recording and Calibration modes."""

        # Clean up old windows
        self._stop_secondary_windows()

        # Update State
        self.is_calibrating = (idx == 1)
        self.is_recording = False

        if self.is_calibrating:
            if self.controller.hardware_triggered:
                self._recording_framerates['trigger'] = self.controller.framerate
                self.controller.framerate = CALIB_HARDWARE_FPS_MAX
                logger.info(f"[Hardware Trigger] Limit applied for calibration: {CALIB_HARDWARE_FPS_MAX} fps")
            else:
                # Save current state and cap framerate
                for cam in self.controller.cameras:
                    self._recording_framerates[cam.name] = cam.framerate

                    if cam.framerate > CALIB_HARDWARE_FPS_MAX:
                        try:
                            cam.framerate = CALIB_HARDWARE_FPS_MAX
                            logger.info(f"[{cam.name}] Limit applied for calibration: {CALIB_HARDWARE_FPS_MAX} fps")
                        except Exception:
                            pass
        else:
            if self.controller.hardware_triggered:
                prev_fps = self._recording_framerates.get('trigger')
                self.controller.framerate = prev_fps
                logger.info(f"[Hardware Trigger] Restored framerate: {prev_fps} fps")
            else:
                # Restore previous framerate (unconstrained)
                for cam in self.controller.cameras:
                    prev_fps = self._recording_framerates.get(cam.name)
                    if prev_fps is not None and cam.framerate != prev_fps:
                        try:
                            cam.framerate = prev_fps
                            logger.info(f"[{cam.name}] Restored framerate: {prev_fps} fps")
                        except Exception:
                            pass

        # Restart windows
        self._start_secondary_windows()

    def _toggle_text_editing(self, checked):
        """Toggle session name editing."""
        if checked:
            # Entering edit mode - only allow if not acquiring
            if self.controller.acquiring:
                self.acq_name_edit_btn.setChecked(False)
                logger.warning("Cannot edit session name while acquisition is running.")
                return
            self.acq_name_textbox.setDisabled(False)
            self.acq_name_edit_btn.setText("Save")
            self.acq_name_textbox.setFocus()
            self.acq_name_textbox.selectAll()
        else:
            # Exiting edit mode - apply the name
            self._apply_session_name()
            self.acq_name_textbox.setDisabled(True)
            self.acq_name_edit_btn.setText("Edit")

    def _apply_session_name(self):
        """Apply the session name from the textbox to the controller."""
        new_name = self.acq_name_textbox.text().strip()
        if not new_name:
            # If empty, restore the current name
            self.acq_name_textbox.setText(self.controller.session_name)
            return

        if new_name == self.controller.session_name:
            # No change
            return

        try:
            self.controller.session_name = new_name
            # Update textbox with the actual name (might be modified if folder existed)
            self.acq_name_textbox.setText(self.controller.session_name)
            # Update path label
            self.current_dir_label.setText(f'{self.controller.full_path.resolve()}')
            logger.info(f"Session name changed to: {self.controller.session_name}")
        except RuntimeError as e:
            # Can't change name while acquiring
            logger.error(f"Cannot change session name: {e}")
            self.acq_name_textbox.setText(self.controller.session_name)
        except Exception as e:
            logger.error(f"Failed to change session name: {e}")
            self.acq_name_textbox.setText(self.controller.session_name)

    # ──────────────────────────────── Main actions ────────────────────────────────

    def _toggle_acquisition(self, checked):
        """Toggle camera acquisition."""
        if checked:
            self.controller.start_acquisition()
            self.button_acquisition.setText("Acquisition ON")
            self.button_acquisition.setIcon(QIcon(icon_capture))
            self.button_acquisition.setStyleSheet(
                f"background-color: {col_green}; color: {col_white};"
            )
            self.button_snapshot.setEnabled(True)
            self.button_recpause.setEnabled(True)
            self.acq_name_edit_btn.setEnabled(False)
            if self.acq_name_edit_btn.isChecked():
                self.acq_name_edit_btn.setChecked(False)
                self.acq_name_textbox.setDisabled(True)
                self.acq_name_edit_btn.setText("Edit")
        else:
            self.controller.stop_acquisition()
            self.button_acquisition.setText("Acquisition off")
            self.button_acquisition.setIcon(QIcon(icon_capture_bw))
            self.button_acquisition.setStyleSheet("")
            self.button_snapshot.setEnabled(False)
            self.button_recpause.setEnabled(False)
            self.acq_name_edit_btn.setEnabled(True)

    def _toggle_recording(self, checked):
        """Toggle recording."""
        if checked:
            self.controller.start_recording()
            self.button_recpause.setText("RECORDING (Space to stop)")
            self.button_recpause.setIcon(QIcon(icon_rec_on))
            self.button_recpause.setStyleSheet(
                f"background-color: {col_red}; color: {col_white};"
            )
            self.is_recording = True
        else:
            self.controller.stop_recording()
            self.button_recpause.setText("Not recording (Space)")
            self.button_recpause.setIcon(QIcon(icon_rec_bw))
            self.button_recpause.setStyleSheet("")
            self.is_recording = False

    def _take_snapshot(self):
        """Take a snapshot."""
        self.controller.take_snapshot()

    # ──────────────────────────────── Secondary windows ────────────────────────────────

    def _start_secondary_windows(self):
        """Create and configure secondary windows."""
        self.calibration_views = {}

        # Create camera windows
        for i, cam in enumerate(self.controller.cameras):
            if self.is_calibrating:
                w = CalibrationVideoWindow(cam, self, self.board_params)
                self.calibration_views[cam.name] = w
            else:
                w = RecordingVideoWindow(cam, self)

            self.video_windows.append(w)

            # Configure visibility button
            btn = self.secondary_windows_visibility_buttons[i]
            btn.setText(f" {w.name.title()} camera")
            btn.setStyleSheet(
                f"border-radius: 5px; padding: 0 10 0 10; "
                f"color: {w.secondary_colour}; background-color: {w.colour};"
            )
            btn.clicked.connect(w.toggle_visibility)
            btn.setChecked(True)

            w.show()

        # Connect coordinator to workers (calibration mode only)
        if self.is_calibrating:
            for cam_name, view in self.calibration_views.items():
                self.coordinator.broadcast_stage.connect(view._worker.set_stage)
                self.coordinator.broadcast_reset.connect(view._worker.reset)
                self.coordinator.broadcast_board_changed.connect(
                    view._detector.configure_new_board
                )
                self.coordinator.broadcast_board_changed.connect(
                    view._worker.configure_new_board
                )
                self.coordinator.broadcast_parameters_loaded.connect(
                    view._on_intrinsics_updated
                )

            # Create 3D view window
            self.viewer_3d = Viewer3D(self)
            self.viewer_3d.show()

            # Create multiview worker
            origin_cam = self.viewer_3d.origin_camera_combo.currentText()

            self.multiview_worker = MultiviewWorker(
                rig=self.rig,
                calibration_board=self.board_params,
                origin_cam=origin_cam
            )
            self.multiview_thread = QThread()
            self.multiview_worker.moveToThread(self.multiview_thread)

            # Connect coordinator to multiview
            self.coordinator.broadcast_stage.connect(self.multiview_worker.set_stage)
            self.coordinator.broadcast_reset.connect(self.multiview_worker.reset)
            self.coordinator.broadcast_board_changed.connect(
                self.multiview_worker.configure_new_board
            )
            self.coordinator.request_refinement.connect(
                self.multiview_worker.trigger_refinement
            )

            # Connect detections to multiview
            for cam_name, view in self.calibration_views.items():
                view._detector.detection_ready.connect(
                    lambda result, name=cam_name: self.multiview_worker.on_detection(name, result),
                    Qt.QueuedConnection
                )

            # Connect 3D scene updates
            self.multiview_worker.scene_updated.connect(
                self.viewer_3d.update_3d_scene
            )

            self.multiview_thread.start()

        self.cascade_windows()

    def _stop_secondary_windows(self):
        """Close and clean up secondary windows."""
        for w in self.video_windows:
            w._force_destroy = True
            w.close()

        if self.viewer_3d:
            self.viewer_3d._force_destroy = True
            self.viewer_3d.close()

        self.video_windows.clear()
        self.calibration_views.clear()
        self.viewer_3d = None

        if self.multiview_thread:
            self.multiview_thread.quit()
            self.multiview_thread.wait()
            self.multiview_thread = None
            self.multiview_worker = None

    def get_visible_windows(self, include_main=False):
        """Get list of visible windows."""
        try:
            windows = [w for w in self.video_windows if w.isVisible()]
            if self.viewer_3d and self.viewer_3d.isVisible():
                windows.append(self.viewer_3d)
            if include_main:
                windows.append(self)
        except RuntimeError as e:
            windows = []
        return windows

    def cascade_windows(self):
        """Arrange windows in a cascade pattern on the selected monitor."""
        self.raise_()  # Bring main window to top

        ax, ay, aw, ah = self._available_screen_space()
        cascade_offset = 30

        visible = self.get_visible_windows(include_main=True)
        num_secondary = len([w for w in visible if w is not self])

        for win in visible:
            frame = win.frameGeometry()
            win_w, win_h = frame.width(), frame.height()

            if win is self:
                # Main window: offset based on number of secondary windows
                offset = cascade_offset * (num_secondary + 1)
            else:
                # Secondary windows: offset based on their index
                idx = getattr(win, 'idx', 0)
                offset = cascade_offset * (idx + 1)

            # Calculate position, constrained to screen bounds
            new_x = max(ax, min(ax + offset, ax + aw - win_w))
            new_y = max(ay, min(ay + offset, ay + ah - win_h))

            win.move(new_x, new_y)

    def _available_screen_space(self):
        """
        Get the available screen space (excluding taskbar/dock) for the selected monitor.
        Returns (x, y, width, height).
        """
        # Default to primary screen
        geom = QGuiApplication.primaryScreen().availableGeometry()

        # Try to find the matching QScreen for our selected monitor
        if self.selected_monitor:
            m = self.selected_monitor
            for screen in QGuiApplication.screens():
                rect = screen.geometry()
                if (rect.x() == m.x and rect.y() == m.y and
                        rect.width() == m.width and rect.height() == m.height):
                    geom = screen.availableGeometry()
                    break

        return geom.x(), geom.y(), geom.width(), geom.height()

    # ──────────────────────────────── Other actions ────────────────────────────────

    def _open_session_folder(self):
        path = self.controller.full_path.resolve()
        try:
            if 'Linux' in platform.system():
                subprocess.Popen(['xdg-open', path])
            elif 'Windows' in platform.system():
                os.startfile(path)
            elif 'Darwin' in platform.system():
                subprocess.Popen(['open', path])
        except:
            pass

    # ──────────────────────────────── Monitors management ────────────────────────────────

    def _set_monitor(self, idx=None):
        """Set the active monitor."""
        if len(self._monitors) > 1 and idx is None:
            self.selected_monitor = next(
                (m for m in self._monitors if m.is_primary),
                self._monitors[0]
            )
        elif idx is not None and idx < len(self._monitors):
            self.selected_monitor = self._monitors[idx]
        else:
            self.selected_monitor = self._monitors[0]

    def _monitor_update(self, idx, event=None):
        """Handle monitor selection click."""
        self._set_monitor(idx)
        self._update_monitors_buttons()

    def _update_monitors_buttons(self):
        """Update monitor selector display."""
        self.monitors_buttons_scene.clear()

        SCALE = 40
        visible_wins = self.get_visible_windows(include_main=True)

        for i, m in enumerate(self._monitors):
            mx, my = m.x // SCALE, m.y // SCALE
            mw, mh = m.width // SCALE, m.height // SCALE

            # Monitor background
            col = '#7f7f7f' if m == self.selected_monitor else '#807f7f7f'
            rect = QGraphicsRectItem(mx, my, mw - 2, mh - 2)
            rect.setBrush(QBrush(QColor(col)))
            rect.setPen(QPen(Qt.PenStyle.NoPen))
            rect.mousePressEvent = partial(self._monitor_update, i)
            rect.setZValue(0)
            self.monitors_buttons_scene.addItem(rect)

            # Window silhouettes
            for win in visible_wins:
                geom = win.geometry()
                center = geom.center()

                if (m.x <= center.x() < m.x + m.width and
                        m.y <= center.y() < m.y + m.height):
                    sx = geom.x() // SCALE
                    sy = geom.y() // SCALE
                    sw = max(1, geom.width() // SCALE)
                    sh = max(1, geom.height() // SCALE)

                    sil = QGraphicsRectItem(sx, sy, sw, sh)
                    sil.setBrush(QBrush(QColor('#b1b1b1b1')))
                    sil.setPen(QPen(Qt.PenStyle.NoPen))
                    sil.setZValue(1)
                    self.monitors_buttons_scene.addItem(sil)

            # Monitor number
            text = QGraphicsTextItem(f"{i}")
            text.setDefaultTextColor(QColor('#ffffffff'))
            text.setFont(QFont(DEFAULT_FONT, 9))
            text.setPos(mx + 2, my + mh - text.boundingRect().height() - 2)
            text.setZValue(2)
            self.monitors_buttons_scene.addItem(text)

    # ──────────────────────────────── Updates ────────────────────────────────

    def _update_main(self):
        """Slow periodic update."""
        self._update_monitors_buttons()

        now = time.monotonic()
        if now - self._tick >= 0.5:
            size = get_size(self.controller.full_path) if self.controller.full_path.is_dir() else 0
            self.frames_saved_label.setText(f'Saved: {pretty_size(size)}')
            self._tick = now

        buffers = np.array(self.controller.nb_buffered_frames)
        pressure = np.nanmean(buffers / self.controller.buffer_size).astype(np.float32)
        bar_val = int(pressure * 100) if np.isfinite(pressure) else 0
        self._mem_pressure_bar.setValue(bar_val)

    # ──────────────────────────────── Shutdown ────────────────────────────────

    def quit(self):
        """Clean shutdown."""

        self.timer_slow.stop()
        self._stop_secondary_windows()

        if self.controller.acquiring:
            self.controller.stop_acquisition()

        self.controller.disconnect_cameras()

        QApplication.instance().quit()

    def closeEvent(self, event):
        """Handle window close."""
        event.ignore()
        self.quit()