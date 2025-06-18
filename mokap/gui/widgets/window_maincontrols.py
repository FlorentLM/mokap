import logging
import os
import sys
import subprocess
from functools import partial
from typing import Tuple
import psutil
import screeninfo
from PySide6.QtCore import QTimer, Qt, Slot, QRect, QThread, QSize
from PySide6.QtGui import QFont, QGuiApplication, QCursor, QBrush, QColor, QPen
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QWidget, QFrame, QHBoxLayout, QLabel,
                               QComboBox, QPushButton, QSizePolicy, QGroupBox, QLineEdit, QScrollArea,
                               QCheckBox, QGraphicsView, QGraphicsScene, QTextEdit, QStatusBar,
                               QProgressBar, QFileDialog, QApplication, QGraphicsRectItem, QGraphicsTextItem)
from mokap.gui import GUI_LOGGER
from mokap.gui.style.commons import *
from mokap.gui.widgets import DEFAULT_BOARD, SLOW_UPDATE_INTERVAL
from mokap.gui.widgets.widgets_base import SnapMixin
from mokap.gui.widgets.windows_live_views import CalibrationLiveView, RecordingLiveView
from mokap.gui.widgets.window_central_calib import CentralCalibrationWindow
from mokap.gui.workers.coordinator import CalibrationCoordinator
from mokap.gui.workers.worker_multiview import MultiviewWorker
from mokap.utils import hex_to_hls, pretty_size
from mokap.utils.datatypes import CalibrationData, IntrinsicsPayload, ExtrinsicsPayload

logger = logging.getLogger(__name__)

class MainControls(QMainWindow, SnapMixin):

    def __init__(self, manager):
        super().__init__()

        self.setWindowTitle('Controls')

        # Heights for the collapsible log panel
        self._compact_height = 0
        self._expanded_height = 0

        self.gui_logger = GUI_LOGGER

        self.manager = manager

        self.board_params = DEFAULT_BOARD

        self.coordinator = CalibrationCoordinator()
        self.multiview_worker = None  # Will be created in _start_secondary_windows
        self.multiview_thread = None

        # Set cameras info
        self.nb_cams = self.manager.nb_cameras
        self._cameras_names = tuple(cam.name for cam in self.manager.cameras)

        # Note: The new camera interface uses .roi to get width/height, not .shape
        self.sources_shapes = {cam.name: (cam.roi[3], cam.roi[2]) for cam in self.manager.cameras}  # (height, width)

        # The colours dict is keyed by serial, but we can map it here to the friendly name
        self.cams_colours = {cam.name: self.manager.colours[cam.unique_id] for cam in self.manager.cameras}
        self.secondary_colours = {k: col_white if hex_to_hls(v)[1] < 60 else col_black for k, v in
                                  self.cams_colours.items()}

        # Identify monitors
        self.selected_monitor = None
        self._monitors = screeninfo.get_monitors()
        self.set_monitor()

        # States
        self.is_editing = False
        self.is_calibrating = False
        self.is_recording = False

        # Refs for the secondary windows
        self.central_calib_window = None
        self.video_windows = []

        # Other things to init
        self._current_buffers = None
        self._mem_pressure = 0.0

        # Build the gui
        self.init_gui()

        # Start the secondary windows
        self._start_secondary_windows()

        # This connection is the single point of truth for UI updates
        self.coordinator.send_to_main.connect(self.route_payload_to_widgets)

        # Setup MainWindow update
        self.timer_slow = QTimer(self)
        self.timer_slow.timeout.connect(self._update_main)
        self.timer_slow.start(int(SLOW_UPDATE_INTERVAL * 1000))

        self._mem_baseline = psutil.virtual_memory().percent

    @property
    def cameras_names(self):
        # Return a copy of the tuple, as a list
        return list(self._cameras_names)

    def get_camera_index(self, unique_id: str) -> int:
        """
        Safely finds the list index for a camera given its unique ID (serial number)
        """
        for i, cam in enumerate(self.manager.cameras):
            if cam.unique_id == unique_id:
                return i

        # This should ideally never happen if the GUI is in sync with the manager
        raise ValueError(f"Could not find camera with unique_id '{unique_id}' in the manager's list.")

    def init_gui(self):

        # Main Layout and central widget setup
        # ------------------------------------
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)

        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        # Create the top (fixed) and bottom (collapsible) containers
        # -----------------------------------------------------------

        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(5)
        top_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        # FIX: Make bottom_container a member variable 'self.bottom_container'
        self.bottom_container = QWidget()
        bottom_layout = QVBoxLayout(self.bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(5)
        self.bottom_container.setVisible(False)  # the entire bottom section starts hidden

        # Populate top container
        # (Toolbar, Acquisition group, Display group)
        # -------------------------------------------

        # Toolbar
        toolbar = QFrame()
        toolbar.setFixedHeight(38)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(3, 0, 3, 0)
        mode_label = QLabel('Mode: ')
        toolbar_layout.addWidget(mode_label)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Recording', 'Calibration'])
        self.mode_combo.currentIndexChanged.connect(self._toggle_calib_record)
        toolbar_layout.addWidget(self.mode_combo, 1)
        toolbar_layout.addStretch(2)
        self.button_exit = QPushButton("Exit (Esc)")
        self.button_exit.clicked.connect(self.quit)
        self.button_exit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.button_exit.setStyleSheet(f"background-color: {col_red}; color: {col_white};")
        toolbar_layout.addWidget(self.button_exit)
        top_layout.addWidget(toolbar)

        # Main content widget (Acquisition and Display panes)
        maincontent = QWidget()
        maincontent_layout = QHBoxLayout(maincontent)

        # Left pane (Acquisition)
        left_pane = QGroupBox("Acquisition")
        left_pane.setMinimumWidth(400)
        left_pane_layout = QVBoxLayout(left_pane)

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
        self.save_dir_current.setStyleSheet(f"color: {col_darkgray};")
        self.save_dir_current.setWordWrap(True)

        folderpath_label_font = QFont()
        folderpath_label_font.setPointSize(10)
        self.save_dir_current.setFont(folderpath_label_font)

        line_2_layout.addWidget(self.save_dir_current, 1)

        self.save_dir_current.setText(f'{self.manager.full_path.resolve()}')
        f_name_and_path_layout.addWidget(line_2)

        line_3 = QWidget()
        line_3_layout = QHBoxLayout(line_3)

        self.button_open_folder = QPushButton("Open folder")
        self.button_open_folder.clicked.connect(self.open_session_folder)

        line_3_layout.addStretch(2)
        line_3_layout.addWidget(self.button_open_folder)

        f_name_and_path_layout.addWidget(line_3)
        left_pane_layout.addWidget(f_name_and_path, 1)
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
        self.button_snapshot.setIcon(icon_snapshot_bw)
        self.button_snapshot.setDisabled(True)

        f_buttons_layout.addWidget(self.button_snapshot, 1)

        self.button_recpause = QPushButton("Not recording (Space to toggle)")
        self.button_recpause.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.button_recpause.setCheckable(True)
        self.button_recpause.clicked.connect(self._toggle_recording)
        self.button_recpause.setIcon(icon_rec_bw)
        self.button_recpause.setDisabled(True)

        f_buttons_layout.addWidget(self.button_recpause, 1)

        left_pane_layout.addWidget(f_buttons, 2)
        maincontent_layout.addWidget(left_pane, 4)

        # Right pane (Display)
        right_pane = QGroupBox("Display")
        right_pane.setMinimumWidth(300)
        right_pane_layout = QVBoxLayout(right_pane)

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
        for i in range(self.nb_cams):
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

        if 'Darwin' in sys.platform:
            self.monitors_buttons.viewport().setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, False)

        monitors_frame_layout.addWidget(self.monitors_buttons)

        self.autotile_button = QPushButton("Auto-tile windows")
        self.autotile_button.clicked.connect(self.autotile_windows)

        monitors_frame_layout.addWidget(self.autotile_button)
        maincontent_layout.addWidget(right_pane, 3)

        top_layout.addWidget(maincontent)

        # Populate the bottom container (Log)
        # ------------------------------------
        self.log_text_area = QTextEdit()
        self.log_text_area.setFont(QFont('consolas', 9))
        self.log_text_area.setReadOnly(True)
        self.log_text_area.setMinimumHeight(150)
        bottom_layout.addWidget(self.log_text_area)
        self.gui_logger.register_text_area(self.log_text_area)

        # Add containers and controls to the main layout
        # ----------------------------------------------
        self.main_layout.addWidget(top_container)

        # Log Button acts as the separator/controller
        self.log_button = QPushButton('Show log ▼')
        self.log_button.setMaximumWidth(100)

        self.log_button.clicked.connect(self.toggle_log_panel)
        self.main_layout.addWidget(self.log_button)
        self.main_layout.addWidget(self.bottom_container)

        # Status Bar
        statusbar = QStatusBar()
        self.setStatusBar(statusbar)

        mem_pressure_label = QLabel('Memory pressure: ')
        mem_pressure_label.setStyleSheet("background-color: transparent;")
        statusbar.addWidget(mem_pressure_label)

        self._mem_pressure_bar = QProgressBar()
        self._mem_pressure_bar.setMaximum(100)

        statusbar.addWidget(self._mem_pressure_bar)

        self.frames_saved_label = QLabel()
        self.frames_saved_label.setText('Saved: (0 bytes)')
        self.frames_saved_label.setStyleSheet("background-color: transparent;")
        statusbar.addPermanentWidget(self.frames_saved_label)

        # Finalize layout and initial size
        self.main_layout.addStretch(1)
        self._update_monitors_buttons()

        # Calculate compact height (with log panel hidden)
        self.adjustSize()
        self._compact_height = self.height()

        self._expanded_height = 600     # why is computing it automatically so hard?? fuck it, hardcoded to 600
        self.bottom_container.setVisible(False)

        # Set the initial fixed size to the compact state
        self.setFixedSize(self.width(), self._compact_height)

    @Slot(CalibrationData)
    def route_payload_to_widgets(self, data: CalibrationData):
        target_name = data.camera_name

        # Route to the specific monocular window
        for w in self.video_windows:
            if isinstance(w, CalibrationLiveView) and w.name == target_name:
                w.handle_payload(data)
                break

    @Slot()
    def on_load_calibration(self):
        """
        Handles loading a single calibration file containing data for all cameras
        """

        # Ensure we are in the right mode (should be impossible to call from recording mode anyway)
        if not self.is_calibrating:
            logger.warning("[MainControls] Must be in Calibration mode to load calibration.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load calibration file",
            str(self.manager.full_path.resolve()),
            "TOML Files (*.toml)"
        )

        if not file_path:
            return

        logger.info(f"[MainControls] Loading calibration from: {file_path}")

        try:
            for cam_name in self.cameras_names:
                # Load intrinsics for the camera
                intrinsics = IntrinsicsPayload.from_file(file_path, cam_name)
                if intrinsics.camera_matrix is not None:
                    self.coordinator.receive_from_main.emit(CalibrationData(cam_name, intrinsics))

                # Load extrinsics for the camera
                extrinsics = ExtrinsicsPayload.from_file(file_path, cam_name)
                if extrinsics.rvec is not None:
                    self.coordinator.receive_from_main.emit(CalibrationData(cam_name, extrinsics))

        except Exception as e:
            logger.error(f"[MainControls] Problem loading calibration file: {e}")

    #  ============= Qt method overrides =============
    def closeEvent(self, event):
        event.ignore()  # we handle shutdown ourselves
        self.quit()

    def quit(self):

        # stop and destroy all secondary windows and their associated threads
        # so no GUI elements try to access camera objects anymore
        self._stop_secondary_windows()

        # Now that all GUI threads are stopped, it's safe to turn off the manager
        if self.manager.acquiring:
            self.manager.stop_acquisition()

        self.manager.disconnect_cameras()

        self.gui_logger.restore()

        # close the main window and exit the application
        self.timer_slow.stop()
        self.close()
        QApplication.instance().quit()

    @Slot()
    def toggle_log_panel(self):
        is_visible = self.bottom_container.isVisible()

        if is_visible:
            # Hide the panel and set the window to its compact height
            self.bottom_container.setVisible(False)
            self.log_button.setText('Show log ▼')
            self.setFixedHeight(self._compact_height)
        else:
            # Show the panel and set the window to its expanded height
            self.bottom_container.setVisible(True)
            self.log_button.setText('Hide log ▲')
            self.setFixedHeight(self._expanded_height)

#  ============= General toggles =============
    def _toggle_calib_record(self):

        if self.is_recording:
            return

        if self.is_calibrating and self.mode_combo.currentIndex() == 0:
            self.is_calibrating = False

            self._stop_secondary_windows()

            if self.manager.acquiring:
                self.button_snapshot.setDisabled(False)
                self.button_recpause.setDisabled(False)

            self._start_secondary_windows()

        elif not self.is_calibrating and self.mode_combo.currentIndex() == 1:
            self.is_calibrating = True

            self._stop_secondary_windows()
            self.button_recpause.setDisabled(True)
            self._start_secondary_windows()

        else:
            pass

    def _toggle_text_editing(self, checked: bool):

        if checked:
            self.acq_name_textbox.setDisabled(False)
            self.acq_name_edit_btn.setText('Set')
            self.acq_name_textbox.setFocus()
            self.is_editing = True
        else:
            self.acq_name_textbox.setDisabled(True)
            self.acq_name_edit_btn.setText('Edit')
            self.manager.session_name = self.acq_name_textbox.text()
            self.save_dir_current.setText(f'{self.manager.full_path.resolve()}')
            self.is_editing = False

    def _toggle_acquisition(self, checked: bool):

        if checked:
            self.manager.on()

            self.save_dir_current.setText(f'{self.manager.full_path.resolve()}')

            self.button_acquisition.setText("Acquiring")
            self.button_acquisition.setIcon(icon_capture)
            self.button_snapshot.setDisabled(False)
            self.acq_name_edit_btn.setDisabled(True)  # disable name editing

            if not self.is_calibrating:
                self.button_recpause.setDisabled(False)

        else:
            # Ensure recording is stopped first
            self._toggle_recording(False)
            self.button_recpause.setChecked(False)  # also update the button's visual state

            self.manager.off()

            # Reset Acquisition folder name
            self.acq_name_textbox.setText('')
            self.save_dir_current.setText('')

            self.button_acquisition.setText("Acquisition off")
            self.button_acquisition.setIcon(icon_capture_bw)
            self.button_snapshot.setDisabled(True)
            self.button_recpause.setDisabled(True)
            self.acq_name_edit_btn.setDisabled(False)  # re-enable name editing

            # Re-enable the framerate sliders when acquisition stops
            if self.manager.hardware_triggered and not self.is_calibrating:
                for w in self.video_windows:
                    if isinstance(w, RecordingLiveView):
                        w.camera_controls_sliders['framerate'].setDisabled(False)

    def _toggle_recording(self, checked: bool):

        if not self.manager.acquiring:
            return

        if checked:
            self.manager.record()
            self.button_recpause.setText("Recording... (Space to toggle)")
            self.button_recpause.setIcon(icon_rec_on)
            self.is_recording = True
        else:
            if self.manager.recording:  # only pause if we are actually recording
                self.manager.pause()

            self.button_recpause.setText("Not recording (Space to toggle)")
            self.button_recpause.setIcon(icon_rec_bw)
            self.is_recording = False

    def _take_snapshot(self):
        """ Takes an instantaneous snapshot from all cameras """
        if self.manager.acquiring:
            self.manager.take_snapshot()

    def open_session_folder(self):
        path = self.manager.full_path.resolve()
        try:
            if 'Linux' in platform.system():
                subprocess.Popen(['xdg-open', path])
            elif 'Windows' in platform.system():
                os.startfile(path)
            elif 'Darwin' in platform.system():
                subprocess.Popen(['open', path])
        except:
            pass

    def _avail_screenspace(self) -> Tuple[int, int, int, int]:
        """
        Finds the QScreen that matches the selected_monitor and returns available space
        """

        # fallback to main screen if nothing matches
        geom = QGuiApplication.primaryScreen().availableGeometry()

        for screen in QGuiApplication.screens():
            rect: QRect = screen.geometry()
            m = self.selected_monitor
            if (
                rect.x() == m.x
                and rect.y() == m.y
                and rect.width() == m.width
                and rect.height() == m.height
            ):
                geom = screen.availableGeometry()

        return geom.x(), geom.y(), geom.width(), geom.height()

    def screen_update(self, val, event):
        # Get current monitor coordinates
        prev_monitor = self.selected_monitor
        # Get current mouse cursor position in relation to window origin
        prev_mouse_pos = QCursor.pos() - self.geometry().topLeft()

        # Set new monitor
        self.set_monitor(val)
        self._update_monitors_buttons()
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
        """ Automatically arranges and resizes windows """
        ax, ay, aw, ah = self._avail_screenspace()

        main_frame = self.frameGeometry()
        main_w, main_h = main_frame.width(), main_frame.height()

        x_main = ax + (aw - main_w) // 2
        y_main = ay + ah - main_h

        self.move(x_main, y_main)

        # if 3D viewer exists and is visible, place it immediately to the right or main windowa
        if self.central_calib_window and self.central_calib_window.isVisible():
            try:
                # if the 3D viewer has its own auto_size()
                self.central_calib_window.auto_size()
            except AttributeError:
                pass

            ogl_frame = self.central_calib_window.frameGeometry()
            ogl_w, ogl_h = ogl_frame.width(), ogl_frame.height()
            x_ogl = x_main + main_w + SPACING
            y_ogl = ay + ah - ogl_h

            self.central_calib_window.move(x_ogl, y_ogl)

        for w in self.visible_windows(include_main=False):
            if w is self.central_calib_window:
                # we just positioned it already
                continue
            try:
                w.auto_size()
                w.snap()
            except Exception:
                # if a window doesn't implement these we just skip it
                pass

    def cascade_windows(self):
        """ Cascade all visible windows """

        # adjust stacking order
        self.raise_()  # Main window on top

        ax, ay, aw, ah = self._avail_screenspace()

        for win in self.visible_windows(include_main=True):
            fg = win.frameGeometry()
            w, h = fg.width(), fg.height()

            # if main window, offset by nb of secondaries * 30 px
            if win is self:
                count_secondaries = len(self.visible_windows(include_main=False))
                dx = 30 * count_secondaries + 30
                dy = 30 * count_secondaries + 30
            else:
                # otherwise use each window idx to stagger them
                dx = 30 * win.idx + 30
                dy = 30 * win.idx + 30

            # constrain to available space
            new_x = min(max(ax, ax + dx), ax + aw - w)
            new_y = min(max(ay, ay + dy), ay + ah - h)

            win.move(new_x, new_y)

    def auto_size(self):
        # Do nothing on the main window
        pass

    def _update_monitors_buttons(self):
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
        windows = [w for w in self.video_windows if w.isVisible()]
        if self.central_calib_window:
            windows += [self.central_calib_window]
        if include_main:
            windows += [self]
        return windows

    def _start_secondary_windows(self):
        if self.is_calibrating:

            # Create the 3D visualization window first
            self.central_calib_window = CentralCalibrationWindow(self)
            self.central_calib_window.show()

            # self.opengl_window.request_board_settings.connect(self.show_global_board_dialog)
            self.central_calib_window.request_load_calibration.connect(self.on_load_calibration)

            origin_name = self.central_calib_window.origin_camera_combo.currentText()

            # Create and start the headless Multiview worker with the correct origin name.
            self.multiview_thread = QThread(self)
            self.multiview_worker = MultiviewWorker(
                self.cameras_names,
                origin_name,
                self.sources_shapes,
                self.board_params
            )
            self.multiview_worker.moveToThread(self.multiview_thread)

            self.multiview_worker.scene_data_ready.connect(self.central_calib_window.on_scene_data_ready)
            self.central_calib_window.run_ba_button.clicked.connect(self.multiview_worker.trigger_refinement)

            self.multiview_thread.start()

            # register the new worker with the coordinator
            self.coordinator.register_worker(self.multiview_worker)

        # Create Monocular/Recording windows and their workers
        for i, cam in enumerate(self.manager.cameras):
            if self.is_calibrating:
                # Pass the global board_params
                w = CalibrationLiveView(cam, self, self.board_params)
                # Register the monocular worker with the coordinator
                self.coordinator.register_worker(w.worker)
            else:
                w = RecordingLiveView(cam, self)

            self.video_windows.append(w)
            self.secondary_windows_visibility_buttons[i].setText(f" {w.name.title()} camera")
            self.secondary_windows_visibility_buttons[i].setStyleSheet(
                f"border-radius: 5px; padding: 0 10 0 10; color: {w.secondary_colour}; background-color: {w.colour};")
            self.secondary_windows_visibility_buttons[i].clicked.connect(w.toggle_visibility)
            self.secondary_windows_visibility_buttons[i].setChecked(True)

            w.show()

        self.cascade_windows()

    def _stop_secondary_windows(self):

        for w in self.video_windows:
            w._force_destroy = True  # signal the window it should really close
            w.close()   # and trigger the closeEvent

        if self.central_calib_window:
            self.central_calib_window._force_destroy = True
            self.central_calib_window.close()

        # clear the lists
        self.video_windows.clear()
        self.central_calib_window = None

        if self.multiview_thread:
            self.multiview_thread.quit()
            self.multiview_thread.wait()
            self.multiview_thread = None
            self.multiview_worker = None

    def _update_main(self):
        # TODO: Reimplement these but faster
        pass
        # # get an estimation of the saved data size
        # try:
        #     # This works for both image sequences and video files
        #     size = sum(f.stat().st_size for f in self.manager.full_path.glob('**/*') if f.is_file())
        #     # self.frames_saved_label.setText(f'Saved frames: {self.manager.saved} ({pretty_size(size)})')
        #     self.frames_saved_label.setText(f'Saved: ({pretty_size(size)})')
        #
        # except FileNotFoundError:
        #     # This can happen if the folder doesn't exist yet
        #     # self.frames_saved_label.setText(f'Saved frames: {self.manager.saved} (0 bytes)')
        #     self.frames_saved_label.setText(f'Saved: (0 bytes)')
        #
        # # Update memory pressure estimation
        # self._mem_pressure += (psutil.virtual_memory().percent - self._mem_baseline) / self._mem_baseline * 100
        # self._mem_pressure_bar.setValue(int(round(self._mem_pressure)))
        # self._mem_baseline = psutil.virtual_memory().percent
