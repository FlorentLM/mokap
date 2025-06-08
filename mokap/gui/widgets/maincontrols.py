import os
import platform
import subprocess
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple

import psutil
import screeninfo
from PIL import Image
from PySide6.QtCore import QTimer, Qt, Slot, QRect, QThread
from PySide6.QtGui import QIcon, QFont, QGuiApplication, QCursor, QBrush, QColor, QPen
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QFrame, QHBoxLayout, QLabel, QComboBox, QPushButton, \
    QSizePolicy, QGroupBox, QLineEdit, QScrollArea, QCheckBox, QGraphicsView, QGraphicsScene, QTextEdit, QStatusBar, \
    QProgressBar, QFileDialog, QApplication, QDialog, QGraphicsRectItem, QGraphicsTextItem

from mokap.gui.widgets import DEFAULT_BOARD, SLOW_UPDATE, GUI_LOGGER
from mokap.gui.widgets.base import SnapMixin
from mokap.gui.widgets.dialogs import BoardParamsDialog
from mokap.gui.widgets.liveviews import Monocular, Recording
from mokap.gui.widgets.opengl import Multiview3D
from mokap.utils import hex_to_rgb, hex_to_hls, pretty_size


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


class MainControls(QMainWindow, SnapMixin):
    INFO_PANEL_MINSIZE_H = 200
    VIDEO_PANEL_MINSIZE_H = 50  # haha
    WINDOW_MIN_W = 630

    # Colours
    col_white = "#ffffff"
    col_white_rgb = hex_to_rgb(col_white)
    col_black = "#000000"
    col_black_rgb = hex_to_rgb(col_black)
    col_lightgray = "#e3e3e3"
    col_lightgray_rgb = hex_to_rgb(col_lightgray)
    col_midgray = "#c0c0c0"
    col_midgray_rgb = hex_to_rgb(col_midgray)
    col_darkgray = "#515151"
    col_darkgray_rgb = hex_to_rgb(col_darkgray)
    col_red = "#FF3C3C"
    col_red_rgb = hex_to_rgb(col_red)
    col_darkred = "#bc2020"
    col_darkred_rgb = hex_to_rgb(col_darkred)
    col_orange = "#FF9B32"
    col_orange_rgb = hex_to_rgb(col_orange)
    col_darkorange = "#cb782d"
    col_darkorange_rgb = hex_to_rgb(col_darkorange)
    col_yellow = "#FFEB1E"
    col_yellow_rgb = hex_to_rgb(col_yellow)
    col_yelgreen = "#A5EB14"
    col_yelgreen_rgb = hex_to_rgb(col_yelgreen)
    col_green = "#00E655"
    col_green_rgb = hex_to_rgb(col_green)
    col_darkgreen = "#39bd50"
    col_darkgreen_rgb = hex_to_rgb(col_green)
    col_blue = "#5ac3f5"
    col_blue_rgb = hex_to_rgb(col_blue)
    col_purple = "#c887ff"
    col_purple_rgb = hex_to_rgb(col_purple)

    def __init__(self, mc):
        super().__init__()

        self.setWindowTitle('Controls')
        if GUI_LOGGER:
            self.gui_logger = GUILogger()
        else:
            self.gui_logger = False

        self.mc = mc

        self.board_params = DEFAULT_BOARD

        self.coordinator = CalibrationCoordinator()
        self.multiview_worker = None  # Will be created in _start_secondary_windows
        self.multiview_thread = None

        self.nb_cams = self.mc.nb_cameras
        self._cameras_names = tuple(cam.name for cam in self.mc.cameras)     # This order is fixed

        # Set cameras info
        self.sources_shapes = {cam.name: np.array(cam.shape)[:2] for cam in self.mc.cameras}
        self.cams_colours = {cam.name: f'#{self.mc.colours[cam.name].lstrip("#")}' for cam in self.mc.cameras}
        self.secondary_colours = {k: self.col_white if hex_to_hls(v)[1] < 60 else self.col_black for k, v in
                                  self.cams_colours.items()}

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
        self.icon_move_bw = QIcon((resources_path / 'move.png').as_posix())     # TODO make an icon - this is a temp one

        # States
        self.is_editing = False
        self.calibration_stage = 0
        self.is_calibrating = False

        self._recording_text = ''

        # Refs for the secondary windows
        self.opengl_window = None
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
        self.timer_slow.start(SLOW_UPDATE)

        self._mem_baseline = psutil.virtual_memory().percent

    @property
    def cameras_names(self):
        # Return a copy of the tuple, as a list
        return list(self._cameras_names)

    def init_gui(self):
        self.MAIN_LAYOUT = QVBoxLayout()
        self.MAIN_LAYOUT.setContentsMargins(5, 5, 5, 5)
        self.MAIN_LAYOUT.setSpacing(5)

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
        self.mode_combo.currentIndexChanged.connect(self._toggle_calib_record)
        toolbar_layout.addWidget(self.mode_combo, 1)    # 1 unit
        toolbar_layout.addStretch(2)    # spacing of 2 units

        # Exit button
        self.button_exit = QPushButton("Exit (Esc)")
        self.button_exit.clicked.connect(self.quit)
        self.button_exit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.button_exit.setStyleSheet(f"background-color: {self.col_red}; color: {self.col_white};")
        toolbar_layout.addWidget(self.button_exit)

        self.MAIN_LAYOUT.addWidget(toolbar)
        # End toolbar

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

        # Now that the UI is ready, refresh the monitors buttons
        self._update_monitors_buttons()

    @Slot(CalibrationData)
    def route_payload_to_widgets(self, data: CalibrationData):
        target_name = data.camera_name
        # Route to the specific monocular window
        for w in self.video_windows:
            if isinstance(w, Monocular) and w.name == target_name:
                w.handle_payload(data)
                break

        # Always send extrinsics/intrinsics to the 3D view
        if self.opengl_window and isinstance(data.payload, (IntrinsicsPayload, ExtrinsicsPayload)):
            self.opengl_window.handle_payload(data)

    @Slot(CalibrationData)
    def route_payload_to_widgets(self, data: CalibrationData):
        """
        Receives ALL data from the coordinator destined for the UI
        """
        target_name = data.camera_name
        payload = data.payload

        # Update the monocular windows
        for w in self.video_windows:
            if isinstance(w, Monocular) and w.name == target_name:
                w.handle_payload(data)

        # Update the 3D view
        if self.opengl_window and isinstance(payload, (IntrinsicsPayload, ExtrinsicsPayload)):
            self.opengl_window.handle_payload(data)

    @Slot()
    def on_load_calibration(self):
        """
        Handles loading a single calibration file containing data for all cameras
        """

        # Ensure we are in the right mode (should be impossible to call from recording mode anyway)
        if not self.is_calibrating:
            print("[MainControls] Must be in Calibration mode to load calibration.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load calibration file",
            str(self.mc.full_path.resolve()),
            "TOML Files (*.toml)"
        )

        if not file_path:
            return

        print(f"[MainControls] Loading calibration from: {file_path}")

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

            # move to the Extrinsics stage
            # self.opengl_window.calibration_stage_combo.setCurrentIndex(1)

        except Exception as e:
            print(f"[MainControls] Error loading calibration file: {e}")

    #  ============= Qt method overrides =============
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

    #  ============= General toggles =============
    def _toggle_calib_record(self):

        if self.is_calibrating and self.mode_combo.currentIndex() == 0:
            self.is_calibrating = False

            self._stop_secondary_windows()

            if self.mc.acquiring:
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

    def _toggle_text_editing(self, override=None):

        if override is None:
            override = self.is_editing

        if not self.is_editing and override is True:
            self.acq_name_textbox.setDisabled(False)
            self.acq_name_edit_btn.setText('Set')
            self.is_editing = False

        elif self.is_editing and override is False:
            self.acq_name_textbox.setDisabled(True)
            self.acq_name_edit_btn.setText('Edit')
            self.mc.session_name = self.acq_name_textbox.text()
            self.save_dir_current.setText(f'{self.mc.full_path.resolve()}')
            self.is_editing = True

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
            if not self.is_calibrating and self.mc.triggered:
                for w in self.video_windows:
                    w.camera_controls_sliders['framerate'].setDisabled(True)

        elif not self.mc.acquiring and override is True:
            self.mc.on()
            if not self.is_calibrating and self.mc.triggered:
                for w in self.video_windows:
                    w.camera_controls_sliders['framerate'].setDisabled(True)
            self.save_dir_current.setText(f'{self.mc.full_path.resolve()}')
            self.button_acquisition.setText("Acquiring")
            self.button_acquisition.setIcon(self.icon_capture)
            self.button_snapshot.setDisabled(False)
            if not self.is_calibrating:
                self.button_recpause.setDisabled(False)

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

    def _take_snapshot(self):
        """
        Takes an instantaneous snapshot from all cameras
        """
        now = datetime.now().strftime('%y%m%d-%H%M%S')
        if self.mc.acquiring:
            arrays = self.mc.get_current_framebuffer()
            for i, arr in enumerate(arrays):
                img = Image.fromarray(arr, mode='RGB' if arr.ndim == 3 else 'L')
                img.save(self.mc.full_path.resolve() / f"snapshot_{now}_{self.mc.cameras[i].name}.bmp")

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

    def show_global_board_dialog(self):
        dlg = BoardParamsDialog(self.board_params, self)
        if dlg.exec_() == QDialog.Accepted:
            new_board_params = dlg.get_values()
            self.board_params = new_board_params

            if self.opengl_window:
                self.opengl_window.update_board_preview(self.board_params)

            # Send the new board to the coordinator to trigger a system reset
            self.coordinator.handle_board_change(self.board_params)

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
        """
        Automatically arranges and resizes windows
        """
        ax, ay, aw, ah = self._avail_screenspace()

        main_frame = self.frameGeometry()
        main_w, main_h = main_frame.width(), main_frame.height()

        x_main = ax + (aw - main_w) // 2
        y_main = ay + ah - main_h

        self.move(x_main, y_main)

        # if 3D viewer exists and is visible, place it immediately to the right or main windowa
        if self.opengl_window and self.opengl_window.isVisible():
            try:
                # if the 3D viewer has its own auto_size()
                self.opengl_window.auto_size()
            except AttributeError:
                pass

            ogl_frame = self.opengl_window.frameGeometry()
            ogl_w, ogl_h = ogl_frame.width(), ogl_frame.height()
            x_ogl = x_main + main_w + SPACING
            y_ogl = ay + ah - ogl_h

            self.opengl_window.move(x_ogl, y_ogl)

        for w in self.visible_windows(include_main=False):
            if w is self.opengl_window:
                # we just positioned it already
                continue
            try:
                w.auto_size()
                w.snap()
            except Exception:
                # if a window doesn't implement these we just skip it
                pass

    def cascade_windows(self):
        """
        Cascade all visible windows
        """

        # adjust stacking order
        self.raise_()  # Main window on top
        if self.opengl_window and self.opengl_window.isVisible():
            self.opengl_window.lower()  # 3D viewer at back

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
        if self.opengl_window:
            windows += [self.opengl_window]
        if include_main:
            windows += [self]
        return windows

    def _start_secondary_windows(self):
        if self.is_calibrating:

            # Create the 3D visualization window first
            self.opengl_window = Multiview3D(self)
            # now that the window exists, set the board params and get the origin camera name
            self.opengl_window.update_board_preview(self.board_params)

            self.opengl_window.request_board_settings.connect(self.show_global_board_dialog)
            self.opengl_window.request_load_calibration.connect(self.on_load_calibration)

            origin_name = self.opengl_window.origin_camera_combo.currentText()

            # Create and start the headless Multiview worker with the correct origin name.
            self.multiview_thread = QThread(self)
            self.multiview_worker = MultiviewWorker(self.cameras_names, origin_name)
            self.multiview_worker.moveToThread(self.multiview_thread)
            self.multiview_thread.start()

            # register the new worker with the coordinator
            self.coordinator.register_worker(self.multiview_worker)

            # Connect the 3D window's BA button to the worker's slot
            self.opengl_window.run_ba_button.clicked.connect(self.multiview_worker.run_bundle_adjustment)
            self.opengl_window.show()

        # Create Monocular/Recording windows and their workers
        for i, cam in enumerate(self.mc.cameras):
            if self.is_calibrating:
                # Pass the global board_params
                w = Monocular(cam, self, self.board_params)
                # Register the monocular worker with the coordinator
                self.coordinator.register_worker(w.worker)
            else:
                w = Recording(cam, self)

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
            w.worker_thread.quit()
            w.worker_thread.wait()
            w._force_destroy = True
            w.close()

        if self.opengl_window:
            self.opengl_window.worker_thread.quit()
            self.opengl_window.worker_thread.wait()
            self.opengl_window.close()
            self.opengl_window.timer_slow.stop()
            self.opengl_window._force_destroy = True
            self.opengl_window.deleteLater()
            self.opengl_window = None

        if self.multiview_thread:
            self.multiview_thread.quit()
            self.multiview_thread.wait()
            self.multiview_thread = None
            self.multiview_worker = None

        self.video_windows.clear()

    def _update_main(self):

        # Get an estimation of the saved data size
        if self.mc._estim_file_size is None:
            self.frames_saved_label.setText(f'Saved frames: {self.mc.saved} (0 bytes)')

        elif self.mc._estim_file_size == -1:
            size = sum(sum(os.path.getsize(os.path.join(res[0], element)) for element in res[2]) for res in
                       os.walk(self.mc.full_path))
            self.frames_saved_label.setText(f'Saved frames: {self.mc.saved} ({pretty_size(size)})')
        else:
            saved = self.mc.saved
            size = sum(self.mc._estim_file_size * saved)
            self.frames_saved_label.setText(f'Saved frames: {saved} ({pretty_size(size)})')

        # Update memory pressure estimation
        self._mem_pressure += (psutil.virtual_memory().percent - self._mem_baseline) / self._mem_baseline * 100
        self._mem_pressure_bar.setValue(int(round(self._mem_pressure)))
        self._mem_baseline = psutil.virtual_memory().percent
