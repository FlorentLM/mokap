from collections import deque
from pathlib import Path
import numpy as np

import cv2
import pyqtgraph as pg
from PySide6.QtCore import Qt, QEvent, Slot, Signal
from PySide6.QtWidgets import QHBoxLayout, QWidget, QVBoxLayout, QGroupBox, QLabel, QSlider, QCheckBox, QSizePolicy, \
    QPushButton, QFileDialog

from mokap.gui.widgets import MAX_PLOT_X
from mokap.gui.widgets.base import PreviewBase
from mokap.gui.workers.monocular import MonocularWorker
from mokap.gui.workers.movement import MovementWorker
from mokap.utils.datatypes import ChessBoard, CharucoBoard, ErrorsPayload, CalibrationData, IntrinsicsPayload


class Recording(PreviewBase):

    def __init__(self, camera, main_window_ref):
        super().__init__(camera, main_window_ref)

        self._n_enabled = False
        self._magnifier_enabled = False

        # Magnification parameters
        self.magn_window_w = 100
        self.magn_window_h = 100
        self.magn_window_x = 10
        self.magn_window_y = 10
        self.magn_target_cx = 0.5   # Target initialised at the centre
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

        self._setup_worker(MovementWorker(self._cam_name))

        # Store worker results and its current state
        self._bboxes = []

        # Finish building the UI by calling the other constructors
        self._init_common_ui()
        self._init_specific_ui()
        self.auto_size()

        # Start worker and timers
        self.worker_thread.start()
        self._start_timers()

    def _init_specific_ui(self):
        """
        This constructor creates the UI elements specific to Recording mode
        """

        # Add mouse click detection to video feed (for the magnifier)
        self.VIDEO_FEED.installEventFilter(self)

        # RIGHT GROUP
        right_group_layout = QHBoxLayout(self.RIGHT_GROUP)
        right_group_layout.setContentsMargins(5, 5, 5, 5)

        self.camera_controls_sliders = {}
        self.camera_controls_sliders_labels = {}
        self.camera_controls_sliders_scales = {}
        self._val_in_sync = {}

        slider_params = [
            ('framerate', (int, 1, int(self._mainwindow.mc.cameras[self.idx].max_framerate), 1, 1)),
            ('exposure', (int, 21, 100000, 5, 1)),  # in microseconds - 100000 microseconds ~ 10 fps
            ('blacks', (float, 0.0, 32.0, 0.5, 3)),
            ('gain', (float, 0.0, 36.0, 0.5, 3)),
            ('gamma', (float, 0.0, 3.99, 0.05, 3))
        ]

        right_group_sliders = QWidget()
        right_group_sliders_layout = QVBoxLayout(right_group_sliders)
        right_group_sliders_layout.setContentsMargins(0, 20, 0, 5)
        right_group_sliders_layout.setSpacing(0)

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

            param_value = getattr(self._mainwindow.mc.cameras[self.idx], label)

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

            right_group_sliders_layout.addWidget(line)

        right_group_layout.addWidget(right_group_sliders)
        right_group_layout.addWidget(sync_groupbox)

        # RIGHT GROUP - Additional elements
        right_group_additional = QWidget()
        right_group_additional_layout = QVBoxLayout(right_group_additional)
        right_group_additional_layout.setContentsMargins(0, 20, 0, 5)
        right_group_additional_layout.setSpacing(0)

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

        right_group_additional_layout.addWidget(line)
        right_group_layout.addWidget(right_group_additional)

    #  ============= Qt method overrides =============
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

    def _annotate(self):

        # Get new coordinates
        h, w = self._display_buffer.shape[:2]

        x_centre, y_centre = w // 2, h // 2
        x_north, y_north = w // 2, 0
        x_south, y_south = w // 2, h
        x_east, y_east = w, h // 2
        x_west, y_west = 0, h // 2

        # Draw crosshair
        cv2.line(self._display_buffer, (x_west, y_west), (x_east, y_east), self._mainwindow.col_white_rgb, 1)
        cv2.line(self._display_buffer, (x_north, y_north), (x_south, y_south), self._mainwindow.col_white_rgb, 1)

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
                                                 self._mainwindow.col_yellow_rgb, 1)

            # Paste the zoom window into the display buffer
            magn_x1 = min(self._display_buffer.shape[0], max(0, self.magn_window_x))
            magn_y1 = min(self._display_buffer.shape[1], max(0, self.magn_window_y))
            magn_x2 = min(self._display_buffer.shape[0], magn_x1 + magn_img.shape[0])
            magn_y2 = min(self._display_buffer.shape[1], magn_y1 + magn_img.shape[1])
            self._display_buffer[magn_x1:magn_x2, magn_y1:magn_y2, :] = magn_img[:magn_x2 - magn_x1, :magn_y2 - magn_y1, :]

            # Add frame around the magnification
            self._display_buffer = cv2.rectangle(self._display_buffer,
                                                 (magn_y1, magn_x1), (magn_y2, magn_x2),
                                                 self._mainwindow.col_yellow_rgb, 1)

        # Position the 'Recording' indicator
        font, txtsiz, txtth = cv2.FONT_HERSHEY_DUPLEX, 1.0, 2
        textsize = cv2.getTextSize(self._mainwindow._recording_text, font, txtsiz, txtth)[0]
        self._display_buffer = cv2.putText(self._display_buffer, self._mainwindow._recording_text,
                                           (int(x_centre - textsize[0] / 2), int(1.5 * y_centre - textsize[1])),
                                           font, txtsiz, self._mainwindow.col_red_rgb, txtth, cv2.LINE_AA)

        # Position the 'Warning' indicator
        if self._warning:
            textsize = cv2.getTextSize(self._warning_text, font, txtsiz, txtth)[0]
            self._display_buffer = cv2.putText(self._display_buffer, self._warning_text,
                                               (int(x_north - textsize[0] / 2), int(y_centre / 2 - textsize[1])),
                                               font, txtsiz, self._mainwindow.col_orange_rgb, txtth, cv2.LINE_AA)

    def _update_fast(self):
        # Grab camera frame
        self._refresh_framebuffer()
        frame = self._frame_buffer
        # frame = self._frame_buffer.copy()

        # if worker is free, send frame
        if not self._worker_busy:
            self._send_frame_to_worker(frame)
        else:
            self._latest_frame = frame  # We overwrite the latest_frame purposefully, no need to queue them

        # Resize viewport to fit current window size
        self._resize_to_display()

        # Resize the image to the new viewport size
        disp_h, disp_w = self._display_buffer.shape[:2]
        scale = min(disp_w / frame.shape[1], disp_h / frame.shape[0])
        display_img = cv2.resize(frame, (0,0), fx=scale, fy=scale)

        # Paste new image into the viewport
        h, w = display_img.shape[:2]
        self._display_buffer[:h, :w] = display_img

        # And fill any remaining viewport space with black
        if h < disp_h:
            self._display_buffer[h:, :] = 0
        if w < disp_w:
            self._display_buffer[:, w:] = 0

        # Fake bounding box
        # for (x, y, bw, bh) in self._bboxes:
        #     sx, sy = int(x*scale), int(y*scale)
        #     sw, sh = int(bw*scale), int(bh*scale)
        #     cv2.rectangle(self._display_buffer, (sx, sy), (sx+sw, sy+sh),
        #                   (0,255,255), 2)

        # Add annotations
        self._annotate()

        # And blit
        self._blit_image()

    @Slot(list)
    def on_worker_result(self, bboxes):
        self._bboxes = bboxes

    #  ============= Recording mode-specific functions =============
    def _toggle_n_display(self):
        if self._n_enabled:
            self.n_button.setStyleSheet('')
            self._n_enabled = False
        else:
            self.n_button.setStyleSheet(f'background-color: #80{self._mainwindow.col_green.lstrip("#")};')
            self._n_enabled = True

    def _toggle_mag_display(self):
        if self._magnifier_enabled:
            self.magn_button.setStyleSheet('')
            # self.magn_slider.setDisabled(True)
            self._magnifier_enabled = False
        else:
            self.magn_button.setStyleSheet(f'background-color: #80{self._mainwindow.col_yellow.lstrip("#")};')
            # self.magn_slider.setDisabled(False)
            self._magnifier_enabled = True

    def update_param(self, label):
        if label == 'framerate' and self._mainwindow.mc.triggered and self._mainwindow.mc.acquiring:
            return

        slider = self.camera_controls_sliders[label]

        new_val_float = slider.value() / self.camera_controls_sliders_scales[label]

        setattr(self._mainwindow.mc.cameras[self.idx], label, new_val_float)

        # And update the slider to the actual new value (can be different from the one requested)
        read_back = getattr(self._mainwindow.mc.cameras[self.idx], label)

        actual_new_val = int(read_back * self.camera_controls_sliders_scales[label])
        slider.setValue(actual_new_val)

        if label == 'exposure':
            # Refresh exposure value for UI display
            self.exposure_value.setText(f"{self._mainwindow.mc.cameras[self.idx].exposure} µs")

            # We also need to update the framerate slider to current resulting fps after exposure change
            self.update_param('framerate')

        elif label == 'framerate':
            # Keep a local copy to warn user if actual framerate is too different from requested fps
            wanted_fps_val = slider.value() / self.camera_controls_sliders_scales[label]
            self._wanted_fps = wanted_fps_val

            if self._mainwindow.mc.triggered:
                self._mainwindow.mc.framerate = self._wanted_fps
            else:
                self._mainwindow.mc.cameras[self.idx].framerate = self._wanted_fps

            new_max = int(self._mainwindow.mc.cameras[self.idx].max_framerate * self.camera_controls_sliders_scales[label])
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

            for window in self._mainwindow.video_windows:
                if window is not self and bool(window._val_in_sync[label].isChecked()):
                    # Apply the window's scale (which should be the same anyway)
                    w_new_val = int(new_val_float * window.camera_controls_sliders_scales[label])
                    window.camera_controls_sliders[label].setValue(w_new_val)
                    window.update_param(label)


class Monocular(PreviewBase):
    request_load = Signal(str)
    request_save = Signal(str)

    def __init__(self, camera, main_window_ref, board_params):
        super().__init__(camera, main_window_ref)

        # Initialize reprojection error data for plotting
        self.reprojection_errors = deque(maxlen=MAX_PLOT_X)

        # The worker annotates the frame by itself so we keep a reference to the latest annotated frame
        self.annotated_frame = None

        # Registration is handled by MainControls
        self._setup_worker(MonocularWorker(board_params, self.idx, self.name, self.source_shape))

        # Finish building the UI by calling the other constructors
        self._init_common_ui()
        self._init_specific_ui()
        self.auto_size()

        self.worker_thread.start()
        self._start_timers()

    def _setup_worker(self, worker_object):
        super()._setup_worker(worker_object)
        # Connect signals for loading/saving which are specific to this worker
        self.request_load.connect(worker_object.load_intrinsics)
        self.request_save.connect(worker_object.save_intrinsics)

    def _init_specific_ui(self):
        """
        This constructor creates the UI elements specific to Calib mode
        """
        layout = QHBoxLayout(self.RIGHT_GROUP)
        layout.setContentsMargins(5, 5, 5, 5)

        # Detection and sampling
        sampling_group = QWidget()
        sampling_layout = QVBoxLayout(sampling_group)

        self.auto_sample_check = QCheckBox("Sample automatically")
        self.auto_sample_check.setChecked(True)
        self.auto_sample_check.stateChanged.connect(self.worker.auto_sample)
        sampling_layout.addWidget(self.auto_sample_check)

        sampling_btns_group = QWidget()
        sampling_btns_layout = QHBoxLayout(sampling_btns_group)

        self.sample_button = QPushButton("Add sample")
        self.sample_button.clicked.connect(self.worker.add_sample)
        self.sample_button.setStyleSheet(f"background-color: {self._mainwindow.col_darkgreen}; color: {self._mainwindow.col_white};")
        sampling_btns_layout.addWidget(self.sample_button)

        self.clear_samples_button = QPushButton("Clear samples")
        self.clear_samples_button.clicked.connect(self.worker.clear_samples)
        sampling_btns_layout.addWidget(self.clear_samples_button)

        sampling_layout.addWidget(sampling_btns_group)

        self.auto_compute_check = QCheckBox("Compute intrinsics automatically")
        self.auto_compute_check.setChecked(True)
        self.auto_compute_check.stateChanged.connect(self.worker.auto_compute)
        sampling_layout.addWidget(self.auto_compute_check)

        intrinsics_btns_group = QWidget()
        intrinsics_btns_layout = QHBoxLayout(intrinsics_btns_group)

        self.compute_intrinsics_button = QPushButton("Compute intrinsics now")
        self.compute_intrinsics_button.clicked.connect(self.worker.compute_intrinsics)
        intrinsics_btns_layout.addWidget(self.compute_intrinsics_button)

        self.clear_intrinsics_button = QPushButton("Clear intrinsics")
        self.clear_intrinsics_button.clicked.connect(self.on_clear_intrinsics)      # Main thread to main thread UI updt
        self.clear_intrinsics_button.clicked.connect(self.worker.clear_intrinsics)  # Send signal to worker
        intrinsics_btns_layout.addWidget(self.clear_intrinsics_button)

        sampling_layout.addWidget(intrinsics_btns_group)

        layout.addWidget(sampling_group)

        # Reprojection Error Plot
        self.error_plot = pg.PlotWidget(title="Reprojection Error")
        self.error_plot.setStyleSheet("background-color: black;")

        self.error_plot.setLabel('left', 'Error (pixels)')
        self.error_plot.setLabel('bottom', 'Sample')
        self.error_plot.showGrid(x=True, y=True)
        self.error_plot_curve = self.error_plot.plot(pen=pg.mkPen(color='y', width=2))
        self.error_plot.setYRange(0.0, 5.0)

        self.video_container_layout.addWidget(self.error_plot, 1)

        calib_io_group = QGroupBox("Load/Save")
        calib_io_group.setMinimumWidth(250)
        calib_io_group.setMaximumWidth(250)
        calib_io_layout = QVBoxLayout(calib_io_group)

        self.load_calib_button = QPushButton("Load intrinsics")
        self.load_calib_button.clicked.connect(self.on_load_parameters)
        calib_io_layout.addWidget(self.load_calib_button)

        self.save_calib_button = QPushButton("Save intrinsics")
        self.save_calib_button.clicked.connect(self.on_save_parameters)
        calib_io_layout.addWidget(self.save_calib_button)

        self.load_save_message = QLabel("")
        self.load_save_message.setMaximumWidth(180)
        self.load_save_message.setWordWrap(True)
        calib_io_layout.addWidget(self.load_save_message)

        layout.addWidget(calib_io_group)

    def _update_fast(self):
        self._refresh_framebuffer()
        frame = self._frame_buffer

        if not self._worker_busy:
            self._send_frame_to_worker(frame)
        else:
            self._latest_frame = frame      # We overwrite the latest_frame purposefully, no need to queue them

        self._resize_to_display()

        # Now scale + display the last annotated frame we got from the worker
        disp_h, disp_w = self._display_buffer.shape[:2]

        if self._worker_blocking:
            # Worker is blocking during computation so it can't annotate
            h, w = frame.shape[:2]
            text = "Computing..."
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 6
            text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            self.annotated_frame = frame

        if self.annotated_frame is not None and self.annotated_frame.size > 0:

            # Resize image to current viewport dimensions
            scale = min(disp_w / self.annotated_frame.shape[1], disp_h / self.annotated_frame.shape[0])
            out = cv2.resize(self.annotated_frame, (0, 0), fx=scale, fy=scale)

            # Paste resized image into display buffer
            h, w = out.shape[:2]
            self._display_buffer[:h, :w] = out
            # Fill any remaining buffer space with black
            if h < disp_h:
                self._display_buffer[h:, :] = 0
            if w < disp_w:
                self._display_buffer[:, w:] = 0
        else:
            self._display_buffer.fill(0)

        self._blit_image()

    @Slot(CalibrationData)
    def handle_payload(self, data: CalibrationData):
        """ Receives data from the main window's router """

        if isinstance(data.payload, ErrorsPayload):
            mean_error = np.mean(data.payload.errors)
            self.reprojection_errors.append(mean_error)
            self.error_plot_curve.setData(list(self.reprojection_errors))
            self.load_save_message.setText(f"Intrinsics updated. Mean err: {mean_error:.3f} px")

        elif isinstance(data.payload, IntrinsicsPayload):
            self.load_save_message.setText(f"Intrinsics updated from {data.camera_name}.")
        # Extrinsics are handled by the worker for visualization, no UI update needed here

    def on_save_parameters(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Intrinsics",
            str(self._mainwindow.mc.full_path.resolve()),
            "TOML Files (*.toml)")

        if file_path:
            self.request_save.emit(file_path)
            self.load_save_message.setText(f"Intrinsics saved to\n{Path(file_path).name}")

    @Slot(np.ndarray)
    def on_worker_result(self, annotated_frame):
        self.annotated_frame = annotated_frame

    def on_clear_intrinsics(self):
        # Clear plot
        self.reprojection_errors.clear()
        self.error_plot_curve.setData(self.reprojection_errors)
        # Clear text
        self.load_save_message.setText('')

    def on_load_parameters(self):
        file_path = self._show_file_dialog(self._mainwindow.mc.full_path.parent)
        if file_path and file_path.is_file():   # Might still be None if the picker did not succeed
            self.request_load.emit(file_path.as_posix())
            self.load_save_message.setText(f"Intrinsics <b>loaded</b> from {file_path.parent}")

    #  ============= Calibration video window functions =============
    def _show_file_dialog(self, startpath: str | Path):
        """
        Opens the small file selector to let the user load a parameters.toml file
        """
        dial = QFileDialog(self)
        dial.setWindowTitle("Choose folder")
        dial.setViewMode(QFileDialog.ViewMode.Detail)
        dial.setDirectory(Path(startpath).resolve().as_posix())

        if dial.exec():
            selected = dial.selectedFiles()
            if selected:
                file = Path(selected[0])
                if file.exists():
                    return file
        return None
