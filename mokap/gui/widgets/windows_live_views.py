from collections import deque
from typing import Tuple
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Slot, Signal, QEvent
from PySide6.QtWidgets import (QHBoxLayout, QWidget, QVBoxLayout, QGroupBox, QLabel,
                               QSlider, QCheckBox, QSizePolicy, QPushButton, QFileDialog,
                               QGraphicsRectItem, QGraphicsItemGroup)
from mokap.gui.style.commons import *
from mokap.gui.widgets import MAX_PLOT_X
from mokap.gui.widgets.widgets_base import LiveViewBase, FastImageItem
from mokap.gui.workers.worker_monocular import MonocularWorker
from mokap.gui.workers.worker_movement import MovementWorker
from mokap.utils import pretty_microseconds
from mokap.utils.datatypes import (ErrorsPayload, CalibrationData, IntrinsicsPayload, ExtrinsicsPayload,
                                   DetectionPayload, ReprojectionPayload, CoveragePayload)


class RecordingLiveView(LiveViewBase):

    def __init__(self, camera, main_window_ref):
        super().__init__(camera, main_window_ref)

        # Magnification parameters
        self.magn_window_w = 100
        self.magn_window_h = 100
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

        # Store parameters for logarithmic sliders
        self.log_slider_params = {}

        # Finish building the UI by calling the other constructors
        self._init_common_ui()
        self._init_specific_ui()
        self.auto_size()

        # Start worker and timers
        self.worker_thread.start()
        self._start_timers()

    def _init_specific_ui(self):
        """ This constructor creates the UI elements specific to Recording mode """

        # --- PyQtGraph overlays ----

        crosshair_pen = pg.mkPen(color='w', style=Qt.DotLine)
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=crosshair_pen)
        self.v_line.setPos(self.source_shape_hw[1] / 2)
        self.h_line.setPos(self.source_shape_hw[0] / 2)

        # we add them to the viewbox not the imageitem (they live on top of the image)
        self.view_box.addItem(self.v_line)
        self.view_box.addItem(self.h_line)

        # Add the text labels
        self.recording_text = pg.TextItem(anchor=(0.5, 0), color=(255, 0, 0))
        self.recording_text.setPos(self.source_shape_hw[1] / 2, self.source_shape_hw[0] / 2)
        self.recording_text.setHtml('<span style="font-size: 16pt; font-weight: bold;">[ ⬤ RECORDING ]</span>')
        self.view_box.addItem(self.recording_text)
        self.recording_text.hide()  # initially hidden

        self.warning_text = pg.TextItem(anchor=(0.5, 0), color=(255, 165, 0))
        self.warning_text.setPos(self.source_shape_hw[1] / 2, 10)
        self.warning_text.setHtml('<span style="font-size: 16pt; font-weight: bold;">[ WARNING ]</span>')
        self.view_box.addItem(self.warning_text)
        self.warning_text.hide()  # also initially hidden

        # Magnifier setup
        self.magnifier_group = QGraphicsItemGroup()
        self.magnifier_item = FastImageItem()

        self.magnifier_border = QGraphicsRectItem()
        self.magnifier_border.setPen(pg.mkPen('y', width=2))

        self.magnifier_group.addToGroup(self.magnifier_item)
        self.magnifier_group.addToGroup(self.magnifier_border)

        self.view_box.addItem(self.magnifier_group)

        self.magnifier_group.hide()

        # Magnifier target
        self.magnifier_source_rect = QGraphicsRectItem()
        self.magnifier_source_rect.setPen(pg.mkPen('y', width=1))
        # this one lives in the main view box, not the group
        self.view_box.addItem(self.magnifier_source_rect)
        self.image_item.setZValue(0)
        self.magnifier_source_rect.setZValue(1)
        self.magnifier_group.setZValue(2)
        self.magnifier_source_rect.hide()

        # Capture mouse event in the graphics widget
        # TODO: Maybe this will need to be moved to the PreviewBase once I add interactive calibration tools
        self.graphics_widget.scene().installEventFilter(self)

        # RIGHT GROUP
        right_group_layout = QHBoxLayout(self.RIGHT_GROUP)
        right_group_layout.setContentsMargins(5, 5, 5, 5)

        self.camera_controls_sliders = {}
        self.camera_controls_sliders_labels = {}
        self.camera_controls_sliders_scales = {}
        self._val_in_sync = {}

        right_group_sliders = QWidget()
        right_group_sliders_layout = QVBoxLayout(right_group_sliders)
        right_group_sliders_layout.setContentsMargins(0, 20, 0, 5)
        right_group_sliders_layout.setSpacing(0)

        sync_groupbox = QGroupBox("Sync")
        sync_groupbox.setContentsMargins(5, 20, 0, 5)
        sync_groupbox_layout = QVBoxLayout(sync_groupbox)

        sync_groupbox_layout.setSpacing(12)

        # The order here determines the order in the GUI
        params_to_create = ['framerate', 'exposure', 'black_level', 'gain', 'gamma']

        for label in params_to_create:
            line = QWidget()
            line_layout = QHBoxLayout(line)
            line_layout.setContentsMargins(1, 1, 1, 1)
            line_layout.setSpacing(2)

            try:
                # Query the camera for all necessary info
                current_range = getattr(self._camera, f"{label}_range")
                current_value = getattr(self._camera, label)

                min_val, max_val = current_range
                param_value = current_value or 0

                # Determine if it's a float or int
                is_float = isinstance(param_value, float) or isinstance(min_val, float)

            except AttributeError:
                # If the camera doesn't have a property (e.g., no gamma), skip creating the slider
                print(f"Camera {self._camera.name} does not support '{label}'. Skipping slider.")
                continue

            slider_label = QLabel(f'{label.title()}:')
            slider_label.setFixedWidth(70)
            slider_label.setContentsMargins(0, 5, 5, 0)
            slider_label.setAlignment(Qt.AlignRight)
            line_layout.addWidget(slider_label)

            slider = QSlider(Qt.Horizontal)

            # For floats, we only want to scale them if their range is small (i.e. they need decimal precision)
            should_scale = is_float and max_val < 1000

            if label == 'exposure':
                slider_min_pos, slider_max_pos = 0, 1000
                slider.setRange(slider_min_pos, slider_max_pos)

                # Store the log mapping parameters
                self.log_slider_params[label] = {
                    'min_val': min_val, 'max_val': max_val,
                    'slider_min': slider_min_pos, 'slider_max': slider_max_pos
                }

                initial_pos = self._log_map(param_value, min_val, max_val, slider_min_pos, slider_max_pos)
                slider.setValue(initial_pos)
                self.camera_controls_sliders_scales[label] = 'log'

                value_text = pretty_microseconds(param_value)

            elif should_scale:
                # For small floats, use a scale to map them to an integer slider
                digits = 2
                scale = 10 ** digits
                slider.setMinimum(int(min_val * scale))
                slider.setMaximum(int(max_val * scale))
                slider.setSingleStep(1)
                slider.setValue(int(param_value * scale))
                self.camera_controls_sliders_scales[label] = scale
                value_text = f"{param_value:.2f}"
            else:
                # For ints and large linear floats (like framerate), it's straightforward
                slider.setMinimum(int(min_val))
                slider.setMaximum(int(max_val))
                slider.setSingleStep(1)
                slider.setValue(int(param_value))
                self.camera_controls_sliders_scales[label] = 1
                value_text = f"{int(param_value)}"

            slider.setMinimumWidth(100)
            slider.valueChanged.connect(lambda value, lbl=label: self._slider_changed(lbl, value))
            slider.sliderReleased.connect(lambda lbl=label: self._slider_released(lbl))
            line_layout.addWidget(slider, 1)

            value_label = QLabel(value_text)
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

        self.n_button = QPushButton('Nothing')
        self.n_button.setCheckable(True)
        self.n_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.n_button.clicked.connect(self._toggle_n_display)
        line_layout.addWidget(self.n_button)

        self.magn_button = QPushButton('Magnification')
        self.magn_button.setCheckable(True)
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

    def _annotate_frame(self):
        """ This is called every frame by _update_display """

        # First, ensure the display buffer has the latest video frame
        if self._latest_frame is None:
            return

        # we copy here before adding annotations
        np.copyto(self._latest_display_frame, self._latest_frame)
        self._latest_frame = None

        if not self.magnifier_group.isVisible():
            return

        view_target_cx = self.magn_target_cx * self._source_width
        view_target_cy = self.magn_target_cy * self._source_height

        source_rect_x = view_target_cx - self.magn_window_w / 2
        source_rect_y = view_target_cy - self.magn_window_h / 2
        source_rect_x = max(0, min(self._source_width - self.magn_window_w, source_rect_x))
        source_rect_y = max(0, min(self._source_height - self.magn_window_h, source_rect_y))

        self.magnifier_source_rect.setRect(source_rect_x, source_rect_y, self.magn_window_w, self.magn_window_h)

        slice_x1 = int(source_rect_x)
        slice_x2 = slice_x1 + self.magn_window_w

        slice_y1 = int(source_rect_y)
        slice_y2 = slice_y1 + self.magn_window_h

        magnifier_source_data = self._latest_display_frame[slice_y1:slice_y2, slice_x1:slice_x2]

        self.magnifier_item.setImageData(magnifier_source_data)

        scale = self.magn_slider.value()
        self.magnifier_item.setScale(scale)

        scaled_rect = self.magnifier_item.mapRectToParent(self.magnifier_item.boundingRect())
        self.magnifier_border.setRect(scaled_rect)

    def eventFilter(self, watched_obj, event):

        if event.type() in [QEvent.GraphicsSceneMousePress,
                            QEvent.GraphicsSceneMouseMove,
                            QEvent.GraphicsSceneMouseRelease]:

            scene_pos = event.scenePos()
            image_pos = self.view_box.mapSceneToView(scene_pos)
            mouse_x = image_pos.x()
            mouse_y = image_pos.y()
            img_h, img_w = self.source_shape_hw
            mouse_x = max(0, min(img_w, mouse_x))
            mouse_y = max(0, min(img_h, mouse_y))
            buttons = event.buttons()

            if event.type() == QEvent.GraphicsSceneMousePress:
                if buttons & Qt.LeftButton: self.left_mouse_btn = True
                if buttons & Qt.RightButton: self.right_mouse_btn = True

            if event.type() == QEvent.GraphicsSceneMouseRelease:
                if event.button() == Qt.LeftButton: self.left_mouse_btn = False
                if event.button() == Qt.RightButton: self.right_mouse_btn = False

            if self.left_mouse_btn:
                self.magn_target_cx = mouse_x / img_w
                self.magn_target_cy = mouse_y / img_h

            if self.right_mouse_btn:
                self.magnifier_group.setPos(mouse_x, mouse_y)

            return True

        # for all other events, pass them on to the base class
        return super().eventFilter(watched_obj, event)

    # TODO: reactivate these
    def set_recording_indicator(self, is_recording):
        """ Shows or hides the recording text """
        self.recording_text.setVisible(is_recording)

    def set_warning_indicator(self, show_warning):
        """ Shows or hides the warning text """
        self.warning_text.setVisible(show_warning)

    def _log_map(self, value, min_val, max_val, slider_min, slider_max):
        """ Maps a value from a log scale to a linear slider position """
        if value <= min_val:
            return slider_min
        if value >= max_val:
            return slider_max

        log_min = np.log(min_val)
        log_max = np.log(max_val)
        log_val = np.log(value)

        # Normalize the log value to 0-1 range
        scale = (log_val - log_min) / (log_max - log_min)
        return int(slider_min + scale * (slider_max - slider_min))

    def _inv_log_map(self, pos, min_val, max_val, slider_min, slider_max):
        """ Maps a linear slider position back to a log scale value """

        if pos <= slider_min:
            return min_val
        if pos >= slider_max:
            return max_val

        log_min = np.log(min_val)
        log_max = np.log(max_val)

        # Normalize the slider position to 0-1 range
        scale = (pos - slider_min) / (slider_max - slider_min)
        log_val = log_min + scale * (log_max - log_min)
        return np.exp(log_val)

    @Slot(str, object)
    def update_slider_value(self, label, value):
        """
        This runs on the main GUI thread and is called by the polling loop in PreviewBase._update_slow
        (when a parameter change is detected on the camera object)
        """
        if label not in self.camera_controls_sliders:
            return

        slider = self.camera_controls_sliders[label]
        scale = self.camera_controls_sliders_scales.get(label, 1)
        value_label = self.camera_controls_sliders_labels[label]

        slider.blockSignals(True)

        if scale == 'log':
            # This logic shouldn't be needed for log sliders as their value isn't clamped by range,
            # but we keep it for correctness.
            params = self.log_slider_params[label]
            current_value = self._inv_log_map(slider.value(), **params)
            value_label.setText(pretty_microseconds(current_value))
        else:
            # Get the new, clamped integer value directly from the slider
            clamped_int_value = slider.value()
            current_value = clamped_int_value / scale
            value_label.setText(f'{int(current_value)}')

        self._last_polled_values[label] = current_value

        if label == 'framerate':
            self._capture_fps_deque.clear()

        slider.blockSignals(False)

    @Slot(str, object)
    def update_slider_range(self, label, value: Tuple[float, float]):
        if label not in self.camera_controls_sliders:
            return

        slider = self.camera_controls_sliders[label]
        scale = self.camera_controls_sliders_scales.get(label, 1)
        min_val, max_val = value

        # Block signals to prevent the slider from re-emitting signals while we modify it.
        slider.blockSignals(True)

        if scale == 'log':
            # Update the stored parameters for the log mapping
            self.log_slider_params[label]['min_val'] = min_val
            self.log_slider_params[label]['max_val'] = max_val
            # The slider's own integer range (0-1000) does not change
        else:
            # For linear sliders, just update the min/max
            slider.setMinimum(int(min_val * scale))
            slider.setMaximum(int(max_val * scale))

        slider.blockSignals(False)

        current_value_from_camera = getattr(self._camera, label)
        self.update_slider_value(label, current_value_from_camera)

    @Slot(list)
    def on_worker_result(self, bboxes):
        self._bboxes = bboxes

    #  ============= Recording mode-specific functions =============
    def _toggle_n_display(self):

        enabled = self.n_button.isChecked()

        if enabled:
            self.n_button.setStyleSheet(f'background-color: #80{col_green.lstrip("#")};')
        else:
            self.magn_button.setStyleSheet('')

    def _toggle_mag_display(self):

        enabled = self.magn_button.isChecked()

        self.magnifier_group.setVisible(enabled)
        self.magnifier_source_rect.setVisible(enabled)

        if enabled:
            self.magn_button.setStyleSheet(f'background-color: #80{col_yellow.lstrip("#")};')
        else:
            self.magn_button.setStyleSheet('')

    def _slider_changed(self, label, int_value):
        """ Updates the text label next to a slider as it's being moved """

        scale = self.camera_controls_sliders_scales.get(label, 1)

        if scale == 'log':
            params = self.log_slider_params[label]
            value_float = self._inv_log_map(int_value, **params)
            self.camera_controls_sliders_labels[label].setText(pretty_microseconds(value_float))
        else:
            value_float = int_value / scale
            if value_float.is_integer():
                self.camera_controls_sliders_labels[label].setText(f'{int(value_float)}')
            else:
                self.camera_controls_sliders_labels[label].setText(f'{value_float:.2f}')

    def _slider_released(self, label):
        """ Called when a slider is used: either sets the parameter on one camera or broadcasts it via the manager """

        slider = self.camera_controls_sliders[label]
        scale = self.camera_controls_sliders_scales.get(label, 1)

        if scale == 'log':
            params = self.log_slider_params[label]
            value = self._inv_log_map(slider.value(), **params)
        else:
            value = slider.value() / scale

        if self._val_in_sync[label].isChecked():
            # Tell the manager to broadcast the setting to everyone
            self._mainwindow.manager.set_all_cameras(label, value)
        else:
            # Set the parameter only on this specific camera
            setattr(self._camera, label, value)

        # After setting the value, immediately read it back
        # to get the actual clamped value that the hardware accepted
        actual_value_set = getattr(self._camera, label)

        # tell the UI to update with this *actual* value
        self.update_slider_value(label, actual_value_set)

        # update the polled value cache. This prevents the slow update loop from redundantly updating on next tick
        self._last_polled_values[label] = actual_value_set

class CalibrationLiveView(LiveViewBase):

    request_load = Signal(str)
    request_save = Signal(str)

    def __init__(self, camera, main_window_ref, board_params):
        super().__init__(camera, main_window_ref)

        # Data stores for plotting
        self.live_error_deque = deque(maxlen=MAX_PLOT_X)
        self.historical_errors_data = []  # tuples of (index, mean, std)

        # Data stores for image annotations
        # TODO: This should be filled instead of recreated
        self.latest_detected_points = np.zeros((0, 2))
        self.latest_reprojected_points = np.zeros((0, 2))
        self.latest_detected_ids = np.array([])

        # Registration is handled by MainControls
        self._setup_worker(MonocularWorker(
            board=board_params,
            cam_name=self.name,
            img_width=self._source_width,
            img_height=self._source_height
        ))

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
        """ This constructor creates the UI elements specific to Calib mode """

        layout = QHBoxLayout(self.RIGHT_GROUP)
        layout.setContentsMargins(5, 5, 5, 5)

        # Overlay text
        self.computing_text = pg.TextItem(anchor=(0.5, 0.5), color=(255, 255, 255))
        self.computing_text.setPos(self.source_shape_hw[1] / 2, self.source_shape_hw[0] / 2)
        self.computing_text.setHtml('<span style="font-size: 16pt; font-weight: bold;">Computing...</span>')
        self.view_box.addItem(self.computing_text)
        self.computing_text.hide()

        # Stats overlay text (top-left) - Use native pyqtgraph styling
        self.stats_text = pg.TextItem(
            color='w',
            anchor=(0, 0),
            fill=pg.mkBrush(0, 0, 0, 120),  # Semi-transparent black fill
            #border=pg.mkPen(col_darkgray)  # Subtle border
        )
        self.stats_text.setPos(10, 10)
        self.view_box.addItem(self.stats_text)

        # Coverage overlay (semi-transparent green grid)
        self.coverage_overlay_item = pg.ImageItem()
        self.view_box.addItem(self.coverage_overlay_item)

        self.perimeter_item = pg.PlotDataItem(
            pen=pg.mkPen(color=(255, 0, 255), width=2),
            connect='all'  # Connects the last point to the first to close the shape
        )
        self.view_box.addItem(self.perimeter_item)

        # Scatter plot for ALL reprojected points (white)
        self.reprojection_points_item = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush('w'), symbol='o', size=5, pxMode=True
        )
        self.view_box.addItem(self.reprojection_points_item)

        # Scatter plot for DETECTED points (yellow)
        self.detection_points_item = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush('y'), symbol='o', size=7, pxMode=True
        )
        self.view_box.addItem(self.detection_points_item)

        # Set Z-order for correct layering
        self.image_item.setZValue(0)
        self.coverage_overlay_item.setZValue(1)
        self.perimeter_item.setZValue(2)
        self.reprojection_points_item.setZValue(3)
        self.detection_points_item.setZValue(4)
        self.stats_text.setZValue(5)
        self.computing_text.setZValue(6)

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
        self.sample_button.setStyleSheet(f"background-color: {col_darkgreen}; color: {col_white};")
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
        self.error_plot.setLabel('bottom', 'Frame / Sample Index')
        self.error_plot.showGrid(x=True, y=True)
        self.error_plot.setYRange(0.0, 5.0)
        self.error_plot.addLegend()

        # Curve for live, per-frame error
        self.live_error_curve = self.error_plot.plot(
            pen=pg.mkPen(color=col_yellow, width=1),
            name="Live Error"
        )

        # Error bars for historical, accepted calibration results
        self.historical_error_bars = pg.ErrorBarItem(
            pen=pg.mkPen(color=col_green, width=2),
            symbol='o',
            symbolPen='w',
            symbolBrush=col_green,
            symbolSize=8,
            name="Accepted Calibrations"
        )
        self.error_plot.addItem(self.historical_error_bars)

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

    def _annotate_frame(self):
        """
        This is called at the high display rate. It overlays the latest
        available annotation onto the latest available video frame
        """

        # first, ensure the display buffer has the latest video frame
        if self._latest_frame is None:
            return

        np.copyto(self._latest_display_frame, self._latest_frame)
        self._latest_frame = None

        # If the worker delivered a new annotated frame, we store it
        # (the annotation frame is persistent)
        if self._worker_blocking:
            self.computing_text.setVisible(True)
            return
        else:
            self.computing_text.setVisible(False)

        if self.latest_detected_points.shape[0] > 0:
            self.detection_points_item.setData(pos=self.latest_detected_points)
        else:
            self.detection_points_item.clear()

        if self.latest_reprojected_points.shape[0] > 0:
            all_points = np.array(self.latest_reprojected_points)

            # inner points (not corners) (white)
            self.reprojection_points_item.setData(pos=all_points[:-4])

            # Perimeter (purple)
            perimeter = np.vstack((all_points[-4:, :], all_points[-4, :]))
            self.perimeter_item.setData(x=perimeter[:, 0], y=perimeter[:, 1])
        else:
            self.reprojection_points_item.clear()
            self.perimeter_item.clear()

    @Slot(CalibrationData)
    def handle_payload(self, data: CalibrationData):
        """ Receives data from the main window's router """

        payload = data.payload

        if isinstance(payload, DetectionPayload):
            self.latest_detected_points = payload.points2D
            return  # early exit for this common payload

        if isinstance(payload, ReprojectionPayload):
            self.latest_reprojected_points = payload.all_points_2d
            self.latest_detected_ids = payload.detected_ids
            return

        if isinstance(payload, CoveragePayload):
            grid = payload.grid.T  # Transpose the grid to match (row, col) expectation
            if grid.any():
                h, w = grid.shape

                # semi-transparent green image from the boolean grid
                rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
                rgba_image[grid] = [0, 150, 0, 100]  # R, G, B, alpha

                self.coverage_overlay_item.setImage(rgba_image, autoLevels=False)
                # position and scale to match the video frame
                self.coverage_overlay_item.setRect(0, 0, self._source_width, self._source_height)
            else:
                self.coverage_overlay_item.clear()

            stats_html = f"""
                            <div style='font-family: sans-serif; font-size: 12pt; padding: 6px;'>
                            Points: {self.latest_detected_points.shape[0]}/{payload.total_points}<br>
                            Coverage: {payload.coverage_percent:.1f} %<br>
                            Samples: {payload.nb_samples}
                            </div>
                            """
            self.stats_text.setHtml(stats_html)

            return

        if isinstance(payload, ErrorsPayload):
            # this is a historical calibration result
            errors = np.asarray(payload.errors)
            if not np.all(np.isfinite(errors)):
                return

            mean_error = np.nanmean(errors)
            std_error = np.nanstd(errors)

            # Add new data point
            calib_index = len(self.historical_errors_data)
            self.historical_errors_data.append((calib_index, mean_error, std_error))

            # Update the error bar plot
            x_vals = [d[0] for d in self.historical_errors_data]
            y_vals = [d[1] for d in self.historical_errors_data]
            std_vals = [d[2] for d in self.historical_errors_data]
            self.historical_error_bars.setData(x=np.array(x_vals), y=np.array(y_vals), top=np.array(std_vals),
                                               bottom=np.array(std_vals))

            self.load_save_message.setText(f"Intrinsics updated. Mean err: {mean_error:.3f} px (std: {std_error:.3f})")

        elif isinstance(payload, ExtrinsicsPayload):
            # This is a live, per-frame error update
            # pyqtgraph handles nans (with gaps) just fine when pose estimation fails
            error_value = payload.error if payload.error is not None else np.nan
            self.live_error_deque.append(error_value)
            self.live_error_curve.setData(list(self.live_error_deque))

        elif isinstance(payload, IntrinsicsPayload):
            # received whenever intrinsics are updated (loading a file, new monocular calculation, or after refinement)
            self.load_save_message.setText(f"Intrinsics updated from {data.camera_name}.")

    def on_save_parameters(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Intrinsics",
            str(self._mainwindow.manager.full_path.resolve()),
            "TOML Files (*.toml)")

        if file_path:
            self.request_save.emit(file_path)
            self.load_save_message.setText(f"Intrinsics saved to\n{Path(file_path).name}")

    def on_clear_intrinsics(self):
        # Clear plot data stores
        self.live_error_deque.clear()
        self.historical_errors_data.clear()

        # Update plots with empty data
        self.live_error_curve.clear()
        self.historical_error_bars.setData(x=np.array([]), y=np.array([]))

        # Clear visualization data and items
        self.latest_detected_points = np.zeros((0, 2))
        self.latest_reprojected_points = np.zeros((0, 2))
        self.detection_points_item.clear()
        self.reprojection_points_item.clear()
        self.perimeter_item.clear()
        self.coverage_overlay_item.clear()

        # Clear text
        self.load_save_message.setText('')
        self.stats_text.setHtml('')

    def on_load_parameters(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Intrinsics",
            str(self._mainwindow.manager.full_path.parent),
            "TOML Files (*.toml)"
        )

        if file_path:
            self.on_clear_intrinsics()
            # then send the request to the worker
            self.request_load.emit(file_path)
            self.load_save_message.setText(f"Intrinsics loading from\n{Path(file_path).name}")