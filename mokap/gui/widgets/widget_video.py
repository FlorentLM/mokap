"""
Live video view windows for Recording and Calibration modes.
"""
import logging
from collections import deque
from pathlib import Path
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Slot, Signal, QThread, QEvent
from PySide6.QtWidgets import (QHBoxLayout, QWidget, QVBoxLayout, QGroupBox, QLabel, QSlider,
                               QCheckBox, QPushButton, QFileDialog, QGraphicsRectItem, QGraphicsItemGroup, QSizePolicy)
from mokap.utils import pretty_microseconds
from mokap.gui.style import *
from mokap.gui.widgets import VideoWindowBase, FastImageItem
from mokap.gui.workers import DetectorWorker, MonocularWorker

logger = logging.getLogger(__name__)


class RecordingVideoWindow(VideoWindowBase):
    """
    Live view for Recording mode.
    """

    def __init__(self, hw_cam, main_window_ref):
        super().__init__(hw_cam, main_window_ref)

        # Magnification parameters
        self.magn_window_w = 100
        self.magn_window_h = 100
        self.magn_target_cx = 0.5
        self.magn_target_cy = 0.5

        # Mouse states
        self.left_mouse_btn = False
        self.right_mouse_btn = False

        # Slider storage
        self.log_slider_params = {}
        self.camera_controls_sliders = {}
        self.camera_controls_sliders_labels = {}
        self.camera_controls_sliders_scales = {}
        self.camera_controls_sync_checks = {}

        # Build UI
        self._init_common_ui()
        self._init_specific_ui()
        self.auto_size()

        # Start timers
        self._start_timers()

    def _init_specific_ui(self):
        """Create Recording-specific UI elements."""

        # ──── ──── Overlays ──── ────
        crosshair_pen = pg.mkPen(color='w', style=Qt.DotLine)
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=crosshair_pen)
        self.v_line.setPos(self.source_shape_hw[1] / 2)
        self.h_line.setPos(self.source_shape_hw[0] / 2)
        self.view_box.addItem(self.v_line)
        self.view_box.addItem(self.h_line)

        # Recording indicator
        self.recording_text = pg.TextItem(anchor=(0.5, 0), color=(255, 0, 0))
        self.recording_text.setPos(self.source_shape_hw[1] / 2, self.source_shape_hw[0] / 2)
        self.recording_text.setHtml(
            '<span style="font-size: 16pt; font-weight: bold;">[ ⬤ RECORDING ]</span>'
        )
        self.view_box.addItem(self.recording_text)
        self.recording_text.hide()

        # Warning indicator
        self.warning_text = pg.TextItem(anchor=(0.5, 0), color=(255, 165, 0))
        self.warning_text.setPos(self.source_shape_hw[1] / 2, 10)
        self.warning_text.setHtml(
            '<span style="font-size: 16pt; font-weight: bold;">[ WARNING ]</span>'
        )
        self.view_box.addItem(self.warning_text)
        self.warning_text.hide()

        # Magnifier
        self.magnifier_group = QGraphicsItemGroup()
        self.magnifier_item = FastImageItem()
        self.magnifier_border = QGraphicsRectItem()
        self.magnifier_border.setPen(pg.mkPen('y', width=2))
        self.magnifier_group.addToGroup(self.magnifier_item)
        self.magnifier_group.addToGroup(self.magnifier_border)
        self.view_box.addItem(self.magnifier_group)
        self.magnifier_group.hide()

        self.magnifier_source_rect = QGraphicsRectItem()
        self.magnifier_source_rect.setPen(pg.mkPen('y', width=1))
        self.view_box.addItem(self.magnifier_source_rect)
        self.magnifier_source_rect.hide()

        # Z-order
        self.image_item.setZValue(0)
        self.magnifier_source_rect.setZValue(1)
        self.magnifier_group.setZValue(2)

        # Mouse events for magnifier
        self.graphics_widget.scene().installEventFilter(self)

        # ──── ──── Right panel layout ──── ────
        right_layout = QHBoxLayout(self.RIGHT_GROUP)
        right_layout.setContentsMargins(5, 5, 5, 5)

        # Sliders column
        sliders_widget = QWidget()
        sliders_layout = QVBoxLayout(sliders_widget)
        sliders_layout.setContentsMargins(0, 20, 0, 5)
        sliders_layout.setSpacing(0)

        # Sync checkboxes column
        sync_group = QGroupBox("Sync")
        sync_group.setContentsMargins(5, 20, 0, 5)
        sync_layout = QVBoxLayout(sync_group)
        sync_layout.setSpacing(12)

        # Create sliders for each parameter
        params = ['framerate', 'exposure', 'black_level', 'gain', 'gamma']

        for label in params:
            try:
                current_range = getattr(self._hw_camera, f"{label}_range")
                current_value = getattr(self._hw_camera, label)
                min_val, max_val = current_range
                param_value = current_value or 0
            except AttributeError:
                continue

            # Determine if the control is actually usable
            is_usable = True
            if label == 'exposure':
                if min_val <= 0:
                    self.camera_controls_sliders_scales[label] = 1
                else:
                    self.camera_controls_sliders_scales[label] = 'log'

            if min_val < 0 or max_val <= 0:
                is_usable = False

            # Slider row
            line = QWidget()
            line_layout = QHBoxLayout(line)
            line_layout.setContentsMargins(1, 1, 1, 1)
            line_layout.setSpacing(2)

            slider_label = QLabel(f"{label.replace('_', ' ').title()}:")
            slider_label.setFixedWidth(70)
            slider_label.setAlignment(Qt.AlignRight)
            line_layout.addWidget(slider_label)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimumWidth(100)

            value_text = "N/A"

            if not is_usable:
                slider.setRange(0, 1)
                slider.setValue(0)
                slider.setEnabled(False)
            else:
                is_float = isinstance(param_value, float) or isinstance(min_val, float)
                should_scale = is_float and max_val < 1000

                if label == 'exposure':
                    slider.setRange(0, 1000)
                    self.log_slider_params[label] = {
                        'min_val': min_val, 'max_val': max_val,
                        'slider_min': 0, 'slider_max': 1000
                    }
                    slider.setValue(self._log_map(param_value, min_val, max_val, 0, 1000))
                    self.camera_controls_sliders_scales[label] = 'log'
                    value_text = pretty_microseconds(param_value)
                elif should_scale:
                    scale = 100
                    slider.setMinimum(int(min_val * scale))
                    slider.setMaximum(int(max_val * scale))
                    slider.setValue(int(param_value * scale))
                    self.camera_controls_sliders_scales[label] = scale
                    value_text = f"{param_value:.2f}"
                else:
                    slider.setMinimum(int(min_val))
                    slider.setMaximum(int(max_val))
                    slider.setValue(int(param_value))
                    self.camera_controls_sliders_scales[label] = 1
                    value_text = f"{int(param_value)}"

                # Connect signals only if usable
                slider.valueChanged.connect(lambda v, lbl=label: self._slider_changed(lbl, v))
                slider.sliderReleased.connect(lambda lbl=label: self._slider_released(lbl))

            line_layout.addWidget(slider, 1)

            value_label = QLabel(value_text)
            value_label.setFixedWidth(50)
            line_layout.addWidget(value_label)

            # Sync checkbox (in separate column)
            sync_check = QCheckBox()
            sync_check.setMaximumWidth(16)
            sync_check.setChecked(True)

            if not is_usable:
                sync_check.setEnabled(False)
                sync_check.setChecked(False)

            if self._hw_camera.hardware_triggered and label == 'framerate':
                sync_check.setDisabled(True)    # Can't be toggled off in hardware sync mode

            sync_layout.addWidget(sync_check)

            # Only add to control dictionaries if usable
            if is_usable:
                self.camera_controls_sliders[label] = slider
                self.camera_controls_sliders_labels[label] = value_label
                self.camera_controls_sync_checks[label] = sync_check

            sliders_layout.addWidget(line)

        right_layout.addWidget(sliders_widget)
        right_layout.addWidget(sync_group)

        # ──── ──── Additional controls (magnifier buttons) ──── ────
        additional_widget = QWidget()
        additional_layout = QVBoxLayout(additional_widget)
        additional_layout.setContentsMargins(0, 20, 0, 5)

        buttons_row = QWidget()
        buttons_row.setMaximumHeight(80)
        buttons_layout = QHBoxLayout(buttons_row)

        self.magn_button = QPushButton('Magnifier')
        self.magn_button.setCheckable(True)
        self.magn_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.magn_button.clicked.connect(self._toggle_magnifier)
        buttons_layout.addWidget(self.magn_button)

        self.magn_slider = QSlider(Qt.Vertical)
        self.magn_slider.setRange(1, 5)
        self.magn_slider.setValue(2)
        buttons_layout.addWidget(self.magn_slider)

        additional_layout.addWidget(buttons_row)
        right_layout.addWidget(additional_widget)

    # ──────────────────────────────── Various helpers ────────────────────────────────

    def _log_map(self, value, min_val, max_val, slider_min, slider_max):
        """Map value to log-scale slider position."""
        if value <= min_val:
            return slider_min
        if value >= max_val:
            return slider_max
        log_min = np.log(min_val)
        log_max = np.log(max_val)
        log_val = np.log(value)
        scale = (log_val - log_min) / (log_max - log_min)
        return int(slider_min + scale * (slider_max - slider_min))

    def _inv_log_map(self, pos, min_val, max_val, slider_min, slider_max):
        """Map log-scale slider position back to value."""
        if pos <= slider_min:
            return min_val
        if pos >= slider_max:
            return max_val
        log_min = np.log(min_val)
        log_max = np.log(max_val)
        scale = (pos - slider_min) / (slider_max - slider_min)
        return np.exp(log_min + scale * (log_max - log_min))

    def _slider_changed(self, label, int_value):
        """Update label text as slider moves (no camera write yet)."""
        scale = self.camera_controls_sliders_scales.get(label, 1)

        if scale == 'log':
            params = self.log_slider_params[label]
            value = self._inv_log_map(int_value, **params)
            text = pretty_microseconds(value)
        else:
            value = int_value / scale
            text = f"{int(value)}" if float(value).is_integer() else f"{value:.2f}"

        self.camera_controls_sliders_labels[label].setText(text)

    def _slider_released(self, label):
        """Apply value to camera(s) when slider is released."""
        slider = self.camera_controls_sliders[label]
        scale = self.camera_controls_sliders_scales.get(label, 1)

        if scale == 'log':
            params = self.log_slider_params[label]
            value = self._inv_log_map(slider.value(), **params)
        else:
            value = slider.value() / scale

        if self.camera_controls_sync_checks[label].isChecked():

            # Apply to all cameras whose sync checkbox is checked
            for window in self._mainwindow.video_windows:
                if not hasattr(window, 'camera_controls_sync_checks'):
                    continue  # Skip non-recording windows
                if window.camera_controls_sync_checks.get(label, None) is None:
                    continue  # This window doesn't have this parameter
                if not window.camera_controls_sync_checks[label].isChecked():
                    continue  # This camera opted out of sync
                if window._hw_camera.hardware_triggered and label == 'framerate':
                    continue  # This should never happen since the checkbox is disabled in hardware mode. But still

                # Set on this camera
                setattr(window._hw_camera, label, value)

                # Update that window's slider UI
                actual = getattr(window._hw_camera, label)
                window._update_slider_from_value(label, actual)
                window._last_polled_values[label] = actual

            # Handle framerate specially for hardware trigger
            if label == 'framerate' and self._mainwindow.controller.hardware_triggered:
                self._mainwindow.controller.framerate = value
        else:
            # Only this camera
            setattr(self._hw_camera, label, value)
            actual = getattr(self._hw_camera, label)
            self._update_slider_from_value(label, actual)
            self._last_polled_values[label] = actual

    def _update_slider_from_value(self, label, value):
        """Update slider position and label from a camera value."""
        slider = self.camera_controls_sliders[label]
        scale = self.camera_controls_sliders_scales.get(label, 1)

        slider.blockSignals(True)
        if scale == 'log':
            params = self.log_slider_params[label]
            slider.setValue(self._log_map(value, **params))
            self.camera_controls_sliders_labels[label].setText(pretty_microseconds(value))
        else:
            slider.setValue(int(value * scale))
            text = f"{int(value)}" if float(value).is_integer() else f"{value:.2f}"
            self.camera_controls_sliders_labels[label].setText(text)
        slider.blockSignals(False)

    # Override the polling hooks from base class
    def _on_slider_value_changed(self, label, value):
        """Called by base class when polled camera value changes."""
        if label in self.camera_controls_sliders:
            self._update_slider_from_value(label, value)

    def _on_slider_range_changed(self, label, new_range):
        """Called by base class when polled camera range changes."""
        if label not in self.camera_controls_sliders:
            return

        slider = self.camera_controls_sliders[label]
        scale = self.camera_controls_sliders_scales.get(label, 1)
        min_val, max_val = new_range

        slider.blockSignals(True)
        if scale == 'log':
            self.log_slider_params[label]['min_val'] = min_val
            self.log_slider_params[label]['max_val'] = max_val
        else:
            slider.setMinimum(int(min_val * scale))
            slider.setMaximum(int(max_val * scale))
        slider.blockSignals(False)

        # Refresh display with current value
        current = getattr(self._hw_camera, label)
        self._update_slider_from_value(label, current)

    def _toggle_magnifier(self):
        enabled = self.magn_button.isChecked()
        self.magnifier_group.setVisible(enabled)
        self.magnifier_source_rect.setVisible(enabled)
        if enabled:
            self.magn_button.setStyleSheet(f'background-color: #80{col_yellow.lstrip("#")};')
        else:
            self.magn_button.setStyleSheet('')

    def set_recording_indicator(self, visible: bool):
        self.recording_text.setVisible(visible)

    def set_warning_indicator(self, visible: bool):
        self.warning_text.setVisible(visible)

    # ──────────────────────────────── Display update ────────────────────────────────

    def eventFilter(self, watched_obj, event):
        """
        Captures mouse events from the graphics scene to control the magnifier.
        Left Drag: Moves the source area (what is being magnified).
        Right Drag: Moves the magnifier overlay position.
        """
        # TODO: Should this move to the base class? Might be useful in Calib mode

        if event.type() in [QEvent.GraphicsSceneMousePress,
                            QEvent.GraphicsSceneMouseMove,
                            QEvent.GraphicsSceneMouseRelease]:

            scene_pos = event.scenePos()
            # Map scene coordinates to the ViewBox (image coordinates)
            image_pos = self.view_box.mapSceneToView(scene_pos)

            mouse_x = image_pos.x()
            mouse_y = image_pos.y()

            img_h, img_w = self.source_shape_hw

            # Clamp coordinates to image bounds
            mouse_x = max(0, min(img_w, mouse_x))
            mouse_y = max(0, min(img_h, mouse_y))

            buttons = event.buttons()

            if event.type() == QEvent.GraphicsSceneMousePress:
                if buttons & Qt.LeftButton:
                    self.left_mouse_btn = True
                if buttons & Qt.RightButton:
                    self.right_mouse_btn = True

            if event.type() == QEvent.GraphicsSceneMouseRelease:
                if event.button() == Qt.LeftButton:
                    self.left_mouse_btn = False
                if event.button() == Qt.RightButton:
                    self.right_mouse_btn = False

            if self.left_mouse_btn:
                # Update target center (normalized 0.0 - 1.0)
                self.magn_target_cx = mouse_x / img_w
                self.magn_target_cy = mouse_y / img_h

            if self.right_mouse_btn:
                # Move the QGraphicsItemGroup
                self.magnifier_group.setPos(mouse_x, mouse_y)

            return True

        return super().eventFilter(watched_obj, event)

    # ──────────────────────────────── Display update ────────────────────────────────

    def _annotate_frame(self):
        """
        Copies the frame to display buffer and handles Magnifier slicing.
        """
        if self._latest_frame is None:
            return

        # Copy raw frame to display buffer
        np.copyto(self._latest_display_frame, self._latest_frame)
        self._latest_frame = None  # mark as consumed

        # Magnifier
        if not self.magnifier_group.isVisible():
            return

        # Calculate coordinates based on target center (0.0-1.0)
        view_target_cx = self.magn_target_cx * self._source_width
        view_target_cy = self.magn_target_cy * self._source_height

        # Calculate top-left corner of the source rectangle
        source_rect_x = view_target_cx - self.magn_window_w / 2
        source_rect_y = view_target_cy - self.magn_window_h / 2

        # Clamp to image boundaries
        source_rect_x = max(0, min(self._source_width - self.magn_window_w, source_rect_x))
        source_rect_y = max(0, min(self._source_height - self.magn_window_h, source_rect_y))

        # Update the yellow rectangle showing WHAT is being magnified
        self.magnifier_source_rect.setRect(source_rect_x, source_rect_y, self.magn_window_w, self.magn_window_h)

        # Slice the numpy array
        slice_x1 = int(source_rect_x)
        slice_x2 = slice_x1 + self.magn_window_w
        slice_y1 = int(source_rect_y)
        slice_y2 = slice_y1 + self.magn_window_h

        # Extract the ROI
        magnifier_source_data = self._latest_display_frame[slice_y1:slice_y2, slice_x1:slice_x2]

        # Update the item inside the magnifier group
        self.magnifier_item.setImageData(magnifier_source_data)

        # Apply zoom scale from the vertical slider
        scale = self.magn_slider.value()
        self.magnifier_item.setScale(scale)

        # Update the border around the magnifier window to match the scale
        scaled_rect = self.magnifier_item.mapRectToParent(self.magnifier_item.boundingRect())
        self.magnifier_border.setRect(scaled_rect)


class CalibrationVideoWindow(VideoWindowBase):
    """
    Live view for Calibration mode.

    Features:
    - Detection overlay (detected corners)
    - Reprojection overlay (computed board corners)
    - Coverage grid overlay
    - Sampling controls
    - Error plots
    - Save/Load intrinsics

    Processing pipeline:
    - DetectorWorker: Runs detection in separate thread, emits detection_ready
    - MonocularWorker: Receives detections, manages calibration logic

    Naming conventions:
    - _hw_camera: Hardware camera (inherited from VideoWindowBase)
    - _cam: Lucida CameraModel for calibration parameters
    """

    # Signal to send frames to the detector
    frame_ready = Signal(np.ndarray, int)

    # Signals for save/load operations
    request_load = Signal(str)
    request_save = Signal(str)

    def __init__(self, hw_cam, main_window_ref, board_params):
        super().__init__(hw_cam, main_window_ref)

        # Data stores for plotting
        self.live_error_deque = deque(maxlen=MAX_PLOT_WIDTH)
        self.historical_errors_data = []

        # Data stores for overlays
        self.latest_detected_points = np.zeros((0, 2))
        self.latest_reprojected_points = np.zeros((0, 2))
        self.latest_detected_ids = np.array([])

        # Lucida CameraModel from the shared rig (for calibration parameters)
        self._cam = self._mainwindow.rig[self._hw_cam_name]

        # ──── ──── Detector (CPU-heavy, separate thread) ──── ────
        self._detector = DetectorWorker(self._cam, board_params)
        self._detector_thread = QThread()
        self._detector.moveToThread(self._detector_thread)
        self._detector_busy = False

        # ──── ──── Calibration worker (stateful, separate thread) ──── ────
        self._worker = MonocularWorker(self._cam, board_params)
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker_blocking = False

        # ──── ──── Wire up signals ──── ────

        # Frame dispatch: this window -> detector
        self.frame_ready.connect(self._detector.handle_frame, Qt.QueuedConnection)

        # Detection flow: detector -> worker
        self._detector.detection_ready.connect(self._worker.on_detection)
        self._detector.finished.connect(self._on_detector_finished)

        # Worker signals -> UI updates
        self._worker.detection_updated.connect(self._on_detection_updated)
        self._worker.coverage_updated.connect(self._on_coverage_updated)
        self._worker.intrinsics_updated.connect(self._on_intrinsics_updated)
        self._worker.pose_updated.connect(self._on_pose_updated)
        self._worker.blocking.connect(self._on_worker_blocking)
        self._worker.stage_changed.connect(self._on_stage_changed)

        # Save/load
        self.request_load.connect(self._worker.load_intrinsics)
        self.request_save.connect(self._worker.save_intrinsics)

        # Start threads
        self._detector_thread.start()
        self._worker_thread.start()

        # Build UI
        self._init_common_ui()
        self._init_specific_ui()
        self.auto_size()

        # Start timers
        self._start_timers()

    def _init_specific_ui(self):
        """Create Calibration-specific UI elements."""
        layout = QHBoxLayout(self.RIGHT_GROUP)
        layout.setContentsMargins(5, 5, 5, 5)

        # ──── ──── Overlays ──── ────

        # Computing indicator
        self.computing_text = pg.TextItem(anchor=(0.5, 0.5), color=(255, 255, 255))
        self.computing_text.setPos(self.source_shape_hw[1] / 2, self.source_shape_hw[0] / 2)
        self.computing_text.setHtml(
            '<span style="font-size: 16pt; font-weight: bold;">Computing...</span>'
        )
        self.view_box.addItem(self.computing_text)
        self.computing_text.hide()

        # Stats overlay
        self.stats_text = pg.TextItem(
            color='w', anchor=(0, 0), fill=pg.mkBrush(0, 0, 0, 120)
        )
        self.stats_text.setPos(10, 10)
        self.view_box.addItem(self.stats_text)

        # Coverage overlay
        self.coverage_overlay_item = pg.ImageItem()
        self.view_box.addItem(self.coverage_overlay_item)

        # Board perimeter
        self.perimeter_item = pg.PlotDataItem(
            pen=pg.mkPen(color=(255, 0, 255), width=2), connect='all'
        )
        self.view_box.addItem(self.perimeter_item)

        # Reprojected points (white)
        self.reprojection_points_item = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush('w'), symbol='o', size=5, pxMode=True
        )
        self.view_box.addItem(self.reprojection_points_item)

        # Detected points (yellow)
        self.detection_points_item = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush('y'), symbol='o', size=7, pxMode=True
        )
        self.view_box.addItem(self.detection_points_item)

        # Z-order
        self.image_item.setZValue(0)
        self.coverage_overlay_item.setZValue(1)
        self.perimeter_item.setZValue(2)
        self.reprojection_points_item.setZValue(3)
        self.detection_points_item.setZValue(4)
        self.stats_text.setZValue(5)
        self.computing_text.setZValue(6)

        # ──── ──── Sampling controls ──── ────
        sampling_group = QWidget()
        sampling_layout = QVBoxLayout(sampling_group)

        self.auto_sample_check = QCheckBox("Sample automatically")
        self.auto_sample_check.setChecked(True)
        self.auto_sample_check.stateChanged.connect(
            lambda state: self._worker.set_auto_sample(state == Qt.Checked)
        )
        sampling_layout.addWidget(self.auto_sample_check)

        sample_btns = QWidget()
        sample_btns_layout = QHBoxLayout(sample_btns)

        self.sample_button = QPushButton("Add sample")
        self.sample_button.clicked.connect(self._worker.add_sample)
        self.sample_button.setStyleSheet(
            f"background-color: {col_darkgreen}; color: {col_white};"
        )
        sample_btns_layout.addWidget(self.sample_button)

        self.clear_samples_button = QPushButton("Clear samples")
        self.clear_samples_button.clicked.connect(self._worker.clear_samples)
        sample_btns_layout.addWidget(self.clear_samples_button)

        sampling_layout.addWidget(sample_btns)

        self.auto_compute_check = QCheckBox("Compute intrinsics automatically")
        self.auto_compute_check.setChecked(True)
        self.auto_compute_check.stateChanged.connect(
            lambda state: self._worker.set_auto_compute(state == Qt.Checked)
        )
        sampling_layout.addWidget(self.auto_compute_check)

        intrinsics_btns = QWidget()
        intrinsics_btns_layout = QHBoxLayout(intrinsics_btns)

        self.compute_intrinsics_button = QPushButton("Compute intrinsics now")
        self.compute_intrinsics_button.clicked.connect(self._worker.compute_intrinsics)
        intrinsics_btns_layout.addWidget(self.compute_intrinsics_button)

        self.clear_intrinsics_button = QPushButton("Clear intrinsics")
        self.clear_intrinsics_button.clicked.connect(self._on_clear_intrinsics)
        self.clear_intrinsics_button.clicked.connect(self._worker.clear_intrinsics)
        intrinsics_btns_layout.addWidget(self.clear_intrinsics_button)

        sampling_layout.addWidget(intrinsics_btns)
        layout.addWidget(sampling_group)

        # ──── ──── Error plot ──── ────
        self.error_plot = pg.PlotWidget(title="Reprojection Error")
        self.error_plot.setStyleSheet("background-color: black;")
        self.error_plot.setLabel('left', 'Error (pixels)')
        self.error_plot.setLabel('bottom', 'Frame / Sample Index')
        self.error_plot.showGrid(x=True, y=True)
        self.error_plot.setYRange(0.0, 5.0)
        self.error_plot.addLegend()

        self.live_error_curve = self.error_plot.plot(
            pen=pg.mkPen(color=col_yellow, width=1), name="Live Error"
        )

        self.historical_error_bars = pg.ErrorBarItem(
            pen=pg.mkPen(color=col_green, width=2),
            symbol='o', symbolPen='w', symbolBrush=col_green, symbolSize=8
        )
        self.error_plot.addItem(self.historical_error_bars)

        self.video_container_layout.addWidget(self.error_plot, 1)

        # ──── ──── Save / load ──── ────
        io_group = QGroupBox("Load/Save")
        io_group.setMinimumWidth(250)
        io_group.setMaximumWidth(250)
        io_layout = QVBoxLayout(io_group)

        self.load_calib_button = QPushButton("Load intrinsics")
        self.load_calib_button.clicked.connect(self._on_load_parameters)
        io_layout.addWidget(self.load_calib_button)

        self.save_calib_button = QPushButton("Save intrinsics")
        self.save_calib_button.clicked.connect(self._on_save_parameters)
        io_layout.addWidget(self.save_calib_button)

        self.load_save_message = QLabel("")
        self.load_save_message.setMaximumWidth(180)
        self.load_save_message.setWordWrap(True)
        io_layout.addWidget(self.load_save_message)

        layout.addWidget(io_group)

    # ──────────────────────────────── Worker control ────────────────────────────────

    def _pause_worker(self):
        """Pause both detector and calibration worker."""
        self._detector.set_paused(True)
        self._worker.set_paused(True)

    def _resume_worker(self):
        """Resume both detector and calibration worker."""
        self._detector.set_paused(False)
        self._worker.set_paused(False)

    # ──────────────────────────────── Frame processing ────────────────────────────────

    def _send_frame_for_processing(self):
        """Send frame to detector thread if pipeline is ready."""
        if self._detector_busy or self._worker_blocking:
            return

        if self._latest_frame is not None:
            frame_idx = self._current_frame_data.get('frame_number', 0)
            self._detector_busy = True
            self.frame_ready.emit(self._latest_frame, frame_idx)

    def _annotate_frame(self):
        """Draw calibration overlays on display buffer."""
        if self._latest_frame is None:
            return

        np.copyto(self._latest_display_frame, self._latest_frame)
        self._latest_frame = None

        if self._worker_blocking:
            self.computing_text.setVisible(True)
            return
        else:
            self.computing_text.setVisible(False)

        # Detection points
        if self.latest_detected_points.shape[0] > 0:
            self.detection_points_item.setData(pos=self.latest_detected_points)
        else:
            self.detection_points_item.clear()

        # Reprojection points
        if self.latest_reprojected_points.shape[0] > 0:
            points = self.latest_reprojected_points
            self.reprojection_points_item.setData(pos=points[:-4])
            perimeter = np.vstack((points[-4:, :], points[-4, :]))
            self.perimeter_item.setData(x=perimeter[:, 0], y=perimeter[:, 1])
        else:
            self.reprojection_points_item.clear()
            self.perimeter_item.clear()

    # ──────────────────────────────── Signal handlers ────────────────────────────────

    @Slot()
    def _on_detector_finished(self):
        """Called when detector finishes processing a frame. Clears busy flag."""
        self._detector_busy = False

    @Slot()
    def _on_detection_updated(self):
        """Handle detection update from worker."""
        det = self._worker.latest_detection

        if det is not None and det.valid:
            self.latest_detected_points = det.detected_points
            self.latest_detected_ids = det.detected_ids
        else:
            self.latest_detected_points = np.zeros((0, 2))
            self.latest_detected_ids = np.array([])

    @Slot()
    def _on_coverage_updated(self):
        """Handle coverage update from worker."""
        tool = self._worker.tool

        grid = tool.coverage_grid
        if grid is not None and grid.any():
            grid_t = grid.T
            h, w = grid_t.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[grid_t > 0] = [0, 150, 0, 100]
            self.coverage_overlay_item.setImage(rgba, autoLevels=False)
            self.coverage_overlay_item.setRect(0, 0, self._source_width, self._source_height)
        else:
            self.coverage_overlay_item.clear()

        stats_html = f"""
            <div style='font-family: sans-serif; font-size: 12pt; padding: 6px;'>
            Points: {self.latest_detected_points.shape[0]}<br>
            Coverage: {tool.current_coverage:.1f} %<br>
            Samples: {tool.sample_count}
            </div>
        """
        self.stats_text.setHtml(stats_html)

    @Slot()
    def _on_intrinsics_updated(self):
        """Handle intrinsics update from worker."""
        cam = self._cam

        with cam.intrinsics.locked():
            rms_per_view = cam.intrinsics.stats.get('rms_per_view', [])

        if not rms_per_view:
            return

        errors = np.asarray(rms_per_view)
        if not np.all(np.isfinite(errors)):
            return

        mean_error = np.nanmean(errors)
        std_error = np.nanstd(errors)

        calib_index = len(self.historical_errors_data)
        self.historical_errors_data.append((calib_index, mean_error, std_error))

        x_vals = [d[0] for d in self.historical_errors_data]
        y_vals = [d[1] for d in self.historical_errors_data]
        std_vals = [d[2] for d in self.historical_errors_data]

        self.historical_error_bars.setData(
            x=np.array(x_vals), y=np.array(y_vals),
            top=np.array(std_vals), bottom=np.array(std_vals)
        )

        self.load_save_message.setText(f"Intrinsics updated.\nMean err: {mean_error:.3f} px")

    @Slot()
    def _on_pose_updated(self):
        """Handle pose update from worker."""
        reprojected = self._worker.tool.reproject()

        if reprojected is not None:
            self.latest_reprojected_points = reprojected
        else:
            self.latest_reprojected_points = np.zeros((0, 2))

    @Slot(bool)
    def _on_worker_blocking(self, is_blocking: bool):
        """Handle blocking state change from worker."""
        self._worker_blocking = is_blocking
        self.computing_text.setVisible(is_blocking)

    @Slot(int)
    def _on_stage_changed(self, stage: int):
        """Handle calibration stage change."""
        is_intrinsics = (stage == 0)

        self.auto_sample_check.setEnabled(is_intrinsics)
        self.auto_compute_check.setEnabled(is_intrinsics)
        self.sample_button.setEnabled(is_intrinsics)
        self.clear_samples_button.setEnabled(is_intrinsics)
        self.compute_intrinsics_button.setEnabled(is_intrinsics)

        if not is_intrinsics:
            self.auto_sample_check.setChecked(False)
            self.auto_compute_check.setChecked(False)

    # ──────────────────────────────── UI actions ────────────────────────────────

    def _on_save_parameters(self):
        """Save intrinsics to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Intrinsics",
            str(self._mainwindow.controller.full_path.resolve()),
            "TOML Files (*.toml)"
        )

        if file_path:
            self.request_save.emit(file_path)
            self.load_save_message.setText(f"Saved to\n{Path(file_path).name}")

    def _on_load_parameters(self):
        """Load intrinsics from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Intrinsics",
            str(self._mainwindow.controller.full_path.parent),
            "TOML Files (*.toml)"
        )

        if file_path:
            self._on_clear_intrinsics()
            self.request_load.emit(file_path)
            self.load_save_message.setText(f"Loading from\n{Path(file_path).name}")

    def _on_clear_intrinsics(self):
        """Clear all calibration data and UI."""
        self.live_error_deque.clear()
        self.historical_errors_data.clear()
        self.live_error_curve.clear()
        self.historical_error_bars.setData(x=np.array([]), y=np.array([]))
        self.latest_detected_points = np.zeros((0, 2))
        self.latest_reprojected_points = np.zeros((0, 2))
        self.detection_points_item.clear()
        self.reprojection_points_item.clear()
        self.perimeter_item.clear()
        self.coverage_overlay_item.clear()
        self.load_save_message.setText('')
        self.stats_text.setHtml('')

    # ──────────────────────────────── Cleanup ────────────────────────────────

    def closeEvent(self, event):
        """Clean up threads on close."""
        if self._force_destroy:
            # Stop detector thread
            self._detector_thread.quit()
            self._detector_thread.wait(2000)

            # Stop worker thread
            self._worker_thread.quit()
            self._worker_thread.wait(2000)

        super().closeEvent(event)