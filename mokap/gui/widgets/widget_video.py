"""
Live video view windows for Recording and Calibration modes.

RecordingVideoWindow: Camera preview with hardware controls
CalibrationVideoWindow: Camera preview with calibration tools
"""
from collections import deque
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Slot, Signal, QThread
from PySide6.QtWidgets import (QHBoxLayout, QWidget, QVBoxLayout, QGroupBox, QLabel, QSlider,
                               QCheckBox, QPushButton, QFileDialog, QGraphicsRectItem, QGraphicsItemGroup)
from mokap.utils import pretty_microseconds
from mokap.gui.style import *
from mokap.gui.widgets import VideoWindowBase, FastImageItem
from mokap.gui.workers import DetectorWorker, MonocularWorker


class RecordingVideoWindow(VideoWindowBase):
    """
    Live view for Recording mode.
    
    Features:
    - Crosshair overlay
    - Magnifier tool
    - Camera hardware control sliders
    - Recording indicator
    """

    def __init__(self, hardware_camera, main_window_ref):
        super().__init__(hardware_camera, main_window_ref)

        # Magnification parameters
        self.magn_window_w = 100
        self.magn_window_h = 100
        self.magn_target_cx = 0.5
        self.magn_target_cy = 0.5

        # Mouse states
        self.left_mouse_btn = False
        self.right_mouse_btn = False

        # Slider parameter storage
        self.log_slider_params = {}
        self.camera_controls_sliders = {}
        self.camera_controls_sliders_labels = {}
        self.camera_controls_sliders_scales = {}
        self._val_in_sync = {}

        # Build UI
        self._init_common_ui()
        self._init_specific_ui()
        self.auto_size()

        # Start timers
        self._start_timers()

    def _init_specific_ui(self):
        """Create Recording-specific UI elements."""
        
        # Overlays
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

        # Warning text
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

        # Right panel: Camera controls
        right_layout = QHBoxLayout(self.RIGHT_GROUP)
        right_layout.setContentsMargins(5, 5, 5, 5)

        sliders_widget = QWidget()
        sliders_layout = QVBoxLayout(sliders_widget)
        sliders_layout.setContentsMargins(0, 5, 0, 5)
        sliders_layout.setSpacing(2)

        # Create sliders for camera parameters
        params = ['framerate', 'exposure', 'black_level', 'gain', 'gamma']
        
        for label in params:
            try:
                current_range = getattr(self._hw_camera, f"{label}_range")
                current_value = getattr(self._hw_camera, label)
                min_val, max_val = current_range
                param_value = current_value or 0
            except AttributeError:
                continue

            line = QWidget()
            line_layout = QHBoxLayout(line)
            line_layout.setContentsMargins(1, 1, 1, 1)
            line_layout.setSpacing(2)

            slider_label = QLabel(f'{label.title()}:')
            slider_label.setFixedWidth(70)
            slider_label.setAlignment(Qt.AlignRight)
            line_layout.addWidget(slider_label)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimumWidth(100)

            is_float = isinstance(param_value, float) or isinstance(min_val, float)
            should_scale = is_float and max_val < 1000

            if label == 'exposure':
                slider.setRange(0, 1000)
                self.log_slider_params[label] = {
                    'min_val': min_val, 'max_val': max_val,
                    'slider_min': 0, 'slider_max': 1000
                }
                initial_pos = self._log_map(param_value, min_val, max_val, 0, 1000)
                slider.setValue(initial_pos)
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

            slider.valueChanged.connect(
                lambda value, lbl=label: self._slider_changed(lbl, value)
            )
            line_layout.addWidget(slider)

            value_label = QLabel(value_text)
            value_label.setFixedWidth(60)
            line_layout.addWidget(value_label)

            self.camera_controls_sliders[label] = slider
            self.camera_controls_sliders_labels[label] = value_label

            sliders_layout.addWidget(line)

        right_layout.addWidget(sliders_widget)

    def _log_map(self, value, min_val, max_val, slider_min, slider_max):
        """Map a value to logarithmic slider position."""
        if value <= min_val:
            return slider_min
        if value >= max_val:
            return slider_max
        
        log_min = np.log(max(min_val, 1))
        log_max = np.log(max_val)
        log_val = np.log(max(value, 1))
        
        return int(slider_min + (log_val - log_min) / (log_max - log_min) * (slider_max - slider_min))

    def _slider_changed(self, label, value):
        """Handle slider value change."""
        scale = self.camera_controls_sliders_scales.get(label, 1)
        
        if scale == 'log':
            params = self.log_slider_params[label]
            log_min = np.log(max(params['min_val'], 1))
            log_max = np.log(params['max_val'])
            ratio = (value - params['slider_min']) / (params['slider_max'] - params['slider_min'])
            actual_value = np.exp(log_min + ratio * (log_max - log_min))
            display_text = pretty_microseconds(actual_value)
        else:
            actual_value = value / scale if scale != 1 else value
            display_text = f"{actual_value:.2f}" if scale != 1 else f"{int(actual_value)}"

        self.camera_controls_sliders_labels[label].setText(display_text)


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
    """

    send_frame_to_detector = Signal(np.ndarray, int)
    request_load = Signal(str)
    request_save = Signal(str)

    def __init__(self, hardware_camera, main_window_ref, board_params):
        super().__init__(hardware_camera, main_window_ref)

        # Data stores for plotting
        self.live_error_deque = deque(maxlen=MAX_PLOT_WIDTH)
        self.historical_errors_data = []

        # Data stores for overlays
        self.latest_detected_points = np.zeros((0, 2))
        self.latest_reprojected_points = np.zeros((0, 2))
        self.latest_detected_ids = np.array([])

        # Get camera model from shared rig
        self._camera_model = self._mainwindow.rig[self._hw_cam_name]

        # Create detector thread
        self._detector = DetectorWorker(self._camera_model, board_params)
        self._detector_thread = QThread()
        self._detector.moveToThread(self._detector_thread)
        self._detector_busy = False

        # Create calibration worker
        self._worker = MonocularWorker(self._camera_model, board_params)
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)

        # Wire detector -> worker
        self._detector.detection_ready.connect(self._worker.on_detection)
        self._detector.finished.connect(self._on_detector_finished)

        # Start threads
        self._detector_thread.start()
        self._worker_thread.start()

        # Connect worker signals to UI
        self._worker.detection_updated.connect(self._on_detection_updated)
        self._worker.coverage_updated.connect(self._on_coverage_updated)
        self._worker.intrinsics_updated.connect(self._on_intrinsics_updated)
        self._worker.pose_updated.connect(self._on_pose_updated)
        self._worker.blocking.connect(self._on_blocking)
        self._worker.stage_changed.connect(self._on_stage_changed)

        # Connect frame handler
        self.send_frame_to_detector.connect(self._detector.handle_frame, Qt.QueuedConnection)

        # Connect save/load
        self.request_load.connect(self._worker.load_intrinsics)
        self.request_save.connect(self._worker.save_intrinsics)

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

        # Overlays
        
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

        # Sampling controls
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

        # Error plot
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

        # Save / load
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

    # ──────────────────────────────── Frame processing ────────────────────────────────

    def _send_frame_for_processing(self):
        """Send frame to detector thread."""
        # Guard: don't send if detector is already processing or worker is blocking
        if self._detector_busy or self._worker_blocking:
            return

        if self._latest_frame is not None:
            frame_idx = self._current_frame_data.get('frame_number', 0)
            self._detector_busy = True
            self.send_frame_to_detector.emit(self._latest_frame, frame_idx)

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
        """Busy state check to avoid queue build up"""
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
        cam = self._camera_model

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
    def _on_blocking(self, is_blocking: bool):
        """Handle blocking state change."""
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
            str(self._mainwindow.manager.full_path.resolve()),
            "TOML Files (*.toml)"
        )

        if file_path:
            self.request_save.emit(file_path)
            self.load_save_message.setText(f"Saved to\n{Path(file_path).name}")

    def _on_load_parameters(self):
        """Load intrinsics from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Intrinsics",
            str(self._mainwindow.manager.full_path.parent),
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
