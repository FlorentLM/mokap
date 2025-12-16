"""
Central calibration window with 3D visualisation and board settings.
"""
import logging
from pathlib import Path
from typing import Union
import cv2
import numpy as np
from PySide6.QtCore import Signal, Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QFrame, QVBoxLayout, QGroupBox, QGridLayout, QLabel,
                               QComboBox, QPushButton, QSpinBox, QDoubleSpinBox, QFileDialog, QWidget)
from pyqtgraph.opengl import GLGridItem, GLViewWidget, GLScatterPlotItem, GLLinePlotItem, GLMeshItem
from mokap.gui.style import *
from mokap.gui.widgets import SharedBase
from lucida.calibration import CharucoBoard, ChessBoard

logger = logging.getLogger(__name__)


class Viewer3D(SharedBase):
    """3D visualisation window."""

    def __init__(self, main_window_ref):
        super().__init__(main_window_ref)

        self._antialiasing = True

        self._cameras_names = tuple(self._mainwindow.cameras_names)

        self._cam_colours_rgba = {
            cam: np.array([*hex_to_rgb(col), 255], dtype=np.uint8)
            for cam, col in self._mainwindow.main_colours.items()
        }
        self._cam_colours_rgba_norm = {
            cam: col / 255 for cam, col in self._cam_colours_rgba.items()
        }

        self.boards_map = {
            "ChArUco": CharucoBoard,
            "Chessboard": ChessBoard
        }
        self.is_editing_board = False

        self._frustum_faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [0, 3, 4], [0, 4, 1],
            [1, 2, 3], [1, 3, 4]
        ])

        # Scale grid and frustum based on board dimensions
        self._gridsize = self._mainwindow.board_params.diagonal * 3.0

        self._percamera_gl_items = {}
        self.global_gl_items = {}

        self._init_ui()
        self._create_gl_items()
        self._connect_signals()
        self._reset_view()

    def _connect_signals(self):
        """Internal signal wiring."""

        # Coordinator -> UI
        self._mainwindow.coordinator.broadcast_stage.connect(self._on_stage_change)
        self._mainwindow.coordinator.broadcast_reset.connect(self._reset_view)

        # UI -> Coordinator
        self.calibration_stage_combo.currentIndexChanged.connect(self._mainwindow.coordinator.set_stage)
        self.run_ba_button.clicked.connect(self._mainwindow.coordinator.trigger_refinement)

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        self.view = GLViewWidget()
        self.view.setWindowTitle('3D viewer')
        self.view.setBackgroundColor('k')
        main_layout.addWidget(self.view, 1)

        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setMaximumWidth(320)
        panel_layout = QVBoxLayout(panel)
        main_layout.addWidget(panel)

        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QGridLayout(controls_group)

        controls_layout.addWidget(QLabel("Stage:"), 0, 0)
        self.calibration_stage_combo = QComboBox()
        self.calibration_stage_combo.addItems(['Intrinsics', 'Extrinsics'])
        self.calibration_stage_combo.currentIndexChanged.connect(
            self._mainwindow.coordinator.set_stage
        )
        controls_layout.addWidget(self.calibration_stage_combo, 0, 1)

        controls_layout.addWidget(QLabel("Origin Cam:"), 1, 0)
        self.origin_camera_combo = QComboBox()
        self.origin_camera_combo.addItems(self._cameras_names)
        self.origin_camera_combo.currentTextChanged.connect(
            self._mainwindow.coordinator.set_origin_camera
        )
        controls_layout.addWidget(self.origin_camera_combo, 1, 1)

        self.run_ba_button = QPushButton("Refine All")
        self.run_ba_button.setStyleSheet(f"background-color: {col_darkgreen}; color: {col_white};")
        self.run_ba_button.clicked.connect(self._mainwindow.coordinator.trigger_refinement)
        controls_layout.addWidget(self.run_ba_button, 2, 0, 1, 2)

        panel_layout.addWidget(controls_group)

        # Board settings
        board_group = QGroupBox("Board Settings")
        board_layout = QGridLayout(board_group)
        board_layout.setColumnStretch(1, 1)

        self.board_preview_label = QLabel()
        self.board_preview_label.setAlignment(Qt.AlignCenter)
        self.board_preview_label.setFixedSize(100, 100)
        board_layout.addWidget(self.board_preview_label, 0, 2, 6, 1)

        board_layout.addWidget(QLabel("Type:"), 0, 0)
        self.board_type_combo = QComboBox()
        self.board_type_combo.addItems(self.boards_map.keys())
        self.board_type_combo.setDisabled(True)
        board_layout.addWidget(self.board_type_combo, 0, 1)

        board_layout.addWidget(QLabel("Grid (RxC):"), 1, 0)
        grid_widget = QWidget()
        grid_layout = QHBoxLayout(grid_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(2, 30)
        self.rows_spin.setDisabled(True)
        grid_layout.addWidget(self.rows_spin)
        grid_layout.addWidget(QLabel("x"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(2, 30)
        self.cols_spin.setDisabled(True)
        grid_layout.addWidget(self.cols_spin)
        board_layout.addWidget(grid_widget, 1, 1)

        board_layout.addWidget(QLabel("Square (cm):"), 2, 0)
        self.sq_len_spin = QDoubleSpinBox()
        self.sq_len_spin.setRange(0.01, 1000.0)
        self.sq_len_spin.setDecimals(2)
        self.sq_len_spin.setSingleStep(0.1)
        self.sq_len_spin.setDisabled(True)
        board_layout.addWidget(self.sq_len_spin, 2, 1)

        self.marker_size_label = QLabel("Marker Size:")
        self.marker_size_spin = QComboBox()
        self.marker_size_spin.addItems(["4", "5", "6", "7"])
        self.marker_size_spin.setDisabled(True)
        board_layout.addWidget(self.marker_size_label, 3, 0)
        board_layout.addWidget(self.marker_size_spin, 3, 1)

        self.margin_label = QLabel("Margin (bits):")
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(1, 10)
        self.margin_spin.setDisabled(True)
        board_layout.addWidget(self.margin_label, 4, 0)
        board_layout.addWidget(self.margin_spin, 4, 1)

        self.padding_label = QLabel("Padding (bits):")
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 10)
        self.padding_spin.setDisabled(True)
        board_layout.addWidget(self.padding_label, 5, 0)
        board_layout.addWidget(self.padding_spin, 5, 1)

        self.edit_board_button = QPushButton("Edit Board")
        self.edit_board_button.setCheckable(True)
        self.edit_board_button.clicked.connect(self._apply_board)
        board_layout.addWidget(self.edit_board_button, 6, 0, 1, 3)

        self.print_board_button = QPushButton("Print Board...")
        self.print_board_button.clicked.connect(self._on_print_board)
        board_layout.addWidget(self.print_board_button, 7, 0, 1, 3)

        for w in [self.board_type_combo, self.rows_spin, self.cols_spin,
                  self.sq_len_spin, self.marker_size_spin, self.margin_spin, self.padding_spin]:
            if hasattr(w, 'valueChanged'):
                w.valueChanged.connect(self._slot_refresh_board_ui)
            elif hasattr(w, 'currentTextChanged'):
                w.currentTextChanged.connect(self._slot_refresh_board_ui)

        panel_layout.addWidget(board_group)

        # I/O
        io_group = QGroupBox("Calibration I/O")
        io_layout = QVBoxLayout(io_group)

        self.load_calib_button = QPushButton("Load from File...")
        self.load_calib_button.clicked.connect(self._on_load_clicked)
        io_layout.addWidget(self.load_calib_button)

        self.save_calib_button = QPushButton("Save to File...")
        self.save_calib_button.clicked.connect(self._on_save_clicked)
        io_layout.addWidget(self.save_calib_button)

        panel_layout.addWidget(io_group)

        panel_layout.addStretch()
        self._refresh_board_ui(self._mainwindow.board_params)

        monitor = self._mainwindow.selected_monitor
        size = min(monitor.height, monitor.width) // 2
        self.resize(size, size)

    def _create_gl_items(self):
        for cam_name in self._cameras_names:
            color = self._cam_colours_rgba_norm[cam_name]
            color_80 = (*color[:3], color[3] * 0.8)
            color_30 = (*color[:3], color[3] * 0.3)

            items = {
                'center': GLScatterPlotItem(pos=np.zeros((1, 3)), color=color, size=10),
                'frustum_mesh': GLMeshItem(
                    vertexes=np.zeros((5, 3)), faces=self._frustum_faces,
                    smooth=self._antialiasing, shader='shaded', glOptions='translucent',
                    drawEdges=True, edgeColor=color_80, color=color_30
                ),
                'optical_axis': GLLinePlotItem(
                    pos=np.zeros((2, 3)), color=color, width=2, antialias=self._antialiasing
                ),
                'detections': GLScatterPlotItem(
                    pos=np.zeros((self._mainwindow.board_params.nb_points, 3)), color=color, size=7, pxMode=True
                )
            }

            for item in items.values():
                item.setVisible(False)
                self.view.addItem(item)
            self._percamera_gl_items[cam_name] = items

        self.global_gl_items['board_3d'] = GLScatterPlotItem(
            pos=np.zeros((self._mainwindow.board_params.nb_points, 3)), color=(1, 0, 1, 0.9), size=8, pxMode=True
        )
        self.global_gl_items['board_3d'].setVisible(False)
        self.view.addItem(self.global_gl_items['board_3d'])

        grid = GLGridItem()
        grid.setSize(self._gridsize * 2, self._gridsize * 2, self._gridsize * 2)
        grid.setSpacing(self._gridsize * 0.1, self._gridsize * 0.1, self._gridsize * 0.1)
        self.global_gl_items['grid'] = grid
        self.view.addItem(grid)
        self.view.opts['distance'] = self._gridsize

    @Slot(dict)
    def update_3d_scene(self, scene_data: dict):
        board_3d = scene_data.get('board_3d')
        frustums_3d = scene_data.get('frustums_3d')
        detections_3d = scene_data.get('detections_3d')
        optical_axes_3d = scene_data.get('optical_axes_3d')
        ready_mask = scene_data.get('ready_mask')

        if frustums_3d is None or optical_axes_3d is None:
            return

        for i, cam_name in enumerate(self._cameras_names):
            self._update_camera_gl_items(
                cam_name, frustums_3d[i], optical_axes_3d[i],
                detections_3d[i] if detections_3d else np.zeros((0, 3)),
                ready_mask[i] if ready_mask is not None else False
            )

        board_plot = self.global_gl_items.get('board_3d')
        if board_3d is not None and board_3d.shape[0] > 0:
            board_plot.setData(pos=board_3d)
            board_plot.setVisible(True)
        else:
            board_plot.setVisible(False)
        pass

    def _update_camera_gl_items(self, cam_name, frustum_points, optical_axis, detection_points, is_ready):
        items = self._percamera_gl_items[cam_name]
        cam_center = optical_axis[0]

        should_show = is_ready

        items['frustum_mesh'].setVisible(should_show)
        items['optical_axis'].setVisible(should_show)

        # Only draw the centre dot if it's visible
        items['center'].setVisible(should_show)

        if should_show:
            items['center'].setData(pos=cam_center[None, :])

        if not should_show:
            items['detections'].setVisible(False)
            return

        items['frustum_mesh'].setMeshData(vertexes=frustum_points, faces=self._frustum_faces)
        items['optical_axis'].setData(pos=optical_axis)
        items['detections'].setData(pos=detection_points)
        items['detections'].setVisible(detection_points.shape[0] > 0)

    def _create_board(self) -> Union['CharucoBoard', 'ChessBoard', None]:

        board_class = self.boards_map[self.board_type_combo.currentText()]

        try:
            if board_class == CharucoBoard:
                return CharucoBoard(
                    rows=self.rows_spin.value(),
                    cols=self.cols_spin.value(),
                    square_length=self.sq_len_spin.value(),
                    marker_size=int(self.marker_size_spin.currentText()),
                    margin=self.margin_spin.value(),
                    padding=self.padding_spin.value()
                )
            elif board_class == ChessBoard:
                return ChessBoard(
                    rows=self.rows_spin.value(), cols=self.cols_spin.value(),
                    square_length=self.sq_len_spin.value()
                )
            else:
                raise AttributeError(f"Board type {board_class} is not supported")

        except Exception as e:
            logger.error(f"Failed to create board: {e}")
            return None

    @Slot()
    def _slot_refresh_board_ui(self):
        if self.is_editing_board:
            board = self._create_board()
            if board:
                self._refresh_board_ui(board)

    def _refresh_board_ui(self, board):
        widgets = [self.board_type_combo, self.rows_spin, self.cols_spin,
                   self.sq_len_spin, self.marker_size_spin, self.margin_spin, self.padding_spin]
        for w in widgets:
            w.blockSignals(True)

        self.board_type_combo.setCurrentText(type(board).__name__)
        self.rows_spin.setValue(board.rows)
        self.cols_spin.setValue(board.cols)
        self.sq_len_spin.setValue(board.square_length)

        is_charuco = type(board) ==CharucoBoard
        if is_charuco:
            self.marker_size_spin.setCurrentText(str(board.markers_size))
            self.margin_spin.setValue(board.margin)
            self.padding_spin.setValue(board.padding)

        for w in widgets:
            w.blockSignals(False)

        for w in [self.marker_size_label, self.marker_size_spin,
                  self.margin_label, self.margin_spin, self.padding_label, self.padding_spin]:
            w.setVisible(is_charuco)

        aspect = board.cols / board.rows
        preview_h, max_w = 100, 120
        preview_w = int(preview_h * aspect)
        if preview_w > max_w:
            preview_w, preview_h = max_w, int(max_w / aspect)
        self.board_preview_label.setFixedSize(preview_w, preview_h)

        board_img = board.to_array()
        if len(board_img.shape) == 3:
            board_img = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
        q_img = QImage(board_img.data, board_img.shape[1], board_img.shape[0],
                       board_img.shape[1], QImage.Format.Format_Grayscale8)
        self.board_preview_label.setPixmap(
            QPixmap.fromImage(q_img).scaled(preview_w, preview_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    @Slot(bool)
    def _apply_board(self, checked):
        self.is_editing_board = checked
        widgets = [self.board_type_combo, self.rows_spin, self.cols_spin,
                   self.sq_len_spin, self.marker_size_spin, self.margin_spin, self.padding_spin]
        for w in widgets:
            w.setEnabled(checked)

        if checked:
            self.edit_board_button.setText("Apply Changes")
            self.edit_board_button.setStyleSheet(f"background-color: {col_orange};")
        else:
            self.edit_board_button.setText("Edit Board")
            self.edit_board_button.setStyleSheet("")
            new_board = self._create_board()
            if new_board:
                self._mainwindow.board_params = new_board
                self._mainwindow.coordinator.handle_board_change(new_board)

    @Slot()
    def _on_print_board(self):
        board = self._mainwindow.board_params
        name = f'{type(board).__name__}_{board.rows}x{board.cols}.svg'
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Printable Board", str(Path.home() / name), "SVG Files (*.svg)"
        )
        if file_path:
            try:
                svg_str = board.to_svg()
                with open(Path(file_path), "w") as f:
                    f.write(svg_str)

            except Exception as e:
                logger.error(f"Failed to save board: {e}")

    @Slot(int)
    def _on_stage_change(self, stage: int):

        self.calibration_stage_combo.blockSignals(True)
        self.calibration_stage_combo.setCurrentIndex(stage)
        self.calibration_stage_combo.blockSignals(False)

        # Hide the board initially when switching stages until detected or solved
        if 'board_3d' in self.global_gl_items:
            self.global_gl_items['board_3d'].setVisible(False)

    @Slot()
    def _reset_view(self):
        """Reset the camera position to look at the centre."""
        self.view.setCameraPosition(distance=self._gridsize * 2, elevation=30, azimuth=45)

    @Slot()
    def _on_load_clicked(self):
        start_dir = str(self._mainwindow.controller.full_path.parent)
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration",
            start_dir,
            "TOML Files (*.toml)"
        )

        if file_path:
            self._mainwindow.coordinator.load_calibration(file_path)

    @Slot()
    def _on_save_clicked(self):
        """Handle save button click locally."""
        start_dir = str(self._mainwindow.controller.full_path.resolve())

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Calibration",
            start_dir,
            "TOML Files (*.toml)"
        )

        if file_path:
            self._mainwindow.coordinator.save_calibration(file_path)