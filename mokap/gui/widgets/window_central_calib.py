import logging
from typing import Union
import cv2
from PySide6.QtCore import Signal, Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QFrame, QVBoxLayout, QGroupBox, QGridLayout, QLabel, QComboBox, QPushButton, \
    QSpinBox, QDoubleSpinBox, QFileDialog, QWidget
from pyqtgraph.opengl import GLGridItem, GLViewWidget, GLScatterPlotItem, GLLinePlotItem, GLMeshItem, MeshData
import numpy as np
import jax.numpy as jnp
from mokap.gui.style.commons import *
from mokap.gui.widgets import BOARD_TYPES
from mokap.gui.widgets.widgets_base import Base
from mokap.utils import hex_to_rgb
from mokap.utils.datatypes import CharucoBoard, ChessBoard
from mokap.utils.geometry.transforms import rotate_points3d, rotate_extrinsics_matrix

logger = logging.getLogger(__name__)


class CentralCalibrationWindow(Base):

    request_board_settings = Signal()
    request_load_calibration = Signal()
    request_save_calibration = Signal()

    def __init__(self, main_window_ref):
        super().__init__(main_window_ref)

        self.nb_cams = main_window_ref.nb_cams
        self.idx = self.nb_cams + 1

        self._cameras_names = self._mainwindow.cameras_names    # Fixed cameras order
        self._origin_camera = self._cameras_names[0]    # default to first one

        self._cam_colours_rgba = {cam: np.array([*hex_to_rgb(col), 255], dtype=np.uint8)
                                  for cam, col in self._mainwindow.cams_colours.items()}
        self._cam_colours_rgba_norm = {cam: col / 255
                                       for cam, col in self._cam_colours_rgba.items()}
        self._sources_shapes = self._mainwindow.sources_shapes

        self.is_editing_board = False

        # Data stores for dynamic points
        self.max_board_points = self._mainwindow.board_params.object_points().shape[0]

        # self.board_points_3d = np.zeros((self.max_board_points, 3))  # Placeholder for global 3D points
        # self.board_points_3d_vis = np.zeros(self.max_board_points, dtype=bool)  # Visibility mask

        # # Per-camera 2D detections
        # self.points_2d = {cam_name: np.zeros((self.max_board_points, 2)) for cam_name in self._cameras_names}
        # self.points_2d_ids = {cam_name: np.array([], dtype=int) for cam_name in self._cameras_names}

        # These are aliases to the (rotated for display) rvec and tvec (filled using the fixed camera order)
        self.optical_axes = np.zeros((self.nb_cams, 3), dtype=np.float32)
        self.cam_positions = np.zeros((self.nb_cams, 3), dtype=np.int32)

        # And a reference to the shared focal point
        self.focal_point = np.zeros((1, 3), dtype=np.float32)

        # Define some constants
        # Points in 2D (as arrays, and using the fixed cameras order)
        self._frustums_points2d = np.stack([np.array([[0, 0],
                                             [self._sources_shapes[cam][1], 0],
                                             [self._sources_shapes[cam][1], self._sources_shapes[cam][0]],
                                             [0, self._sources_shapes[cam][0]]], dtype=np.int32)
                                   for cam in self._cameras_names])
        self._centres_points2d = self._frustums_points2d[:, 2, :] / 2.0

        # Faces as triangles
        self._frustum_faces = np.array([[0, 1, 2], [0, 2, 3]])
        self._volume_verts = np.array([
            [-1., -1., -1.],
            [ 1., -1., -1.],
            [ 1.,  1., -1.],
            [-1.,  1., -1.],
            [-1., -1.,  1.],
            [ 1., -1.,  1.],
            [ 1.,  1.,  1.],
            [-1.,  1.,  1.]
        ], dtype=np.float32)
        self._volume_faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Bottom
            [4, 5, 6], [4, 6, 7],  # Top
            [0, 1, 5], [0, 5, 4],  # Front
            [1, 2, 6], [1, 6, 5],  # Right
            [2, 3, 7], [2, 7, 6],  # Back
            [3, 0, 4], [3, 4, 7]   # Left
        ], dtype=np.int32)

        # References to displayed items
        self._static_items = []
        self._refreshable_items = []
        self._frustum_depth = 200
        self._gridsize = 100
        self._antialiasing = True

        # Dictionary to hold persistent GL items
        self.camera_gl_items = {}
        self.global_gl_items = {}

        # Finish building the UI and initialise the 3D scene
        self._init_ui()

        # Create GL items after the UI exists
        self._create_gl_items()

        # Add the grid now bc no need to update it later
        self.grid = GLGridItem()
        self.grid.setSize(self._gridsize * 2, self._gridsize * 2, self._gridsize * 2)
        self.grid.setSpacing(self._gridsize * 0.1, self._gridsize * 0.1, self._gridsize * 0.1)
        self.view.addItem(self.grid)
        self.view.opts['distance'] = self._gridsize

    def _init_ui(self):
        self.view = GLViewWidget()
        self.view.setWindowTitle('3D viewer')
        self.view.setBackgroundColor('k')

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.view, 1)  # The 3D view takes up most space
        self.setLayout(main_layout)

        # Create a vertical control panel on the right
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_panel.setMaximumWidth(320)
        panel_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel)

        # Group 1: Main controls
        controls_group = QGroupBox("Controls")
        controls_layout = QGridLayout(controls_group)
        controls_layout.addWidget(QLabel("Stage:"), 0, 0)

        self.calibration_stage_combo = QComboBox()
        self.calibration_stage_combo.addItems(['Intrinsics', 'Extrinsics'])
        self.calibration_stage_combo.currentIndexChanged.connect(self._mainwindow.coordinator.set_stage)

        controls_layout.addWidget(self.calibration_stage_combo, 0, 1)
        controls_layout.addWidget(QLabel("Origin Cam:"), 1, 0)

        self.origin_camera_combo = QComboBox()
        self.origin_camera_combo.addItems(self._cameras_names)
        self.origin_camera_combo.currentTextChanged.connect(self._mainwindow.coordinator.set_origin_camera)

        controls_layout.addWidget(self.origin_camera_combo, 1, 1)

        self.run_ba_button = QPushButton("Refine All")
        self.run_ba_button.setStyleSheet(f"background-color: {col_darkgreen}; color: {col_white};")

        controls_layout.addWidget(self.run_ba_button, 2, 0, 1, 2)
        panel_layout.addWidget(controls_group)

        # Group 2: Board settings
        board_group = QGroupBox("Board Settings")
        board_layout = QGridLayout(board_group)
        board_layout.setColumnStretch(1, 1)  # Give controls column priority

        # Board preview
        self.board_preview_label = QLabel()
        self.board_preview_label.setAlignment(Qt.AlignCenter)
        self.board_preview_label.setFixedSize(100, 100)
        self.board_preview_label.setStyleSheet("background-color: transparent; border: none;")
        board_layout.addWidget(self.board_preview_label, 0, 2, 6, 1)

        # Board Type
        board_layout.addWidget(QLabel("Type:"), 0, 0, 1, 2)
        self.board_type_combo = QComboBox()
        self.board_type_combo.addItems(BOARD_TYPES.keys())
        self.board_type_combo.setDisabled(True)
        board_layout.addWidget(self.board_type_combo, 0, 1, 1, 1)

        # Rows and Cols in a single horizontal layout
        board_layout.addWidget(QLabel("Grid (RxC):"), 1, 0)
        grid_widget = QWidget()
        grid_widget_layout = QHBoxLayout(grid_widget)
        grid_widget_layout.setContentsMargins(0, 0, 0, 0)

        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(2, 30)
        self.rows_spin.setDisabled(True)

        grid_widget_layout.addWidget(self.rows_spin)
        grid_widget_layout.addWidget(QLabel("x"))

        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(2, 30)
        self.cols_spin.setDisabled(True)

        grid_widget_layout.addWidget(self.cols_spin)
        board_layout.addWidget(grid_widget, 1, 1)

        # Square Length
        board_layout.addWidget(QLabel("Square (cm):"), 2, 0)
        self.sq_len_spin = QDoubleSpinBox()
        self.sq_len_spin.setRange(0.01, 1000.0)
        self.sq_len_spin.setDecimals(2)
        self.sq_len_spin.setSingleStep(0.1)
        self.sq_len_spin.setDisabled(True)
        board_layout.addWidget(self.sq_len_spin, 2, 1)

        # Markers size
        self.marker_size_label = QLabel("Marker Size:")
        self.marker_size_spin = QComboBox()
        self.marker_size_spin.addItems(["4", "5", "6", "7"])
        self.marker_size_spin.setDisabled(True)
        board_layout.addWidget(self.marker_size_label, 3, 0)
        board_layout.addWidget(self.marker_size_spin, 3, 1)

        # Margin
        self.margin_label = QLabel("Margin (bits):")
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(1, 10)
        self.margin_spin.setDisabled(True)
        board_layout.addWidget(self.margin_label, 4, 0)
        board_layout.addWidget(self.margin_spin, 4, 1)

        # Padding
        self.padding_label = QLabel("Padding (bits):")
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 10)
        self.padding_spin.setDisabled(True)
        board_layout.addWidget(self.padding_label, 5, 0)
        board_layout.addWidget(self.padding_spin, 5, 1)

        # Edit/Apply button
        self.edit_board_button = QPushButton("Edit Board")
        self.edit_board_button.setCheckable(True)
        self.edit_board_button.clicked.connect(self._apply_board)
        board_layout.addWidget(self.edit_board_button, 6, 0, 1, 3)

        # Print button
        self.print_board_button = QPushButton("Print Board...")
        self.print_board_button.clicked.connect(self._on_print_board)

        # connection for live preview updates
        self.board_type_combo.currentTextChanged.connect(self._slot_refresh_board_ui)
        self.rows_spin.valueChanged.connect(self._slot_refresh_board_ui)
        self.cols_spin.valueChanged.connect(self._slot_refresh_board_ui)
        self.sq_len_spin.valueChanged.connect(self._slot_refresh_board_ui)
        self.marker_size_spin.currentTextChanged.connect(self._slot_refresh_board_ui)
        self.margin_spin.valueChanged.connect(self._slot_refresh_board_ui)
        self.padding_spin.valueChanged.connect(self._slot_refresh_board_ui)

        board_layout.addWidget(self.print_board_button, 7, 0, 1, 3)
        panel_layout.addWidget(board_group)

        # Group 3: Calibration files I/O
        io_group = QGroupBox("Calibration I/O")
        io_layout = QVBoxLayout(io_group)

        self.load_calib_button = QPushButton("Load from File...")
        self.load_calib_button.clicked.connect(self.request_load_calibration)
        io_layout.addWidget(self.load_calib_button)

        self.save_calib_button = QPushButton("Save to File...")
        self.save_calib_button.clicked.connect(self.request_save_calibration)
        io_layout.addWidget(self.save_calib_button)
        panel_layout.addWidget(io_group)

        # Final UI setup
        panel_layout.addStretch()

        self._refresh_board_ui(self._mainwindow.board_params)

        # Set initial window size
        if self._mainwindow.selected_monitor.height < self._mainwindow.selected_monitor.width:
            h = w = self._mainwindow.selected_monitor.height // 2
        else:
            h = w = self._mainwindow.selected_monitor.width // 2
        self.resize(w, h)
        self.show()

    def update_scene_single_camera(self, cam_idx: int):
        # simplified version of the scene update method to draw one camera
        # (i.e. not a full back-projection of all cameras)

        # TODO: implement this
        # just call the full update for now
        self.update_scene()

    def _create_board(self) -> Union[CharucoBoard, ChessBoard, None]:
        """ Reads UI widgets and returns a new board object, or None on failure """

        board_class = BOARD_TYPES[self.board_type_combo.currentText()]
        try:
            if board_class == CharucoBoard:
                return CharucoBoard(
                    rows=self.rows_spin.value(), cols=self.cols_spin.value(),
                    square_length=self.sq_len_spin.value(),
                    markers_size=int(self.marker_size_spin.currentText()),
                    margin=self.margin_spin.value(), padding=self.padding_spin.value()
                )
            else:
                return ChessBoard(
                    rows=self.rows_spin.value(), cols=self.cols_spin.value(),
                    square_length=self.sq_len_spin.value()
                )

        except Exception as e:
            logger.error(f"Failed to create board from UI: {e}")
            return None

    @Slot()
    def _slot_refresh_board_ui(self):
        """ A small wrapper slot for calling _refresh_board_ui """
        if self.is_editing_board:
            board = self._create_board()
            if board:
                self._refresh_board_ui(board)

    def _refresh_board_ui(self, board: Union[CharucoBoard, ChessBoard]):
        """ Updates all UI elements to reflect the state of the given board object """

        for w in [self.board_type_combo, self.rows_spin, self.cols_spin, self.sq_len_spin, self.marker_size_spin,
                  self.margin_spin, self.padding_spin]:
            w.blockSignals(True)

        self.board_type_combo.setCurrentText(board.kind.title())
        self.rows_spin.setValue(board.rows)
        self.cols_spin.setValue(board.cols)
        self.sq_len_spin.setValue(board.square_length)

        is_charuco = isinstance(board, CharucoBoard)

        if is_charuco:
            self.marker_size_spin.setCurrentText(str(board.markers_size))
            self.margin_spin.setValue(board.margin)
            self.padding_spin.setValue(board.padding)

        for w in [self.board_type_combo, self.rows_spin, self.cols_spin, self.sq_len_spin, self.marker_size_spin,
                  self.margin_spin, self.padding_spin]:
            w.blockSignals(False)

        # Update widget visibility
        for w in [self.marker_size_label, self.marker_size_spin, self.margin_label, self.margin_spin,
                  self.padding_label, self.padding_spin]:
            w.setVisible(is_charuco)

        # Update preview image with aspect ratio correction
        aspect_ratio = board.cols / board.rows
        preview_h, max_w = 100, 120
        preview_w = int(preview_h * aspect_ratio)
        if preview_w > max_w:
            preview_w, preview_h = max_w, int(max_w / aspect_ratio)

        self.board_preview_label.setFixedSize(preview_w, preview_h)
        board_img = board.to_image((preview_w * 2, preview_h * 2))
        if len(board_img.shape) == 3:
            board_img = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
        q_img = QImage(board_img.data, board_img.shape[1], board_img.shape[0], board_img.shape[1],
                       QImage.Format.Format_Grayscale8)
        self.board_preview_label.setPixmap(
            QPixmap.fromImage(q_img).scaled(preview_w, preview_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @Slot(bool)
    def _apply_board(self, checked: bool):
        """ Handles the 'Edit Board' / 'Apply Changes' button """
        self.is_editing_board = checked

        for w in [self.board_type_combo, self.rows_spin, self.cols_spin, self.sq_len_spin, self.marker_size_spin,
                  self.margin_spin, self.padding_spin]:
            w.setEnabled(checked)

        if checked:
            self.edit_board_button.setText("Apply Changes")
            self.edit_board_button.setStyleSheet(f"background-color: {col_orange};")
        else:
            self.edit_board_button.setText("Edit Board")
            self.edit_board_button.setStyleSheet("")

            new_board = self._create_board()
            if new_board:
                # Commit successful changes to the system
                self._mainwindow.board_params = new_board
                self._mainwindow.coordinator.handle_board_change(new_board)
                logger.info("Board parameters applied successfully.")

            else:
                # On failure, revert UI to last known good state
                logger.error("Could not apply board settings. Reverting UI.")
                self._refresh_board_ui(self._mainwindow.board_params)

    @Slot()
    def _on_print_board(self):
        """ Opens a dialog to save the current board as a printable SVG file """
        board = self._mainwindow.board_params

        # Suggest a filename based on board parameters
        if isinstance(board, CharucoBoard):
            suggested_name = f'Charuco_{board.rows}x{board.cols}_sq{board.square_length}cm_mk{board.marker_length}cm.svg'
        else:
            suggested_name = f'Chessboard_{board.rows}x{board.cols}_sq{board.square_length}cm.svg'

        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Printable Board",
            str(Path.home() / suggested_name),  # start in user's home directory
            "SVG Files (*.svg)"
        )

        if file_path:
            try:
                save_dir = Path(file_path).parent
                board.to_file(save_dir)
                logger.info(f"Successfully saved printable board to {save_dir}")

            except Exception as e:
                logger.error(f"Could not save board to file: {e}")

    def _create_gl_items(self):

        # Create global items once
        self.global_gl_items['board_3d'] = GLScatterPlotItem(
            pos=np.zeros((self.max_board_points, 3)),
            color=(1, 0, 1, 0.9), size=8, pxMode=True
        )
        self.global_gl_items['board_3d'].setVisible(False)
        self.view.addItem(self.global_gl_items['board_3d'])
        # TODO: Add volume of trust here

        for i, cam_name in enumerate(self._cameras_names):
            color = self._cam_colours_rgba_norm[cam_name]
            color_translucent_80 = (*color[:3], color[3] * 0.8)
            color_translucent_50 = (*color[:3], color[3] * 0.5)

            detections_scatter = GLScatterPlotItem(pos=np.zeros((self.max_board_points, 3)),
                                                   color=color, size=7, pxMode=True)
            detections_scatter.setVisible(False)    # initially hidden

            # Create items with placeholder data
            center_scatter = GLScatterPlotItem(pos=np.zeros((1, 3)), color=color, size=10)

            # 4 corners of frustum + camera center = 5 vertices. 4 lines from center to corners.
            frustum_lines = GLLinePlotItem(pos=np.zeros((8, 3)), color=color, width=1, antialias=self._antialiasing,
                                           mode='lines')

            frustum_mesh = GLMeshItem(vertexes=np.zeros((4, 3)), faces=self._frustum_faces,
                                      smooth=self._antialiasing, shader='shaded', glOptions='translucent',
                                      drawEdges=True, edgeColor=color_translucent_80, color=color_translucent_50)

            # Dashed lines for optical axis requires are slow
            # so we replace them with a solid line for now
            # TODO: custom shader for that
            optical_axis_line = GLLinePlotItem(pos=np.zeros((2, 3)), color=color, width=2, antialias=self._antialiasing)

            # Store references to the items
            self.camera_gl_items[cam_name] = {
                'center': center_scatter,
                'frustum_lines': frustum_lines,
                'frustum_mesh': frustum_mesh,
                'optical_axis': optical_axis_line,
                'detections': detections_scatter
            }

            # Add items to the view
            self.view.addItem(self.camera_gl_items[cam_name]['center'])
            self.view.addItem(self.camera_gl_items[cam_name]['frustum_lines'])
            self.view.addItem(self.camera_gl_items[cam_name]['frustum_mesh'])
            self.view.addItem(self.camera_gl_items[cam_name]['optical_axis'])
            self.view.addItem(self.camera_gl_items[cam_name]['detections'])

    @Slot(dict)
    def on_scene_data_ready(self, scene_data: dict):
        """ Receives pre-computed 3D data from the worker and updates GL items """

        all_E = scene_data['extrinsics']
        all_frustums_3d = scene_data['frustums_3d']
        all_centers_3d = scene_data['centers_3d']
        all_detections_3d = scene_data['detections_3d']

        for i, cam_name in enumerate(self._cameras_names):
            if not np.all(np.isfinite(all_frustums_3d[i])):
                continue

            self._update_camera_gl(
                cam_name=cam_name,
                extrinsics_mat=all_E[i],
                frustum_points3d=all_frustums_3d[i],
                center_point3d=all_centers_3d[i],
                detection_points3d=all_detections_3d[i]
            )

    def _update_camera_gl(self, cam_name: str, extrinsics_mat: jnp.ndarray, frustum_points3d: jnp.ndarray, center_point3d: jnp.ndarray, detection_points3d: jnp.ndarray):
        """ Only updates GL item data with pre-calculated 3D points
        All points are rotated 180 degrees on Y for correct visualization """

        if not np.all(np.isfinite(frustum_points3d)):
            return

        gl_items = self.camera_gl_items[cam_name]

        # Rotate all incoming points for visualization
        E_rot = rotate_extrinsics_matrix(extrinsics_mat, 180, axis='y')
        cam_center_pos_rot = E_rot[:3, 3].reshape(1, -1)
        frustum_points3d_rot = rotate_points3d(frustum_points3d, 180, axis='y')
        center_point3d_rot = rotate_points3d(center_point3d, 180, axis='y')
        detection_points3d_rot = rotate_points3d(detection_points3d, 180, axis='y')

        # Update GL Items with new data
        gl_items['center'].setData(pos=cam_center_pos_rot)
        gl_items['frustum_mesh'].setMeshData(vertexes=frustum_points3d_rot)
        gl_items['optical_axis'].setData(pos=np.vstack([cam_center_pos_rot, center_point3d_rot]))
        gl_items['detections'].setData(pos=detection_points3d_rot)
        gl_items['detections'].setVisible(detection_points3d_rot.shape[0] > 0)

        # Update frustum lines connecting center to corners
        line_verts = np.empty((8, 3))
        line_verts[0:2] = np.vstack([cam_center_pos_rot, frustum_points3d_rot[0]])
        line_verts[2:4] = np.vstack([cam_center_pos_rot, frustum_points3d_rot[1]])
        line_verts[4:6] = np.vstack([cam_center_pos_rot, frustum_points3d_rot[2]])
        line_verts[6:8] = np.vstack([cam_center_pos_rot, frustum_points3d_rot[3]])

        gl_items['frustum_lines'].setData(pos=line_verts)

    # TODO: These should migrate to the Multiview Worker
    def update_3d_points(self):
        """ Updates the global 3D board points scatter plot """
        board_plot = self.global_gl_items['board_3d']

        # Get only the visible points
        visible_points = self.board_points_3d[self.board_points_3d_vis]

        if visible_points.shape[0] > 0:
            points3d_rot = rotate_points3d(visible_points, 180, axis='y')
            board_plot.setData(pos=points3d_rot)
            board_plot.setVisible(True)
        else:
            board_plot.setVisible(False)

    def add_cube(self, center: np.ndarray, size: float | np.ndarray, color=(1, 1, 1, 0.5)):
        """
        Add a cube centered at "center" with the given "size"
        TODO: we prob want a rectangular cuboid or an ellipsoid instead
        """

        color_translucent_50 = np.copy(color)
        color_translucent_50[3] *= 0.5

        color = tuple(color)
        color_translucent_50 = tuple(color_translucent_50)

        hsize = np.asarray(size) * 0.5
        if hsize.shape == ():
            hsize = np.array([hsize, hsize, hsize])
        if len(hsize) != 3:
            raise AttributeError("Volume size must be a scalar or a vector 3!")

        vertices = (self._volume_verts * hsize) + np.asarray(center)
        meshdata = MeshData(vertexes=vertices, faces=self._volume_faces)
        cube = GLMeshItem(meshdata=meshdata,
                             smooth=self._antialiasing,
                             shader='shaded',
                             glOptions='translucent',
                             drawEdges=True,
                             edgeColor=color,
                             color=color_translucent_50)
        self._refreshable_items.append(cube)

    def add_focal_point(self, color=(1, 1, 1, 1)):
        color = tuple(color)
        focal_scatter = GLScatterPlotItem(pos=self.focal_point, color=color, size=5)
        self._refreshable_items.append(focal_scatter)

    def add_dashed_line(self, start: np.ndarray, end: np.ndarray, dash_length=5.0, gap_length=5.0, color=(1, 1, 1, 1), width=1, antialias=True):

        color = tuple(color)

        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        vec = end - start
        total_length = np.linalg.norm(vec)
        if total_length == 0:
            return
        direction = vec / total_length
        step = dash_length + gap_length

        num_steps = int(total_length // step)

        for i in range(num_steps + 1):
            seg_start = start + i * step * direction
            seg_end = seg_start + dash_length * direction

            # clamp the segment end so it doesn't overshoot
            if np.linalg.norm(seg_end - start) > total_length:
                seg_end = end
            line_seg = GLLinePlotItem(pos=np.array([seg_start, seg_end]),
                                         color=color,
                                         width=width,
                                         antialias=antialias)
            self._refreshable_items.append(line_seg)

