from PySide6.QtCore import Signal, Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QFrame, QVBoxLayout, QGroupBox, QGridLayout, QLabel, QComboBox, QPushButton
from pyqtgraph.opengl import GLGridItem, GLViewWidget, GLScatterPlotItem, GLLinePlotItem, GLMeshItem, MeshData
import numpy as np

from mokap.gui.style.commons import *
from mokap.gui.widgets import VERBOSE
from mokap.gui.widgets.base import Base
from mokap.utils import hex_to_rgb
from mokap.utils.datatypes import CalibrationData, ExtrinsicsPayload, IntrinsicsPayload, DetectionPayload
from mokap.utils.geometry.projective import back_projection_batched, back_projection
from mokap.utils.geometry.transforms import extrinsics_matrix, rotate_points3d, rotate_extrinsics_matrix


class Multiview3D(Base):

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

        # Data stores for visualization remain the same
        self._cameras_matrices = np.zeros((self.nb_cams, 3, 3), dtype=np.float32)
        self._dist_coeffs = np.zeros((self.nb_cams, 14), dtype=np.float32)
        self._rvecs = np.zeros((self.nb_cams, 3), dtype=np.float32)
        self._tvecs = np.zeros((self.nb_cams, 3), dtype=np.float32)

        # Data stores for dynamic points
        self.max_board_points = self._mainwindow.board_params.object_points().shape[0]
        self.board_points_3d = np.zeros((self.max_board_points, 3))  # Placeholder for global 3D points
        self.board_points_3d_vis = np.zeros(self.max_board_points, dtype=bool)  # Visibility mask

        # Per-camera 2D detections
        self.points_2d = {cam_name: np.zeros((self.max_board_points, 2)) for cam_name in self._cameras_names}
        self.points_2d_ids = {cam_name: np.array([], dtype=int) for cam_name in self._cameras_names}

        # These will be aliases to the (rotated for display) rvec and tvec (filled using the fixed camera order)
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
        control_panel.setMaximumWidth(250)
        panel_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel)

        # Group for calibration controls
        controls_group = QGroupBox("Controls")
        controls_layout = QGridLayout()
        controls_group.setLayout(controls_layout)

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

        self.run_ba_button = QPushButton("Run Bundle Adjustment")
        self.run_ba_button.setStyleSheet(
            f"background-color: {col_darkgreen}; color: {col_white};")
        controls_layout.addWidget(self.run_ba_button, 2, 0, 1, 2)
        panel_layout.addWidget(controls_group)

        # Group for Board and I/O controls
        io_group = QGroupBox("Global Settings")
        io_layout = QVBoxLayout(io_group)

        # board preview
        self.board_preview_label = QLabel()
        self.board_preview_label.setAlignment(Qt.AlignCenter)
        io_layout.addWidget(self.board_preview_label)

        self.board_settings_button = QPushButton("Edit board")
        self.board_settings_button.clicked.connect(self.request_board_settings)
        io_layout.addWidget(self.board_settings_button)

        self.load_calib_button = QPushButton("Load calibration")
        self.load_calib_button.clicked.connect(self.request_load_calibration)
        io_layout.addWidget(self.load_calib_button)

        self.save_calib_button = QPushButton("Save calibration")
        self.save_calib_button.clicked.connect(self.request_save_calibration)
        io_layout.addWidget(self.save_calib_button)

        panel_layout.addWidget(io_group)

        panel_layout.addStretch()        # Pushes everything to the top

        # If landscape screen
        if self._mainwindow.selected_monitor.height < self._mainwindow.selected_monitor.width:
            h = w = self._mainwindow.selected_monitor.height // 2
        else:
            h = w = self._mainwindow.selected_monitor.width // 2

        self.resize(h, w)
        self.show()

    @Slot(CalibrationData)
    def handle_payload(self, data: CalibrationData):

        if VERBOSE:
            print(f'[3D Widget] Received {data.camera_name} {data.payload}')

        cam_idx = self._cameras_names.index(data.camera_name)
        needs_redraw = False

        if isinstance(data.payload, ExtrinsicsPayload):
            self._rvecs[cam_idx] = data.payload.rvec
            self._tvecs[cam_idx] = data.payload.tvec
            needs_redraw = True

        elif isinstance(data.payload, IntrinsicsPayload):
            self._cameras_matrices[cam_idx] = data.payload.camera_matrix
            dist_len = len(data.payload.dist_coeffs)
            self._dist_coeffs[cam_idx, :dist_len] = data.payload.dist_coeffs
            needs_redraw = True

        elif isinstance(data.payload, DetectionPayload):
            self.points_2d[data.camera_name] = data.payload.points2D
            self.points_2d_ids[data.camera_name] = data.payload.pointsIDs
            self._update_camera_gl(cam_idx, data.camera_name)

        # TODO: Triangulated board points
        # Placeholder global 3D points update
        # elif isinstance(data.payload, TriangulatedPointsPayload):
        #     self.board_points_3d = data.payload.points3D
        #     self.board_points_3d_vis = data.payload.visibility_mask
        #     self.update_3d_points()

        if needs_redraw:
            self.update_scene()

    def update_board_preview(self, board_params):
        max_w, max_h = 100, 100
        board_img = board_params.to_image((max_w * 2, max_w * 2))
        h, w = board_img.shape
        q_img = QImage(board_img.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        bounded_pixmap = pixmap.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.board_preview_label.setPixmap(bounded_pixmap)

    def _create_gl_items(self):
        """
        Creates all GL items for all cameras ONCE and adds them to the scene.
        """
        for i, cam_name in enumerate(self._cameras_names):
            color = self._cam_colours_rgba_norm[cam_name]
            color_translucent_80 = (*color[:3], color[3] * 0.8)
            color_translucent_50 = (*color[:3], color[3] * 0.5)

            detections_scatter = GLScatterPlotItem(pos=np.zeros((self.max_board_points, 3)),
                                                   color=color, size=7, pxMode=True)
            detections_scatter.setVisible(False)    # Initially hidden

            # Create items with placeholder data
            center_scatter = GLScatterPlotItem(pos=np.zeros((1, 3)), color=color, size=10)

            # 4 corners of frustum + camera center = 5 vertices. 4 lines from center to corners.
            frustum_lines = GLLinePlotItem(pos=np.zeros((8, 3)), color=color, width=1, antialias=self._antialiasing,
                                           mode='lines')

            frustum_mesh = GLMeshItem(vertexes=np.zeros((4, 3)), faces=self._frustum_faces,
                                      smooth=self._antialiasing, shader='shaded', glOptions='translucent',
                                      drawEdges=True, edgeColor=color_translucent_80, color=color_translucent_50)

            # Dashed line for optical axis requires multiple line items, which is inefficient to update.
            # We'll replace it with a single, solid line for now for performance.
            # If dashes are required, a custom shader is the performant way.
            optical_axis_line = GLLinePlotItem(pos=np.zeros((2, 3)), color=color, width=2, antialias=self._antialiasing)

            # Store references to the items
            self.camera_gl_items[cam_name] = {
                'center': center_scatter,
                'frustum_lines': frustum_lines,
                'frustum_mesh': frustum_mesh,
                'optical_axis': optical_axis_line,
                'detections': detections_scatter
            }

            # --- Create Global Items ---
            # Create scatter plot for 3D board points
            self.global_gl_items['board_3d'] = GLScatterPlotItem(pos=np.zeros((self.max_board_points, 3)),
                                                                 color=(1, 0, 1, 0.9), size=8, pxMode=True)
            self.global_gl_items['board_3d'].setVisible(False)  # Initially hidden

            # TODO: Add volume

            # Add items to the view
            self.view.addItem(center_scatter)
            self.view.addItem(frustum_lines)
            self.view.addItem(frustum_mesh)
            self.view.addItem(optical_axis_line)
            self.view.addItem(detections_scatter)
            self.view.addItem(self.global_gl_items['board_3d'])

    def update_scene(self):

        # # --- PROFILING CODE START ---
        # profiler = cProfile.Profile()
        # profiler.enable()
        # # --- PROFILING CODE END ---

        origin_name = self._mainwindow.coordinator._origin_camera_name

        # --- Prepare Batched Data ---
        # Note: self._rvecs and self._tvecs are already batched (5, 3)
        all_E = extrinsics_matrix(self._rvecs, self._tvecs)  # (5, 4, 4)
        all_K = self._cameras_matrices  # (5, 3, 3)
        all_D = self._dist_coeffs  # (5, 14)

        # --- Perform Batched Calculations ---
        all_frustums_3d = back_projection_batched(
            self._frustums_points2d,  # (5, 4, 2)
            self._frustum_depth,  # scalar, will be broadcast
            all_K,  # (5, 3, 3)
            all_E,  # (5, 4, 4)
            all_D  # (5, 14)
        ) # (5, 4, 3)

        # same for the centers
        # all_centers_3d = back_projection_batched(
        #     self._centres_points2d,  # (5, 2) -> will be treated as (5, 1, 2)
        #     self._frustum_depth,
        #     all_K, all_E, all_D
        # )
        # # Result will be (5, 1, 3), so we squeeze it
        # all_centers_3d = all_centers_3d.squeeze(axis=1)

        # TODO: Compute volume

        # Loop through the results and update each camera's visualization
        # This Python loop is fine because all the heavy math is already done
        for i, cam_name in enumerate(self._cameras_names):
            # We pass the pre-computed 3D points for this specific camera to the update helper.
            self._update_camera_gl(i, cam_name, frustum_points3d=all_frustums_3d[i])

        # # --- PROFILING CODE START ---
        # profiler.disable()
        # stats = pstats.Stats(profiler).sort_stats('cumtime')  # Sort by cumulative time spent
        # stats.print_stats(20)  # Print the top 20 time-consuming functions
        # # --- PROFILING CODE END ---

    def _update_camera_gl(self, cam_idx: int, cam_name: str, frustum_points3d: np.ndarray):
        """
        Updates the data of existing GL items for a specific camera.
        """
        # Retrieve the persistent GL items for this camera
        gl_items = self.camera_gl_items[cam_name]

        # --- We already have the frustum points, just rotate them for visualization ---
        # The JAX rotation function is fast enough to be called in a loop.
        frustum_points3d_rot = rotate_points3d(frustum_points3d, 180, axis='y')

        # --- We still need to calculate the camera's rotated center and axis ---
        E = extrinsics_matrix(self._rvecs[cam_idx], self._tvecs[cam_idx])
        E_rot = rotate_extrinsics_matrix(E, 180, axis='y')
        center_pos = E_rot[:3, 3].reshape(1, -1)

        # Calculate the optical axis endpoint
        # Using the non-vmapped function for a single point
        centre3d = back_projection(self._centres_points2d[cam_idx], self._frustum_depth,
                                           self._cameras_matrices[cam_idx], E, self._dist_coeffs[cam_idx])
        centre3d_rot = rotate_points3d(centre3d, 180, axis='y')

        # --- Update GL Items with new data ---

        # Update camera center scatter plot
        gl_items['center'].setData(pos=center_pos)

        # Update the frustum mesh, with a safety check
        if frustum_points3d_rot.shape == (4, 3):
            gl_items['frustum_mesh'].setMeshData(vertexes=frustum_points3d_rot)
            gl_items['frustum_mesh'].setVisible(True)
        else:
            gl_items['frustum_mesh'].setVisible(False)

        # Update the optical axis line
        gl_items['optical_axis'].setData(pos=np.vstack([center_pos, centre3d_rot]))

        # Update the lines from the camera center to the frustum corners
        line_verts = np.empty((8, 3))
        if frustum_points3d_rot.shape == (4, 3):
            line_verts[0:2] = np.vstack([center_pos, frustum_points3d_rot[0]])
            line_verts[2:4] = np.vstack([center_pos, frustum_points3d_rot[1]])
            line_verts[4:6] = np.vstack([center_pos, frustum_points3d_rot[2]])
            line_verts[6:8] = np.vstack([center_pos, frustum_points3d_rot[3]])
            gl_items['frustum_lines'].setData(pos=line_verts)
            gl_items['frustum_lines'].setVisible(True)
        else:
            gl_items['frustum_lines'].setVisible(False)

        # Update the 2D detections scatter plot (this logic remains the same)
        ids = self.points_2d_ids.get(cam_name)  # Use .get for safety
        if ids is not None and len(ids) > 0:
            points2d_subset = self.points_2d[cam_name][:len(ids)]

            # Use the single, non-vmapped back projection here too
            points3d_detections = back_projection(points2d_subset, self._frustum_depth * 0.95,
                                                          self._multi_cameras_matrices[cam_idx], E,
                                                          self._multi_dist_coeffs[cam_idx])
            points3d_rot = rotate_points3d(points3d_detections, 180, axis='y')

            gl_items['detections'].setData(pos=points3d_rot)
            gl_items['detections'].setVisible(True)
        else:
            gl_items['detections'].setVisible(False)

    def update_3d_points(self):
        """
        Updates the global 3D board points scatter plot
        """
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

    @Slot(CalibrationData)
    def handle_payload(self, data: CalibrationData):
        print(f'[Main Thread] Received {data.camera_name} {data.payload}')

        cam_idx = self._cameras_names.index(data.camera_name)

        if isinstance(data.payload, ExtrinsicsPayload):
            self._rvecs[cam_idx, :] = data.payload.rvec
            self._tvecs[cam_idx, :] = data.payload.tvec

        elif isinstance(data.payload, IntrinsicsPayload):
            self._cameras_matrices[cam_idx, :, :] = data.payload.camera_matrix
            self._dist_coeffs[cam_idx, :len(data.payload.dist_coeffs)] = data.payload.dist_coeffs

        self.update_scene()
