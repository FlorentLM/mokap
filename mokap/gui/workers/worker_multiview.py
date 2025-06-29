import logging

import jax
import numpy as np
import jax.numpy as jnp
from typing import List, Optional, Union, Dict
from PySide6.QtCore import QTimer, Slot, Signal
from numpy.typing import ArrayLike
from mokap.calibration.multiview import MultiviewCalibrationTool
from mokap.gui.workers.workers_base import CalibrationProcessingWorker
from mokap.utils.datatypes import (CalibrationData, DetectionPayload, ExtrinsicsPayload, IntrinsicsPayload,
                                   ChessBoard, CharucoBoard)
from mokap.utils.geometry.projective import back_projection_batched, back_projection
from mokap.utils.geometry.transforms import extrinsics_matrix, invert_rtvecs, rotate_extrinsics_matrix, rotate_points3d, \
    rotate_extrinsics_matrices, Rmat_from_angle, invert_extrinsics_matrix, rotate_points3d_sets, rodrigues

logger = logging.getLogger(__name__)


class MultiviewWorker(CalibrationProcessingWorker):

    scene_data_ready = Signal(dict)

    def __init__(self,
                 cameras_names:     List[str],
                 origin_camera:     str,
                 sources_shapes_wh: Dict[str, ArrayLike],
                 calibration_board: Union[ChessBoard, CharucoBoard]):
        super().__init__(name='multiview')

        # Configuration and static data
        self._cameras_names = cameras_names
        self._orig_cam_name = origin_camera
        self._orig_cam_idx = self._cameras_names.index(self._orig_cam_name)

        self._C = len(cameras_names)

        self._sources_shapes_wh = sources_shapes_wh  # Expects {cam_name: (w, h)}
        self.calibration_board = calibration_board

        # Store board object points in homogenous coordinates for transforms
        self._object_points_hom = np.hstack([
            self.calibration_board.object_points,
            np.ones((self.calibration_board.object_points.shape[0], 1))
        ])

        self.multiview_tool: Optional[MultiviewCalibrationTool] = None

        # Local state for visualisation
        # This worker holds the "master copy" of all camera parameters for the scene
        self._intrinsics_ready = np.zeros(self._C, dtype=bool)
        self._cameras_matrices = np.array([np.eye(3)] * self._C, dtype=np.float32)
        self._dist_coeffs = np.zeros((self._C, 8), dtype=np.float32)

        # We start with no extrinsics, they will be estimated or loaded
        self._rvecs_c2w = np.zeros((self._C, 3), dtype=np.float32)
        self._tvecs_c2w = np.zeros((self._C, 3), dtype=np.float32)
        self._rvecs_c2w[self._orig_cam_idx] = np.zeros(3)  # Origin is fixed at (0, 0, 0)
        self._tvecs_c2w[self._orig_cam_idx] = np.zeros(3)

        # Define static 2D points for frustum visualization
        img_points_2d = np.array([
            [[w / 2, h / 2],    # Principal point (for optical axis)
             [0, 0],            # Top-left corner
             [w, 0],            # Top-right corner
             [w, h],            # Bottom-right corner
             [0, h]             # Bottom-left corner
             ] for w, h in sources_shapes_wh.values()], dtype=np.float32)

        self._img_points_2d = jnp.asarray(img_points_2d)  # (C, 5, 2)

        # Buffer for per-frame 2D detections for visualization
        self._points_2d: Dict[str, np.ndarray] = {name: np.zeros((0, 2)) for name in self._cameras_names}
        self._points_ids: Dict[str, np.ndarray] = {name: np.array([]) for name in self._cameras_names}

        # Empty arrays to emit nodata when needed
        self._nopoints_2d = np.zeros((0, 2))
        self._nopoints_ids = np.array([])
        self._nopoints_3d = np.zeros((0, 3))

        self._frustum_depth = 200.0 # TODO: compute this automatically

        self._min_ba_samples = 100  # TODO: GUI access to these
        self._max_ba_samples = 150

        # Timer for sending scene data to the 3D view
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(33)  # ~ 30 Hz
        self.update_timer.timeout.connect(self._compute_3d_scene)

        self.update_timer.start()

    def _try_create_tool(self):
        """ Creates the MultiviewCalibrationTool once all intrinsics are available """

        if self.multiview_tool is None and self._current_stage > 0 and np.all(self._intrinsics_ready):
            logger.info(
                f"[{self.name.title()}] All intrinsics received for Stage {self._current_stage}. Creating Multiview tool.")

            # Convert dict to ordered list for the tool
            image_sizes_wh_list = [self._sources_shapes_wh[name] for name in self._cameras_names]

            self.multiview_tool = MultiviewCalibrationTool(
                nb_cameras=self._C,
                images_sizes_wh=np.array(image_sizes_wh_list),
                origin_idx=self._orig_cam_idx,
                init_cam_matrices=self._cameras_matrices,
                init_dist_coeffs=self._dist_coeffs,
                object_points=self.calibration_board.object_points,
                min_detections=self._min_ba_samples,
                max_detections=self._max_ba_samples
            )

    @Slot(CalibrationData)
    def on_payload_received(self, data: CalibrationData):
        """ Handles all incoming payloads and updates the worker's state """

        cam_idx = self._cameras_names.index(data.camera_name)
        payload = data.payload

        # Set the 'ready' flag as soon as intrinsics arrive
        if isinstance(payload, IntrinsicsPayload):
            self._cameras_matrices[cam_idx] = payload.camera_matrix

            d_len = len(payload.dist_coeffs)
            self._dist_coeffs[cam_idx, :d_len] = payload.dist_coeffs

            # Mark this camera as ready for rendering
            self._intrinsics_ready[cam_idx] = True

            # If we are already in stage 1, we might need to create the tool
            # (if intrinsics were loaded from a file after switching)
            self._try_create_tool()

        # Only process extrinsics during the initial seeding phase. After that, the tool takes over.
        elif self._current_stage == 0 and isinstance(payload, ExtrinsicsPayload):

            if payload.rvec is not None and payload.tvec is not None:
                r_c2w, t_c2w = invert_rtvecs(payload.rvec, payload.tvec)
                self._rvecs_c2w[cam_idx] = r_c2w
                self._tvecs_c2w[cam_idx] = t_c2w

        # Accept detection payloads in any stage (for visualization)
        elif isinstance(payload, DetectionPayload):
            # In stage > 0, also register the detection with the BA tool
            if self._current_stage > 0 and self.multiview_tool:
                self.multiview_tool.register(cam_idx, payload)

            # Always store the 2D points for visualization, regardless of stage
            self._points_2d[data.camera_name] = payload.points2D if payload.points2D is not None else self._nopoints_2d
            self._points_ids[data.camera_name] = payload.pointsIDs if payload.pointsIDs is not None else self._nopoints_ids

    def _compute_3d_scene(self):
        """ Periodically calculates and emits all data needed for the 3D view """

        # Determine which set of parameters to use for visualization
        if self.multiview_tool and self.multiview_tool.is_refined:
            Ks, Ds = self.multiview_tool.refined_intrinsics
            rs_c2w, ts_c2w = self.multiview_tool.refined_extrinsics
            # In this case, all intrinsics are guaranteed to be valid
            ready_mask = np.ones(self._C, dtype=bool)

        elif self.multiview_tool and self.multiview_tool.is_estimated:
            Ks, Ds = self.multiview_tool.intrinsics
            rs_c2w, ts_c2w = self.multiview_tool.extrinsics
            # In this case, all intrinsics are guaranteed to be valid
            ready_mask = np.ones(self._C, dtype=bool)

        else:
            # Before estimation, use the initial parameters stored in the worker
            Ks, Ds, rs_c2w, ts_c2w = self._cameras_matrices, self._dist_coeffs, self._rvecs_c2w, self._tvecs_c2w
            # Here, we must rely on the live-updated ready flag
            ready_mask = self._intrinsics_ready.astype(bool)

        if Ks is None or rs_c2w is None or ts_c2w is None:
            # nothing to draw, early exit
            return

        Es_c2w = extrinsics_matrix(rs_c2w, ts_c2w)
        cam_centres = Es_c2w[:, :3, 3]

        # Back-project the 5 points (principal + 4 corners) into 3D space
        frustums_points_all = back_projection_batched(self._img_points_2d,
                                                      40,
                                                      Ks, Es_c2w, jnp.zeros_like(Ds),   # TODO: No need to create this at each call
                                                      distortion_model='full')

        # Safety Check
        # Mask out frustums for cameras whose intrinsics haven't arrived yet
        # This prevents rendering a valid pose with an invalid (identity) K matrix
        masked_frustums = jnp.where(
            ready_mask[:, None, None],
            frustums_points_all,
            cam_centres[:, None, :]  # Collapse the frustum to a single point if not ready
        )

        # Proceed with the rest of the rendering using the safe, masked data
        principal_points = masked_frustums[:, 0, :]
        frustum_corners = masked_frustums[:, 1:, :]

        frustums_points_3d = jnp.concatenate([cam_centres[:, None, :], frustum_corners], axis=1)
        optical_axes_3d = jnp.stack([cam_centres, principal_points], axis=1)

        # --- Stage-dependent visualisation logic ---
        board_3d = None
        detections_3d = [self._nopoints_3d] * self._C

        # Stage 0: Board is at origin, cameras "orbit" around it
        if self._current_stage == 0:
            board_3d = self.calibration_board.object_points

            # Detections are points on the board so their 3D coords are known from their IDs
            temp_detections = []
            for i in range(self._C):
                ids = self._points_ids[self._cameras_names[i]]
                if ids.shape[0] > 0 and board_3d is not None:
                    # Select the corresponding 3D points from the master list
                    temp_detections.append(board_3d[ids])
                else:
                    temp_detections.append(self._nopoints_3d)
            detections_3d = temp_detections

        # Stage > 0: Cameras are static, the board moves
        elif self._current_stage > 0:

            if self.multiview_tool and self.multiview_tool.current_board_pose is not None:
                # If we have a valid board pose, transform the board object points
                board_pose = self.multiview_tool.current_board_pose
                board_3d = (board_pose @ self._object_points_hom.T).T[:, :3]

                # Detections are the specific points from the transformed board
                temp_detections = []
                for i in range(self._C):
                    ids = self._points_ids[self._cameras_names[i]]
                    if ids.shape[0] > 0 and board_3d is not None:
                        temp_detections.append(board_3d[ids])
                    else:
                        temp_detections.append(self._nopoints_3d)
                detections_3d = temp_detections
            # else:
            # In Stage > 0 but without a board pose yet, board_3d remains None
            # and detections_3d remains a list of empty arrays. This is fine,
            # as it will correctly show just the static cameras until a board is detected

        def to_gl(points):
            if points is None or points.shape[0] == 0:
                return points
            return rotate_points3d(points, 180, 'x')

        def to_gl_batch(points_batch):
            if points_batch is None or points_batch.shape[0] == 0:
                return points_batch
            return rotate_points3d_sets(points_batch, 180, 'x')

        scene_data = {
            'ready_mask': ready_mask,
            'board_3d': to_gl(board_3d),
            'frustums_points_3d': to_gl_batch(frustums_points_3d),
            'optical_axes_3d': to_gl(optical_axes_3d),
            'detections_3d': [to_gl(d) for d in detections_3d]
        }
        self.scene_data_ready.emit(scene_data)

    @Slot()
    def trigger_refinement(self):
        """ Slot connected to the GUI button to trigger the final BA """

        if self._paused or self.multiview_tool is None:
            logger.warning("Cannot trigger refinement: Worker paused or tool not initialized.")
            return

        logger.info(f"[{self.name.title()}] Attempting to run final Bundle Adjustment.")

        self.blocking.emit(True)
        success = self.multiview_tool.refine_all()
        self.blocking.emit(False)

        if success:
            logger.info(f"[{self.name.title()}] Bundle Adjustment successful. Emitting refined parameters.")

            K_opts, D_opts = self.multiview_tool.refined_intrinsics
            r_opts, t_opts = self.multiview_tool.refined_extrinsics

            # Update worker's internal state with the new best parameters
            self._cameras_matrices, self._dist_coeffs = K_opts, D_opts
            self._rvecs_c2w, self._tvecs_c2w = r_opts, t_opts

            # Emit the final results for other workers and for saving
            for i, cam_name in enumerate(self._cameras_names):
                self.send_payload.emit(CalibrationData(cam_name, IntrinsicsPayload(K_opts[i], D_opts[i])))
                self.send_payload.emit(CalibrationData(cam_name, ExtrinsicsPayload(r_opts[i], t_opts[i])))
        else:
            logger.error(f"[{self.name.title()}] Bundle Adjustment failed.")

    @Slot()
    def reset(self):
        """ Resets the worker to its initial state """
        super().reset()

        self.multiview_tool = None
        self._intrinsics_ready.fill(False)

        # Reset poses to default
        self._rvecs_c2w = np.zeros((self._C, 3), dtype=np.float32)
        self._tvecs_c2w = np.zeros((self._C, 3), dtype=np.float32)

        # Ensure origin camera is correctly initialized after reset
        self._rvecs_c2w[self._orig_cam_idx] = np.zeros(3)
        self._tvecs_c2w[self._orig_cam_idx] = np.zeros(3)

        logger.debug(f"[{self.name.title()}] Worker has been reset.")

    @Slot(int)
    def set_stage(self, stage: int):
        super().set_stage(stage)

        # When moving back to stage 0, we must perform a full reset
        if stage == 0:
            self.reset()

        # When moving to stage 1, try to create the tool
        # This will use the final intrinsics gathered during stage 0
        else:
            self._try_create_tool()