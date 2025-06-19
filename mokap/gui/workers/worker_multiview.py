import logging
import numpy as np
from typing import List, Optional, Union, Dict
from PySide6.QtCore import QTimer, Slot, Signal
from numpy.typing import ArrayLike
from mokap.calibration.multiview import MultiviewCalibrationTool
from mokap.gui.workers.workers_base import CalibrationProcessingWorker
from mokap.utils.datatypes import (CalibrationData, DetectionPayload, ExtrinsicsPayload, IntrinsicsPayload,
                                   ChessBoard, CharucoBoard)
from mokap.utils.geometry.projective import back_projection_batched, back_projection
from mokap.utils.geometry.transforms import extrinsics_matrix

logger = logging.getLogger(__name__)


class MultiviewWorker(CalibrationProcessingWorker):

    scene_data_ready = Signal(dict)

    def __init__(self,
                 cameras_names:     List[str],
                 origin_camera:     str,
                 sources_shapes_wh: ArrayLike,
                 calibration_board: Union[ChessBoard, CharucoBoard]):
        super().__init__(name='multiview')

        # Configuration and static data
        self._cameras_names = cameras_names
        self._orig_cam_name = origin_camera
        self._orig_cam_idx = self._cameras_names.index(self._orig_cam_name)

        self._C = len(cameras_names)

        self._sources_shapes_wh = sources_shapes_wh  # Expects {cam_name: (w, h)}
        self.calibration_board = calibration_board

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

        # Buffer for per-frame 2D detections for visualization
        self._points_2d: Dict[str, np.ndarray] = {name: np.zeros((0, 2)) for name in self._cameras_names}

        # Empty arrays to emit nodata when needed
        self._nopoints_2d = np.zeros((0, 2))
        self._nopoints_3d = np.zeros((0, 3))

        self._frustum_depth = 200.0 # TODO: compute this automatically

        # Timer for sending scene data to the 3D view
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(33)  # ~ 30 Hz
        self.update_timer.timeout.connect(self.process_and_emit_3d_data)

    def _try_create_tool(self):
        """ Creates the MultiviewCalibrationTool once all intrinsics are available """

        if self.multiview_tool is None and np.all(self._intrinsics_ready) :
            logger.info(f"[{self.name.title()}] All intrinsics received. Creating Multiview tool.")

            image_sizes_wh = np.array([self._sources_shapes_wh[name] for name in self._cameras_names])

            self.multiview_tool = MultiviewCalibrationTool(
                nb_cameras=self._C,
                images_sizes_wh=image_sizes_wh,
                origin_idx=self._orig_cam_idx,
                init_cam_matrices=self._cameras_matrices,
                init_dist_coeffs=self._dist_coeffs,
                object_points=self.calibration_board.object_points,
                min_detections=50,  # TODO: Make configurable
                max_detections=300
            )
            self.update_timer.start()  # Start emitting 3D scene data

    @Slot(MultiviewCalibrationTool)
    def initialize_tool(self, tool: MultiviewCalibrationTool):
        """ Receives the fully configured tool from the Coordinator when calibration stage > 0 """
        # TODO: Rename this, make it consistent with the monocular tool's names

        self.multiview_tool = tool
        logger.debug(f"[{self.name.title()}] Multiview tool initialized and ready.")

    @Slot(CalibrationData)
    def on_payload_received(self, data: CalibrationData):
        """ Handles all incoming payloads and updates the worker's state """

        cam_idx = self._cameras_names.index(data.camera_name)
        payload = data.payload

        if isinstance(payload, IntrinsicsPayload):
            # Update local state for this camera
            self._cameras_matrices[cam_idx] = payload.camera_matrix
            d_len = len(payload.dist_coeffs)
            self._dist_coeffs[cam_idx, :d_len] = payload.dist_coeffs
            self._intrinsics_ready[cam_idx] = True

            # Check if all intrinsics are now ready to create the tool
            self._try_create_tool()

        elif isinstance(payload, DetectionPayload) and self.multiview_tool:
            # Pass detections directly to the tool
            self.multiview_tool.register(cam_idx, payload)

            # Also store the 2D points for visualization
            self._points_2d[data.camera_name] = payload.points2D if payload.points2D is not None else self._nopoints_2d

    def process_and_emit_3d_data(self):
        """ Periodically calculates and emits all data needed for the 3D view """

        # This state is updated by incoming payloads or after a successful BA

        # Determine which set of parameters to use for visualization
        if self.multiview_tool and self.multiview_tool.is_refined:
            K, D = self.multiview_tool.refined_intrinsics
            r, t = self.multiview_tool.refined_extrinsics

        elif self.multiview_tool and self.multiview_tool._estimated:
            K, D = self.multiview_tool.intrinsics
            r, t = self.multiview_tool.extrinsics
        else:
            # Before estimation, use the initial parameters stored in the worker
            K, D, r, t = self._cameras_matrices, self._dist_coeffs, self._rvecs_c2w, self._tvecs_c2w

        # Early exit if there's nothing to draw
        if K is None or r is None:
            return

        all_E = extrinsics_matrix(r, t) # TODO: This expects jax arrays - make we dont convert back and forth to numpy for no reason

        # Frustum visualization
        # TODO: No need to initialise these arrays over and over - they need to be stored
        frustums_pts = np.array([ [[0, 0], [w, 0], [w, h], [0, h]]
            for w, h in self._sources_shapes_wh.values()], dtype=np.float32)
        centers_pts = np.array([[w / 2, h / 2] for w, h in self._sources_shapes_wh.values()], dtype=np.float32)

        # TODO: Why two calls to back_projection_batched? We could have one stacked array and one call, then slice it
        all_frustums_3d = back_projection_batched(frustums_pts, self._frustum_depth, K, all_E, D)
        all_centers_3d = back_projection_batched(centers_pts[:, np.newaxis, :], self._frustum_depth, K, all_E,
                                                 D).squeeze(axis=1)

        # Detections visualization
        all_detections_3d = [
            back_projection(self._points_2d[name], self._frustum_depth * 0.95, K[i], all_E[i], D[i])
            if self._points_2d[name].shape[0] > 0 else self._nopoints_3d
            for i, name in enumerate(self._cameras_names)
        ]

        self.scene_data_ready.emit({
            'extrinsics': all_E,
            'frustums_3d': all_frustums_3d,
            'centers_3d': all_centers_3d,
            'detections_3d': all_detections_3d
        })

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

            K_opts = np.asarray(K_opts)
            D_opts = np.asarray(D_opts)
            r_opts = np.asarray(r_opts)
            t_opts = np.asarray(t_opts)

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

        self.update_timer.stop()
        self.multiview_tool = None
        self._intrinsics_ready.fill(False)

        logger.debug(f"[{self.name.title()}] Worker has been reset.")

    @Slot(int)
    def set_stage(self, stage: int):
        super().set_stage(stage)

        # If we move back to stage 0, we must destroy the tool and stop the timer
        if stage == 0 and self.multiview_tool is not None:
            logger.debug(f"[{self.name.title()}] Resetting. Multiview tool destroyed.")
            self.update_timer.stop()
            self.multiview_tool = None
