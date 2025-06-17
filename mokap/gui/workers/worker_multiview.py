import logging
import numpy as np
from typing import List, Optional, Union
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
            cameras_names: List[str],
            origin_camera_name: str,
            sources_shapes: ArrayLike,
            board_params: Union[ChessBoard, CharucoBoard]):
        super().__init__(name='multiview')

        self._cameras_names = cameras_names
        self._origin_camera_name = origin_camera_name
        self.multiview_tool: Optional[MultiviewCalibrationTool] = None  # will be initialized later

        # Keep a local copy of camera parameters
        self._cameras_matrices = None
        self._dist_coeffs = None
        self._rvecs = None
        self._tvecs = None
        self._sources_shapes = sources_shapes
        self._points_2d = {cam_name: np.zeros((board_params.object_points.shape[0], 2)) for cam_name in
                           self._cameras_names}
        self._points_2d_ids = {cam_name: np.array([], dtype=int) for cam_name in self._cameras_names}
        self._frustum_depth = 200

        self.update_timer = QTimer(self)
        self.update_timer.setInterval(33)  # ~ 30 Hz for scene updates
        self.update_timer.timeout.connect(self.process_and_emit_3d_data)

        self.update_timer.start()

    @Slot(MultiviewCalibrationTool)
    def initialize_tool(self, tool: MultiviewCalibrationTool):
        """ Receives the fully configured tool from the Coordinator when calibration stage > 0 """
        # TODO: Rename this, make it consistent with the monocular tool's names

        self.multiview_tool = tool
        logger.debug(f"[{self.name.title()}] Multiview tool initialized and ready.")

    @Slot(CalibrationData)
    def on_payload_received(self, data: CalibrationData):

        logger.debug(f'[{self.name.title()}] Received (from Coordinator): ({data.camera_name}) {data.payload}')

        # if the tool isn't ready, do nothing
        if self.multiview_tool is None:
            return

        payload = data.payload
        cam_idx = self._cameras_names.index(data.camera_name)

        # update local state based on incoming data
        if isinstance(payload, DetectionPayload):
            self.multiview_tool.register(cam_idx, payload)

            # self._points_2d[data.camera_name] = payload.points2D
            # self._points_2d_ids[data.camera_name] = payload.pointsIDs

        # elif isinstance(payload, ExtrinsicsPayload) and payload.rvec is not None:
        #     self._rvecs[cam_idx] = payload.rvec
        #     self._tvecs[cam_idx] = payload.tvec

        # elif isinstance(payload, IntrinsicsPayload):
        #     self._cameras_matrices[cam_idx] = payload.camera_matrix
        #     dist_len = len(payload.dist_coeffs)
        #     self._dist_coeffs[cam_idx, :dist_len] = payload.dist_coeffs

    def process_and_emit_3d_data(self):
        """ Performs all calculations and emits the results for the 3D view
        Works in both Stage 0 (using individual parameters) and Stage 1 (using tool's global parameters)
        """

        if self.multiview_tool and self.multiview_tool.is_refined:
            all_K, all_D = self.multiview_tool.refined_intrinsics
            all_r, all_t = self.multiview_tool.refined_extrinsics

        elif self.multiview_tool and self.multiview_tool._estimated:
            all_K, all_D = self.multiview_tool.intrinsics
            all_r, all_t = self.multiview_tool.extrinsics

        else:
            all_K, all_D = self._cameras_matrices, self._dist_coeffs
            all_r, all_t = self._rvecs, self._tvecs

        if all_r is None or np.all(all_r == 0):
            return

        all_E = extrinsics_matrix(all_r, all_t)

        frustums_points2d = np.stack(
            [np.array([[0, 0], [shape[1], 0], [shape[1], shape[0]], [0, shape[0]]], dtype=np.int32) for shape in
             self._sources_shapes.values()])
        centres_points2d = frustums_points2d[:, 2, :] / 2.0

        all_frustums_3d = back_projection_batched(frustums_points2d, self._frustum_depth, all_K, all_E, all_D)
        all_centers_3d = back_projection_batched(centres_points2d, self._frustum_depth, all_K, all_E, all_D).squeeze(
            axis=1)

        all_detections_3d = [
            back_projection(self._points_2d[name], self._frustum_depth * 0.95, all_K[i], all_E[i], all_D[i]) if len(
                ids) > 0 else np.zeros((0, 3)) for i, (name, ids) in enumerate(self._points_2d_ids.items())]

        scene_data = {
            'extrinsics': all_E,
            'frustums_3d': all_frustums_3d,
            'centers_3d': all_centers_3d,
            'detections_3d': all_detections_3d
        }
        self.scene_data_ready.emit(scene_data)

    @Slot()
    def trigger_refinement(self):
        """
        Slot to be connected to a GUI button. Triggers the final BA
        """
        if self._paused or self.multiview_tool is None:
            return

        logger.info(f"[{self.name.title()}] Attempting to run final Bundle Adjustment.")

        self.blocking.emit(True)
        success = self.multiview_tool.refine_all()
        self.blocking.emit(False)

        if success:
            logger.info(f"[{self.name.title()}] Bundle Adjustment successful. Emitting refined results.")

            # emit the final results
            K_opts, D_opts = self.multiview_tool.refined_intrinsics
            r_opts, t_opts = self.multiview_tool.refined_extrinsics

            for i, cam_name in enumerate(self._cameras_names):
                # Send refined intrinsics
                intr_payload = IntrinsicsPayload(K_opts[i], D_opts[i])
                self.send_payload.emit(CalibrationData(cam_name, intr_payload))

                # Send refined extrinsics
                extr_payload = ExtrinsicsPayload(r_opts[i], t_opts[i])
                self.send_payload.emit(CalibrationData(cam_name, extr_payload))
        else:
            logger.info(f"[{self.name.title()}] Bundle Adjustment failed.")

    @Slot()
    def reset(self):
        """ Destroys the tool and stops the timer, resets the worker's state """
        super().reset()

        if self.multiview_tool is not None:
            logger.debug(f"[{self.name.title()}] Resetting. Multiview tool destroyed.")

            self.update_timer.stop()
            self.multiview_tool = None

    @Slot(int)
    def set_stage(self, stage: int):
        super().set_stage(stage)

        # If we move back to stage 0, we must destroy the tool and stop the timer
        if stage == 0 and self.multiview_tool is not None:
            logger.debug(f"[{self.name.title()}] Resetting. Multiview tool destroyed.")
            self.update_timer.stop()
            self.multiview_tool = None
