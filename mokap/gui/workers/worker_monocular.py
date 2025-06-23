import logging
from typing import Union, Optional
import numpy as np
from PySide6.QtCore import Slot
from numpy.typing import ArrayLike
from mokap.calibration.monocular import MonocularCalibrationTool
from mokap.gui.workers.workers_base import CalibrationProcessingWorker
from mokap.utils.datatypes import (ChessBoard, CharucoBoard, CalibrationData, ErrorsPayload,
                                   IntrinsicsPayload, ExtrinsicsPayload, DetectionPayload, CoveragePayload,
                                   ReprojectionPayload, DistortionModel)

logger = logging.getLogger(__name__)

class MonocularWorker(CalibrationProcessingWorker):

    def __init__(self,
                 board: Union[ChessBoard, CharucoBoard],
                 cam_name: str,
                 img_width: int,
                 img_height: int):
        super().__init__(name=cam_name)

        self.cam_name = cam_name
        self.img_w = img_width
        self.img_h = img_height

        # Configuration from GUI (with defaults)
        # TODO: These need to be configurable from the GUI
        self._cam_th_focal: Optional[float] = 60.0
        self._sensor_size: Optional[Union[float, str]] = '1/2.9"'

        # Policy settings for auto-calibration
        # TODO: These need to be configurable from the GUI
        self._auto_sample = True
        self._auto_compute = True
        self._fix_aspect_ratio = False
        self._distortion_model: DistortionModel = 'standard'
        self._coverage_threshold = 75.0
        self._area_threshold = 0.2
        self._min_stack_for_calib = 20

        # Tool initialization
        self.monocular_tool: MonocularCalibrationTool = self._create_tool(board)

        # A flag to avoid looping over failures
        self._last_calib_failed = False
        # flags to prevent sending redundant empty payloads
        self._was_detecting = False
        self._was_reprojecting = False

        # Payloads for empty/failed cases
        self.empty_detection = DetectionPayload(-1, np.zeros((0, 2)), np.array([]))
        self.empty_reprojection = ReprojectionPayload(np.zeros((0, 2)), np.array([]))

    def _create_tool(self, board: Union[ChessBoard, CharucoBoard]):
        """ Creates and configures a new MonocularCalibrationTool """

        logger.debug(f"[{self.name.title()}] Creating new calibration tool.")

        return MonocularCalibrationTool(
            calibration_board=board,
            imsize_hw=(self.img_h, self.img_w),
            min_stack=self._min_stack_for_calib,
            focal_mm=self._cam_th_focal,
            sensor_size=self._sensor_size
        )

    @Slot(object)
    def configure_new_board(self, board: Union[ChessBoard, CharucoBoard]):
        """ Slot to handle a full reset when board parameters change """

        logger.debug(f"[{self.name.title()}] Received new board parameters. Recreating tool.")

        # Re-create the tool with the new parameters
        self.monocular_tool = self._create_tool(board)

    @Slot()
    def reset(self):
        """ Resets the state of the monocular worker and its tool """
        super().reset()

        self._last_calib_failed = False
        self._was_detecting = False
        self._was_reprojecting = False

        if self.monocular_tool:
            self.monocular_tool.clear_stacks()
            self.monocular_tool.clear_intrinsics()

            # Notify the UI that errors and data are reset/invalid
            self.send_payload.emit(
                CalibrationData(self.cam_name, ErrorsPayload(np.array([np.inf])))
            )
            self.send_payload.emit(
                CalibrationData(self.cam_name, CoveragePayload(grid=np.zeros((1, 1), dtype=bool),
                                                               coverage_percent=0.0,
                                                               nb_samples=0,
                                                               total_points=self.monocular_tool.detector.board.nb_points))
            )

    @Slot(int)
    def set_stage(self, stage: int):
        super().set_stage(stage)
        self.monocular_tool.clear_stacks()

    @Slot(CalibrationData)
    def on_payload_received(self, data: CalibrationData):
        """ This slot receives data intended for this worker (i.e. only intrinsics payloads), regardless of origin """
        super().on_payload_received(data)
        payload = data.payload

        if isinstance(payload, IntrinsicsPayload):
            # This is the single entry point for updating intrinsics
            self.monocular_tool.set_intrinsics(payload.camera_matrix, payload.dist_coeffs, payload.errors)

    @Slot(np.ndarray, int)
    def handle_frame(self, frame: ArrayLike, frame_idx: int):
        """ The main processing loop for each frame. """
        if self._paused or not self.monocular_tool:
            return

        # --- Detection ---
        self.monocular_tool.detect(frame)

        is_detecting_now = self.monocular_tool.has_detection
        if is_detecting_now:
            # Always send payload when we have a new detection
            det_payload = DetectionPayload(frame_idx, *self.monocular_tool.detection)
            self.send_payload.emit(CalibrationData(self.cam_name, det_payload))

        elif self._was_detecting:
            # if we are NOT detecting now, but were last frame, send one empty payload to clear UI
            self.send_payload.emit(CalibrationData(self.cam_name, self.empty_detection))

        # Update state for next frame
        self._was_detecting = is_detecting_now

        # --- Stage 0: Intrinsic Calibration policy ---
        if self._current_stage == 0:
            # Policy for auto-sampling
            if self._auto_sample:
                # The worker decides to try registering, the tool just executes
                if self.monocular_tool.register_sample(min_new_area=self._area_threshold):

                    self._last_calib_failed = False # clear the flag since we anned a sample

                    # If a sample was added, update the UI
                    self.send_coverage_update()

            # Policy for auto-computation
            if self._auto_compute and not self._last_calib_failed:
                is_ready = (self.monocular_tool.pct_coverage >= self._coverage_threshold and
                            self.monocular_tool.curr_nb_samples >= self._min_stack_for_calib)

                if is_ready:

                    self.blocking.emit(True)  # Block UI during computation
                    # The worker calls the computation, the tool just executes
                    success = self.monocular_tool.compute_intrinsics(
                        fix_aspect_ratio=self._fix_aspect_ratio,
                        distortion_model=self._distortion_model
                    )
                    self.blocking.emit(False)

                    self._last_calib_failed = not success

                    if self._last_calib_failed:
                        logger.debug("Auto-computation of intrinsics failed. Will not re-attempt until new samples are added.")
                        self.monocular_tool.clear_grid()
                    else:
                        self.monocular_tool.clear_stacks()  # this also clears the grid

                    # Regardless of success, send out the update to refresh UI
                    self.send_intrinsics_update()

        # --- Pose and reprojection (for all stages) ---

        is_reprojecting_now = False  # assume false by default
        if self.monocular_tool.has_intrinsics:

            self.monocular_tool.compute_extrinsics()

            # Send reprojection payload for visualization
            if self.monocular_tool.has_extrinsics:

                is_reprojecting_now = True

                # Send extrinsics payload (will be None if pose failed)
                rvec, tvec = self.monocular_tool.extrinsics
                pose_error = self.monocular_tool.pose_error

                self.send_payload.emit(
                    CalibrationData(self.cam_name, ExtrinsicsPayload(rvec=rvec, tvec=tvec, error=pose_error))
                )

                # Reproject points
                self.monocular_tool.reproject()

                reproj_payload = ReprojectionPayload(
                    all_points_2d=self.monocular_tool.reprojected_points2d,
                    detected_ids=self.monocular_tool.detection[1]
                )
                self.send_payload.emit(CalibrationData(self.cam_name, reproj_payload))

            # check if the reprojection state has changed
            if not is_reprojecting_now and self._was_reprojecting:
                # if we are not reprojecting now, but were last frame, send one empty payload to clear UI
                self.send_payload.emit(CalibrationData(self.cam_name, self.empty_reprojection))

            # Update state for next frame
            self._was_reprojecting = is_reprojecting_now

        self.finished.emit()

    def send_coverage_update(self):
        """ Sends the current coverage and sample count to the UI """

        if self.monocular_tool:

            payload = CoveragePayload(
                grid=self.monocular_tool.grid,
                coverage_percent=self.monocular_tool.pct_coverage,
                nb_samples=self.monocular_tool.curr_nb_samples,
                total_points=self.monocular_tool.detector.board.nb_points
            )
            self.send_payload.emit(CalibrationData(self.cam_name, payload))

    def send_intrinsics_update(self):
        """ Sends the current intrinsics and their errors to the UI and coordinator """

        if self.monocular_tool and self.monocular_tool.has_intrinsics:

            # Send errors
            self.send_payload.emit(
                CalibrationData(self.cam_name, ErrorsPayload(self.monocular_tool.intrinsics_errors))
            )

            # Send intrinsics themselves
            K, D = self.monocular_tool.intrinsics
            intr_payload = IntrinsicsPayload(K, D, self.monocular_tool.intrinsics_errors)
            self.send_payload.emit(CalibrationData(self.cam_name, intr_payload))

    # Other slots for manual control
    @Slot()
    def add_sample(self):
        self.monocular_tool.register_sample(min_new_area=0.0)

    @Slot()
    def clear_samples(self):
        self.monocular_tool.clear_stacks()
        # After clearing, emit a payload to update the UI
        self.send_payload.emit(
            CalibrationData(self.cam_name, CoveragePayload(
                grid=self.monocular_tool.grid,
                coverage_percent=self.monocular_tool.pct_coverage,
                nb_samples=self.monocular_tool.curr_nb_samples,
                total_points=self.monocular_tool.detector.board.nb_points
            ))
        )

    @Slot()
    def compute_intrinsics(self):

        self.blocking.emit(True)
        success = self.monocular_tool.compute_intrinsics(
            fix_aspect_ratio=self._fix_aspect_ratio,
            distortion_model=self._distortion_model
        )
        self.blocking.emit(False)

        if success:
            self.send_intrinsics_update()

    @Slot()
    def clear_intrinsics(self):
        self.monocular_tool.clear_intrinsics()
        self.monocular_tool.clear_stacks()

    @Slot(bool)
    def auto_sample(self, value: bool):
        self._auto_sample = value

    @Slot(bool)
    def auto_compute(self, value: bool):
        self._auto_compute = value

    @Slot(str)
    def load_intrinsics(self, file_path: str):
        intrinsics = IntrinsicsPayload.from_file(file_path, self.cam_name)

        # Disable auto modes and clear existing samples/coverage
        self.auto_sample(False)
        self.auto_compute(False)
        self.clear_samples()

        # Set the intrinsics in the tool
        self.monocular_tool.set_intrinsics(intrinsics.camera_matrix, intrinsics.dist_coeffs, intrinsics.errors)
        # And forward to the coordinator (it will route to the Multiview and UI)
        self.send_payload.emit(
            CalibrationData(self.cam_name, intrinsics)
        )

    @Slot(str)
    def save_intrinsics(self, file_path: str):
        if self.monocular_tool.has_intrinsics:
            data = CalibrationData(self.cam_name, IntrinsicsPayload(*self.monocular_tool.intrinsics,
                                                                    self.monocular_tool.intrinsics_errors))
            data.to_file(file_path)
