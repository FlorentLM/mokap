import logging
from typing import Union, Optional
import numpy as np
from PySide6.QtCore import Slot
from numpy.typing import ArrayLike
from mokap.calibration.monocular import MonocularCalibrationTool
from mokap.gui.workers.workers_base import CalibrationProcessingWorker
from mokap.utils.datatypes import (ChessBoard, CharucoBoard, CalibrationData, ErrorsPayload,
                                   IntrinsicsPayload, ExtrinsicsPayload, DetectionPayload)

logger = logging.getLogger(__name__)

class MonocularWorker(CalibrationProcessingWorker):

    def __init__(self,
            board_params: Union[ChessBoard, CharucoBoard],
            cam_idx: int,
            cam_name: str,
            img_width: int,
            img_height: int):
        super().__init__(name=cam_name)

        self.camera_idx = cam_idx
        self.cam_name = cam_name
        self.img_w = img_width
        self.img_h = img_height

        # TODO: These need to be configurable from the GUI
        self._cam_th_focal: Optional[float] = 60
        self._sensor_size: Optional[Union[float, str]] = '1/2.9"'

        # the tool and board points are initialized by the _create_tool method
        self.monocular_tool: Optional[MonocularCalibrationTool] = None
        self.board_object_points: Optional[np.ndarray] = None
        self._create_tool(board_params)

        # TODO: These need to be configurable from the GUI
        self._auto_sample = True
        self._auto_compute = True
        self._simple_focal = False
        self._simple_distortion = False
        self._complex_distortion = False    # TODO: Replace the 3 flags with the Literal like in MultiviewCalibTool
        self._coverage_threshold = 75
        self._stack_length_threshold = 20

    def _create_tool(self, board_params: Union[ChessBoard, CharucoBoard]):
        """ Create and configure a new MonocularCalibrationTool. Used in init or after calibration board change """

        logger.debug(f"[{self.name.title()}] Received new board parameters.")

        # Update the worker's own reference to the board points for the coordinator
        self.board_object_points = board_params.object_points()

        # Create a brand new MonocularCalibrationTool instance
        self.monocular_tool = MonocularCalibrationTool(
            board_params=board_params,
            imsize_hw=(self.img_h, self.img_w),
            focal_mm=self._cam_th_focal,
            sensor_size=self._sensor_size
        )
        self.monocular_tool.set_visualisation_scale(2)  # TODO: This will be removed once visualisation is taken out

    @Slot(object)
    def configure_new_board(self, board_params: Union[ChessBoard, CharucoBoard]):
        """ Slot to handle a full reset when board parameters change """

        logger.debug(f"[{self.name.title()}] Received new board parameters. Recreating tool.")

        # Re-create the tool with the new parameters
        self._create_tool(board_params)

    @Slot()
    def reset(self):
        """ Resets the state of the monocular worker and its tool """
        super().reset()

        if self.monocular_tool:
            self.monocular_tool.clear_stacks()
            self.monocular_tool.clear_intrinsics()
            # Notify the UI that errors are now reset/invalid
            self.send_payload.emit(CalibrationData(self.cam_name, ErrorsPayload(np.array([np.inf]))))

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
        if self._paused:
            return

        # we received a *reference* to the latest frame so a copy is necessary to avoid it being overwritten
        local_frame = frame.copy()

        self.monocular_tool.process_frame(local_frame)
        self.monocular_tool.detect()    # TODO: Detection can be moved out of this class it's much cleaner

        # Stage 0: Compute initial intrinsics
        if self._current_stage == 0:
            if self._auto_sample:
                self.monocular_tool.auto_register_area_based(area_threshold=0.2)

            if self._auto_compute:
                if self.monocular_tool.coverage >= self._coverage_threshold and self.monocular_tool.nb_samples > self._stack_length_threshold:
                    self.blocking.emit(True)

                r = self.monocular_tool.auto_compute_intrinsics(    # TODO: the auto_compute is a bit useless if we check the thresholds here
                    coverage_threshold=self._coverage_threshold,
                    stack_length_threshold=self._stack_length_threshold,
                    simple_focal=self._simple_focal,
                    simple_distortion=self._simple_distortion,
                    complex_distortion=self._complex_distortion
                )
                self.blocking.emit(False)

                if r:
                    self.send_payload.emit(
                        CalibrationData(self.cam_name, ErrorsPayload(self.monocular_tool.intrinsics_errors))
                    )

                    if self.monocular_tool.has_intrinsics:
                        self.send_payload.emit(
                            CalibrationData(self.cam_name, IntrinsicsPayload(*self.monocular_tool.intrinsics))
                        )

        # This extrinsics computation is only used to provide visualisation of the board pose (perimeter) and plotting
        if self.monocular_tool.has_intrinsics:
            self.monocular_tool.compute_extrinsics()

            # always send extrinsics payload so the GUI knows the current state
            rvec, tvec = self.monocular_tool.extrinsics_np()
            pose_error = self.monocular_tool.pose_error     # this will be nan if no pose

            self.send_payload.emit(
                CalibrationData(self.cam_name, ExtrinsicsPayload(rvec=rvec, tvec=tvec, error=pose_error))
            )

        # in all stages, if a detection happened, send its data for visualization
        if self.monocular_tool.has_detection:
            # we use this payload for visualization AND to feed the multiview processing
            self.send_payload.emit(
                CalibrationData(self.cam_name, DetectionPayload(frame_idx, *self.monocular_tool.detection))
            )
        else:
            # when no detection, pyqtgraph still wants a 2D array
            empty_points = np.zeros((0, 2), dtype=np.float32)
            empty_ids = np.array([], dtype=int)
            self.send_payload.emit(
                CalibrationData(self.cam_name, DetectionPayload(frame_idx, empty_points, empty_ids))
            )

        self.finished.emit()

    # Other slots for manual control
    @Slot()
    def add_sample(self):
        self.monocular_tool.register_sample()

    @Slot()
    def clear_samples(self):
        self.monocular_tool.clear_stacks()

    @Slot()
    def compute_intrinsics(self):
        self.monocular_tool.compute_intrinsics()

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
