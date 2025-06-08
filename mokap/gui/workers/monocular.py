from typing import Union
import numpy as np
from PySide6.QtCore import Signal, Slot
from numpy.typing import ArrayLike
from mokap.gui.workers import DEBUG_SIGNALS_FLOW
from mokap.gui.workers.base import CalibrationProcessingWorker
from mokap.utils.datatypes import ChessBoard, CharucoBoard, CalibrationData, ErrorsPayload, IntrinsicsPayload, ExtrinsicsPayload, DetectionPayload


class MonocularWorker(CalibrationProcessingWorker):

    annotations = Signal(np.ndarray)       # Annotations are already burned into the image, so emit the whole thing

    def __init__(self, board_params: Union[ChessBoard, CharucoBoard], cam_idx: int, cam_name: str, cam_shape: ArrayLike):
        super().__init__(name=cam_name)
        self.camera_idx = cam_idx
        self.cam_name = cam_name
        self.cam_shape = cam_shape

        # The tool and board points are initialized by the _create_tool method
        self.monocular_tool: MonocularCalibrationTool = None
        self.board_object_points: np.ndarray = None
        self._create_tool(board_params)

        # Flags for worker function
        self._auto_sample = True
        self._auto_compute = True

    def _create_tool(self, board_params: Union[ChessBoard, CharucoBoard]):
        """
        Create and configure a new MonocularCalibrationTool. Used in init or after calibration board change
        """
        if DEBUG_SIGNALS_FLOW:
            print(f"[{self.name.title()}] Received new board parameters. Recreating tool.")

        # Update the worker's own reference to the board points for the coordinator
        self.board_object_points = board_params.object_points()

        # Create a brand new MonocularCalibrationTool instance
        self.monocular_tool = MonocularCalibrationTool(
            board_params=board_params,
            imsize_hw=self.cam_shape[:2],
            focal_mm=60,
            sensor_size='1/2.9"'
        )
        self.monocular_tool.set_visualisation_scale(2)

    @Slot(object)
    def update_board(self, board_params: Union[ChessBoard, CharucoBoard]):
        """
        Slot to handle a full reset when board parameters change
        """
        if DEBUG_SIGNALS_FLOW:
            print(f"[{self.name.title()}] Received new board parameters. Recreating tool.")

        # Re-create the tool with the new parameters
        self._create_tool(board_params)

        # Clear any old data and notify the UI
        self.monocular_tool.clear_stacks()
        self.monocular_tool.clear_intrinsics()
        self.send_payload.emit(CalibrationData(self.cam_name, ErrorsPayload(np.array([np.inf]))))

    @Slot(int)
    def set_stage(self, stage: int):
        super().set_stage(stage)
        self.monocular_tool.clear_stacks()

    @Slot(CalibrationData)
    def _handle_payload(self, data: CalibrationData):
        super()._handle_payload(data)
        payload = data.payload

        if isinstance(payload, IntrinsicsPayload):
            self.monocular_tool.set_intrinsics(payload.camera_matrix, payload.dist_coeffs, payload.errors)

        # extrinsics setting is typically for visualisation only
        elif isinstance(payload, ExtrinsicsPayload):
            self.monocular_tool.set_extrinsics(payload.rvec, payload.tvec)

    @Slot(np.ndarray, int)
    def handle_frame(self, frame: ArrayLike, frame_idx: int):
        if self._paused:
            return

        self.monocular_tool.detect(frame)

        # Stage 0: Compute initial intrinsics
        if self._current_stage == 0:
            if self._auto_sample:
                self.monocular_tool.auto_register_area_based(area_threshold=0.2, nb_points_threshold=4)
            if self._auto_compute:
                coverage_threshold, stack_length_threshold = 80, 20
                if self.monocular_tool.coverage >= coverage_threshold and self.monocular_tool.nb_samples > stack_length_threshold:
                    self.blocking.emit(True)
                r = self.monocular_tool.auto_compute_intrinsics(
                    coverage_threshold=coverage_threshold,
                    stack_length_threshold=stack_length_threshold,
                    simple_focal=False, simple_distortion=False, complex_distortion=False
                )
                self.blocking.emit(False)
                if r:
                    self.send_payload.emit(
                        CalibrationData(self.cam_name, ErrorsPayload(self.monocular_tool.intrinsics_errors)))
                    if self.monocular_tool.has_intrinsics:
                        self.send_payload.emit(
                            CalibrationData(self.cam_name, IntrinsicsPayload(*self.monocular_tool.intrinsics)))

        # This is only used to provide visualisation of the board pose (perimeter)
        if self.monocular_tool.has_intrinsics:
            self.monocular_tool.compute_extrinsics()

        # Stage 1 and 2: In these stages, the worker's ONLY job is to send detections to the Multiview system
        if self._current_stage >= 1:
            if self.monocular_tool.has_detection:
                self.send_payload.emit(
                    CalibrationData(self.cam_name, DetectionPayload(frame_idx, *self.monocular_tool.detection))
                )

        annotated = self.monocular_tool.visualise(errors_mm=True)
        self.annotations.emit(annotated)
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
