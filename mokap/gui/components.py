import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from mokap.calibration import MonocularCalibrationTool, MultiviewCalibrationTool
from mokap.utils import fileio
from dataclasses import dataclass
from typing import Dict, Any, Literal, List, Annotated
from pathlib import Path

DEBUG = True

@dataclass
class ErrorsPayload:
    errors: List[float]

@dataclass
class PosePayload:
    """
    Monocular estimation of the extrinsics (camera pose)
    """
    frame: int
    rvec: np.ndarray
    tvec: np.ndarray

@dataclass
class DetectionPayload:
    """
    Monocular detection of points 2D
    """
    frame: int
    points2D: np.ndarray
    pointsIDs: np.ndarray

@dataclass
class IntrinsicsPayload:
    """
    Monocular intrinsics parameters
    """
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray

@dataclass
class ExtrinsicsPayload:
    """
    Multiview extrinsics parameters (global arrangement)
    """
    rvec: np.ndarray
    tvec: np.ndarray

@dataclass
class CalibrationData:
    """
    Encapsulation of a payload with the camera name
    """
    camera_name: str
    payload: IntrinsicsPayload | ExtrinsicsPayload | DetectionPayload | PosePayload | ErrorsPayload


##

class BaseWorker(QObject):

    finished = Signal()
    error = Signal(Exception)
    blocking = Signal(bool)     # Tell the main thread when doing a blocking function

    send_payload = Signal(CalibrationData)      # Single output signal - will always go to the coordinator
    receive_payload = Signal(CalibrationData)   # Single input signal - will always be accessed by the coordinator

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self._paused = False

        # We connect through an intermediary internal signal so that we can call receive_payload from the outside
        # without having to specifically connect to _handle_payload
        self.receive_payload.connect(self._handle_payload)

    def set_paused(self, paused: bool):
        self._paused = paused

    @Slot(CalibrationData)
    def _handle_payload(self, data: CalibrationData):
        print(f'[Worker {self.name}] Received {data.payload} (origin: {data.camera_name})')

class MainWorker(BaseWorker):
    """
    This worker lives in its own thread and does stuff on the full resolution image
    """

    annotations = Signal(list)

    def __init__(self, cam_name):
        super().__init__(cam_name)

    @Slot(np.ndarray)
    def process_frame(self, frame):
        if self._paused:
            return

        # Fake motiuon detection
        bboxes = self._do_motion_detection(frame)

        # Emit the results back to the main thread
        self.annotations.emit(bboxes)

        # Emit the finished signal to tell we're done
        self.finished.emit()

    def _do_motion_detection(self, frame):
        # TESTING - Fake bounding box around (100,100) of size (50,40)
        return [(100, 100, 50, 40)]

class MonocularWorker(BaseWorker):
    """
    This worker lives in its own thread and does monocular detection/calibration
    """

    annotations = Signal(np.ndarray)       # Annotations are already burned into the image, so emit the whole thing

    def __init__(self, board_params, cam_idx, cam_name, cam_shape):
        super().__init__(cam_name)

        self.camera_idx = cam_idx
        self.cam_name = cam_name
        self.cam_shape = cam_shape

        self.monocular_tool = MonocularCalibrationTool(
            board_params=board_params,
            imsize_hw=self.cam_shape[:2],  # pass frame size so it can track coverage
            focal_mm=60,  # TODO - UI field for these
            sensor_size='1/2.9"'  #
        )
        self.monocular_tool.set_visualisation_scale(2)

        # Flags for worker function
        self.auto_sample = True
        self.auto_compute = True

        self._current_stage = 0

    @Slot(np.ndarray, int)
    def process_frame(self, frame, frame_id):
        if self._paused:
            return

        # Detect
        self.monocular_tool.detect(frame)

        # Intrinsics stage: Auto-sample and compute intrinsics
        if self._current_stage == 0:
            if self.auto_sample:
                self.monocular_tool.auto_register_area_based(area_threshold=0.2, nb_points_threshold=4)

            if self.auto_compute:
                # TODO - expose these in the UI
                coverage_threshold = 80
                stack_length_threshold = 20

                # Intrinsics computation takes time - singal blocking state
                if self.monocular_tool.coverage >= coverage_threshold and self.monocular_tool.nb_samples > stack_length_threshold:
                    self.blocking.emit(True)

                r = self.monocular_tool.auto_compute_intrinsics(
                    coverage_threshold=coverage_threshold,
                    stack_length_threshold=stack_length_threshold,
                    simple_focal=True,
                    simple_distortion=True,
                    complex_distortion=False
                )
                self.blocking.emit(False)

                if r:
                    # If the intrinsics have been updated, emit the new errors...
                    self.send_payload.emit(
                        CalibrationData(self.cam_name, ErrorsPayload(self.monocular_tool.last_best_errors))
                    )

                    # ...and the intrinsics themselves
                    if self.monocular_tool.has_intrinsics:
                        self.send_payload.emit(
                            CalibrationData(self.cam_name, IntrinsicsPayload(*self.monocular_tool.intrinsics))
                        )

        # Compute extrinsics (will only do anything if the monocular tool has intrinsics)
        self.monocular_tool.compute_extrinsics()  # We do it in all modes to highlight the board perimeters
        # TODO - Maybe use the refined extrinsics instead, when we have them?

        # Extrinsics stage: Estimate pose (i.e. monocularly-estimated extrinsics) and emit
        if self._current_stage == 1:
            if self.monocular_tool.has_extrinsics:
                self.send_payload.emit(
                    CalibrationData(self.cam_name, PosePayload(frame_id, *self.monocular_tool.extrinsics))
                )

        # Refining stage: Emit points detections
        if self._current_stage == 2:
            if self.monocular_tool.has_detection:
                self.send_payload.emit(
                    CalibrationData(self.cam_name, DetectionPayload(frame_id, *self.monocular_tool.detection))
                )

        # Visualize (this returns the annotated image in full resolution)
        annotated = self.monocular_tool.visualise(errors_mm=True)

        # Emit the annotated frame to the main thread
        self.annotations.emit(annotated)

        # Emit the end signal to tell this thread is free
        self.finished.emit()

    @Slot(bool)
    def set_auto_sample(self, value):
        self.auto_sample = value

    @Slot(bool)
    def set_auto_compute(self, value):
        self.auto_compute = value

    @Slot()
    def add_sample(self):
        self.monocular_tool.register_sample()

    @Slot()
    def clear_samples(self):
        self.monocular_tool.clear_stacks()

    @Slot()
    def clear_intrinsics(self):
        self.monocular_tool.clear_intrinsics()
        self.monocular_tool.clear_stacks()

    @Slot(int)
    def set_stage(self, stage):
        self._current_stage = stage
        self.monocular_tool.clear_stacks()

class MultiviewWorker(BaseWorker):

    # signal_return_computed_poses = Signal(np.ndarray, np.ndarray)  # Send current camera poses back to main thread
    # signal_return_computed_points = Signal(np.ndarray)           # Send points 3d back to main thread

    def __init__(self, multiview_calib):
        super().__init__('multiview')
        self.multiview_calib = multiview_calib

    # @Slot(int, int, np.ndarray, np.ndarray)
    # def on_received_detection(self, frame_idx, cam_idx, points2d, points_ids):
    #     # when we recieve a detection from the monocular worker
    #     self.multiview_calib.register_detection(frame_idx, cam_idx, points2d, points_ids)
    #     if DEBUG:
    #         print(f"[DEBUG] [MultiCalibWorker] Registered detection for cam {cam_idx}\r")
    #
    # @Slot(int, int, np.ndarray, np.ndarray)
    # def on_received_camera_pose(self, frame_idx, cam_idx, rvec, tvec):
    #     # when we recieve a pose from the monocular worker
    #     self.multiview_calib.register_extrinsics(frame_idx, cam_idx, rvec, tvec)
    #     if DEBUG:
    #         print(f"[DEBUG] [MultiCalibWorker] Registered extrinsics for cam {cam_idx}\r")
    #
    # @Slot(int, np.ndarray, np.ndarray)
    # def on_updated_intrinsics(self, cam_idx, cam_mat, dist_coeff):
    #     self.multiview_calib.register_intrinsics(cam_idx, cam_mat, dist_coeff)
    #     if DEBUG:
    #         print(f'[DEBUG] [MultiCalibWorker] Intrinsics updated for camera {cam_idx}\r')

    @Slot()
    def compute(self):
        if self._paused:
            return

        # # Estimate extrinsics
        # self.multiview_calib.compute_estimation()
        #
        # rvecs, tvecs = self.multiview_calib.extrinsics
        # if rvecs is not None and tvecs is not None:
        #     # Send them back to the main thread
        #     self.signal_return_computed_poses.emit(rvecs, tvecs)
        pass

    @Slot(int)
    def set_origin_camera(self, value: int):
        self.multiview_calib.origin_camera = value

class CalibrationCoordinator(QObject):
    def __init__(self):
        super().__init__()
        self._workers = {}
        self.name = 'coordinator'

    def register_worker(self, worker: BaseWorker):
        self._workers[worker.name] = worker
        # The coordinator is always the one to receive payloads first, so direct connections here
        worker.send_payload.connect(self._route_data)

    @Slot(CalibrationData)
    def _route_data(self, data: CalibrationData):
        """
        Slot to receive and forward data to the corresponding private method
        """
        sending_worker = self.sender()

        if isinstance(data.payload, IntrinsicsPayload):
            print(f'[Coordinator] Received {data.camera_name} intrinsics payload from worker {sending_worker.name}')
            self._send_to('multiview', data)

        elif isinstance(data.payload, DetectionPayload):
            print(f'[Coordinator] Received {data.camera_name} extrinsics payload from worker {sending_worker.name}')

    def _send_to(self, worker_name, data: CalibrationData):
        # We go via the 'receive_payload' signal on the worker thread, so no direct connection required
        self._workers[worker_name].receive_payload.emit(data)
