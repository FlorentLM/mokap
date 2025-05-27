import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=5)
from PySide6.QtCore import QObject, Signal, Slot
from mokap.calibration import MonocularCalibrationTool, MultiviewCalibrationTool
from mokap.utils.datatypes import (BoardParams, CalibrationData, IntrinsicsPayload, ExtrinsicsPayload,
                                   ErrorsPayload, OriginCameraPayload, PosePayload, DetectionPayload)
from numpy.typing import ArrayLike


class CalibrationWorker(QObject):
    """ Base class for all the Calibration workers. Sends/Receives CalibrationData objects. """
    error = Signal(Exception)

    send_payload = Signal(CalibrationData)      # Single output signal - will always go to the coordinator
    receive_payload = Signal(CalibrationData)   # Single input signal - will always be accessed by the coordinator

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = name

        # We connect through an intermediary internal signal so that we can call receive_payload from the outside
        # without having to specifically connect to _handle_payload
        self.receive_payload.connect(self._handle_payload)

    @Slot(CalibrationData)
    def _handle_payload(self, data: CalibrationData):
        # This signal is coming from the receive_payload proxy, which always comes from the Coordinator
        print(f'[{self.name.title()}] Received (from Coordinator): ({data.camera_name}) {data.payload}')

class ProcessingWorker(QObject):
    """ Base class for all processing workers. Can be paused. """

    finished = Signal()
    blocking = Signal(bool)     # Report when doing a blocking function

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._paused = False

    def set_paused(self, paused: bool):
        self._paused = paused

    @Slot(np.ndarray, int)
    def handle_frame(self, frame, frame_idx):
        if self._paused:
            return

class MovementWorker(ProcessingWorker):

    annotations = Signal(list)

    def __init__(self, name: str):
        super().__init__()
        self.name: str = name

    #  ============= Slots for direct Main Thread signals =============
    @Slot(np.ndarray, int)
    def handle_frame(self, frame, frame_idx):
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

class CalibrationProcessingWorker(CalibrationWorker, ProcessingWorker):
    """ Base class for calibration processing workers (i.e. MonocularWorker and MultiviewWorker) """

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self._current_stage = 0

    @Slot(int)
    def set_stage(self, stage: int):
        """ Update worker's internal stage (can be overridden) """
        self._current_stage = stage
        print(f"[{self.name.title()}] Received calibration stage: {stage}")

class MonocularWorker(CalibrationProcessingWorker):

    annotations = Signal(np.ndarray)       # Annotations are already burned into the image, so emit the whole thing

    def __init__(self, board_params: BoardParams, cam_idx: int, cam_name: str, cam_shape: ArrayLike):
        super().__init__(name=cam_name)
        self.camera_idx = cam_idx
        self.cam_name = cam_name
        self.cam_shape = cam_shape

        self.monocular_tool = MonocularCalibrationTool(
            board_params=board_params,
            imsize_hw=self.cam_shape[:2],   # pass frame size so it can track coverage
            focal_mm=60,                    # TODO - UI field for these
            sensor_size='1/2.9"'  #
        )
        self.monocular_tool.set_visualisation_scale(2)

        # Flags for worker function
        self._auto_sample = True
        self._auto_compute = True

        self._current_stage = 0

    #  ============= Slots for Coordinator signals =============
    @Slot(int)
    def set_stage(self, stage: int):
        """ Extended stage handling with calibration tool reset """
        super().set_stage(stage)
        self.monocular_tool.clear_stacks()

    @Slot(CalibrationData)
    def _handle_payload(self, data: CalibrationData):
        super()._handle_payload(data)  # Keep debug print

        payload = data.payload

        if isinstance(payload, IntrinsicsPayload):
            self.monocular_tool.set_intrinsics(payload.camera_matrix, payload.dist_coeffs, payload.errors)

    #  ============= Slots for direct Main Thread signals =============
    @Slot(np.ndarray, int)
    def handle_frame(self, frame: ArrayLike, frame_idx: int):
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
                    simple_focal=False,
                    simple_distortion=False,
                    complex_distortion=False
                )
                self.blocking.emit(False)

                if r:
                    # If the intrinsics have been updated, emit the new errors...
                    self.send_payload.emit(
                        CalibrationData(self.cam_name, ErrorsPayload(self.monocular_tool.intrinsics_errors))
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
                    CalibrationData(self.cam_name, PosePayload(frame_idx, *self.monocular_tool.extrinsics))
                )

        # Refining stage: Emit points detections
        if self._current_stage == 2:
            if self.monocular_tool.has_detection:
                self.send_payload.emit(
                    CalibrationData(self.cam_name, DetectionPayload(frame_idx, *self.monocular_tool.detection))
                )

        # Visualize (this returns the annotated image in full resolution)
        annotated = self.monocular_tool.visualise(errors_mm=True)

        # Emit the annotated frame to the main thread
        self.annotations.emit(annotated)

        # Emit the end signal to tell this thread is free
        self.finished.emit()

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

class MultiviewWorker(CalibrationProcessingWorker):

    def __init__(self, cameras_names, origin_camera_name):
        super().__init__(name='multiview')
        self.name = 'multiview'

        self._cameras_names = cameras_names.copy()
        self._origin_camera_idx = self._cameras_names.index(origin_camera_name)

        self.multiview_tool = MultiviewCalibrationTool(len(self._cameras_names),
                                                       origin_camera_idx=self._origin_camera_idx,
                                                       min_poses=3)

    @Slot(CalibrationData)
    def _handle_payload(self, data: CalibrationData):
        super()._handle_payload(data)  # Keep debug print

        if isinstance(data.payload, IntrinsicsPayload):
            self.multiview_tool.register_intrinsics(
                self._cameras_names.index(data.camera_name),
                data.payload.camera_matrix,
                data.payload.dist_coeffs
            )

        elif isinstance(data.payload, DetectionPayload):
            self.multiview_tool.register_detection(
                data.payload.frame,
                self._cameras_names.index(data.camera_name),
                data.payload.points2D,
                data.payload.pointsIDs
            )

        elif isinstance(data.payload, PosePayload):
            self.multiview_tool.register_pose(
                data.payload.frame,
                self._cameras_names.index(data.camera_name),
                data.payload.rvec,
                data.payload.tvec
            )

        elif isinstance(data.payload, OriginCameraPayload):
            self._origin_camera_idx = self._cameras_names.index(data.payload.camera_name)
            self.multiview_tool.origin_camera = self._origin_camera_idx

    @Slot()
    def compute(self):
        if self._paused:
            return

        if self._current_stage == 2:
            # Estimate extrinsics
            self.multiview_tool.estimate_extrinsics()

            if self.multiview_tool.has_extrinsics:
                # If this worked, send them back to the main thread
                rvecs, tvecs = self.multiview_tool.extrinsics
                for cam_idx, cam_name in enumerate(self._cameras_names):
                    self.send_payload.emit(CalibrationData(cam_name, ExtrinsicsPayload(rvecs[cam_idx, :], tvecs[cam_idx, :])))
        else:
            pass

class CalibrationCoordinator(QObject):

    broadcast_stage = Signal(int)
    send_to_main = Signal(CalibrationData)
    receive_from_main = Signal(CalibrationData)  # New signal for main thread

    def __init__(self):
        super().__init__()
        self.name = 'coordinator'
        self._current_stage = 0

        self._workers = {}

        self.receive_from_main.connect(self._handle_payload_from_main)

    def register_worker(self, worker: CalibrationProcessingWorker):
        self._workers[worker.name] = worker
        self.broadcast_stage.connect(worker.set_stage)
        worker.set_stage(self._current_stage)
        # The coordinator is always the one to receive payloads first, so direct connections here
        worker.send_payload.connect(self._route_payload)

        self._current_stage = 0

    @Slot(int)
    def set_stage(self, stage: int):
        """ Main thread entry point for stage changes """
        self._current_stage = stage
        # Broadcast to all workers
        self.broadcast_stage.emit(stage)
        print(f"[{self.name.title()}] Broadcasting calibration stage: {stage}")

    @Slot(str)
    def set_origin_camera(self, camera_name: str):
        """ Main thread entry point for origin camera change """
        data = CalibrationData('multiview', OriginCameraPayload(camera_name))   # a bit redundant...
        self._send_to_worker('multiview', data)    # Direct to multiview worker

    @Slot(CalibrationData)
    def _handle_payload_from_main(self, data: CalibrationData):
        """ Main thread entry point for CalibrationData """

        print(f'[{self.name.title()}] Received (from Main): ({data.camera_name}) {data.payload}')

        if isinstance(data.payload, IntrinsicsPayload) or isinstance(data.payload, ExtrinsicsPayload):
            # Send to respective MonocularWorker and MultiviewWorker
            self._send_to_worker(data.camera_name, data)
            self._send_to_worker('multiview', data)

    @Slot(CalibrationData)
    def _route_payload(self, data: CalibrationData):

        sending_worker = self.sender()
        # This signal carries the source sender still
        print(f'[{self.name.title()}] Received (from {sending_worker.name.title()}): ({data.camera_name}) {data.payload}')

        if isinstance(data.payload, IntrinsicsPayload):
            if sending_worker.name == 'multiview':
                # This is the refined intrinsics -> send to main thread
                self.send_to_main.emit(data)  # Use the sigle signal towards main thread

            elif sending_worker.name == data.camera_name and self._current_stage == 0:
                # This comes from a monocular worker, so forward to the aggregator if we're in stage 0
                self._send_to_worker('multiview', data)

        elif isinstance(data.payload, ExtrinsicsPayload):
            if sending_worker.name == 'multiview':
                # This is the refined extrinsics -> send to main thread
                self.send_to_main.emit(data)  # Use the sigle signal towards main thread

        elif isinstance(data.payload, PosePayload):
            # Must always come from a monocular worker, and only route if stage 1
            if sending_worker == data.camera_name and self._current_stage == 1:
                self._send_to_worker('multiview', data)

        elif isinstance(data.payload, DetectionPayload):
            # Must always come from a monocular worker, and only route if stage 2
            if sending_worker == data.camera_name and self._current_stage == 2:
                self._send_to_worker('multiview', data)

        elif isinstance(data.payload, ErrorsPayload):
            # Must always come from a monocular worker, and only route if stage 0
            if sending_worker == data.camera_name and self._current_stage == 0:
                self.send_to_main.emit(data)

    def _send_to_worker(self, worker_name, data: CalibrationData):
        # We go via the 'receive_payload' signal on the worker thread, so no direct connection required
        self._workers[worker_name].receive_payload.emit(data)
