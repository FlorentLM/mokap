import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=5)
# Use a QTimer for non-blocking updates from the MultiviewWorker
from PySide6.QtCore import QObject, Signal, Slot, QTimer
from mokap.calibration import MonocularCalibrationTool, MultiviewCalibrationTool
from mokap.utils.datatypes import (ChessBoard, CharucoBoard, CalibrationData, IntrinsicsPayload, ExtrinsicsPayload,
                                   ErrorsPayload, OriginCameraPayload, PosePayload, DetectionPayload)
from typing import Union, List, Dict
from numpy.typing import ArrayLike


DEBUG_SIGNALS_FLOW = True
VERBOSE = True
GUI_UPDATE_TIMER = 200

# ===================================================================
# BASE CLASSES TODO: need to be moved to a separate file
# ===================================================================

class CalibrationWorker(QObject):
    """ Base class for all the Calibration workers. Sends/Receives CalibrationData objects. """
    error = Signal(Exception)
    send_payload = Signal(CalibrationData)
    receive_payload = Signal(CalibrationData)

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = name
        self.receive_payload.connect(self._handle_payload)

    @Slot(CalibrationData)
    def _handle_payload(self, data: CalibrationData):
        if DEBUG_SIGNALS_FLOW:
            print(f'[{self.name.title()}] Received (from Coordinator): ({data.camera_name}) {data.payload}')


class ProcessingWorker(QObject):
    """ Base class for all processing workers. Can be paused. """
    finished = Signal()
    blocking = Signal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._paused = False

    def set_paused(self, paused: bool):
        self._paused = paused

    @Slot(np.ndarray, int)
    def handle_frame(self, frame, frame_idx):
        if self._paused:
            return


class CalibrationProcessingWorker(CalibrationWorker, ProcessingWorker):
    """ Base class for calibration processing workers (i.e. MonocularWorker and MultiviewWorker) """
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self._current_stage = 0

    @Slot(int)
    def set_stage(self, stage: int):
        self._current_stage = stage
        if DEBUG_SIGNALS_FLOW:
            print(f"[{self.name.title()}] Received calibration stage: {stage}")


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


# ===================================================================
# MONOCULAR WORKER
# ===================================================================

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


# ===================================================================
# MULTIVIEW WORKER
# ===================================================================

class MultiviewWorker(CalibrationProcessingWorker):

    def __init__(self, cameras_names: List[str], origin_camera_name: str):
        super().__init__(name='multiview')
        self.name = 'multiview'  # redundant

        self._cameras_names = cameras_names
        self._origin_camera_name = origin_camera_name
        self.multiview_tool: MultiviewCalibrationTool = None  # Will be initialized later

        self.update_timer = QTimer(self)
        self.update_timer.setInterval(GUI_UPDATE_TIMER)
        self.update_timer.timeout.connect(self.emit_online_extrinsics)

    @Slot(MultiviewCalibrationTool)
    def initialize_tool(self, tool: MultiviewCalibrationTool):
        """
        Receives the fully configured tool from the Coordinator when calibration stage > 0
        """
        self.multiview_tool = tool
        self.update_timer.start()
        if DEBUG_SIGNALS_FLOW:
            print(f"[{self.name.title()}] Multiview tool initialized and ready.")

    @Slot(CalibrationData)
    def _handle_payload(self, data: CalibrationData):
        """
        This is ONLY for receiving detection data during online refinement
        """
        if DEBUG_SIGNALS_FLOW:
            print(f'[{self.name.title()}] Received (from Coordinator): ({data.camera_name}) {data.payload}')

        # If tool isn't ready or payload isn't a detection, do nothing
        if self.multiview_tool is None or not isinstance(data.payload, DetectionPayload):
            return

        # core of the online process: register the detection
        cam_idx = self._cameras_names.index(data.camera_name)
        self.multiview_tool.register(cam_idx, data.payload)

    def emit_online_extrinsics(self):
        """
        Called by the QTimer. Checks if the tool has estimated extrinsics and emits them
        """
        if self.multiview_tool and self.multiview_tool._estimated:
            rvecs, tvecs = self.multiview_tool.initial_extrinsics
            for i, cam_name in enumerate(self._cameras_names):
                payload = ExtrinsicsPayload(rvecs[i], tvecs[i])
                # This goes to the Coordinator, which will forward it to the main thread UI
                self.send_payload.emit(CalibrationData(cam_name, payload))

    @Slot()
    def run_bundle_adjustment(self):
        """
        Slot to be connected to a GUI button. Triggers the final BA
        """
        if self._paused or self.multiview_tool is None:
            return

        print(f"[{self.name.title()}] Attempting to run final Bundle Adjustment.")
        self.blocking.emit(True)

        # TODO: Get these flags from the UI
        success = self.multiview_tool.refine_all(
            simple_focal=False,
            simple_distortion=False,
            complex_distortion=False,
            shared=False
        )
        self.blocking.emit(False)

        if success:
            print(f"[{self.name.title()}] Bundle Adjustment successful. Emitting refined results.")
            # Emit the final, highly accurate results
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
            print(f"[{self.name.title()}] Bundle Adjustment failed.")

    @Slot(int)
    def set_stage(self, stage: int):
        super().set_stage(stage)

        # If we move back to stage 0, we must destroy the tool and stop the timer
        if stage == 0 and self.multiview_tool is not None:
            print(f"[{self.name.title()}] Resetting. Multiview tool destroyed.")
            self.update_timer.stop()
            self.multiview_tool = None


# ===================================================================
# COORDINATOR (Refactored to be the orchestrator)
# ===================================================================

class CalibrationCoordinator(QObject):
    broadcast_stage = Signal(int)
    send_to_main = Signal(CalibrationData)
    receive_from_main = Signal(CalibrationData)
    broadcast_new_board = Signal(object)

    def __init__(self):
        super().__init__()
        self.name = 'coordinator'
        self._current_stage = 0
        self._workers: Dict[str, CalibrationProcessingWorker] = {}

        self._cameras_names: List[str] = []
        self._origin_camera_name: str = None
        self._initial_intrinsics: Dict[str, IntrinsicsPayload] = {}
        self.receive_from_main.connect(self._handle_payload_from_main)

    def register_worker(self, worker: CalibrationProcessingWorker):
        self._workers[worker.name] = worker
        self.broadcast_stage.connect(worker.set_stage)
        worker.send_payload.connect(self._route_payload)

        # ### NEW ### Collect camera names on registration
        if isinstance(worker, MonocularWorker):
            self._cameras_names.append(worker.name)
            self.broadcast_new_board.connect(worker.update_board)

        worker.set_stage(self._current_stage)

    @Slot(int)
    def set_stage(self, stage: int):
        if stage == self._current_stage:
            return  # No change

        self._current_stage = stage
        if DEBUG_SIGNALS_FLOW:
            print(f"[{self.name.title()}] Broadcasting calibration stage: {stage}")

        # ### NEW ### Orchestration logic
        if self._current_stage == 1:
            self._initialize_multiview_tool()

        self.broadcast_stage.emit(stage)

    @Slot(object)
    def handle_board_change(self, new_board_params: Union[ChessBoard, CharucoBoard]):
        """
        Receives new board parameters from the GUI and triggers a system-wide reset.
        """
        if DEBUG_SIGNALS_FLOW:
            print(f"[{self.name.title()}] Board parameters changed. Triggering full system reset.")

        # Reset internal state
        self._initial_intrinsics.clear()

        # Force the entire system back to the beginning (Stage 0)
        # We call the method directly to ensure the logic runs before the signal is emitted
        self.set_stage(0)

        # Broadcast the new board parameters to all MonocularWorkers so they can recreate their tools
        self.broadcast_new_board.emit(new_board_params)

    @Slot(str)
    def set_origin_camera(self, camera_name: str):
        if camera_name in self._cameras_names:
            self._origin_camera_name = camera_name
            print(f"[{self.name.title()}] Origin camera set to: {camera_name}")
        else:
            print(f"[{self.name.title()}] Error: Unknown camera name '{camera_name}' for origin.")

    def _initialize_multiview_tool(self):
        """
        Gathers all required data, creates the MultiviewCalibrationTool, and sends it to the MultiviewWorker
        """
        multiview_worker = self._workers.get('multiview')
        if not multiview_worker:
            print("[Coordinator] Error: MultiviewWorker not registered.")
            return

        # Check if we have all initial intrinsics
        if len(self._initial_intrinsics) != len(self._cameras_names):
            print(
                f"[Coordinator] Error: Cannot start extrinsics stage. Have {len(self._initial_intrinsics)}/{len(self._cameras_names)} initial intrinsics.")
            # Revert stage to 0 to prevent getting stuck
            self.set_stage(0)
            return

        print("[Coordinator] All initial intrinsics received. Initializing Multiview tool.")

        # Gather data for the tool's constructor
        init_cam_matrices = []
        init_dist_coeffs = []
        images_sizes_wh = []
        for name in self._cameras_names:
            worker = self._workers[name]
            intr = self._initial_intrinsics[name]
            init_cam_matrices.append(intr.camera_matrix)
            init_dist_coeffs.append(intr.dist_coeffs)
            images_sizes_wh.append((worker.cam_shape[1], worker.cam_shape[0]))  # w, h order

        # This assumes the first registered monocular worker's board is representative
        object_points = self._workers[self._cameras_names[0]].board_object_points
        origin_idx = self._cameras_names.index(self._origin_camera_name)

        # Create the tool
        tool = MultiviewCalibrationTool(
            nb_cameras=len(self._cameras_names),
            images_sizes_wh=np.array(images_sizes_wh),
            origin_idx=origin_idx,
            init_cam_matrices=np.array(init_cam_matrices),
            init_dist_coeffs=np.array(init_dist_coeffs),
            object_points=object_points,
            intrinsics_window=10,
            min_detections=15,
            max_detections=100,
            debug_print=VERBOSE
        )

        # Send the tool to the worker
        multiview_worker.initialize_tool(tool)

    @Slot(CalibrationData)
    def _handle_payload_from_main(self, data: CalibrationData):
        """
        Handles payloads sent directly from the main thread, typically from loading a file
        This method acts as a router to dispatch the data to the correct workers
        """
        if DEBUG_SIGNALS_FLOW:
            print(f'[{self.name.title()}] Received (from Main): ({data.camera_name}) {data.payload}')

        payload = data.payload
        target_worker_name = data.camera_name

        if target_worker_name not in self._workers:
            print(f"[Coordinator] Warning: Received payload for unknown worker '{target_worker_name}'.")
            return

        # Use the target worker's public 'receive_payload' signal to send the data
        worker = self._workers[target_worker_name]
        worker.receive_payload.emit(data)

        # Route back to the main thread for UI update
        # The MainControls widget will catch this and update the correct UI element
        self.send_to_main.emit(data)

        # Additionally, if the payload is intrinsics, we need to update our internal cache
        # so that the MultiviewTool can be created correctly later
        if isinstance(payload, IntrinsicsPayload):
            self._initial_intrinsics[data.camera_name] = payload
            # If the multiview worker already exists, it also needs the update
            if 'multiview' in self._workers:
                self._workers['multiview'].receive_payload.emit(data)

    @Slot(CalibrationData)
    def _route_payload(self, data: CalibrationData):
        """
        This method is for payloads coming FROM workers
        """

        sending_worker_name = self.sender().name
        if DEBUG_SIGNALS_FLOW:
            print(
                f'[{self.name.title()}] Received (from {sending_worker_name.title()}): ({data.camera_name}) {data.payload}')

        # Route based on payload type and current stage
        payload = data.payload

        # --- Inter-worker routing ---

        if isinstance(payload, IntrinsicsPayload):
            # This can come from a MonocularWorker (initial) or MultiviewWorker (refined)
            if sending_worker_name != 'multiview':  # From a MonocularWorker
                # Store it for the eventual creation of the MultiviewTool
                self._initial_intrinsics[data.camera_name] = payload
            # In all cases, forward to the main thread for UI updates
            self.send_to_main.emit(data)

        elif isinstance(payload, ExtrinsicsPayload):
            # This should ONLY come from the MultiviewWorker
            # Forward to main thread for UI updates
            self.send_to_main.emit(data)

        elif isinstance(payload, DetectionPayload):
            # This comes from a MonocularWorker. If we are in the right stage, route it.
            if self._current_stage >= 1:
                self._workers['multiview'].receive_payload.emit(data)

        elif isinstance(payload, ErrorsPayload):
            # This comes from a MonocularWorker's initial calibration.
            # Forward to main thread for UI updates.
            self.send_to_main.emit(data)

        # --- And always forward relevant data to the Main Thread for UI updates ---
        if isinstance(payload, (IntrinsicsPayload, ExtrinsicsPayload, ErrorsPayload)):
            self.send_to_main.emit(data)