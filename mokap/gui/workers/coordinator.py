import logging
from typing import Dict, List, Union
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from mokap.calibration.multiview import MultiviewCalibrationTool
from mokap.gui.workers.workers_base import CalibrationProcessingWorker
from mokap.gui.workers.worker_monocular import MonocularWorker
from mokap.utils.datatypes import ChessBoard, CharucoBoard, CalibrationData, IntrinsicsPayload


logger = logging.getLogger(__name__)


class CalibrationCoordinator(QObject):

    send_to_main = Signal(CalibrationData)
    receive_from_main = Signal(CalibrationData)

    broadcast_stage = Signal(int)
    broadcast_new_board = Signal(object)
    broadcast_reset = Signal()

    def __init__(self):
        super().__init__()
        self.name = 'coordinator'
        self._current_stage = 0
        self._workers: Dict[str, CalibrationProcessingWorker] = {}

        self._cameras_names: List[str] = []
        self._origin_camera_name: str = None
        self._initial_intrinsics: Dict[str, IntrinsicsPayload] = {}

        self.receive_from_main.connect(self.on_payload_from_main)

    def register_worker(self, worker: CalibrationProcessingWorker):
        self._workers[worker.name] = worker
        self.broadcast_stage.connect(worker.set_stage)

        worker.send_payload.connect(self.on_payload_from_worker)
        self.broadcast_reset.connect(worker.reset)

        if isinstance(worker, MonocularWorker):
            self._cameras_names.append(worker.name)
            self.broadcast_new_board.connect(worker.configure_new_board)

        worker.set_stage(self._current_stage)

    @Slot(int)
    def set_stage(self, stage: int):

        if stage == self._current_stage:
            return

        self._current_stage = stage
        logger.debug(f"[{self.name.title()}] Broadcasting calibration stage: {stage}")

        if self._current_stage == 1:
            self._initialize_multiview_tool()

        self.broadcast_stage.emit(stage)

    @Slot(object)
    def handle_board_change(self, new_board_params: Union[ChessBoard, CharucoBoard]):
        """ Receives new board parameters from the GUI and triggers a system-wide reset """

        logger.debug(f"[{self.name.title()}] Board parameters changed. Triggering full system reset.")

        self._initial_intrinsics.clear()
        self.set_stage(0)    # force the entire system back to the beginning
        self.broadcast_reset.emit() # tell all workers to reset their internal state

        # Broadcast the new board parameters to all MonocularWorkers so they can recreate their tools
        self.broadcast_new_board.emit(new_board_params)

    @Slot(str)
    def set_origin_camera(self, camera_name: str):

        if camera_name in self._cameras_names:
            self._origin_camera_name = camera_name
            logger.info(f"[{self.name.title()}] Origin camera set to: {camera_name}")

        else:
            logger.error(f"[{self.name.title()}] Unknown camera name '{camera_name}' for origin.")

    def _initialize_multiview_tool(self):
        """ Gathers all required data, creates the MultiviewCalibrationTool, and sends it to the MultiviewWorker """

        multiview_worker = self._workers.get('multiview')

        if not multiview_worker:
            logger.error("[Coordinator] MultiviewWorker not registered.")
            return

        # Check if we have all initial intrinsics
        if len(self._initial_intrinsics) != len(self._cameras_names):
            logger.error(f"[Coordinator] Cannot start extrinsics stage. Have {len(self._initial_intrinsics)}/{len(self._cameras_names)} initial intrinsics.")

            # Revert stage to 0 to prevent getting stuck
            self.set_stage(0)
            return

        logger.debug("[Coordinator] All initial intrinsics received. Initializing Multiview tool.")

        # Gather data for the tool's constructor
        init_cam_matrices = []
        init_dist_coeffs = []
        images_sizes_wh = []
        for name in self._cameras_names:
            worker = self._workers[name]
            intr = self._initial_intrinsics[name]
            init_cam_matrices.append(intr.camera_matrix)
            init_dist_coeffs.append(intr.dist_coeffs)
            images_sizes_wh.append((worker.cam_width, worker.cam_height))

        # This assumes the first registered monocular worker's board is representative
        object_points = self._workers[self._cameras_names[0]].board_object_points
        origin_idx = self._cameras_names.index(self._origin_camera_name)

        tool = MultiviewCalibrationTool(
            nb_cameras=len(self._cameras_names),
            images_sizes_wh=np.array(images_sizes_wh),
            origin_idx=origin_idx,
            init_cam_matrices=np.array(init_cam_matrices),
            init_dist_coeffs=np.array(init_dist_coeffs),
            object_points=object_points,
            min_detections=15,      # TODO: Move these into the init, and have GUI controls for them
            max_detections=100
        )

        # Send the tool to the worker
        multiview_worker.initialize_tool(tool)

    @Slot(CalibrationData)
    def on_payload_from_main(self, data: CalibrationData):
        """
        Handles payloads sent directly from the main thread, typically from loading a file
        This method acts as a router to dispatch the data to the correct workers
        """

        logger.debug(f'[{self.name.title()}] Received (from Main): ({data.camera_name}) {data.payload}')

        payload = data.payload
        target_worker_name = data.camera_name

        if target_worker_name not in self._workers:
            logger.debug(f"[Coordinator] Warning: Received payload for unknown worker '{target_worker_name}'.")
            return

        # Use the target worker's public 'receive_payload' signal to send the data
        worker = self._workers[target_worker_name]
        worker.receive_payload.emit(data)

        # Route back to the main thread for UI update
        # The MainControls widget will catch this and update the correct UI element
        self.send_to_main.emit(data)

        # Additionally, if the payload is intrinsics, we need to update the internal cache
        if isinstance(payload, IntrinsicsPayload):
            self._initial_intrinsics[data.camera_name] = payload

    @Slot(CalibrationData)
    def on_payload_from_worker(self, data: CalibrationData):
        """
        This method is for payloads coming from workers.
        It routes data to other workers and to the main thread for UI updates
        """

        sending_worker_name = self.sender().name
        logger.debug(
                f'[{self.name.title()}] Received (from {sending_worker_name.title()}): ({data.camera_name}) {data.payload}')

        # Route based on payload type and current stage
        payload = data.payload

        # always send all payloads to the main UI thread for plots, text fields, etc
        self.send_to_main.emit(data)

        # Inter-worker routing
        # If the payload comes from a MonocularWorker, it should be sent to the MultiviewWorker
        if sending_worker_name != 'multiview':
            if 'multiview' in self._workers:
                self._workers['multiview'].receive_payload.emit(data)

        # If the payload is refined intrinsics from Multiview, send it back to the
        # relevant MonocularWorker so it can update its internal tool state
        if sending_worker_name == 'multiview' and isinstance(payload, IntrinsicsPayload):
            target_worker = self._workers.get(data.camera_name)
            if isinstance(target_worker, MonocularWorker):
                target_worker.receive_payload.emit(data)