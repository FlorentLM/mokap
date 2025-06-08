from typing import Dict, List, Union

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from mokap.calibration.multiview import MultiviewCalibrationTool
from mokap.gui.workers.base import CalibrationProcessingWorker
from mokap.gui.workers.monocular import MonocularWorker
from mokap.utils.datatypes import ChessBoard, CharucoBoard, CalibrationData, IntrinsicsPayload, ExtrinsicsPayload, DetectionPayload, ErrorsPayload

DEBUG_SIGNALS_FLOW = True
VERBOSE = True


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
