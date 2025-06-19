import logging
from typing import Dict, List, Union, Optional
from PySide6.QtCore import QObject, Signal, Slot
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
    request_refinement = Signal()

    def __init__(self):
        super().__init__()
        self.name = 'coordinator'
        self._current_stage = 0
        self._workers: Dict[str, CalibrationProcessingWorker] = {}

        self._cameras_names: List[str] = []
        self._origin_camera_name: Optional[str] = None
        self._initial_intrinsics: Dict[str, IntrinsicsPayload] = {}

        self.receive_from_main.connect(self.on_payload_from_main)

    def register_worker(self, worker: CalibrationProcessingWorker):
        self._workers[worker.name] = worker
        self.broadcast_stage.connect(worker.set_stage)

        worker.send_payload.connect(self.on_payload_from_worker)
        self.broadcast_reset.connect(worker.reset)

        if worker.name == 'multiview':
            self.request_refinement.connect(worker.trigger_refinement)

        if isinstance(worker, MonocularWorker):
            self._cameras_names.append(worker.name)
            self.broadcast_new_board.connect(worker.configure_new_board)

            # if this is the first camera being registered, set it as the default origin
            if self._origin_camera_name is None:
                self.set_origin_camera(worker.name)

        worker.set_stage(self._current_stage)

    @Slot(int)
    def set_stage(self, stage: int):

        if stage == self._current_stage:
            return

        if stage_index == 1:  # Extrinsics stage
            self.main_window.origin_camera_combo.setEnabled(False)
        else:  # Intrinsics stage
            self.origin_camera_combo.setEnabled(True)

        self._current_stage = stage
        logger.debug(f"[{self.name.title()}] Broadcasting calibration stage: {stage}")

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

    @Slot(CalibrationData)
    def on_payload_from_main(self, data: CalibrationData):
        """
        Handles payloads sent directly from the main thread, typically from loading a file
        This method acts as a router to dispatch the data to the correct workers
        """
        logger.debug(f'[{self.name.title()}] Received (from Main): ({data.camera_name}) {data.payload}')

        payload = data.payload
        target_worker_name = data.camera_name

        # If it's a generic payload for all cameras, find all relevant workers
        if target_worker_name == 'all':
            target_workers = list(self._workers.values())
        elif target_worker_name in self._workers:
            target_workers = [self._workers[target_worker_name]]
        else:
            logger.warning(f"[Coordinator] Received payload for unknown worker '{target_worker_name}'.")
            return

        for worker in target_workers:
            worker.receive_payload.emit(data)

        self.send_to_main.emit(data)

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

        payload = data.payload

        # always send all payloads to the main UI thread for plots, text fields, etc
        self.send_to_main.emit(data)

        # Inter-worker routing
        # If the payload comes from a MonocularWorker, it should always be sent to the MultiviewWorker
        if sending_worker_name != 'multiview':

            # Block live-computed intrinsics from monocular workers during extrinsics estimation
            if isinstance(payload, IntrinsicsPayload) and self._current_stage > 0:
                logger.debug(
                    f"[{self.name.title()}] Dropping live intrinsics from {sending_worker_name} during stage {self._current_stage}.")
                return  # stop processing this payload further

            # For all other payloads (Detections, Exrinsics in stage 0), forward to MultiviewWorker
            if 'multiview' in self._workers:
                self._workers['multiview'].receive_payload.emit(data)

        # MultiviewWorker -> MonocularWorker (for refined intrinsics/extrinsics)
        if sending_worker_name == 'multiview' and isinstance(payload, IntrinsicsPayload):
            target_worker = self._workers.get(data.camera_name)
            if isinstance(target_worker, MonocularWorker):
                target_worker.receive_payload.emit(data)