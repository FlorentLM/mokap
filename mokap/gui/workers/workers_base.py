import logging
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from mokap.utils.datatypes import CalibrationData

logger = logging.getLogger(__name__)


class CalibrationWorker(QObject):
    """ Base class for all the Calibration workers
     Sends/Receives CalibrationData objects """

    error = Signal(Exception)
    send_payload = Signal(CalibrationData)
    receive_payload = Signal(CalibrationData)

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = name
        self.receive_payload.connect(self.on_payload_received)

    @Slot(CalibrationData)
    def on_payload_received(self, data: CalibrationData):
        """ Handles incoming payloads routed from the Coordinator """
        logger.debug(f'[{self.name.title()}] Received (from Coordinator): ({data.camera_name}) {data.payload}')


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

    @Slot()
    def reset(self):
        """ Resets the worker's internal state (typically when changing calibration stages or board params)
        Implemented by subclasses """

        logger.debug(f"[{self.name.title()}] Received reset signal.")

        self.set_stage(0) # default reset behavior is to go back to stage 0

    @Slot(int)
    def set_stage(self, stage: int):
        self._current_stage = stage
        logger.debug(f"[{self.name.title()}] Received calibration stage: {stage}")