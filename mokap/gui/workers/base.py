import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from mokap.gui.workers import DEBUG_SIGNALS_FLOW
from mokap.utils.datatypes import CalibrationData


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
