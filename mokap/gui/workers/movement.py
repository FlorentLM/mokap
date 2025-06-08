import numpy as np
from PySide6.QtCore import Signal, Slot
from mokap.gui.workers.base import ProcessingWorker


class MovementWorker(ProcessingWorker):

    annotations = Signal(list)

    def __init__(self, name: str):
        super().__init__()
        self.name: str = name

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
