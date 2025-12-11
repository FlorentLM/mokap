"""
Stateless Detector Threads

These threads handle the CPU-intensive detection work and pass results
back to the main calibration workers via signals.

Key design:
- Detectors are stateless (per Lucida's design)
- Each camera gets its own detector thread
- Detection results are emitted as signals, not stored
- The calibration tool/worker decides what to do with detections
"""
import logging
from typing import Union
import numpy as np
from numpy.typing import ArrayLike
from PySide6.QtCore import QObject, Signal, Slot
from lucida import CameraModel
from lucida.calibration import CharucoBoard, ChessBoard, CharucoDetector, ChessboardDetector
from mokap.gui.workers import DetectionResult

logger = logging.getLogger(__name__)


class DetectorWorker(QObject):
    """
    Stateless detector that runs in a separate thread.
    Receives frames, runs detection, emits results.
    """

    # Emitted when detection is complete (success or failure)
    detection_ready = Signal(object)  # DetectionResult
    
    # Emitted when frame processing is done (for synchronisation)
    finished = Signal()

    def __init__(
            self,
            camera: 'CameraModel',
            calibration_board: Union['ChessBoard', 'CharucoBoard'],
    ):
        super().__init__()
        
        self._camera = camera
        self._board = calibration_board

        # Create the appropriate detector
        if type(calibration_board) is CharucoBoard:
            self._detector = CharucoDetector(calibration_board)

        elif type(calibration_board) is ChessBoard:
            self._detector = ChessboardDetector(calibration_board)

        else:
            raise TypeError(f"Unsupported board type: {type(calibration_board)}")
        
        self._paused = False

    @property
    def name(self) -> str:
        return f"detector_{self._camera.name}"

    @property 
    def camera_name(self) -> str:
        return self._camera.name

    @Slot(np.ndarray, int)
    def handle_frame(self, frame: ArrayLike, frame_idx: int):
        """
        Process a frame and emit detection results.
        """
        if self._paused:
            self.finished.emit()
            return
        
        # Run detection (the K, D and refine_markers args are ignored by the detector if it's a Chessboard)
        detection = self._detector.detect(
            frame,
            K=self._camera.K,
            D=self._camera.D,
            refine_markers=True,
            refine_points=True,
        )

        # Package result
        result = DetectionResult(
            frame_idx=frame_idx,
            image_points=detection.image_points,
            valid=detection.valid
        )

        self.detection_ready.emit(result)
        self.finished.emit()

    @Slot(object)
    def configure_new_board(self, board: Union['ChessBoard', 'CharucoBoard']):
        """Update the calibration board (recreates detector)."""
        self._board = board
        
        if type(self._board) is CharucoBoard:
            self._detector = CharucoDetector(board)

        elif type(self._board) is ChessBoard:
            self._detector = ChessboardDetector(board)

        else:
            raise TypeError(f"Unsupported board type: {type(self._board)}")
        
        logger.debug(f"[{self.name}] Detector reconfigured for new board.")

    def set_paused(self, paused: bool):
        self._paused = paused