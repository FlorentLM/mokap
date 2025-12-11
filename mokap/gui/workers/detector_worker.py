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
            camera: CameraModel,
            calibration_board: Union[ChessBoard, CharucoBoard],
    ):
        super().__init__()
        
        self._camera = camera
        self._board = calibration_board

        # Create the appropriate detector
        if isinstance(calibration_board, CharucoBoard) and not isinstance(calibration_board, ChessBoard):
            self._detector = CharucoDetector(calibration_board)

        elif isinstance(calibration_board, ChessBoard):
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
    def configure_new_board(self, board: Union[ChessBoard, CharucoBoard]):
        """Update the calibration board (recreates detector)."""
        self._board = board
        
        if isinstance(board, CharucoBoard) and not isinstance(board, ChessBoard):
            self._detector = CharucoDetector(board)
        else:
            self._detector = ChessboardDetector(board)
        
        logger.debug(f"[{self.name}] Detector reconfigured for new board.")

    def set_paused(self, paused: bool):
        self._paused = paused


# class MultiDetectorManager(QObject):
#     """
#     Manages multiple detector threads for a camera rig.
#
#     Provides a convenient interface for:
#     - Creating detector threads for each camera
#     - Routing frames to the correct detector
#     - Collecting detection results
#
#     Usage:
#         manager = MultiDetectorManager(rig, board)
#         manager.start_all()
#
#         # Route frames
#         for cam_name, frame in frames.items():
#             manager.submit_frame(cam_name, frame, frame_idx)
#
#         # Collect results (connect to this signal)
#         manager.detection_ready.connect(on_detection)
#     """
#
#     # Emitted when any detector produces a result
#     # Includes camera name for routing: (camera_name, DetectionResult)
#     detection_ready = Signal(str, object)
#
#     def __init__(
#             self,
#             cameras: list,  # List of CameraModel
#             calibration_board: Union[ChessBoard, CharucoBoard],
#     ):
#         super().__init__()
#
#         from PySide6.QtCore import QThread
#
#         self._board = calibration_board
#         self._detectors: dict[str, DetectorThread] = {}
#         self._threads: dict[str, QThread] = {}
#
#         for camera in cameras:
#             # Create detector
#             detector = DetectorThread(camera, calibration_board)
#
#             # Create thread
#             thread = QThread()
#             detector.moveToThread(thread)
#
#             # Connect detection results (add camera name for routing)
#             detector.detection_ready.connect(
#                 lambda result, cam=camera.name: self.detection_ready.emit(cam, result)
#             )
#
#             self._detectors[camera.name] = detector
#             self._threads[camera.name] = thread
#
#     def start_all(self):
#         """Start all detector threads."""
#         for thread in self._threads.values():
#             if not thread.isRunning():
#                 thread.start()
#
#     def stop_all(self):
#         """Stop all detector threads."""
#         for thread in self._threads.values():
#             thread.quit()
#             thread.wait()
#
#     def submit_frame(self, camera_name: str, frame: np.ndarray, frame_idx: int):
#         """Submit a frame to the appropriate detector."""
#         if camera_name in self._detectors:
#             # This will be processed in the detector's thread
#             self._detectors[camera_name].handle_frame(frame, frame_idx)
#
#     def get_detector(self, camera_name: str) -> Optional[DetectorThread]:
#         """Get a specific detector thread."""
#         return self._detectors.get(camera_name)
#
#     def set_paused(self, paused: bool):
#         """Pause/unpause all detectors."""
#         for detector in self._detectors.values():
#             detector.set_paused(paused)
#
#     @Slot(object)
#     def configure_new_board(self, board: Union[ChessBoard, CharucoBoard]):
#         """Update board on all detectors."""
#         self._board = board
#         for detector in self._detectors.values():
#             detector.configure_new_board(board)
