"""
The worker focuses on:
- Registering detections with the calibration tool
- Managing calibration policy (auto-sample, auto-compute)
- Pose estimation
- Save/Load operations
"""
import logging
from typing import Union, Optional
from PySide6.QtCore import QObject, Signal, Slot
from lucida import CameraModel
from lucida.calibration import MonocularCalibrationTool, CharucoBoard, ChessBoard
from mokap.gui.workers import DetectionResult

logger = logging.getLogger(__name__)


class MonocularWorker(QObject):
    """
    Worker for monocular camera calibration.
    
    Receives detection results from a separate DetectorThread.
    Manages calibration logic and updates CameraModel.
    
    Signals emitted are for UI refresh only, no data payloads.
    """

    # Lifecycle signals
    finished = Signal()              # Emitted when processing is complete
    error = Signal(Exception)        # Emitted on errors
    blocking = Signal(bool)          # Emitted when long computation starts/ends

    # UI refresh signals
    detection_updated = Signal()     # Detection state changed
    coverage_updated = Signal()      # Sample added, coverage changed  
    intrinsics_updated = Signal()    # Intrinsics were recomputed
    pose_updated = Signal()          # Pose estimation updated
    
    # Stage management
    stage_changed = Signal(int)

    def __init__(
            self,
            camera_model: CameraModel,
            calibration_board: Union[ChessBoard, CharucoBoard],
            min_samples: int = 15,
            max_samples: int = 100
    ):
        super().__init__()

        self._camera_model = camera_model
        self._board = calibration_board
        
        # Create the calibration tool
        self._tool = MonocularCalibrationTool(
            camera=self._camera_model,
            calibration_board=self._board,
            min_samples=min_samples,
            max_samples=max_samples
        )

        # Worker state
        self._paused = False
        self._current_stage = 0

        # Store latest detection for UI access
        self._latest_detection: Optional[DetectionResult] = None

        # Policy settings (TODO: make configurable from GUI)
        self._auto_sample = True
        self._auto_compute = True
        self._coverage_threshold = 75.0
        self._area_threshold = 0.1
        
        # Flag to avoid looping over calibration failures
        self._last_calib_failed = False

    @property
    def latest_detection(self) -> Optional[DetectionResult]:
        """Most recent detection result for UI display."""
        return self._latest_detection

    # ──────────────────────────────── Handle detections ────────────────────────────────

    @Slot(object)
    def on_detection(self, result: DetectionResult):
        """
        Handle detection results from DetectorThread.
        This is called when a detection arrives.
        """
        if self._paused:
            self.finished.emit()
            return

        # Store for UI access
        self._latest_detection = result

        if result.valid:
            # Register with calibration tool
            accepted = self._tool.register_detection(
                result.image_points,
                min_area=self._area_threshold if self._auto_sample else -1
            )
            
            self.detection_updated.emit()

            if accepted:
                self._last_calib_failed = False
                self.coverage_updated.emit()

            # Intrinsics calibration
            if self._current_stage == 0 and self._auto_compute and not self._last_calib_failed:
                if self._should_compute_intrinsics():
                    self._do_compute_intrinsics()

            # Pose estimation (all stages)
            if self._tool.estimate_pose():
                self.pose_updated.emit()

        else:
            # Invalid detection: still notify UI to clear overlays
            self.detection_updated.emit()

        self.finished.emit()

    def _should_compute_intrinsics(self) -> bool:
        """Check if we have enough data to attempt calibration."""
        return (
            self._tool.current_coverage >= self._coverage_threshold and
            self._tool.sample_count >= self._tool._min_samples
        )

    def _do_compute_intrinsics(self):
        """Run intrinsics computation with UI blocking."""

        self.blocking.emit(True)
        
        success = self._tool.compute_intrinsics(keep_samples=False)
        
        self.blocking.emit(False)
        
        self._last_calib_failed = not success
        
        if success:
            self.intrinsics_updated.emit()
            self.coverage_updated.emit()  # Coverage was cleared
        else:
            logger.debug(f"[{self.name}] Auto-computation failed. Waiting for new samples.")

    # ────────────────────────────────  Manual controls ────────────────────────────────

    @Slot()
    def add_sample(self):
        """Manually add current detection as a sample."""

        if self._latest_detection is not None and self._latest_detection.valid:
            self._tool.register_detection(
                self._latest_detection.image_points,
                min_area=-1  # Force accept
            )
            self.coverage_updated.emit()

    @Slot()
    def clear_samples(self):
        """Clear all accumulated samples."""
        self._tool.clear_samples()
        self.coverage_updated.emit()

    @Slot()
    def compute_intrinsics(self):
        """Manually trigger intrinsics computation."""
        self._do_compute_intrinsics()

    @Slot()
    def clear_intrinsics(self):
        """Reset intrinsics to initial guess."""
        self._tool.clear_intrinsics()
        self._tool.clear_samples()
        self.intrinsics_updated.emit()
        self.coverage_updated.emit()

    # ────────────────────────────────  Configuration ────────────────────────────────

    @Slot(bool)
    def set_auto_sample(self, enabled: bool):
        self._auto_sample = enabled

    @Slot(bool)
    def set_auto_compute(self, enabled: bool):
        self._auto_compute = enabled

    @Slot(object)
    def configure_new_board(self, board: Union[ChessBoard, CharucoBoard]):
        """Handle board parameter changes: recreate the tool."""
        logger.debug(f"[{self.name}] Board changed, recreating tool.")
        
        self._board = board
        self._tool = MonocularCalibrationTool(
            camera=self._camera_model,
            calibration_board=self._board,
            min_samples=self._tool._min_samples,
            max_samples=self._tool._max_samples
        )
        
        self._latest_detection = None
        self.coverage_updated.emit()
        self.intrinsics_updated.emit()

    # ────────────────────────────────  State management ────────────────────────────────

    def set_paused(self, paused: bool):
        self._paused = paused

    @Slot(int)
    def set_stage(self, stage: int):
        if stage == self._current_stage:
            return
            
        self._current_stage = stage
        self._tool.clear_samples()
        
        logger.debug(f"[{self.name}] Stage changed to {stage}")
        self.stage_changed.emit(stage)
        self.coverage_updated.emit()

    @Slot()
    def reset(self):
        """Full reset of worker state."""
        self._last_calib_failed = False
        self._latest_detection = None
        self._tool.clear_samples()
        self._tool.clear_intrinsics()
        
        self.intrinsics_updated.emit()
        self.coverage_updated.emit()

    # ────────────────────────────────  Save / load ────────────────────────────────

    @Slot(str)
    def load_intrinsics(self, file_path: str):
        """Load intrinsics from a TOML file."""
        try:
            loaded_cam = CameraModel.load(file_path)
            
            with self._camera_model.intrinsics.locked():
                self._camera_model.intrinsics.K = loaded_cam.intrinsics.K
                self._camera_model.intrinsics.D = loaded_cam.intrinsics.D
                self._camera_model.intrinsics.rms = loaded_cam.intrinsics.rms
                self._camera_model.intrinsics.stats = loaded_cam.intrinsics.stats.copy()

            self._auto_sample = False
            self._auto_compute = False
            self.clear_samples()
            
            self.intrinsics_updated.emit()
            
        except Exception as e:
            logger.error(f"[{self.name}] Failed to load intrinsics: {e}")
            self.error.emit(e)

    @Slot(str)
    def save_intrinsics(self, file_path: str):
        """Save current camera intrinsics to a TOML file."""
        try:
            self._camera_model.save(file_path)
        except Exception as e:
            logger.error(f"[{self.name}] Failed to save intrinsics: {e}")
            self.error.emit(e)

    # ────────────────────────────────  Properties ────────────────────────────────

    @property
    def name(self):
        return self._camera_model.name

    @property
    def tool(self):
        return self._tool