import logging
from typing import List, Optional, Union
from PySide6.QtCore import QObject, Signal, Slot
from lucida import CameraRig
from lucida.calibration import ChessBoard, CharucoBoard

logger = logging.getLogger(__name__)


class CalibrationCoordinator(QObject):
    """
    Coordinates calibration stages across all workers.
    
    Responsibilities:
    - Manage calibration stage transitions (Stage 0: Intrinsics, Stage 1: Extrinsics)
    - Broadcast board parameter changes
    - Track which camera is the origin for multiview calibration
    """

    # Broadcast signals
    broadcast_stage = Signal(int)
    broadcast_board_changed = Signal(object)  # ChessBoard or CharucoBoard
    broadcast_reset = Signal()
    broadcast_parameters_loaded = Signal()

    # Signal to request multiview refinement
    request_refinement = Signal()

    def __init__(self, rig: 'CameraRig'):
        super().__init__()
        
        self._rig = rig
        self._current_stage = 0

        self._origin_camera: Optional[str] = rig.metadata.get('origin_camera')

        if not self._origin_camera and len(rig) > 0:
            self._origin_camera = rig[0].name

    @property
    def rig(self) -> 'CameraRig':
        return self._rig

    @property
    def current_stage(self) -> int:
        return self._current_stage

    @property
    def origin_camera(self) -> Optional[str]:
        return self._origin_camera

    @property
    def camera_names(self) -> List[str]:
        return [cam.name for cam in self._rig]

    # ────────────────────────────────  Stage management ────────────────────────────────

    @Slot(int)
    def set_stage(self, stage: int):
        """
        Set the calibration stage.
        
        Stage 0: Intrinsic calibration (monocular)
        Stage 1: Extrinsic calibration (multiview)
        """
        if stage == self._current_stage:
            return

        logger.info(f"[Coordinator] Stage transition: {self._current_stage} -> {stage}")
        
        # Going back to stage 0 requires a full reset
        if stage == 0:
            self.broadcast_reset.emit()

        self._current_stage = stage
        self.broadcast_stage.emit(stage)

    # ──────────────────────────────── Board management ────────────────────────────────

    @Slot(object)
    def handle_board_change(self, new_board: Union['ChessBoard', 'CharucoBoard']):
        """
        Handle calibration board parameter changes.
        
        This triggers a full system reset since all accumulated
        calibration data is invalidated.
        """
        logger.info("[Coordinator] Board parameters changed. Triggering full reset.")
        
        # Force back to stage 0
        self._current_stage = 0
        self.broadcast_stage.emit(0)
        
        # Tell all workers to reset and use new board
        self.broadcast_reset.emit()
        self.broadcast_board_changed.emit(new_board)

    # ──────────────────────────────── Origin camera ────────────────────────────────

    @Slot(str)
    def set_origin_camera(self, camera_name: str):
        """Set which camera serves as the world origin for multiview calibration."""
        try:
            # Verify camera exists
            self._rig.get_index(camera_name)
            self._origin_camera = camera_name
            logger.info(f"[Coordinator] Origin camera set to: {camera_name}")
        except KeyError:
            logger.error(f"[Coordinator] Unknown camera: {camera_name}")

    # ──────────────────────────────── Multiview refinement ────────────────────────────────

    @Slot()
    def trigger_refinement(self):
        """Request the multiview worker to run bundle adjustment."""
        if self._current_stage == 0:
            logger.warning("[Coordinator] Cannot refine in Intrinsics stage.")
            return
            
        logger.info("[Coordinator] Requesting multiview refinement...")
        self.request_refinement.emit()

    # ──────────────────────────────── Calibration I/O ────────────────────────────────

    @Slot(str)
    def load_calibration(self, file_path: str):
        """
        Load calibration from a TOML file and update the CameraRig.
        """
        try:
            loaded_rig = CameraRig.load(file_path)

            has_all_extrinsics = True
            with self._rig.locked():
                for cam in self._rig:

                    if cam.name not in loaded_rig:
                        continue

                    loaded_cam = loaded_rig[cam.name]

                    # Update Intrinsics
                    cam.intrinsics.K = loaded_cam.intrinsics.K
                    cam.intrinsics.D = loaded_cam.intrinsics.D
                    cam.intrinsics.rms = loaded_cam.intrinsics.rms
                    cam.intrinsics.stats = loaded_cam.intrinsics.stats.copy()

                    # Update Extrinsics
                    cam.extrinsics.T = loaded_cam.extrinsics.T

                    if not cam.extrinsics.is_set and not cam.name == self._origin_camera:
                        has_all_extrinsics = False

            logger.info(f"[Coordinator] Loaded calibration from {file_path}")

            # Decide on Stage Switch
            if has_all_extrinsics and len(self._rig) > 1:
                logger.info("[Coordinator] Loaded valid extrinsics. Switching to Extrinsics mode.")
                self.set_stage(1)
            else:
                self.set_stage(0)

            # Notify everyone
            self.broadcast_parameters_loaded.emit()

            # If we are in extrinsics mode, the Multiview worker must reset
            if self._current_stage == 1:
                self.broadcast_reset.emit()

        except Exception as e:
            logger.error(f"[Coordinator] Failed to load calibration: {e}")

    @Slot(str)
    def save_calibration(self, file_path: str):
        """
        Save the current CameraRig configuration to a TOML file.
        """
        try:
            self._rig.save(file_path)
            logger.info(f"[Coordinator] Saved calibration to {file_path}")
        except Exception as e:
            logger.error(f"[Coordinator] Failed to save calibration: {e}")