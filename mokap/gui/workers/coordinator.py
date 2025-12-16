import logging
from typing import Optional, Union
from PySide6.QtCore import QObject, Signal, Slot
from lucida import CameraRig
from lucida.calibration import ChessBoard, CharucoBoard

logger = logging.getLogger(__name__)


class CalibrationCoordinator(QObject):
    """
    Coordinates calibration stages across all workers.
    """

    # Broadcast signals
    broadcast_stage = Signal(int)
    broadcast_board_changed = Signal(object)  # ChessBoard or CharucoBoard
    broadcast_reset = Signal()
    broadcast_parameters_loaded = Signal()
    broadcast_origin_camera_changed = Signal(str)

    # Signal to request multiview refinement
    request_refinement = Signal()

    def __init__(self, rig: 'CameraRig'):
        super().__init__()
        
        self._rig = rig
        self._current_stage = 0

        self._origin_cam: Optional[str] = rig.metadata.get('origin_camera')

        if not self._origin_cam and len(rig) > 0:
            self._origin_cam = rig[0].name

    @property
    def rig(self) -> 'CameraRig':
        return self._rig

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
            self._origin_cam = camera_name
            logger.info(f"[Coordinator] Origin camera set to: {camera_name}")
            self.broadcast_origin_camera_changed.emit(camera_name)

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
                for cam in loaded_rig:

                    if cam.name not in self._rig:
                        continue

                    self._rig[cam.name] = cam.copy()

                    if not cam.extrinsics.is_set and not cam.name == self._origin_cam:
                        has_all_extrinsics = False

            del loaded_rig

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