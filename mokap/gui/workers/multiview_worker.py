"""
- Receives DetectionResults directly from DetectorThreads
- MultiviewCalibrationTool updates CameraRig directly (thread-safe)
- 3D scene computation using CameraRig properties
"""
import logging
from typing import Dict, Optional, Union, List
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from lucida import CameraRig
from lucida.calibration import MultiviewCalibrationTool, CharucoBoard, ChessBoard
from lucida.geometry import transform_points
from mokap.gui.workers import IndexedDetection

logger = logging.getLogger(__name__)


class MultiviewWorker(QObject):
    """
    Worker for multi-camera extrinsic calibration and 3D visualisation.
    
    Responsibilities:
    - Receive detections from DetectorThreads
    - Register detections with MultiviewCalibrationTool
    - Run bundle adjustment
    - Compute 3D scene data for visualisation
    
    The CameraRig is the single source of truth for all camera parameters.
    The MultiviewCalibrationTool updates the rig directly (Lucida's rig is thread-safe)
    """

    # Lifecycle signals
    finished = Signal()
    error = Signal(Exception)
    blocking = Signal(bool)

    # UI refresh signals
    scene_updated = Signal(dict)  # 3D scene data for OpenGL view
    coverage_updated = Signal()  # Sample count changed
    refinement_complete = Signal(bool)  # BA finished (success/failure)

    # Stage management
    stage_changed = Signal(int)

    def __init__(
            self,
            rig: 'CameraRig',
            calibration_board: Union['ChessBoard', 'CharucoBoard'],
            origin_cam: Optional[Union[int, str]] = None,
            min_samples: int = 100,
            max_samples: int = 300,
    ):
        super().__init__()

        self._rig = rig
        self._board = calibration_board
        self._origin_cam = self._rig.get_name(origin_cam if origin_cam is not None else 0)
        self._min_samples = min_samples
        self._max_samples = max_samples

        # Tool is created when entering Extrinsics stage    # TODO: Should be destroyed when leaving extrinsics stage?
        self._tool: Optional[MultiviewCalibrationTool] = None

        # Worker state
        self._paused = False
        self._current_stage = 0

        # Store latest detections for visualization (per camera)
        self._latest_detections: Dict[str, Optional['IndexedDetection']] = {
            cam.name: None for cam in self._rig
        }

        # TODO: Use the rig's internal coord transform methods
        self._T_disp = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ], dtype=np.float32).T

        # Timer for 3D scene updates
        self._scene_timer = QTimer(self)
        self._scene_timer.setInterval(100)
        self._scene_timer.timeout.connect(self._compute_3d_scene)
        self._scene_timer.start()

    # ──────────────────────────────── Internal helpers ────────────────────────────────

    def _create_tool(self):
        logger.info("[Multiview] Creating MultiviewCalibrationTool...")

        self._tool = MultiviewCalibrationTool(
            rig=self._rig,
            calibration_board=self._board,
            origin_cam=self._origin_cam,
            min_samples=self._min_samples,
            max_samples=self._max_samples,
        )

    # ────────────────────────────────  3D scene computation ────────────────────────────────

    def _compute_3d_scene(self):
        """Compute 3D scene data for the OpenGL visualisation."""

        if self._paused:
            return

        frustum_depth = self._board.diagonal * 5.0
        axis_length = self._board.diagonal * 2.5

        with self._rig.locked():
            C = len(self._rig)
            cam_centers = self._rig.centers

            # TODO: Ready mask should be computed differently based on intrinsics or extrinsics stage...
            ready_mask = [cam.extrinsics.is_set for cam in self._rig]

            # Ensure origin camera is always drawn as "ready" since it is implicitly at Identity
            try:
                if self._origin_cam:
                    origin_idx = self._rig.get_index(self._origin_cam)
                    ready_mask[origin_idx] = True
            except KeyError:
                pass

        frustums_3d = np.zeros((C, 5, 3))
        optical_axes_3d = np.zeros((C, 2, 3))

        for i, cam in enumerate(self._rig):
            center = cam_centers[i]

            if not ready_mask[i]:
                frustums_3d[i] = center
                optical_axes_3d[i] = center
                continue

            points_2d = np.vstack([cam.image_corners, cam.c])  # 4 image corners + image centre + principal point
            _, directions = cam.raycast(points_2d)

            # TODO: Use the image centre too

            frustums_3d[i, 0] = center
            frustums_3d[i, 1:] = center + directions[:4] * frustum_depth
            optical_axes_3d[i, 0] = center
            optical_axes_3d[i, 1] = center + directions[4] * axis_length

        # Board position depends on stage
        if self._current_stage == 0:
            board_3d = self._board.object_points
        elif self._tool is not None and self._tool.current_object_pose is not None:
            board_3d = transform_points(self._board.object_points, self._tool.current_object_pose)
        else:
            board_3d = None

        detections_3d = self._get_detections(board_3d) if board_3d is not None else [np.zeros((0, 3))] * C

        scene_data = {
            'ready_mask': ready_mask,
            'cam_centers': self._to_gl(cam_centers),
            'frustums_3d': self._to_gl(frustums_3d),
            'optical_axes_3d': self._to_gl(optical_axes_3d),
            'board_3d': self._to_gl(board_3d),
            'detections_3d': [self._to_gl(d) for d in detections_3d],
        }
        self.scene_updated.emit(scene_data)

    def _to_gl(self, points: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Convert to OpenGL coordinates (rotate 180° around X) using Matrix."""
        if points is None or points.size == 0:
            return points

        return points @ self._T_disp

    # ──────────────────────────────── Handle detections ────────────────────────────────

    def _get_detections(self, board_3d: np.ndarray) -> List[np.ndarray]:
        """Get 3D detection positions when board is at origin."""

        # TODO: Use the tool's reproject method instead??

        detections = []

        for cam in self._rig:
            det = self._latest_detections.get(cam.name)

            if det is not None and det.valid:
                # Detection IDs map directly to board points
                ids = det.detected_ids
                if len(ids) > 0:
                    detections.append(board_3d[ids])
                else:
                    detections.append(np.zeros((0, 3)))
            else:
                detections.append(np.zeros((0, 3)))

        return detections

    @Slot(str, object)
    def on_detection(self, camera_name: str, result: 'IndexedDetection'):
        """
        Receive detection from a DetectorThread.
        """
        if self._paused:
            return

        # Store for visualisation
        self._latest_detections[camera_name] = result if result.valid else None

        # In Extrinsics Stage, register with the calibration tool
        if self._current_stage > 0 and self._tool is not None and result.valid:
            cam_idx = self._rig.get_index(camera_name)

            accepted = self._tool.register_detection(
                cam_idx=cam_idx,
                frame_idx=result.frame_idx,
                detection=result.image_points
            )

            if accepted:
                self.coverage_updated.emit()

        self.finished.emit()

    # ──────────────────────────────── State management ────────────────────────────────

    def set_paused(self, paused: bool):
        self._paused = paused

    @Slot(int)
    def set_stage(self, stage: int):
        """Handle stage transitions."""
        if stage == self._current_stage:
            return

        logger.info(f"[Multiview] Stage transition: {self._current_stage} -> {stage}")

        if stage == 0:
            # Going back to Intrinsics stage: full reset
            self.reset()
            self._tool = None
        else:
            # Entering Extrinsics stage: create the tool
            self._create_tool()

        self._current_stage = stage
        self.stage_changed.emit(stage)

    @Slot()
    def reset(self):
        """Full reset of worker state."""
        logger.debug("[Multiview] Resetting worker...")

        self._tool = None
        self._latest_detections = {cam.name: None for cam in self._rig}
        self._current_stage = 0

        self.coverage_updated.emit()

    @Slot()
    def refresh_from_rig(self):
        """
        Called when the CameraRig has been updated externally.
        """
        self._compute_3d_scene()

    # ────────────────────────────────  Manual controls ────────────────────────────────

    # TODO: Should have clear_samples

    @Slot()
    def trigger_refinement(self):
        """Run bundle adjustment to refine all camera parameters."""
        if self._paused:
            logger.warning("[Multiview] Cannot refine: worker is paused.")
            return

        if self._tool is None:
            logger.warning("[Multiview] Cannot refine: tool not initialized (still in Stage 0?).")
            return

        if self._tool.sample_count < self._min_samples:
            logger.warning(
                f"[Multiview] Cannot refine: not enough samples "
                f"({self._tool.sample_count}/{self._min_samples})."
            )
            return

        logger.info(f"[Multiview] Starting Bundle Adjustment with {self._tool.sample_count} samples...")

        self.blocking.emit(True)
        success = self._tool.refine()
        self.blocking.emit(False)

        if success:
            logger.info("[Multiview] Bundle Adjustment successful.")
            # Just notify the UI to refresh
            self.refinement_complete.emit(True)
        else:
            logger.error("[Multiview] Bundle Adjustment failed.")
            self.refinement_complete.emit(False)

    # ──────────────────────────────── Configuration ────────────────────────────────

    # TODO: Should probably have set_auto_sample and set_auto_compute?

    @Slot(object)
    def configure_new_board(self, board: Union['ChessBoard', 'CharucoBoard']):
        """Handle board parameter changes."""
        logger.debug("[Multiview] Board changed, resetting...")

        self._board = board

        # Reset tool if it exists
        if self._tool is not None:
            self._tool = None
            if self._current_stage > 0:
                self._create_tool()

        self.coverage_updated.emit()

    @Slot(str)
    def set_origin_camera(self, camera_name: str):
        """Change which camera is the world origin."""
        try:
            self._origin_cam = camera_name

            # If the tool exists, let it handle the rig update so it can preserve samples
            if self._tool is not None:
                self._tool.update_origin(camera_name)
            else:
                # Fallback if no tool exists
                self._rig.set_origin(camera_name)
                logger.info(f"[Multiview] Origin camera set to: {camera_name}")

            # Force scene update
            self._compute_3d_scene()

        except KeyError:
            logger.error(f"[Multiview] Unknown camera: {camera_name}")

    # ──────────────────────────────── Properties ────────────────────────────────

    @property
    def name(self):
        return 'multiview'

    @property
    def tool(self):
        return self._tool

    @property
    def current_coverage(self) -> List[float]:
        """Coverage percentage for each camera."""
        if self._tool is not None:
            return self._tool.current_coverage
        return [0.0] * len(self._rig)