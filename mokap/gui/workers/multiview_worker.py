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
from lucida.geometry import transform_points, rotate_points
from mokap.gui.workers import DetectionResult

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
    scene_updated = Signal(dict)    # 3D scene data for OpenGL view
    coverage_updated = Signal()     # Sample count changed
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
        self._origin_cam = origin_cam if origin_cam is not None else 0
        self._min_samples = min_samples
        self._max_samples = max_samples

        # Tool is created when entering Extrinsics stage    # TODO: Should be destroyed when leaving extrinsics stage?
        self._tool: Optional[MultiviewCalibrationTool] = None

        # Worker state
        self._paused = False
        self._current_stage = 0

        # Store latest detections for visualization (per camera)
        self._latest_detections: Dict[str, Optional['DetectionResult']] = {
            cam.name: None for cam in self._rig
        }

        # Precompute board points in homogeneous coords for transforms
        self._object_points_hom = np.hstack([
            self._board.object_points,
            np.ones((self._board.object_points.shape[0], 1))
        ])

        # Timer for 3D scene updates
        self._scene_timer = QTimer(self)
        self._scene_timer.setInterval(100)
        self._scene_timer.timeout.connect(self._compute_3d_scene)
        self._scene_timer.start()

    # ──────────────────────────────── Handle detections ────────────────────────────────

    @Slot(str, object)
    def on_detection(self, camera_name: str, result: 'DetectionResult'):
        """
        Receive detection from a DetectorThread.
        """
        if self._paused:
            return

        # Store for visualization
        self._latest_detections[camera_name] = result if result.valid else None

        # In Stage 1+, register with the calibration tool
        if self._current_stage > 0 and self._tool is not None and result.valid:
            cam_idx = self._rig.get_index(camera_name)
            
            accepted = self._tool.register_detection(
                cam_idx=cam_idx,
                frame_idx=result.frame_idx,
                points2d=result.image_points
            )
            
            if accepted:
                self.coverage_updated.emit()

        self.finished.emit()

    # ────────────────────────────────  Bundle Adjustment ────────────────────────────────

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

    # ────────────────────────────────  3D scene computation ────────────────────────────────

    def _compute_3d_scene(self):
        """
        Compute 3D scene data for the OpenGL visualization.
        """
        if self._paused:
            return

        with self._rig.locked():
            K = self._rig.K.copy()
            T_c2w = self._rig.T_c2w.copy()

            # Check which cameras have valid intrinsics
            ready_mask = np.array([cam.fx > 1.0 for cam in self._rig])

        C = len(self._rig)

        # Camera centers from extrinsics
        cam_centers = T_c2w[:, :3, 3]

        # Compute frustum corners by unprojecting image corners
        frustums_3d = self._compute_frustums(K, T_c2w, ready_mask)
        optical_axes_3d = self._compute_optical_axes(K, T_c2w, cam_centers)

        # Board and detection positions depend on stage
        board_3d = None
        detections_3d = [np.zeros((0, 3))] * C

        if self._current_stage == 0:
            # Intrinsics stage: Board at origin, cameras move around it
            board_3d = self._board.object_points

            # Detections are on the stationary board
            detections_3d = self._get_detections(board_3d)

        else:
            # Extrinsics stage: Cameras fixed, board moves
            if self._tool is not None and self._tool.current_object_pose is not None:
                # Transform board by current pose
                T_o2w = self._tool.current_object_pose
                board_3d = transform_points(self._board.object_points, T_o2w)

                # Detections are on the transformed board
                detections_3d = self._get_detections(board_3d)

        # Apply OpenGL coordinate transform (flip Y and Z)
        scene_data = {
            'ready_mask': ready_mask,
            'cam_centers': self._to_gl(cam_centers),
            'frustums_3d': self._to_gl_batch(frustums_3d),
            'optical_axes_3d': self._to_gl_batch(optical_axes_3d),
            'board_3d': self._to_gl(board_3d),
            'detections_3d': [self._to_gl(d) for d in detections_3d],
        }

        self.scene_updated.emit(scene_data)

    def _compute_frustums(self, K, T_c2w, ready_mask) -> np.ndarray:
        """Compute frustum corner positions for each camera."""
        C = len(self._rig)
        frustum_depth = 200.0  # TODO: compute automatically

        frustums = []
        for i, cam in enumerate(self._rig):
            w, h = cam.image_size

            if not ready_mask[i]:
                # Camera not ready - return degenerate frustum at center
                center = T_c2w[i, :3, 3]
                frustums.append(np.tile(center, (5, 1)))
                continue

            # Image corners
            corners_2d = np.array([
                [0, 0], [w, 0], [w, h], [0, h]
            ], dtype=np.float32)

            # Unproject to rays and extend to frustum_depth
            origins, directions = cam.raycast(corners_2d)
            corners_3d = origins + directions * frustum_depth

            # Frustum = [center, corner0, corner1, corner2, corner3]
            center = T_c2w[i, :3, 3]
            frustum = np.vstack([center[None, :], corners_3d])
            frustums.append(frustum)

        return np.array(frustums)  # (C, 5, 3)

    def _compute_optical_axes(self, K, T_c2w, cam_centers) -> np.ndarray:
        """Compute optical axis lines for each camera."""
        axis_length = 100.0  # TODO: compute automatically

        axes = []
        for i, cam in enumerate(self._rig):
            center = cam_centers[i]

            # Principal point
            w, h = cam.image_size
            pp = np.array([[w / 2, h / 2]], dtype=np.float32)

            # Unproject principal point
            _, direction = cam.raycast(pp)
            endpoint = center + direction[0] * axis_length

            axes.append([center, endpoint])

        return np.array(axes)  # (C, 2, 3)

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

    @staticmethod
    def _to_gl(points: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Convert to OpenGL coordinates (rotate 180° around X)."""
        if points is None or points.size == 0:
            return points
        return rotate_points(points, angle_degrees=180, axis=[1.0, 0.0, 0.0])

    @staticmethod
    def _to_gl_batch(points: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Convert batch of points to OpenGL coordinates."""
        if points is None or points.size == 0:
            return points
        return rotate_points(points, angle_degrees=180, axis=[1.0, 0.0, 0.0])

    # ──────────────────────────────── Stage management ────────────────────────────────

    @Slot(int)
    def set_stage(self, stage: int):
        """Handle stage transitions."""
        if stage == self._current_stage:
            return

        logger.info(f"[Multiview] Stage transition: {self._current_stage} -> {stage}")

        if stage == 0:
            # Going back to Intrinsics stage: full reset    # TODO: Maybe delete it instead?
            self.reset()
        else:
            # Entering Extrinsics stage: create the tool
            self._create_tool()

        self._current_stage = stage
        self.stage_changed.emit(stage)

    def _create_tool(self):
        logger.info("[Multiview] Creating MultiviewCalibrationTool...")

        self._tool = MultiviewCalibrationTool(
            rig=self._rig,
            calibration_board=self._board,
            origin_cam=self._origin_cam,
            min_samples=self._min_samples,
            max_samples=self._max_samples,
        )

    @Slot()
    def reset(self):
        """Full reset of worker state."""
        logger.debug("[Multiview] Resetting worker...")

        self._tool = None
        self._latest_detections = {cam.name: None for cam in self._rig}
        self._current_stage = 0

        self.coverage_updated.emit()

    # ──────────────────────────────── Configuration ────────────────────────────────

    @Slot(object)
    def configure_new_board(self, board: Union['ChessBoard', 'CharucoBoard']):
        """Handle board parameter changes."""
        logger.debug("[Multiview] Board changed, resetting...")

        self._board = board
        self._object_points_hom = np.hstack([
            board.object_points,
            np.ones((board.object_points.shape[0], 1))
        ])

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
            self._origin_cam = self._rig.get_index(camera_name)
            logger.info(f"[Multiview] Origin camera set to: {camera_name}")

            # Recreate tool with new origin if in Extrinsics stage
            if self._tool is not None:
                self._create_tool()

        except KeyError:
            logger.error(f"[Multiview] Unknown camera: {camera_name}")

    def set_paused(self, paused: bool):
        self._paused = paused

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