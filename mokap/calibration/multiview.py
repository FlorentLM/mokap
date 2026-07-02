import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import gc
import psutil

import numpy as np
from mokap.geometry.backend import xp, ArrayLike, set_at

from mokap.calibration.bundle_adjustment import covariance_from_std, run_bundle_adjustment
from mokap.calibration.common import solve_pnp_robust

from mokap.utils.datatypes import DetectionPayload, DistortionModel
from mokap.geometry import (
    project_to_cameras,
    reprojection_errors,
    project_to_cameras_multi,
    quaternion_average,
    average_qtposes,
    compute_bounds,
    flip_transform_180,
    compose_transform_matrix,
    decompose_transform_matrix,
    quaternion_from_vector,
    quaternion_from_matrix,
    vector_from_quaternion,
    invert_transform,
    quaternion_distance,
    transform_points
)

logger = logging.getLogger(__name__)


# TODO: The DetectionPayload dataclass is completely redundant now

@dataclass
class SingleviewDetection:
    """
    Represents a single camera's detection of the calibration object.
    """
    frame_idx: int

    cam_idx: int
    T_o2c: xp.ndarray       # (4, 4)
    points2D: np.ndarray    # (N, 2)
    pointsIDs: np.ndarray   # (N,)


@dataclass
class MultiviewDetection:
    """
    Represents a multi-camera detection of the calibration object.
    """
    frame_idx: int

    cam_indices: xp.ndarray     # (n_C,)
    T_o2c: xp.ndarray           # (n_C, 4, 4)
    points2D: xp.ndarray        # (n_C, N, 2)
    visibility_mask: xp.ndarray # (n_C, N)


@dataclass
class BundleAdjustmentConfig:
    distortion_model: DistortionModel

    # Flags
    fix_intrinsics: bool
    fix_extrinsics: bool
    fix_object_points: bool
    fix_object_poses: bool
    fix_aspect_ratio: bool
    shared_intrinsics: bool

    radial_penalty: float

    # Covariance parameters
    sigma_f: float
    sigma_c: float
    sigma_d: float
    sigma_r: float
    sigma_t: float


class MultiviewCalibrationTool:
    """
    Multi-camera calibration tool.

    This tool processes detections of a calibration object from multiple cameras,
    estimates camera extrinsics, and performs global bundle adjustment to refine
    both intrinsics and extrinsics.

    - Online extrinsics estimation from multi-view detections
    - Temporal pose disambiguation using rotation history
    - Three-stage graduated bundle adjustment
    - Quality control based on reprojection errors
    """

    def __init__(
            self,
            nb_cameras: int,
            images_sizes: ArrayLike,
            origin_cam_idx: int,
            K_init: ArrayLike,
            D_init: ArrayLike,
            object_points: ArrayLike,
            min_detections: int = 100,
            max_detections: int = 100,
            angular_thresh: float = 10.0,  # in degrees
            translational_thresh: float = 10.0,  # in object_points' units
            distortion_model: DistortionModel = 'standard'
    ):
        """
        Initialise the multiview calibration tool.

        Args:
            nb_cameras: Number of cameras in the rig
            images_sizes: Image sizes (height, width) for each camera or single size for all
            origin_cam_idx: Index of the origin camera (fixed at identity transform)
            K_init: Initial camera matrices (3x3) or array of matrices
            D_init: Initial distortion coefficients or array of coefficients
            object_points: Known 3D points on calibration object (N, 3)
            min_detections: Minimum number of frames required for bundle adjustment
            max_detections: Maximum number of frames to keep in buffer
            angular_thresh: Angular threshold for pose consensus (degrees)
            translational_thresh: Translation threshold for pose consensus (same units as object_points)
            distortion_model: Distortion model to use
        """

        self.nb_cameras = nb_cameras
        self.origin_cam_idx = origin_cam_idx
        self._distortion_model = distortion_model
        self._images_sizes_hw = self._validate_image_sizes(images_sizes)
        self._angular_thresh_rad: float = np.deg2rad(angular_thresh)
        self._translational_thresh: float = translational_thresh

        # Initialise 3D object model
        self._object_points = xp.asarray(object_points, dtype=xp.float32)  # in object-local coordinates

        # Intrinsics state
        self._K = self._validate_camera_matrices(K_init)
        self._D = self._validate_distortion_coeffs(D_init)

        # Extrinsics state
        self._has_extrinsics = np.zeros(nb_cameras, dtype=bool)
        self._has_extrinsics[self.origin_cam_idx] = True  # origin camera is ths origin so it has extrinsics immediately

        identity = xp.eye(4, dtype=xp.float32)
        self._camera_poses = xp.repeat(identity[None, ...], nb_cameras, axis=0)   # current estimate of camera poses as T mats, in c2w

        # Object pose tracking
        self._current_object_pose: Optional[xp.ndarray] = None  # latest object pose as a T mat, in o2w
        # We only keep a short history to disambiguate 180-degree flips using temporal continuity
        self._object_poses_stack = deque(maxlen=20)

        # Detection buffers
        self._detection_buffer: List[Dict[int, SingleviewDetection]] = [{} for _ in range(nb_cameras)]
        self._current_frame_indices = np.full(nb_cameras, -1, dtype=int)  # latest detection per camera

        # Bundle adjustment samples buffers
        self._samples = deque(maxlen=max_detections)
        self._min_detections = min_detections

        # Refinement results
        self._is_refined = False

        self._refined_intrinsics = None
        self._refined_cam_poses_c2w = None      # T matrices (C, 4, 4)
        self._refined_object_poses_o2w = None   # T matrices (P, 4, 4)

        self._points2d_final = None
        self._visibility_final = None
        self._volume_of_trust = None

        # Quality control thresholds
        self._max_frame_rms_threshold = 5.0  # in pixels

    def _validate_image_sizes(self, images_sizes_hw: ArrayLike) -> np.ndarray:

        images_sizes_hw = np.asarray(images_sizes_hw)

        if images_sizes_hw.ndim == 2 and images_sizes_hw.shape[0] == self.nb_cameras:
            return images_sizes_hw[:, :2]
        elif images_sizes_hw.ndim == 1 and 2 <= images_sizes_hw.shape[0] <= 3:
            logger.debug('Only one size passed, assuming identical image size for all cameras.')
            return np.asarray([images_sizes_hw[:2]] * self.nb_cameras)
        else:
            raise AttributeError("Can't understand image size.")

    def _validate_camera_matrices(self, K_init: ArrayLike) -> xp.ndarray:

        K_np = np.asarray(K_init)

        if K_np.ndim == 2:
            logger.debug("A single camera matrix was provided. Broadcasting to all cameras.")
            K = xp.asarray([K_np] * self.nb_cameras, dtype=xp.float32)
        else:
            K = xp.asarray(K_np, dtype=xp.float32)

        if K.shape != (self.nb_cameras, 3, 3):
            raise ValueError(f"Shape mismatch for init_cam_matrices. Expected ({self.nb_cameras}, 3, 3), got {K.shape}")
        return K

    def _validate_distortion_coeffs(self, D_init: ArrayLike) -> xp.ndarray:

        D_np = np.asarray(D_init)

        if D_np.ndim == 1:
            logger.debug("A single set of distortion coeffs was provided. Broadcasting to all cameras.")
            D = xp.asarray([D_np] * self.nb_cameras, dtype=xp.float32)
        else:
            D = xp.asarray(D_np, dtype=xp.float32)

        if D.shape[0] != self.nb_cameras:
            raise ValueError(f"Shape mismatch for init_dist_coeffs. Expected ({self.nb_cameras}, D), got {D.shape}")
        return D

    def register(self, cam_idx: int, detection: DetectionPayload):
        """
        Register a new detection from a camera:
        - Validates the detection (enough points, successful PnP)
        - Stores the detection in a frame buffer
        - Triggers frame processing when detections from multiple cameras are available

        Args:
            cam_idx: Index of the camera
            detection: Detection payload containing points and frame number
        """

        if detection.pointsIDs is None or detection.points2D is None:
            return

        if len(detection.pointsIDs) < 4:
            return

        # Estimate object-to-camera pose
        success, T_o2c, errors_dict = solve_pnp_robust(
            points3d=self._object_points[detection.pointsIDs],
            points2d=detection.points2D,
            K=self._K[cam_idx],
            D=self._D[cam_idx]
        )

        if not success:
            return

        self._current_frame_indices[cam_idx] = detection.frame_idx

        # Store detection in frame buffer
        cam_detection = SingleviewDetection(
            frame_idx=detection.frame_idx,
            cam_idx=cam_idx,
            T_o2c=T_o2c,
            points2D=detection.points2D,
            pointsIDs=detection.pointsIDs
        )
        self._detection_buffer[cam_idx][detection.frame_idx] = cam_detection

        # Process any complete frames
        self._flush_frames()

    def _flush_frames(self):
        """Process all stale frames that have detections from multiple cameras."""

        for frame_num in self._find_stale_frames():
            # Collect all cameras that detected this frame
            detections = []
            for cam_idx in range(self.nb_cameras):
                if frame_num in self._detection_buffer[cam_idx]:
                    detection = self._detection_buffer[cam_idx].pop(frame_num)
                    detections.append(detection)

            # Only process frames with detections from at least 2 cameras
            if len(detections) >= 2:
                self._process_frame(detections)

    def _find_stale_frames(self) -> List[int]:
        """
        Find frame numbers that are behind the minimum processed frame across
        cameras that have actually seen something.
        """
        valid_indices = self._current_frame_indices[self._current_frame_indices >= 0]

        # If no camera has seen anything, nothing is stale
        if len(valid_indices) == 0:
            return []

        global_min = int(valid_indices.min())

        # Collect all pending frame numbers
        pending_frames = set()
        for buffer in self._detection_buffer:
            pending_frames.update(buffer.keys())

        # Return frames that are behind the global minimum
        return [f for f in pending_frames if f < global_min]

    def _consolidate_frame_data(self, detections: List[SingleviewDetection]) -> MultiviewDetection:
        """
        Convert list of single view camera detections into multi-view arrays.
        """

        n_C = len(detections)   # number of cameras with a detection in this frame
        N = self._object_points.shape[0]

        frame_indices = np.zeros(n_C, dtype=np.int32)
        cam_indices = np.zeros(n_C, dtype=np.int32)
        T_o2c = np.zeros((n_C, 4, 4), dtype=np.float32)
        points2d = np.zeros((n_C, N, 2), dtype=np.float32)
        visibility_mask = np.zeros((n_C, N), dtype=bool)

        for i, det in enumerate(detections):
            frame_indices[i] = det.frame_idx
            cam_indices[i] = det.cam_idx
            T_o2c[i, :, :] = det.T_o2c
            points2d[i, det.pointsIDs, :] = det.points2D
            visibility_mask[i, det.pointsIDs] = True

        assert np.all(frame_indices == frame_indices[0])

        return MultiviewDetection(
            frame_idx=int(frame_indices[0]),
            cam_indices=cam_indices,
            T_o2c=T_o2c,
            points2D=points2d,
            visibility_mask=visibility_mask
        )

    def _process_frame(self, detections: List[SingleviewDetection]):
        """
        Process a complete frame with detections from multiple cameras.
            - Estimates the object pose in world coordinates
            - Disambiguates 180-degree PnP ambiguities using pose history
            - Validates pose consensus across views
            - Updates camera extrinsics
            - Does quality control based on reprojection error

        Args:
            detections: List of SingleviewDetection objects from different cameras
        """
        if not any(self._has_extrinsics):
            self._current_object_pose = None
            return

        mv_detection = self._consolidate_frame_data(detections)

        # Check which cameras in this detection have known extrinsics
        active_and_known = xp.array([self._has_extrinsics[cam_idx] for cam_idx in mv_detection.cam_indices])

        if not xp.any(active_and_known):
            return

        # Extract object-to-camera transforms
        T_c2w_ank = self._camera_poses[mv_detection.cam_indices[active_and_known]]
        T_o2c_ank = mv_detection.T_o2c[active_and_known]

        # Generates the object-to-world votes
        T_o2w_votes = T_c2w_ank @ T_o2c_ank

        # Disambiguate using pose history if available, and find a consensus
        T_o2w_votes = self._disambiguate_poses(T_o2w_votes, T_c2w_ank, T_o2c_ank)
        T_o2w = self._consensus_poses_strict(T_o2w_votes)   # uses the strict one online estimation
        if T_o2w is None:
            return

        # Calculate new camera extrinsics for the currently active (and known) cameras
        T_c2o_detected = invert_transform(mv_detection.T_o2c)
        T_c2w_detected = T_o2w @ T_c2o_detected

        # Quality control
        if not self._validate_frame_quality(mv_detection, T_o2w, T_c2w_detected):
            # Frame rejected
            self._current_object_pose = None
            return

        # Append new agreed uppon object pose
        self._current_object_pose = T_o2w
        self._object_poses_stack.append(T_o2w)

        # Update extrinsics with new estimate
        for i, cam_idx in enumerate(mv_detection.cam_indices):

            if cam_idx != self.origin_cam_idx:  # never update origin camera
                self._camera_poses = set_at(self._camera_poses, cam_idx, T_c2w_detected[i])
                self._has_extrinsics[cam_idx] = True

        # Store the accepted sample for bundle adjustment
        self._samples.append(mv_detection)

    def _disambiguate_poses(
            self,
            T_o2w_votes: xp.ndarray,
            T_c2w_known: xp.ndarray,
            T_o2c_known: xp.ndarray
    ) -> xp.ndarray:
        """
        Resolve 180-degree PnP ambiguities using pose history.

        PnP can have a 180-degree rotation ambiguity. This method uses the
        recent pose history to determine which solution is more consistent.

        Args:
            T_o2w_votes: Initial object-to-world transforms (N, 4, 4)
            T_c2w_known: Camera-to-world transforms for cameras with known extrinsics
            T_o2c_known: Object-to-camera transforms for those cameras

        Returns:
            Disambiguated object-to-world transforms
        """
        if len(self._object_poses_stack) == 0:
            return T_o2w_votes

        # Compute reference rotation from history
        history_transforms = xp.stack(list(self._object_poses_stack))
        history_quats = quaternion_from_matrix(history_transforms)
        q_ref = quaternion_average(history_quats)

        # Get alternative PnP solutions (180-degree flip)
        T_o2c_alt = flip_transform_180(T_o2c_known)
        T_o2w_votes_alt = T_c2w_known @ T_o2c_alt

        # Compare distances to reference
        q_votes = quaternion_from_matrix(T_o2w_votes)
        q_votes_alt = quaternion_from_matrix(T_o2w_votes_alt)

        dist_original = quaternion_distance(q_votes, q_ref)
        dist_alt = quaternion_distance(q_votes_alt, q_ref)

        # Select closer solution
        use_alt_mask = dist_alt < dist_original
        T_o2w_votes = xp.where(use_alt_mask[:, None, None], T_o2w_votes_alt, T_o2w_votes)

        nb_corrected = xp.sum(use_alt_mask)
        if nb_corrected > 0:
            logger.debug(f"[FLIP_CORRECTED] Corrected {nb_corrected} PnP results using stable reference.")

        return T_o2w_votes

    def _consensus_poses_lenient(self, T_o2w_votes: xp.ndarray) -> xp.ndarray:
        """
        Find consensus object pose from multiple camera views using a more
        lenient average method (Markley's method, no outlier filtering).
        """

        # Decompose transforms for q and t
        r_stack, t_stack = decompose_transform_matrix(T_o2w_votes)
        q_stack = quaternion_from_vector(r_stack)
        q_avg = quaternion_average(q_stack)
        t_avg = xp.median(t_stack, axis=0)

        # Re compose final transform
        return compose_transform_matrix(vector_from_quaternion(q_avg), t_avg)

    def _consensus_poses_strict(self, T_o2w_votes: xp.ndarray) -> Optional[xp.ndarray]:
        """
        Find consensus object pose from multiple camera views using strict thresholds.
        Uses IRLS-style outlier rejection.
        """

        # Decompose transforms for q and t
        r_stack, t_stack = decompose_transform_matrix(T_o2w_votes)
        q_stack = quaternion_from_vector(r_stack)
        qt_stack = xp.concatenate([q_stack, t_stack], axis=1)

        # IRLS-style filtering of the poses with multiple iterations
        q_avg, t_avg, success = average_qtposes(
            qt_stack=qt_stack,
            thresh_radians=self._angular_thresh_rad,
            thresh_distance=self._translational_thresh,
            iters=3
        )

        if not success:
            logger.debug(f"[CONSENSUS_FAIL] Frame rejected. Could not find a consistent"
                         f" object pose among {qt_stack.shape[0]} views.")
            return None

        # Re compose final transform
        return compose_transform_matrix(vector_from_quaternion(q_avg), t_avg)

    def _validate_frame_quality(
            self,
            frame_data: MultiviewDetection,
            T_o2w: xp.ndarray,
            T_c2w_new: xp.ndarray
    ) -> bool:
        """
        Validate frame quality using reprojection error.

        Args:
            frame_data: Vectorized frame data
            T_o2w: Object-to-world transform
            T_c2w_new: New camera-to-world transforms

        Returns:
            True if frame passes quality check, False otherwise
        """
        # Transform object points to world
        world_pts = transform_points(self._object_points, T_o2w)

        # Project to cameras
        T_w2c_new = invert_transform(T_c2w_new)
        K_batch = self._K[frame_data.cam_indices]
        D_batch = self._D[frame_data.cam_indices]

        reproj_pts, reproj_mask = project_to_cameras(
            points3d=world_pts,
            T_w2c=T_w2c_new,
            K=K_batch,
            D=D_batch,
            distortion_model=self._distortion_model
        )

        # Calculate reprojection error
        effective_visibility = frame_data.visibility_mask * reproj_mask

        errors_dict = reprojection_errors(
            points2d_observed=frame_data.points2D,
            points2d_reprojected=reproj_pts,
            visibility_mask=effective_visibility
        )

        frame_rms = errors_dict['rms_euclidean']

        if frame_rms > self._max_frame_rms_threshold:
            logger.debug(f"[QUALITY_REJECT] Frame {frame_data.frame_idx} rejected. High Euclidean RMS: {frame_rms:.2f}px")
            return False

        logger.debug(f"[ACCEPTED] Frame {frame_data.frame_idx} Euclidean RMS: {frame_rms:.2f} px.")
        return True


    def refine(self) -> bool:
        """
        Perform global three-stage bundle adjustment over all collected samples.

        This implements a graduated non-convexity approach:
        - Stage 1: Establish stable geometry with shared intrinsics and no distortion
        - Stage 2: Refine per-camera intrinsics with simple distortion model
        - Stage 3: Final polish with full distortion model

        Handles memory constraints by reducing sample count if MemoryError occurs.

        Returns:
            True if bundle adjustment succeeded, False otherwise
        """
        if not all(self._has_extrinsics):
            logger.error("[BA] Initial extrinsics have not been estimated yet.")
            return False

        P = self.sample_count
        if P < self._min_detections:
            logger.error(f"[BA] Not enough samples. Have {P}, need {self._min_detections}.")
            return False

        logger.debug(f"[BA] Starting 3-Stage Bundle Adjustment with {P} samples.")

        ba_configs = self._create_ba_stage_configs()

        current_P = P
        while current_P >= self._min_detections:
            try:
                logger.info(f"[BA] Attempting Bundle Adjustment with {current_P} samples.")

                self._prepare_ba_data(current_P)
                final_results = self._run_multistage_ba(ba_configs)
                self._store_ba_results(final_results)
                logger.info(f"Bundle adjustment complete using {current_P} samples.")
                return True

            except MemoryError:
                gc.collect()
                logger.warning(f"[BA] Memory error with {current_P} samples. Reducing and retrying.")
                current_P = int(current_P * 0.9)

            except RuntimeError as e:
                logger.error(f"[BA] {e}. Could not converge even with {current_P} samples. Aborting.")
                return False

        logger.error(f"[BA] Failed to complete bundle adjustment.")
        return False

    def _create_ba_stage_configs(self) -> List[BundleAdjustmentConfig]:
        """
        Create configuration for all three stages of bundle adjustment.

        Stage 1: Anchor to datasheet specs with shared intrinsics
        Stage 2: Per-camera intrinsics with simple distortion
        Stage 3: Full refinement with complete distortion model
        """

        return [
            # Stage 1: Global geometry with shared intrinsics, no distortion
            BundleAdjustmentConfig(
                distortion_model='none',
                fix_intrinsics=False,
                fix_extrinsics=False,
                fix_object_points=True,
                fix_object_poses=False,
                fix_aspect_ratio=True,
                shared_intrinsics=True,
                radial_penalty=0.0,
                sigma_f=10.0,  # Allow 10px deviation from datasheet
                sigma_c=0.2,  # Lock principal point to center
                sigma_d=1.0,  # Unused (distortion 'none')
                sigma_r=10.0,  # Loose extrinsics (finding the pose)
                sigma_t=1000.0,  # Loose translation
            ),
            # Stage 2: Per-camera intrinsics with simple distortion
            BundleAdjustmentConfig(
                distortion_model='simple',
                fix_intrinsics=False,
                fix_extrinsics=False,
                fix_object_points=True,
                fix_object_poses=False,
                fix_aspect_ratio=True,
                shared_intrinsics=False,
                radial_penalty=2.0,
                sigma_f=1.0,  # Keep close to global average from Stage 1
                sigma_c=1.0,  # Allow slight PP shift per camera
                sigma_d=0.1,  # Allow small distortion (prevent overfitting)
                sigma_r=1.0,  # Allow extrinsics to adjust
                sigma_t=50.0,
            ),
            # Stage 3: Final polish with full distortion model
            BundleAdjustmentConfig(
                distortion_model=self._distortion_model,
                fix_intrinsics=False,
                fix_extrinsics=False,
                fix_object_points=True,
                fix_object_poses=False,
                fix_aspect_ratio=True,
                shared_intrinsics=False,
                radial_penalty=4.0,
                sigma_f=1.0,
                sigma_c=1.0,
                sigma_d=0.05,  # Tighten distortion further
                sigma_r=0.05,  # Approx 2.8 degrees
                sigma_t=10.0,
            )
        ]

    def _prepare_ba_data(self, P: int) -> None:
        """Prepare data buffers for bundle adjustment."""

        C = self.nb_cameras
        N = self._object_points.shape[0]

        # Get most recent samples
        current_samples: List[MultiviewDetection] = list(self._samples)[-P:]

        # Init buffers
        pts2d_buf = np.zeros((C, P, N, 2), dtype=np.float32)
        vis_buf = np.zeros((C, P, N), dtype=bool)
        T_object_w_buf = []

        for sample_idx, mv_detection in enumerate(current_samples):

            cams_in_sample = mv_detection.cam_indices

            # Fill buffers
            for i, cam_idx in enumerate(cams_in_sample):
                pts2d_buf[cam_idx, sample_idx, :, :] = mv_detection.points2D[i]
                vis_buf[cam_idx, sample_idx, :] = mv_detection.visibility_mask[i]

            # Estimate initial object poses for BA initialisation
            T_o2c = mv_detection.T_o2c
            T_c2w = self._camera_poses[cams_in_sample]
            T_o2w_votes = T_c2w @ T_o2c

            T_object_w_buf.append(self._consensus_poses_lenient(T_o2w_votes))

        # Store as xp arrays
        self._ba_points2d = xp.asarray(pts2d_buf)
        self._ba_visibility = xp.asarray(vis_buf)
        self._ba_object_poses = xp.stack(T_object_w_buf)
        self._ba_num_samples = P


    def _run_multistage_ba(self, configs: List[BundleAdjustmentConfig]) -> Dict:
        """Run three-stage bundle adjustment."""

        current_K = self._K
        current_D = self._D
        current_camera_poses = self._camera_poses
        current_object_poses = self._ba_object_poses

        results = None
        stage_names = ["Anchoring to specs", "Per-camera refinement", "Final polish"]

        for s, config in enumerate(configs):

            logger.debug(f"[BA] >>> STAGE {s + 1}: {stage_names[s]} ({self._ba_num_samples} frames)")

            # Create covariance matrices for this stage
            cov_extr, cov_intr = covariance_from_std(
                nb_cameras=self.nb_cameras,
                distortion_model=config.distortion_model,
                fix_aspect_ratio=config.fix_aspect_ratio,
                shared_intrinsics=config.shared_intrinsics,
                sigma_f=config.sigma_f,
                sigma_c=config.sigma_c,
                sigma_d=config.sigma_d,
                sigma_r=config.sigma_r,
                sigma_t=config.sigma_t
            )

            # Run bundle adjustment for this stage
            success, results = run_bundle_adjustment(
                K=current_K,
                D=current_D,
                cameras_poses=current_camera_poses,
                images_sizes_hw=self._images_sizes_hw,
                points2d_observed=self._ba_points2d,
                visibility_mask=self._ba_visibility,
                object_points=self._object_points,
                object_poses=current_object_poses,
                origin_cam_idx=self.origin_cam_idx,
                distortion_model=config.distortion_model,
                fix_intrinsics=config.fix_intrinsics,
                fix_extrinsics=config.fix_extrinsics,
                fix_object_points=config.fix_object_points,
                fix_object_poses=config.fix_object_poses,
                fix_aspect_ratio=config.fix_aspect_ratio,
                shared_intrinsics=config.shared_intrinsics,
                covariance_intrinsics=cov_intr,
                covariance_extrinsics=cov_extr,
                radial_penalty=config.radial_penalty,
                stage=s+1
            )

            if not success:
                raise RuntimeError(f"BA Stage {s} failed.")

            # Use results as input for next stage
            current_K = results['K_opt']
            current_D = results['D_opt']
            current_camera_poses = results['camera_poses_opt']
            current_object_poses = results['object_poses_opt']

        return results

    def _store_ba_results(self, results: Dict) -> None:

        self._refined_intrinsics = (results['K_opt'], results['D_opt'])
        self._refined_cam_poses_c2w = results['camera_poses_opt']  # (C, 4, 4)
        self._refined_object_poses_o2w = results['object_poses_opt']   # (P, 4, 4)

        self._points2d_final = self._ba_points2d
        self._visibility_final = self._ba_visibility

        self._is_refined = True
        self.volume_of_trust()
        self._samples.clear()

        # Clean up BA buffers
        self._ba_points2d = None
        self._ba_visibility = None
        self._ba_object_poses = None

    def volume_of_trust(
            self,
            threshold: float = 1.0,
            iqr_factor: float = 1.5
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Calculate the "volume of trust" - the 3D region with reliable reprojection.

        This computes the bounding box of points with low reprojection error,
        indicating the spatial region where the calibration is most accurate.

        Args:
            threshold: Error threshold in pixels
            iqr_factor: IQR factor for outlier detection

        Returns:
            Dictionary with 'x', 'y', 'z' keys containing (min, max) tuples
        """

        if not self._is_refined:
            return None

        # Transform all object poses to world coordinates
        object_points_world = transform_points(self._object_points, self._refined_object_poses_o2w)

        # Invert all camera poses to world-to-camera
        T_cam_w2c = invert_transform(self._refined_cam_poses_c2w)  # (C, 4, 4)

        # Project all 3D points into all cameras
        reprojected_pts, valid_depth_mask = project_to_cameras_multi(
            object_points_world,
            T_cam_w2c,
            *self._refined_intrinsics,
            distortion_model=self._distortion_model
        )

        # Get reprojection errors
        effective_visibility = self._visibility_final * valid_depth_mask

        errors_dict = reprojection_errors(
            self._points2d_final,
            reprojected_pts,
            effective_visibility,
            per_point_errors=True
        )
        points_errors = errors_dict['mre_per_point']

        # Compute reliable bounding box
        volume = compute_bounds(
            object_points_world,
            points_errors,
            error_threshold=threshold,
            iqr_factor=iqr_factor
        )
        volume = {k: (float(v[0]), float(v[1])) for k, v in volume.items()}
        if volume:
            print("[ Volume of Trust ]")
            print(f"X range: {volume['x'][0]:.2f} to {volume['x'][1]:.2f} mm")
            print(f"Y range: {volume['y'][0]:.2f} to {volume['y'][1]:.2f} mm")
            print(f"Z range: {volume['z'][0]:.2f} to {volume['z'][1]:.2f} mm")

            self._volume_of_trust = volume

        return self._volume_of_trust

    @property
    def sample_count(self) -> int:
        """Number of samples currently stored for bundle adjustment."""
        return len(self._samples)

    @property
    def intrinsics(self) -> Tuple[xp.ndarray, xp.ndarray]:
        if self._is_refined:
            return self._refined_intrinsics
        else:
            return self._K, self._D

    @property
    def camera_poses(self) -> Optional[xp.ndarray]:
        if self._is_refined:
            return self._refined_cam_poses_c2w
        elif all(self._has_extrinsics):
            return self._camera_poses
        else:
            return None

    @property
    def object_poses(self) -> Optional[xp.ndarray]:
        """
        Get object poses.
        Returns refined poses after BA, or the current pose history otherwise.
        """
        if self._is_refined:
            return self._refined_object_poses_o2w
        elif len(self._object_poses_stack) > 0:
            return xp.stack(list(self._object_poses_stack))
        else:
            return None

    @property
    def current_object_pose(self) -> Optional[xp.ndarray]:
        """Most recent object pose in world coordinates."""
        return self._current_object_pose
