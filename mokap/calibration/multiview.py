import logging
from collections import deque
from typing import Tuple, Dict, Optional, List, Any
import numpy as np
import gc
import psutil
from jax import numpy as jnp
from jax.typing import ArrayLike
from mokap.calibration import bundle_adjustment
from mokap.calibration.common import solve_pnp_robust
from mokap.utils.datatypes import DetectionPayload
from mokap.utils.geometry.projective import (project_to_multiple_cameras, reprojection_errors,
                                             project_multiple_to_multiple)
from mokap.utils.geometry.fitting import (quaternion_average, filter_rt_samples, reliability_bounds_3d_iqr)
from mokap.utils.geometry.transforms import (extrinsics_matrix, extmat_to_rtvecs,
                                             axisangle_to_quaternion_batched, quaternion_to_axisangle,
                                             invert_extrinsics_matrix, invert_rtvecs)

logger = logging.getLogger(__name__)


class MultiviewCalibrationTool:
    def __init__(self,
                 nb_cameras:            int,
                 images_sizes_wh:       ArrayLike,
                 origin_idx:            int,
                 init_cam_matrices:     ArrayLike,
                 init_dist_coeffs:      ArrayLike,
                 object_points:         ArrayLike,
                 min_detections:        int = 100,
                 max_detections:        int = 100,
                 angular_thresh:        float = 10.0,   # in degrees
                 translational_thresh:  float = 10.0,   # in object_points' units
    ):

        self.nb_cameras = nb_cameras
        self.origin_idx = origin_idx

        images_sizes_wh = np.asarray(images_sizes_wh)
        if images_sizes_wh.ndim == 2 and images_sizes_wh.shape[0] == self.nb_cameras:
            self._images_sizes_wh = images_sizes_wh[:, :2]
        elif images_sizes_wh.ndim == 1 and 2 <= images_sizes_wh.shape[0] <= 3:
            logger.debug('Only one size passed, assuming identical image size for all cameras.')
            self._images_sizes_wh = np.asarray([images_sizes_wh[:2]] * self.nb_cameras)
        else:
            raise AttributeError("Can't understand image size.")

        self._angular_thresh_rad: float = np.deg2rad(angular_thresh)
        self._translational_thresh: float = translational_thresh

        # Known 3D board model points (N, 3)
        self._object_points = jnp.asarray(object_points, dtype=jnp.float32)
        self._board_pts_hom = jnp.hstack([
            self._object_points, jnp.ones((self._object_points.shape[0], 1), dtype=jnp.float32)
        ])

        # buffers for incoming frames
        self._detection_buffer = [dict() for _ in range(nb_cameras)]
        self._last_frame = np.full(nb_cameras, -1, dtype=int)

        # State for extrinsics (camera-to-world)
        self._has_extrinsics = np.zeros(nb_cameras, dtype=bool)
        self._rvecs_c2w: jnp.ndarray = jnp.zeros((nb_cameras, 3), dtype=jnp.float32)
        self._tvecs_c2w: jnp.ndarray = jnp.zeros((nb_cameras, 3), dtype=jnp.float32)

        # State for board pose (world)
        self._latest_board_pose_w: Optional[jnp.ndarray] = None

        # intrinsics state

        init_cam_matrices_np = np.asarray(init_cam_matrices)
        if init_cam_matrices_np.ndim == 2:
            logger.debug("A single camera matrix was provided. Broadcasting to all cameras.")
            self._cam_matrices = jnp.asarray([init_cam_matrices_np] * self.nb_cameras, dtype=jnp.float32)
        else:
            self._cam_matrices = jnp.asarray(init_cam_matrices_np, dtype=jnp.float32)

        init_dist_coeffs_np = np.asarray(init_dist_coeffs)
        if init_dist_coeffs_np.ndim == 1:
            logger.debug("A single set of distortion coeffs was provided. Broadcasting to all cameras.")
            self._dist_coeffs = jnp.asarray([init_dist_coeffs_np] * self.nb_cameras, dtype=jnp.float32)
        else:
            self._dist_coeffs = jnp.asarray(init_dist_coeffs_np, dtype=jnp.float32)

        if self._cam_matrices.shape != (self.nb_cameras, 3, 3):
            raise ValueError(
                f"Shape mismatch for init_cam_matrices. Expected ({self.nb_cameras}, 3, 3), got {self._cam_matrices.shape}")
        if self._dist_coeffs.shape[0] != self.nb_cameras:
            raise ValueError(
                f"Shape mismatch for init_dist_coeffs. Expected ({self.nb_cameras}, D), got {self._dist_coeffs.shape}")

        # triangulation & BA buffers
        self.ba_samples = deque(maxlen=max_detections)
        self._min_detections = min_detections

        # bs results
        self._refined = False
        self._refined_intrinsics = None
        self._refined_extrinsics = None
        self._refined_board_poses = None
        self._points2d = None
        self._visibility_mask = None
        self._volume_of_trust = None

    def _find_stale_frames(self):
        global_min = int(self._last_frame.min())

        pending = set()

        for buf in self._detection_buffer:
            pending.update(buf.keys())

        stale = [f for f in pending if f < global_min]
        return stale

    def _flush_frames(self):
        for f in self._find_stale_frames():
            cams = [c for c in range(self.nb_cameras) if f in self._detection_buffer[c]]
            if len(cams) < 2:
                for c in cams:
                    self._detection_buffer[c].pop(f, None)
                continue

            entries = [(c, *self._detection_buffer[c].pop(f)) for c in cams]
            self._process_frame(entries)

    def _gather_frame_data(self, entries):
        """ Gathers and pads frame data into JAX arrays for vectorized processing """

        C = len(entries)
        N = self._object_points.shape[0]

        cam_indices = np.array([c for c, _, _, _ in entries], dtype=np.int32)

        gt_points_padded_np = np.zeros((C, N, 2), dtype=np.float32)
        visibility_mask_np = np.zeros((C, N), dtype=bool)

        for i, (_, _, points2d, pointsids) in enumerate(entries):
            gt_points_padded_np[i, pointsids, :] = points2d
            visibility_mask_np[i, pointsids] = True

        return jnp.asarray(cam_indices), jnp.asarray(gt_points_padded_np), jnp.asarray(visibility_mask_np)

    def _process_frame(self, entries: List[Tuple[int, Any, Any, Any]]):

        if not any(self._has_extrinsics):
            # Reset latest board pose if no extrinsics are known
            self._latest_board_pose_w = None
            return

        cam_indices, gt_points_padded, visibility_mask = self._gather_frame_data(entries)

        # Filter entries based on which cameras have known extrinsics
        known_mask = jnp.array([self._has_extrinsics[c] for c in cam_indices])
        if not jnp.any(known_mask):
            return

        E_b2c_all = jnp.stack([entry[1] for entry in entries])

        # Estimate board-to-world pose (E_b2w) by averaging poses from known cameras
        E_c2w_known = extrinsics_matrix(self._rvecs_c2w[cam_indices[known_mask]],
                                        self._tvecs_c2w[cam_indices[known_mask]])
        E_b2c_known = E_b2c_all[known_mask]
        E_b2w_votes = E_c2w_known @ E_b2c_known

        r_stack, t_stack = extmat_to_rtvecs(E_b2w_votes)
        q_stack = axisangle_to_quaternion_batched(r_stack)
        rt_stack = jnp.concatenate([q_stack, t_stack], axis=1)

        q_avg, t_avg, success = filter_rt_samples(
            rt_stack=rt_stack,
            ang_thresh=self._angular_thresh_rad,
            trans_thresh=self._translational_thresh
        )

        if not success:
            logger.debug(f"[CONSENSUS_FAIL] Frame rejected. Could not find a consistent board pose among {rt_stack.shape[0]} views.")
            # (this happens for instance if the initial camera extrinsics are very inaccurate)

            # Reset latest board pose on failure
            self._latest_board_pose_w = None
            return

        E_b2w = extrinsics_matrix(quaternion_to_axisangle(q_avg), t_avg)

        # Store the successful board pose
        self._latest_board_pose_w = E_b2w

        # Calculate the potential new camera-to-world poses for ALL cameras in this frame
        E_c2b_all = invert_extrinsics_matrix(E_b2c_all)
        E_c2w_new = E_b2w @ E_c2b_all
        r_c2w_new, t_c2w_new = extmat_to_rtvecs(E_c2w_new)

        # --- Quality Control using the new poses ---
        world_pts = (E_b2w @ self._board_pts_hom.T).T[:, :3]

        # Get world-to-camera transforms from the NEWLY CALCULATED camera poses
        r_w2c_new, t_w2c_new = invert_rtvecs(r_c2w_new, t_c2w_new)
        K_batch = self._cam_matrices[cam_indices]
        D_batch = self._dist_coeffs[cam_indices]

        reproj_pts = project_to_multiple_cameras(
            world_pts,
            r_w2c_new,
            t_w2c_new,
            K_batch,
            D_batch,
            distortion_model='full'
        )

        errors_dict = reprojection_errors(gt_points_padded, reproj_pts, visibility_mask)
        mean_frame_error = errors_dict['rms']

        FRAME_ERROR_THRESHOLD = 5.0
        if mean_frame_error > FRAME_ERROR_THRESHOLD:
            logger.debug(f"[QUALITY_REJECT] Frame rejected. High reproj error: {mean_frame_error:.2f}px")

            # Reset latest board pose on quality rejection
            self._latest_board_pose_w = None
            return

        logger.debug(f"[ACCEPTED] Frame mean: {mean_frame_error:.2f} px.")

        # --- Update camera extrinsics and buffer data ---
        # if the quality check passed we store the new poses to the class state
        for i, cam_idx in enumerate(cam_indices):
            # We update all cameras except the origin
            if cam_idx != self.origin_idx:
                self._rvecs_c2w = self._rvecs_c2w.at[cam_idx].set(r_c2w_new[i])
                self._tvecs_c2w = self._tvecs_c2w.at[cam_idx].set(t_c2w_new[i])

            # of course we still mark all the cameras including the origin as having extrinsics
            self._has_extrinsics[cam_idx] = True

        # Buffer data for final BA and intrinsics refinement
        self.ba_samples.append(entries)

    def register(self, cam_idx: int, detection: DetectionPayload):

        if detection.pointsIDs is None or detection.points2D is None:
            return

        if len(detection.pointsIDs) < 4:
            return

        # Reestimate the board-to-camera pose and validate it
        success, rvec, tvec, pose_errors = solve_pnp_robust(
            object_points=np.asarray(self._object_points[detection.pointsIDs]),
            image_points=detection.points2D,
            camera_matrix=np.asarray(self._cam_matrices[cam_idx]),
            dist_coeffs=np.asarray(self._dist_coeffs[cam_idx])
        )

        # if PnP fails, return
        if not success:
            return

        #--- From here on, rvec and tvec should be sane ---

        f = detection.frame
        self._last_frame[cam_idx] = f

        E_b2c = extrinsics_matrix(jnp.asarray(rvec), jnp.asarray(tvec))
        self._detection_buffer[cam_idx][f] = (E_b2c, detection.points2D, detection.pointsIDs)

        if cam_idx == self.origin_idx and not self._has_extrinsics[cam_idx]:
            self._has_extrinsics[cam_idx] = True
            # The pose is already (0, 0, 0) we can continue

        self._flush_frames()

    def refine_all(self) -> bool:
        """
        Performs a global, three-stage bundle adjustment (BA) over all collected samples
        (Sort of graduated non-convexity process)

        - Stage 1: Solves for a stable global geometry with shared intrinsics and no distortion
        - Stage 2: Refines per-camera intrinsics (still no distortion)
        - Stage 3: Performs a full refinement with all parameters (including distortion)
        """

        if not all(self._has_extrinsics):
            logger.error("[BA] Initial extrinsics have not been estimated yet.")
            return False

        P = self.ba_sample_count
        if P < self._min_detections:
            logger.error(f"[BA] Not enough samples for bundle adjustment. Have {P}, need {self._min_detections}.")
            return False

        logger.debug(f"[BA] Starting 3-Stage Bundle Adjustment with {P} samples.")

        C = self.nb_cameras
        N = self._object_points.shape[0]

        # Little safeguard to avoid filling up the RAM because of the jacobian
        # (it grows quadratically with the nb of samples)

        current_P = self.ba_sample_count
        ba_succeeded = False
        final_results = None

        while current_P >= self._min_detections:
            try:
                logger.info(f"[BA] Attempting Bundle Adjustment with {current_P} samples.")

                current_samples = list(self.ba_samples)[-current_P:]

                # --------Prepare the data-----------

                pts2d_buf = np.zeros((C, current_P, N, 2), dtype=np.float32)
                vis_buf = np.zeros((C, current_P, N), dtype=bool)

                for p_idx, entries in enumerate(current_samples):
                    for cam_idx, _, pts2D, ids in entries:
                        pts2d_buf[cam_idx, p_idx, ids, :] = pts2D
                        vis_buf[cam_idx, p_idx, ids] = True

                # Initial guess for board poses (from online estimation)
                r_board_w_list, t_board_w_list = [], []
                E_c2w_all = extrinsics_matrix(self._rvecs_c2w, self._tvecs_c2w)

                for p_idx, entries in enumerate(current_samples):

                    cam_indices_in_frame = jnp.array([c for c, _, _, _ in entries])

                    E_b2c_in_frame = jnp.stack([E_b2c for _, E_b2c, _, _ in entries])
                    E_c2w_in_frame = E_c2w_all[cam_indices_in_frame]

                    E_b2w_votes = E_c2w_in_frame @ E_b2c_in_frame

                    r_stack, t_stack = extmat_to_rtvecs(E_b2w_votes)
                    q_stack = axisangle_to_quaternion_batched(r_stack)

                    #  here we use the simple, lenient average for BA initialization
                    #  (Because the spread of this cluster is a direct result of the accumulated errors
                    #  during online camera pose estimates - which are unavoidable!!
                    #  The hardcore filter used online would likely jusyt eliminate everyone here)
                    r_board_w_list.append(quaternion_to_axisangle(quaternion_average(q_stack)))
                    t_board_w_list.append(jnp.median(t_stack, axis=0))

                # Start with the online estimates
                cam_r_online = self._rvecs_c2w
                cam_t_online = self._tvecs_c2w
                K_online = self._cam_matrices
                D_online = self._dist_coeffs
                board_r_online = jnp.stack(r_board_w_list)
                board_t_online = jnp.stack(t_board_w_list)

                pts2d_buf = jnp.asarray(pts2d_buf)
                vis_buf = jnp.asarray(vis_buf)

                self._points2d, self._visibility_mask = pts2d_buf, vis_buf  # store points for this run

                # STAGE 1: Ideal pinhole world (shared intrinsics, no distortion)   (wellllll maybe better with simple dist)
                # ---------------------------------------------------------------
                logger.debug(f"[BA] >>> STAGE 1: Consolidating cameras position with {current_P} frames...")
                success_s1, results_s1 = bundle_adjustment.run_bundle_adjustment(

                    K_online, D_online, cam_r_online, cam_t_online, board_r_online, board_t_online,

                    pts2d_buf, vis_buf,

                    self._object_points,
                    self._images_sizes_wh,

                    max_frames=current_P,

                    shared_intrinsics=True,
                    fix_aspect_ratio=True,
                    distortion_model='none',        # Fixes distortion params to zero
                    # distortion_model='simple',

                    fix_focal_principal=False,
                    fix_distortion=True,
                    fix_extrinsics=False,
                    fix_board_poses=False,

                    radial_penalty=0.0      # for fisrst stage we want to consider all points
                )
                if not success_s1:
                    raise RuntimeError("BA Stage 1 failed.")

                # STAGE 2: Per-camera pinhole world (shared intrinsics, simple distortion)
                # ------------------------------------------------------------------------
                logger.debug(f"[BA] >>> STAGE 2: Consolidating per-camera intrinsics with {current_P} frames...")

                K_s2_init, D_s2_init = results_s1['K_opt'], results_s1['D_opt']
                cam_r_s2_init, cam_t_s2_init = results_s1['cam_r_opt'], results_s1['cam_t_opt']
                board_r_s2_init, board_t_s2_init = results_s1['board_r_opt'], results_s1['board_t_opt']

                success_s2, results_s2 = bundle_adjustment.run_bundle_adjustment(

                    K_s2_init, D_s2_init, cam_r_s2_init, cam_t_s2_init, board_r_s2_init, board_t_s2_init,

                    pts2d_buf, vis_buf,

                    self._object_points,
                    self._images_sizes_wh,

                    max_frames=current_P,

                    shared_intrinsics=False,    # Now we optimize per-camera
                    fix_aspect_ratio=False,     # we allow fx and fy to differ
                    distortion_model='simple',
                    fix_focal_principal=False,
                    fix_distortion=True,
                    fix_extrinsics=False,
                    fix_board_poses=False,

                    radial_penalty=2.0 # for second stage we want to start penalising points too far from the working volume
                )
                if not success_s2:
                    raise RuntimeError("BA Stage 2 failed.")


                # STAGE 3: Real world (Full extrinsics + intrinsics refinement with distortion)
                # -----------------------------------------------------------------------------
                logger.debug(f"[BA] >>> STAGE 3: Full refinement with {current_P} frames...")

                K_s3_init, D_s3_init = results_s2['K_opt'], results_s2['D_opt']
                cam_r_s3_init, cam_t_s3_init = results_s2['cam_r_opt'], results_s2['cam_t_opt']
                board_r_s3_init, board_t_s3_init = results_s2['board_r_opt'], results_s2['board_t_opt']

                success_s3, final_results_attempt = bundle_adjustment.run_bundle_adjustment(

                    K_s3_init, D_s3_init, cam_r_s3_init, cam_t_s3_init, board_r_s3_init, board_t_s3_init,

                    pts2d_buf, vis_buf,

                    self._object_points,
                    self._images_sizes_wh,

                    max_frames=current_P,

                    shared_intrinsics=False,
                    fix_aspect_ratio=False,
                    distortion_model='standard',  # Use the 5-parameter model
                    # distortion_model='full',  # Use the 8-parameter model

                    fix_focal_principal=False,
                    fix_distortion=False,
                    fix_extrinsics=False,
                    fix_board_poses=False,

                    radial_penalty=4.0   # now we kinda want to ignore the points far from the working volume
                )
                if not success_s3:
                    raise RuntimeError("BA Stage 3 failed.")

                # If we reach here, all stages were successful
                ba_succeeded = True
                final_results = final_results_attempt

                break

            except MemoryError:
                gc.collect()  # Force garbage collection
                mem = psutil.virtual_memory()
                logger.warning(
                    f"[BA] Memory error encountered with {current_P} samples. "
                    f"RAM usage: {mem.percent}% ({mem.used / 1e9:.2f}/{mem.total / 1e9:.2f} GB). "
                    f"Reducing sample count and retrying."
                )

                # Reduce sample count by 10% for the next attempt
                current_P = int(current_P * 0.9)
                continue

            except RuntimeError as e:
                logger.error(f"[BA] {e}. Could not converge even with {current_P} samples. Aborting.")
                return False

        if ba_succeeded and final_results is not None:
            logger.info(f"Bundle adjustment complete using {current_P} samples. Storing refined parameters.")

            # store the globally optimized results
            final_K = final_results['K_opt']
            final_D = final_results['D_opt']
            final_cam_r = final_results['cam_r_opt']
            final_cam_t = final_results['cam_t_opt']
            final_board_r = final_results['board_r_opt']
            final_board_t = final_results['board_t_opt']

            self._refined_intrinsics = (final_K, final_D)
            self._refined_extrinsics = (final_cam_r, final_cam_t)
            self._refined_board_poses = (final_board_r, final_board_t)

            self._refined = True
            self.volume_of_trust()
            self.ba_samples.clear()
            return True
        else:
            logger.error(f"[BA] Failed to complete bundle adjustment. "
                         f"Minimum sample requirement is {self._min_detections}, but failed even after reducing to {current_P}.")
            return False

    @property
    def intrinsics(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._cam_matrices, self._dist_coeffs

    @property
    def extrinsics(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # TODO: mask these using self._has_extrinsics to only return the non-zero ones (well and the origin which is 0)
        return self._rvecs_c2w, self._tvecs_c2w

    @property
    def is_estimated(self) -> bool:
        return all(self._has_extrinsics)

    @property
    def current_board_pose(self) -> Optional[jnp.ndarray]:
        return self._latest_board_pose_w

    @property
    def refined_intrinsics(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._refined_intrinsics

    @property
    def refined_extrinsics(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._refined_extrinsics

    @property
    def refined_board_poses(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._refined_board_poses

    @property
    def image_points(self):
        return self._points2d, self._visibility_mask

    def volume_of_trust(self,
            threshold:  float = 1.0,
            iqr_factor: float = 1.5
    ) -> Optional[Dict[str, Tuple[float, float]]]:

        if self._refined:

            # calculate the 3D world coordinates of all point instances using the refined poses
            E_b2w_all_opt = extrinsics_matrix(*self._refined_board_poses)
            world_pts_all_instances = jnp.einsum('pij,nj->pni', E_b2w_all_opt, self._board_pts_hom)[:, :, :3]

            # Reprojection and error calculation

            # Get all necessary parameters for projection
            observed_pts2d, visibility_mask = self.image_points

            # Invert camera-to-world poses to get world-to-camera poses for projection
            r_w2c, t_w2c = invert_rtvecs(*self._refined_extrinsics)

            # Project all 3D points into all cameras
            reprojected_pts = project_multiple_to_multiple(
                world_pts_all_instances,
                r_w2c, t_w2c,
                *self._refined_intrinsics,
                distortion_model='full'
            )

            # Compute the raw, per-point Euclidean distance errors
            raw_errors = jnp.linalg.norm(observed_pts2d - reprojected_pts, axis=-1)
            all_errors = jnp.where(visibility_mask, raw_errors, jnp.nan)    # And mark non-observed points as nan

            # And compute the reliable bounding box using the world points and their errors
            volume_of_trust = reliability_bounds_3d_iqr(
                world_pts_all_instances,
                all_errors,
                error_threshold_px=threshold,
                iqr_factor=iqr_factor
            )

            # Convert back to floats to save
            volume_of_trust = {k: (float(v[0]), float(v[1])) for k, v in volume_of_trust.items()}

            if volume_of_trust:
                print("--- Volume of Trust ---")
                print(f"X range: {volume_of_trust['x'][0]:.2f} to {volume_of_trust['x'][1]:.2f} mm")
                print(f"Y range: {volume_of_trust['y'][0]:.2f} to {volume_of_trust['y'][1]:.2f} mm")
                print(f"Z range: {volume_of_trust['z'][0]:.2f} to {volume_of_trust['z'][1]:.2f} mm")

                self._volume_of_trust = volume_of_trust

            return self._volume_of_trust

    @property
    def ba_sample_count(self) -> int:
        return len(self.ba_samples)

    @property
    def is_refined(self) -> bool:
        return self._refined