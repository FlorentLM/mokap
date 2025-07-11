import logging
from collections import deque
from typing import Tuple, Dict, Optional, List, Any

import jax
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
from mokap.utils.geometry.fitting import (quaternion_average, filter_rt_samples, reliability_bounds_3d_iqr,
                                          generate_ambiguous_pose, generate_multiple_ambiguous_poses)
from mokap.utils.geometry.transforms import (extrinsics_matrix, extmat_to_rtvecs,
                                             axisangle_to_quaternion_batched, quaternion_to_axisangle,
                                             invert_extrinsics_matrix, invert_rtvecs, quaternions_angular_distance,
                                             axisangle_to_quaternion)

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
        self._board_pose_history = deque(maxlen=10)

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
            self._latest_board_pose_w = None
            return

        cam_indices, gt_points_padded, visibility_mask = self._gather_frame_data(entries)

        known_mask = jnp.array([self._has_extrinsics[c] for c in cam_indices])
        if not jnp.any(known_mask):
            return

        # --- Initial Board Pose Estimation ---
        E_b2c_all = jnp.stack([entry[1] for entry in entries])
        E_c2w_known = extrinsics_matrix(self._rvecs_c2w[cam_indices[known_mask]],
                                        self._tvecs_c2w[cam_indices[known_mask]])
        E_b2c_known = E_b2c_all[known_mask]

        # Initial vote for the board's pose based on currently known cameras
        E_b2w_votes = E_c2w_known @ E_b2c_known

        # --- Temporal disambiguation using a stable ref pose ---
        if len(self._board_pose_history) > 0:

            history_r, history_t = extmat_to_rtvecs(jnp.stack(list(self._board_pose_history)))
            history_q = axisangle_to_quaternion_batched(history_r)

            # We average the rotation (via quaternions) and translation separately
            q_ref = quaternion_average(history_q)
            # t_ref = jnp.mean(history_t, axis=0)   # these are not super useful, the quaternion is the most important
            # r_ref = quaternion_to_axisangle(q_ref)

            # Get the alternative PnP solutions (180-degree flip)
            r_b2c_known, t_b2c_known = extmat_to_rtvecs(E_b2c_known)
            r_b2c_alt, t_b2c_alt = generate_multiple_ambiguous_poses(r_b2c_known, t_b2c_known)
            E_b2c_alt = extrinsics_matrix(r_b2c_alt, t_b2c_alt)

            # Calculate world poses for both the original and the alternative PnP result
            E_b2w_votes_alt = E_c2w_known @ E_b2c_alt

            # For each vote determine which (original or alternative) is closer to the stable ref
            r_votes, _ = extmat_to_rtvecs(E_b2w_votes)
            q_votes = axisangle_to_quaternion_batched(r_votes)

            r_votes_alt, _ = extmat_to_rtvecs(E_b2w_votes_alt)
            q_votes_alt = axisangle_to_quaternion_batched(r_votes_alt)

            # Calculate angular distance to the reference for both sets of poses
            dist_original = jax.vmap(lambda q: quaternions_angular_distance(q, q_ref))(q_votes) # TODO: move the vmaps to the geometry module
            dist_alt = jax.vmap(lambda q: quaternions_angular_distance(q, q_ref))(q_votes_alt)

            # Choose the best pose for each camera view
            use_alt_mask = dist_alt < dist_original
            E_b2w_votes = jnp.where(use_alt_mask[:, None, None], E_b2w_votes_alt, E_b2w_votes)

            num_corrected = jnp.sum(use_alt_mask)
            if num_corrected > 0:
                logger.debug(f"[FLIP_CORRECTED] Corrected {num_corrected} PnP results using stable reference.")

        # --- Averaging and Quality Control ---
        r_stack, t_stack = extmat_to_rtvecs(E_b2w_votes)
        q_stack = axisangle_to_quaternion_batched(r_stack)
        rt_stack = jnp.concatenate([q_stack, t_stack], axis=1)

        q_avg, t_avg, success = filter_rt_samples(
            rt_stack=rt_stack,
            ang_thresh=self._angular_thresh_rad,
            trans_thresh=self._translational_thresh
        )

        if not success:
            logger.debug(
                f"[CONSENSUS_FAIL] Frame rejected. Could not find a consistent board pose among {rt_stack.shape[0]} views.")
            self._latest_board_pose_w = None  # invalidate the single-frame pose
            return

        # --- Update state with the new good pose ---
        E_b2w = extrinsics_matrix(quaternion_to_axisangle(q_avg), t_avg)
        self._latest_board_pose_w = E_b2w
        self._board_pose_history.append(E_b2w)

        E_c2b_all = invert_extrinsics_matrix(E_b2c_all)
        E_c2w_new = E_b2w @ E_c2b_all
        r_c2w_new, t_c2w_new = extmat_to_rtvecs(E_c2w_new)

        world_pts = (E_b2w @ self._board_pts_hom.T).T[:, :3]

        r_w2c_new, t_w2c_new = invert_rtvecs(r_c2w_new, t_c2w_new)
        K_batch = self._cam_matrices[cam_indices]
        D_batch = self._dist_coeffs[cam_indices]

        reproj_pts, reproj_mask = project_to_multiple_cameras(
            world_pts,
            r_w2c_new,
            t_w2c_new,
            K_batch,
            D_batch,
            distortion_model='full'
        )

        effective_visibility = visibility_mask * reproj_mask

        errors_dict = reprojection_errors(gt_points_padded, reproj_pts, effective_visibility)
        mean_frame_error = errors_dict['rms']

        FRAME_ERROR_THRESHOLD = 5.0
        if mean_frame_error > FRAME_ERROR_THRESHOLD:
            logger.debug(f"[QUALITY_REJECT] Frame rejected. High reproj error: {mean_frame_error:.2f}px")
            # if the frame is bad, we should not have added it to the history. So we dump it. TODO: that's a bit suboptimal but that'll do for now
            if len(self._board_pose_history) > 0 and jnp.all(self._board_pose_history[-1] == E_b2w):
                self._board_pose_history.pop()
            self._latest_board_pose_w = None
            return

        logger.debug(f"[ACCEPTED] Frame mean: {mean_frame_error:.2f} px.")

        # Commit the new extrinsics to the main state for all cameras in this frame
        for i, cam_idx in enumerate(cam_indices):
            if cam_idx != self.origin_idx:  # Never update the origin camera
                self._rvecs_c2w = self._rvecs_c2w.at[cam_idx].set(r_c2w_new[i])
                self._tvecs_c2w = self._tvecs_c2w.at[cam_idx].set(t_c2w_new[i])
            self._has_extrinsics[cam_idx] = True

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

        # The origin camera's extrinsics are fixed at identity, so its flag is always true
        # This only needs to be set once
        if not self._has_extrinsics[self.origin_idx]:
            self._has_extrinsics[self.origin_idx] = True

        self._flush_frames()

    def refine_all(self) -> bool:
        """
        Performs a global, three-stage bundle adjustment (BA) over all collected samples
        (Sort of graduated non-convexity process)

        - Stage 1: Solves for a stable global geometry with shared intrinsics and no distortion
        - Stage 2: Refines per-camera intrinsics (still no distortion)
        - Stage 3: Performs a full refinement with all parameters (including distortion)
        """

        # def _log_params(stage_name, results_dict, initial_K, initial_D):
        #     """ Helper to log key parameters and their deviation from initial values """
        #     K_opt, D_opt = results_dict['K_opt'], results_dict['D_opt']
        #
        #     # Calculate average focal length and principal point
        #     fx_avg = jnp.mean(K_opt[:, 0, 0])
        #     fy_avg = jnp.mean(K_opt[:, 1, 1])
        #     cx_avg = jnp.mean(K_opt[:, 0, 2])
        #     cy_avg = jnp.mean(K_opt[:, 1, 2])
        #
        #     # Calculate max absolute distortion coefficients across all cameras
        #     # Only check the coeffs relevant to the current model to avoid noise from unused params
        #     # This assumes your BA stages correctly zero-out or handle unused coeffs
        #     n_d = D_opt.shape[1]  # Let's just check all available
        #     max_dist_coeffs = jnp.max(jnp.abs(D_opt[:, :n_d]), axis=0)
        #
        #     # Calculate deviation from initial parameters
        #     K_init_avg_f = (jnp.mean(initial_K[:, 0, 0]) + jnp.mean(initial_K[:, 1, 1])) / 2.0
        #     K_opt_avg_f = (fx_avg + fy_avg) / 2.0
        #     f_drift_percent = 100 * (K_opt_avg_f - K_init_avg_f) / K_init_avg_f
        #
        #     D_init_max_abs = jnp.max(jnp.abs(initial_D), axis=0)
        #     D_drift = max_dist_coeffs - D_init_max_abs
        #
        #     # Using logger.info for high visibility during debugging
        #     logger.info(f"--- [BA] End of {stage_name} ---")
        #     logger.info(f"  Avg Focal Length (fx, fy): ({fx_avg:.2f}, {fy_avg:.2f}) px")
        #     logger.info(f"  Focal Length Drift: {f_drift_percent:+.2f}% from initial guess")
        #     logger.info(f"  Avg Principal Point (cx, cy): ({cx_avg:.2f}, {cy_avg:.2f}) px")
        #     logger.info(f"  Max Distortion Coeffs (abs): {np.array2string(np.asarray(max_dist_coeffs), precision=4)}")
        #     logger.info(
        #         f"  Distortion Drift (max abs):   {np.array2string(np.asarray(D_drift), precision=4, sign='+')}")

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

        ba_succeeded = False
        final_results = None

        # Priors weights to prevent the BA from overfittign
        priors_stage1 = {
            'intrinsics': {
                'focal_length': 0.1,    # weak, just to prevent the *average* focal length from drifting into nonsense
                'principal_point': 5.0, # This can be quite strong, most modern lenses have it very close to the centre
                'distortion': 0.0
            },
            'extrinsics': {
                'rotation': 0.0,
                'translation': 0.0
            }
        }
        priors_stage2 = {
            'intrinsics': {
                'focal_length': 1.0,    # quite strong. Keeps each camera's focal length from deviating from the average found in Stage 1
                'principal_point': 0.1, # weak, but still here to keep the principal point near the image center
                'distortion': 0.5       # medium, keeps the initial distortion terms small and well-behaved
            },
            'extrinsics': {             # extrinsics priors off
                'rotation': 0.0,
                'translation': 0.0
            }
        }
        priors_stage3 = {
            'intrinsics': {
                'focal_length': 1.0,    # still strong. This is critical to avoid overfitting. TODO: Could be stronger maybe?
                'principal_point': 0.1, # same as in stage 2. Modern cameras with modern lenses should be pretty centered...
                'distortion': 0.1       # Relaxed from stage 2. We want to refine these a bit more.
            },
            'extrinsics': { # We assume by then the geometry is pretty good, so we set priors on the extrinsics
                            # This prevents a single camera with poor visibility in some frames from drifting

                # A weight of ~ 700 on radians is comparable to a weight of 0.1 on mm for a target tolerance of 0.5 deg / 1.0 mm
                # This keeps camera poses very stable, allowing only tiny final adjustments
                # TODO: Maybe we want to do this scaling inside the bundle_adjustment module and only expose normalised weights here?
                'rotation': 700,
                'translation': 0.1
            }
        }

        # The try except loop is a little safeguard to avoid filling up the RAM because of the jacobian
        # (it grows quadratically with the nb of samples)
        current_P = self.ba_sample_count
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

                    # here we use the simple, lenient average for BA initialization
                    # (Because the spread of this cluster is a direct result of the accumulated errors
                    # during online camera pose estimates - which are unavoidable!!
                    # The hardcore filter used online would likely jusyt eliminate everyone here)
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

                # STAGE 1: Ideal pinhole world (shared intrinsics, no distortion)
                # ---------------------------------------------------------------
                # Here we care only about the overall camera layout and the average 3D structure of the scene
                #
                logger.debug(f"[BA] >>> STAGE 1: Consolidating cameras position with {current_P} frames...")
                success_s1, results_s1 = bundle_adjustment.run_bundle_adjustment(

                    K_online, D_online, cam_r_online, cam_t_online, board_r_online, board_t_online,

                    pts2d_buf, vis_buf,

                    self._object_points,
                    self._images_sizes_wh,
                    origin_idx=self.origin_idx,

                    max_frames=current_P,

                    priors=priors_stage1,

                    shared_intrinsics=True,         # This is critical: Forces a single camera model for all views
                    fix_aspect_ratio=True,          # this is a simplification: it assumes fx = fy
                    distortion_model='simple',      # we use a simple model...
                    fix_distortion=True,            # ...but don;t optimize it. it's frozen it at 0

                    # Free parameters we want to solve for
                    fix_camera_matrix=False,
                    fix_extrinsics=False,
                    fix_board_poses=False,

                    radial_penalty=0.0      # for fisrst stage we want to consider all points, even at the edge
                )
                if not success_s1:
                    raise RuntimeError("BA Stage 1 failed.")

                # _log_params("Stage 1 (Shared Pinhole)", results_s1, K_online, D_online)

                # STAGE 2: Per-camera pinhole world (shared intrinsics, simple distortion)
                # ------------------------------------------------------------------------
                # Here we relax the shared model and start refining the per-camera details, but we use priors
                # to keep them from deviating wildly from the stable average we found in Stage 1
                #
                logger.debug(f"[BA] >>> STAGE 2: Consolidating per-camera intrinsics with {current_P} frames...")

                K_s2_init, D_s2_init = results_s1['K_opt'], results_s1['D_opt']
                cam_r_s2_init, cam_t_s2_init = results_s1['cam_r_opt'], results_s1['cam_t_opt']
                board_r_s2_init, board_t_s2_init = results_s1['board_r_opt'], results_s1['board_t_opt']

                success_s2, results_s2 = bundle_adjustment.run_bundle_adjustment(

                    K_s2_init, D_s2_init, cam_r_s2_init, cam_t_s2_init, board_r_s2_init, board_t_s2_init,

                    pts2d_buf, vis_buf,

                    self._object_points,
                    self._images_sizes_wh,
                    origin_idx=self.origin_idx,

                    max_frames=current_P,

                    priors=priors_stage2,

                    shared_intrinsics=False,    # Critical: we now optimize per-camera intrinsics
                    fix_aspect_ratio=False,     # We relax the aspect ratio constraint
                    distortion_model='simple',  # we use a simple 4-parameter model...
                    fix_distortion=False,       # ... and start optimizing for it

                    # Free parameters we want to solve for
                    fix_camera_matrix=False,
                    fix_extrinsics=False,
                    fix_board_poses=False,

                    radial_penalty=2.0 # for second stage we want to start penalising points too far from the working volume
                )
                if not success_s2:
                    raise RuntimeError("BA Stage 2 failed.")

                # _log_params("Stage 2 (Per-Cam Pinhole)", results_s2, K_online, D_online)

                # STAGE 3: Real world (Full extrinsics + intrinsics refinement with distortion)
                # -----------------------------------------------------------------------------
                # Everything should be close to the correct solution. We enable the most complex distortion models
                # (like full or rational) and let all parameters adjust simultaneously for the final polish
                #
                logger.debug(f"[BA] >>> STAGE 3: Full refinement with {current_P} frames...")

                K_s3_init, D_s3_init = results_s2['K_opt'], results_s2['D_opt']
                cam_r_s3_init, cam_t_s3_init = results_s2['cam_r_opt'], results_s2['cam_t_opt']
                board_r_s3_init, board_t_s3_init = results_s2['board_r_opt'], results_s2['board_t_opt']

                success_s3, final_results_attempt = bundle_adjustment.run_bundle_adjustment(

                    K_s3_init, D_s3_init, cam_r_s3_init, cam_t_s3_init, board_r_s3_init, board_t_s3_init,

                    pts2d_buf, vis_buf,

                    self._object_points,
                    self._images_sizes_wh,
                    origin_idx=self.origin_idx,

                    max_frames=current_P,

                    priors=priors_stage3,       # Priors are mega important at this stage

                    shared_intrinsics=False,    # Still optimising for this independently
                    fix_aspect_ratio=False,     # Still letting fx and fy be independent
                    distortion_model='full',    # Now we use a more elaborate model...
                    fix_distortion=False,       # ...and of course we let it be optimised

                    # These ones are ofc still optimised
                    fix_camera_matrix=False,
                    fix_extrinsics=False,
                    fix_board_poses=False,

                    radial_penalty=4.0   # now we kinda want to ignore the points far from the working volume
                )
                if not success_s3:
                    raise RuntimeError("BA Stage 3 failed.")

                # _log_params("Stage 3 (Full Model)", final_results_attempt, K_online, D_online)

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
            reprojected_pts, valid_depth_mask = project_multiple_to_multiple(
                world_pts_all_instances,
                r_w2c, t_w2c,
                *self._refined_intrinsics,
                distortion_model='full'
            )

            effective_visibility = visibility_mask * valid_depth_mask

            # Compute the raw, per-point Euclidean distance errors
            raw_errors = jnp.linalg.norm(observed_pts2d - reprojected_pts, axis=-1)
            all_errors = jnp.where(effective_visibility, raw_errors, jnp.nan)    # And mark non-observed points as nan

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