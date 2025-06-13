from collections import deque
from typing import Tuple, Dict, Optional, List, Any
import cv2
import numpy as np
from functools import partial
import jax
from jax import numpy as jnp
from jax.typing import ArrayLike
from mokap.calibration import bundle_adjustment
from mokap.utils.datatypes import DetectionPayload
from mokap.utils.geometry.projective import compute_volume_of_trust, project_to_multiple_cameras
from mokap.utils.geometry.fitting import quaternion_average, filter_rt_samples
from mokap.utils.geometry.transforms import extrinsics_matrix, extmat_to_rtvecs, axisangle_to_quaternion_batched, \
    quaternion_to_axisangle, invert_extrinsics_matrix, invert_rtvecs


class MultiviewCalibrationTool:
    def __init__(self,
                 nb_cameras:            int,
                 images_sizes_wh:       ArrayLike,
                 origin_idx:            int,
                 init_cam_matrices:     ArrayLike,
                 init_dist_coeffs:      ArrayLike,
                 object_points:         ArrayLike,
                 intrinsics_window:     int = 10,
                 min_detections:        int = 15,
                 max_detections:        int = 100,
                 angular_thresh:        float = 10.0,   # in degrees
                 translational_thresh:  float = 10.0,   # in object_points' units
                 debug_print = True):

        # TODO: Typing and optimising this class

        self._debug_print = debug_print

        self.nb_cameras = nb_cameras
        self.origin_idx = origin_idx

        images_sizes_wh = np.asarray(images_sizes_wh)
        assert images_sizes_wh.ndim == 2 and images_sizes_wh.shape[0] == self.nb_cameras
        self._images_sizes_wh = images_sizes_wh[:, :2]

        self._angular_thresh_rad: float = np.deg2rad(angular_thresh)
        self._translational_thresh: float = translational_thresh

        # Known 3D board model points (N, 3)
        board_points = np.asarray(object_points, dtype=np.float32)
        self._object_points = jnp.asarray(board_points)
        self._board_pts_hom = jnp.asarray(np.hstack([board_points,
                                                     np.ones((board_points.shape[0], 1), dtype=np.float32)]))

        # buffers for incoming frames
        self._detection_buffer = [dict() for _ in range(nb_cameras)]
        self._last_frame = np.full(nb_cameras, -1, dtype=int)

        # extrinsics state (camera-to-world)
        self._has_ext: List[bool] = [False] * nb_cameras
        self._rvecs_c2w: jnp.ndarray = jnp.zeros((nb_cameras, 3), dtype=jnp.float32)
        self._tvecs_c2w: jnp.ndarray = jnp.zeros((nb_cameras, 3), dtype=jnp.float32)
        self._estimated: bool = False

        # intrinsics state
        self._cam_matrices: jnp.ndarray = jnp.asarray(init_cam_matrices, dtype=jnp.float32)
        self._dist_coeffs: jnp.ndarray = jnp.asarray(init_dist_coeffs, dtype=jnp.float32)

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

    def register(self, cam_idx: int, detection: DetectionPayload):

        if detection.pointsIDs is None or detection.points2D is None:
            return

        if len(detection.pointsIDs) < 4:
            return

        # Reestimate the board-to-camera pose and validate it
        try:
            success, rvec, tvec = cv2.solvePnP(
                objectPoints=np.asarray(self._object_points[detection.pointsIDs]),
                imagePoints=detection.points2D,
                cameraMatrix=np.asarray(self._cam_matrices[cam_idx]),
                distCoeffs=np.asarray(self._dist_coeffs[cam_idx]),
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            # if PnP fails, return
            if not success:
                return

            rvec, tvec = rvec.squeeze(), tvec.squeeze()

            # if PnP placed the board behind the camera, return
            if tvec[2] <= 0:
                return

        except cv2.error:
            return

        #--- From here on, rvec and tvec should be sane ---

        f = detection.frame
        self._last_frame[cam_idx] = f

        E_b2c = extrinsics_matrix(jnp.asarray(rvec), jnp.asarray(tvec))
        self._detection_buffer[cam_idx][f] = (E_b2c, detection.points2D, detection.pointsIDs)

        if cam_idx == self.origin_idx and not self._has_ext[cam_idx]:
            self._has_ext[cam_idx] = True
            # The pose is already (0, 0, 0) we can continue

        self._flush_frames()

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

        if not any(self._has_ext):
            return

        cam_indices, gt_points_padded, visibility_mask = self._gather_frame_data(entries)

        # Filter entries based on which cameras have known extrinsics
        known_mask = jnp.array([self._has_ext[c] for c in cam_indices])
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
            return

        E_b2w = extrinsics_matrix(quaternion_to_axisangle(q_avg), t_avg)

        # Calculate the potential new camera-to-world poses for ALL cameras in this frame
        E_c2b_all = invert_extrinsics_matrix(E_b2c_all)
        E_c2w_new = E_b2w @ E_c2b_all
        r_c2w_new, t_c2w_new = extmat_to_rtvecs(E_c2w_new)

        # --- Quality Control using the NEW poses ---
        world_pts = (E_b2w @ self._board_pts_hom.T).T[:, :3]

        # Get world-to-camera transforms from the NEWLY CALCULATED camera poses
        r_w2c_new, t_w2c_new = invert_rtvecs(r_c2w_new, t_c2w_new)
        K_batch = self._cam_matrices[cam_indices]
        D_batch = self._dist_coeffs[cam_indices]

        reproj_pts = project_to_multiple_cameras(world_pts, r_w2c_new, t_w2c_new, K_batch, D_batch)

        # Compute error only on visible points
        errors = jnp.linalg.norm(reproj_pts - gt_points_padded, axis=-1)
        visible_errors_sum = jnp.sum(jnp.where(visibility_mask, errors, 0.0))
        total_visible_points = jnp.sum(visibility_mask)
        mean_frame_error = jnp.where(total_visible_points > 0, visible_errors_sum / total_visible_points, jnp.inf)

        FRAME_ERROR_THRESHOLD = 5.0
        if mean_frame_error > FRAME_ERROR_THRESHOLD:
            if self._debug_print:
                print(f"[QUALITY_REJECT] Frame rejected. High reproj error: {mean_frame_error:.2f}px")
            return

        # --- Update camera extrinsics and buffer data ---
        # if the quality check passed we store the new poses to the class state
        for i, cam_idx in enumerate(cam_indices):
            # We update all cameras except the origin
            if cam_idx != self.origin_idx:
                self._rvecs_c2w = self._rvecs_c2w.at[cam_idx].set(r_c2w_new[i])
                self._tvecs_c2w = self._tvecs_c2w.at[cam_idx].set(t_c2w_new[i])

            # of course we still mark all the cameras including the origin as having extrinsics
            self._has_ext[cam_idx] = True

        self._estimated = True

        if self._debug_print:
            per_cam_vis_pts = jnp.sum(visibility_mask, axis=1)
            per_cam_err_sum = jnp.sum(jnp.where(visibility_mask, errors, 0.0), axis=1)
            per_cam_mean_err = jnp.where(per_cam_vis_pts > 0, per_cam_err_sum / per_cam_vis_pts, 0.0)
            print(f"[REPROJ_ERR] Frame mean: {mean_frame_error:.2f}px. Per-cam: " +
                  ", ".join([f"{c}:{e:.2f}px" for c, e in zip(cam_indices, per_cam_mean_err)]))

        # Buffer data for final BA and intrinsics refinement
        self.ba_samples.append(entries)

    def refine_all(self) -> bool:
        """
        Performs a global, three-stage bundle adjustment (BA) over all collected samples
        (Sort of graduated non-convexity process)

        - Stage 1: Solves for a stable global geometry with shared intrinsics and no distortion
        - Stage 2: Refines per-camera intrinsics (still no distortion)
        - Stage 3: Performs a full refinement with all parameters (including distortion)
        """

        if not self._estimated:
            print("[BA] Error: Initial extrinsics have not been estimated yet.")
            return False

        P = self.ba_sample_count
        if P < self._min_detections:
            print(f"[BA] Not enough samples for bundle adjustment. Have {P}, need {self._min_detections}.")
            return False

        if self._debug_print:
            print(f"[BA] Starting 3-Stage Bundle Adjustment with {P} samples.")

        C = self.nb_cameras
        N = self._object_points.shape[0]

        # --------Prepare the data-----------

        pts2d_buf = np.zeros((C, P, N, 2), dtype=np.float32)
        vis_buf = np.zeros((C, P, N), dtype=bool)

        for p_idx, entries in enumerate(self.ba_samples):
            for cam_idx, _, pts2D, ids in entries:
                pts2d_buf[cam_idx, p_idx, ids, :] = pts2D
                vis_buf[cam_idx, p_idx, ids] = True

        # Initial guess for board poses (from online estimation)
        r_board_w_list, t_board_w_list = [], []
        E_c2w_all = extrinsics_matrix(self._rvecs_c2w, self._tvecs_c2w)

        for p_idx, entries in enumerate(self.ba_samples):
            # Get the camera indices and their corresponding board-to-camera poses for THIS FRAME
            cam_indices_in_frame = jnp.array([c for c, _, _, _ in entries])
            E_b2c_in_frame = jnp.stack([E_b2c for _, E_b2c, _, _ in entries])

            # Get the camera-to-world poses ONLY for the cameras in this frame
            E_c2w_in_frame = E_c2w_all[cam_indices_in_frame]

            # Correctly compose poses for this frame's views
            E_b2w_votes = E_c2w_in_frame @ E_b2c_in_frame

            # Remove potantial outliers
            r_stack, t_stack = extmat_to_rtvecs(E_b2w_votes)
            q_stack = axisangle_to_quaternion_batched(r_stack)
            rt_stack = jnp.concatenate([q_stack, t_stack], axis=1)

            q_avg, t_avg, success = filter_rt_samples(
                rt_stack=rt_stack,
                ang_thresh=self._angular_thresh_rad,
                trans_thresh=self._translational_thresh
            )
            r_avg = quaternion_to_axisangle(q_avg)

            # handle the rare case where filtering might fail (all votes are outliers)
            if not success:
                r_avg, t_avg = r_stack[0], t_stack[0]

            r_board_w_list.append(r_avg)
            t_board_w_list.append(t_avg)

        # Start with the online estimates
        cam_r_online = self._rvecs_c2w
        cam_t_online = self._tvecs_c2w
        K_online = self._cam_matrices
        D_online = self._dist_coeffs
        board_r_online = jnp.stack(r_board_w_list)
        board_t_online = jnp.stack(t_board_w_list)

        pts2d_buf = jnp.asarray(pts2d_buf)
        vis_buf = jnp.asarray(vis_buf)

        # STAGE 1: Ideal Pinhole World (Shared Intrinsics, simple distortion)
        # ----------------------------
        print("\n" + "=" * 80)
        print(">>> STAGE 1: BA on Ideal World (Shared Intrinsics, simple distortion)")
        print("=" * 80)
        success_s1, results_s1 = bundle_adjustment.run_bundle_adjustment(

            K_online, D_online, cam_r_online, cam_t_online, board_r_online, board_t_online,

            pts2d_buf, vis_buf,

            self._object_points,
            self._images_sizes_wh,

            radial_penalty=0.0,   # for fisrst stage we want to consider all points

            shared_intrinsics=True,
            fix_aspect_ratio=True,
            distortion_model='simple',    # Fixes distortion params to zero

            # What to optimize:
            fix_focal_principal=False,
            fix_distortion=True,        # Redundant with model='none', but explicit
            fix_extrinsics=False,
            fix_board_poses=False
        )

        if not success_s1:
            print("[ERROR] BA Stage 1 failed. Aborting calibration.")
            return False

        # Use results of S1 as initial guess for S2
        K_s2_init = jnp.asarray(results_s1['K_opt'])
        D_s2_init = jnp.asarray(results_s1['D_opt'])
        cam_r_s2_init = jnp.asarray(results_s1['cam_r_opt'])
        cam_t_s2_init = jnp.asarray(results_s1['cam_t_opt'])
        board_r_s2_init = jnp.asarray(results_s1['board_r_opt'])
        board_t_s2_init = jnp.asarray(results_s1['board_t_opt'])

        # STAGE 2: Per-camera pinhole world (Per-camera intrinsics, no Distortion)
        # ---------------------------------
        print("\n" + "=" * 80)
        print(">>> STAGE 2: BA on Pinhole World (Per-Camera Intrinsics, No Distortion)")
        print("=" * 80)

        success_s2, results_s2 = bundle_adjustment.run_bundle_adjustment(

            K_s2_init, D_s2_init, cam_r_s2_init, cam_t_s2_init, board_r_s2_init, board_t_s2_init,

            pts2d_buf, vis_buf, self._object_points, self._images_sizes_wh,

            radial_penalty=2.0,   # for second stage we want to start penalising points too far from the working volume

            shared_intrinsics=False,  # Now we optimize per-camera
            fix_aspect_ratio=False,   # we allow fx and fy to differ
            distortion_model='none',

            fix_focal_principal=False,
            fix_distortion=True,
            fix_extrinsics=False,
            fix_board_poses=False
        )

        if not success_s2:
            print("[ERROR] BA Stage 2 failed. Aborting calibration.")
            return False

        # Use results of stage 2 as initial guess for stage 3
        K_s3_init = jnp.asarray(results_s2['K_opt'])
        D_s3_init = jnp.asarray(results_s2['D_opt'])
        cam_r_s3_init = jnp.asarray(results_s2['cam_r_opt'])
        cam_t_s3_init = jnp.asarray(results_s2['cam_t_opt'])
        board_r_s3_init = jnp.asarray(results_s2['board_r_opt'])
        board_t_s3_init = jnp.asarray(results_s2['board_t_opt'])

        # STAGE 3: Real world (Full Refinement with Distortion)
        # -------------------
        print("\n" + "=" * 80)
        print(">>> STAGE 3: BA on Real World (Full Refinement with Distortion)")
        print("=" * 80)

        success_s3, final_results = bundle_adjustment.run_bundle_adjustment(

            K_s3_init, D_s3_init, cam_r_s3_init, cam_t_s3_init, board_r_s3_init, board_t_s3_init,

            pts2d_buf, vis_buf, self._object_points, self._images_sizes_wh,

            radial_penalty=4.0,     # now we kinda want to ignore the points far from the working volume

            shared_intrinsics=False,
            fix_aspect_ratio=False,
            # distortion_model='standard',  # Use the 5-parameter model
            distortion_model='full',  # Use the 8-parameter model

            fix_focal_principal=False,
            fix_distortion=False,
            fix_extrinsics=False,
            fix_board_poses=False,

            priors_weight=0.0
        )

        # Finish
        # -----------
        if success_s3:
            print("\nBundle adjustment complete. Storing refined parameters.")

            # store the globally optimized results
            self._refined_intrinsics = (final_results['K_opt'], final_results['D_opt'])
            self._refined_extrinsics = (final_results['cam_r_opt'], final_results['cam_t_opt'])
            self._refined_board_poses = (final_results['board_r_opt'], final_results['board_t_opt'])
            self._points2d, self._visibility_mask = np.asarray(pts2d_buf), np.asarray(vis_buf)

            self._refined = True

            volume_of_trust = compute_volume_of_trust(
                jnp.asarray(final_results['K_opt']), jnp.asarray(final_results['D_opt']),
                jnp.asarray(final_results['cam_r_opt']), jnp.asarray(final_results['cam_t_opt']),
                jnp.asarray(final_results['board_r_opt']), jnp.asarray(final_results['board_t_opt']),
                pts2d_buf, vis_buf,
                self._object_points,
                error_threshold_px=1.5,
                percentile=1.0      # 99th percentile
            )

            if volume_of_trust:
                print("--- Volume of Trust ---")
                print(f"X range: {volume_of_trust['x'][0]:.2f} to {volume_of_trust['x'][1]:.2f} mm")
                print(f"Y range: {volume_of_trust['y'][0]:.2f} to {volume_of_trust['y'][1]:.2f} mm")
                print(f"Z range: {volume_of_trust['z'][0]:.2f} to {volume_of_trust['z'][1]:.2f} mm")

            self.ba_samples.clear()

            return True
        else:
            print("[ERROR] Final BA Stage 3 failed. Aborting calibration.")
            return False

    @property
    def initial_extrinsics(self):
        return np.array(self._rvecs_c2w), np.array(self._tvecs_c2w)

    @property
    def refined_intrinsics(self):
        return self._refined_intrinsics

    @property
    def refined_extrinsics(self):
        return self._refined_extrinsics

    @property
    def refined_board_poses(self):
        return self._refined_board_poses

    @property
    def image_points(self):
        return self._points2d, self._visibility_mask

    @property
    def ba_sample_count(self):
        return len(self.ba_samples)