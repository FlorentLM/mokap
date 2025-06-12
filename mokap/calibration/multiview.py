from collections import deque
from typing import Tuple, Dict, Optional
import cv2
import numpy as np
from jax import numpy as jnp
from numpy.typing import ArrayLike
from mokap.calibration import bundle_adjustment
from mokap.utils.datatypes import DetectionPayload
from mokap.utils.geometry.fitting import quaternion_average, filter_rt_samples
from mokap.utils.geometry.transforms import extrinsics_matrix, extmat_to_rtvecs, axisangle_to_quaternion_batched, \
    quaternion_to_axisangle, invert_extrinsics_matrix, invert_rtvecs


class MultiviewCalibrationTool:
    def __init__(self,
                 nb_cameras: int,
                 images_sizes_wh: ArrayLike,
                 origin_idx: int,
                 init_cam_matrices: ArrayLike,
                 init_dist_coeffs: ArrayLike,
                 object_points: ArrayLike,
                 intrinsics_window: int = 10,
                 min_detections: int = 15,
                 max_detections: int = 100,
                 angular_thresh: float = 10.0,         # in degrees
                 translational_thresh: float = 10.0,   # in object_points' units
                 debug_print = True):

        # TODO: Typing and optimising this class

        self._debug_print = debug_print

        self.nb_cameras = nb_cameras
        self.origin_idx = origin_idx

        images_sizes_wh = np.asarray(images_sizes_wh)
        assert images_sizes_wh.ndim == 2 and images_sizes_wh.shape[0] == self.nb_cameras
        self.images_sizes_wh = images_sizes_wh[:, :2]

        self._angular_thresh = angular_thresh
        self._translational_thresh = translational_thresh

        # Known 3D board model points (N, 3)
        self._object_points = np.asarray(object_points, dtype=np.float32)

        # buffers for incoming frames
        self._detection_buffer = [dict() for _ in range(nb_cameras)]
        self._last_frame = np.full(nb_cameras, -1, dtype=int)

        # extrinsics state
        self._has_ext = [False] * nb_cameras
        self._rvecs_cam2world = jnp.zeros((nb_cameras, 3), dtype=jnp.float32)
        self._tvecs_cam2world = jnp.zeros((nb_cameras, 3), dtype=jnp.float32)
        self._estimated = False

        # intrinsics state & buffer per camera
        self._cam_matrices = [np.asarray(init_cam_matrices[c], dtype=np.float32) for c in range(nb_cameras)]
        self._dist_coeffs = [np.asarray(init_dist_coeffs[c], dtype=np.float32) for c in range(nb_cameras)]
        self._intrinsics_buffer = {c: deque(maxlen=intrinsics_window) for c in range(nb_cameras)}
        self._intrinsics_window = intrinsics_window

        # triangulation & BA buffers
        self.ba_samples = deque(maxlen=max_detections)
        self.min_detections = min_detections

        self._refined = False

        # bs results
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

        # Get the current best intrinsics for this camera
        K = self._cam_matrices[cam_idx]
        D = self._dist_coeffs[cam_idx]

        # Prepare the 3D and 2D points for solvePnP
        ids = detection.pointsIDs
        obj_pts_subset = self._object_points[ids]
        img_pts_subset = detection.points2D

        # Reestimate the board-to-camera pose and validate it
        try:
            success, rvec, tvec = cv2.solvePnP(
                objectPoints=obj_pts_subset,
                imagePoints=img_pts_subset,
                cameraMatrix=K,
                distCoeffs=D,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            # If PnP fails, return
            if not success:
                return

            rvec = rvec.squeeze()
            tvec = tvec.squeeze()

            # if PnP placed the board behind the camera, return
            if tvec[2] <= 0:
                return

        except cv2.error:
            return

        #--- From here on, we know rvec tvec are sane ---

        f = detection.frame
        self._last_frame[cam_idx] = f

        E_b2c = extrinsics_matrix(
            jnp.asarray(rvec), jnp.asarray(tvec)
        )
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

    def _process_frame(self, entries):

        if not any(self._has_ext):
            return

        # Recompute board-to-camera poses for this frame
        recomputed_entries = []
        for cam_idx, _, pts2D, ids in entries:
            K = self._cam_matrices[cam_idx]
            D = self._dist_coeffs[cam_idx]
            obj_pts_subset = self._object_points[np.asarray(ids).flatten()]
            img_pts_subset = np.asarray(pts2D)
            if len(ids) < 4: continue

            success, rvec, tvec = cv2.solvePnP(obj_pts_subset, img_pts_subset, K, D)
            if not success or tvec.squeeze()[2] <= 0:
                continue

            rvec, tvec = rvec.flatten(), tvec.flatten()
            E_b2c = extrinsics_matrix(jnp.asarray(rvec), jnp.asarray(tvec))
            recomputed_entries.append((cam_idx, E_b2c, pts2D, ids))

        if not recomputed_entries:
            return

        known = [e for e in recomputed_entries if self._has_ext[e[0]]]

        # Estimate board-to-world pose
        E_b2w = None
        if known:
            E_votes = []
            for cam_idx, E_b2c, _, _ in known:
                E_c2w_current = extrinsics_matrix(
                    self._rvecs_cam2world[cam_idx], self._tvecs_cam2world[cam_idx]
                )
                E_votes.append(E_c2w_current @ E_b2c)

            E_stack = jnp.stack(E_votes, axis=0)

            # Convert all pose votes into (quaternion, translation) format
            r_stack, t_stack = extmat_to_rtvecs(E_stack)
            q_stack = axisangle_to_quaternion_batched(r_stack)
            rt_stack = jnp.concatenate([q_stack, t_stack], axis=1)  # (num_known, 7)

            # robust filtering and averaging
            q_med, t_med, success = filter_rt_samples(
                rt_stack=rt_stack,
                ang_thresh=np.deg2rad(self._angular_thresh),
                trans_thresh=self._translational_thresh
            )

            if not success:
                return

            E_b2w = extrinsics_matrix(
                quaternion_to_axisangle(q_med), t_med
            )

            # Update all non-origin camera extrinsics based on the new E_b2w
            # (this new state will be used in the next iteration of this loop)
            if E_b2w is not None:
                for cam_idx, E_b2c, _, _ in recomputed_entries:
                    if cam_idx != self.origin_idx:
                        E_c2b = invert_extrinsics_matrix(E_b2c)
                        E_c2w = E_b2w @ E_c2b
                        r_c2w, t_c2w = extmat_to_rtvecs(E_c2w)
                        self._rvecs_cam2world = self._rvecs_cam2world.at[cam_idx].set(r_c2w)
                        self._tvecs_cam2world = self._tvecs_cam2world.at[cam_idx].set(t_c2w)
                    self._has_ext[cam_idx] = True

        # Final buffering (using the stabilised state)
        if E_b2w is not None:

            board_pts_hom = np.hstack([self._object_points, np.ones((self._object_points.shape[0], 1))])
            world_pts = (E_b2w @ board_pts_hom.T).T[:, :3]

            # --- DEBUG / Quality control ---
            total_err = 0
            total_pts = 0
            for cam_idx, _, pts2D, ids in recomputed_entries:
                # Project board points into this camera's view
                r_w2c, t_w2c = invert_rtvecs(self._rvecs_cam2world[cam_idx], self._tvecs_cam2world[cam_idx])
                obj_pts_world = world_pts[ids]
                proj_pts, _ = cv2.projectPoints(
                    np.asarray(obj_pts_world).reshape(-1, 1, 3),
                    np.asarray(r_w2c), np.asarray(t_w2c),
                    self._cam_matrices[cam_idx], self._dist_coeffs[cam_idx]
                )
                err = np.linalg.norm(proj_pts.reshape(-1, 2) - np.asarray(pts2D), axis=1)
                total_err += np.sum(err)
                total_pts += len(err)

            mean_frame_error = (total_err / total_pts) if total_pts > 0 else float('inf')

            # If the mean error for this multi-view frame is too high, skip it
            FRAME_ERROR_THRESHOLD = 5.0
            if mean_frame_error > FRAME_ERROR_THRESHOLD:
                if self._debug_print:
                    print(f"[QUALITY_REJECT] Frame rejected due to high reproj error: {mean_frame_error:.2f}px")
                return  # skip buffering this low-quality frame

            # --- End of DEBUG / Quality control block ---


            for cam_idx, E_b2c, pts2D, ids in recomputed_entries:
                uv_obs = np.asarray(pts2D, dtype=np.float32)

                if cam_idx == self.origin_idx:
                    r_b2c, t_b2c = extmat_to_rtvecs(E_b2c)
                    obj_pts_local = self._object_points[ids]
                    proj_pts, _ = cv2.projectPoints(
                        np.asarray(obj_pts_local).reshape(-1, 1, 3), np.asarray(r_b2c), np.asarray(t_b2c),
                        self._cam_matrices[cam_idx], self._dist_coeffs[cam_idx]
                    )
                else:
                    uv_obs = np.asarray(pts2D, dtype=np.float32)

                    # Get the world->camera transform for this camera
                    current_r_c2w = self._rvecs_cam2world[cam_idx]
                    current_t_c2w = self._tvecs_cam2world[cam_idx]
                    r_w2c, t_w2c = invert_rtvecs(current_r_c2w, current_t_c2w)

                    # Get the 3D points in the world frame
                    obj_pts_world = world_pts[ids]

                    # Project them and calculate error
                    proj_pts, _ = cv2.projectPoints(
                        np.asarray(obj_pts_world).reshape(-1, 1, 3), np.asarray(r_w2c), np.asarray(t_w2c),
                        self._cam_matrices[cam_idx], self._dist_coeffs[cam_idx]
                    )

                uv_proj = proj_pts.reshape(-1, 2)
                errs = np.linalg.norm(uv_proj - uv_obs, axis=1)
                if self._debug_print:
                    print(f"[REPROJ_ERR] cam={cam_idx}, mean={errs.mean():.2f}px, max={errs.max():.2f}px")

            self._estimated = True

        # Buffer data for final BA
        for cam_idx, _, pts2D, ids in recomputed_entries:
            if len(ids) >= 6:
                board_pts_subset = self._object_points[ids].astype(np.float32)
                img_pts = np.asarray(pts2D, dtype=np.float32)
                self._intrinsics_buffer[cam_idx].append((board_pts_subset, img_pts))

        self.ba_samples.append(recomputed_entries)

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
        if P < self.min_detections:
            print(f"[BA] Not enough samples for bundle adjustment. Have {P}, need {self.min_detections}.")
            return False

        if self._debug_print:
            print(f"[BA] Starting 3-Stage Bundle Adjustment with {P} samples.")

        C = self.nb_cameras
        N = self._object_points.shape[0]

        # --------Prepare the data-----------

        pts2d_buf = np.full((C, P, N, 2), 0.0, dtype=np.float32)
        vis_buf = np.zeros((C, P, N), dtype=bool)

        for p_idx, entries in enumerate(self.ba_samples):
            for cam_idx, _, pts2D, ids in entries:
                pts2d_buf[cam_idx, p_idx, ids, :] = pts2D
                vis_buf[cam_idx, p_idx, ids] = True

        # Initial guess for board poses (from online estimation)
        r_board_w_list, t_board_w_list = [], []
        E_c2w_all = extrinsics_matrix(self._rvecs_cam2world, self._tvecs_cam2world)

        for p_idx, entries in enumerate(self.ba_samples):
            E_b2w_votes = [E_c2w_all[c] @ E_b2c for c, E_b2c, _, _ in entries]
            E_stack = jnp.stack(E_b2w_votes, axis=0)
            r_stack, t_stack = extmat_to_rtvecs(E_stack)
            q_stack = axisangle_to_quaternion_batched(r_stack)
            r_board_w_list.append(quaternion_to_axisangle(quaternion_average(q_stack)))
            t_board_w_list.append(jnp.median(t_stack, axis=0))

        # Start with the online estimates
        cam_r_online = np.asarray(self._rvecs_cam2world)
        cam_t_online = np.asarray(self._tvecs_cam2world)
        K_online = np.stack(self._cam_matrices)
        D_online = np.stack(self._dist_coeffs)
        board_r_online = np.asarray(jnp.stack(r_board_w_list))
        board_t_online = np.asarray(jnp.stack(t_board_w_list))

        # self._debug_cam_r = cam_r_online.copy()
        # self._debug_cam_t = cam_t_online.copy()
        # self._debug_K = K_online.copy()
        # self._debug_D = D_online.copy()
        # self._debug_board_r = board_r_online.copy()
        # self._debug_board_t = board_t_online.copy()
        # self._debug_points2d = pts2d_buf.copy()
        # self._debug_pointsids = vis_buf.copy()

        # STAGE 1: Ideal Pinhole World (Shared Intrinsics, simple distortion)
        # ----------------------------
        print("\n" + "=" * 80)
        print(">>> STAGE 1: BA on Ideal World (Shared Intrinsics,  simple distortion)")
        print("=" * 80)
        success_s1, results_s1 = bundle_adjustment.run_bundle_adjustment(
            K_online, D_online, cam_r_online, cam_t_online, board_r_online, board_t_online,
            pts2d_buf, vis_buf,

            self._object_points, self.images_sizes_wh,

            radial_penalty=0.0,   # for fisrst stage we want to consider all points

            shared_intrinsics=True,
            fix_aspect_ratio=True,
            distortion_model='none',    # Fixes distortion params to zero

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
        K_s2_init = results_s1['K_opt']
        D_s2_init = results_s1['D_opt']
        cam_r_s2_init = results_s1['cam_r_opt']
        cam_t_s2_init = results_s1['cam_t_opt']
        board_r_s2_init = results_s1['board_r_opt']
        board_t_s2_init = results_s1['board_t_opt']

        # STAGE 2: Per-camera pinhole world (Per-camera intrinsics, no Distortion)
        # ---------------------------------
        print("\n" + "=" * 80)
        print(">>> STAGE 2: BA on Pinhole World (Per-Camera Intrinsics, No Distortion)")
        print("=" * 80)

        success_s2, results_s2 = bundle_adjustment.run_bundle_adjustment(
            K_s2_init, D_s2_init, cam_r_s2_init, cam_t_s2_init, board_r_s2_init, board_t_s2_init,
            pts2d_buf, vis_buf, self._object_points, self.images_sizes_wh,

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
        K_s3_init = results_s2['K_opt']
        D_s3_init = results_s2['D_opt']
        cam_r_s3_init = results_s2['cam_r_opt']
        cam_t_s3_init = results_s2['cam_t_opt']
        board_r_s3_init = results_s2['board_r_opt']
        board_t_s3_init = results_s2['board_t_opt']

        # STAGE 3: Real world (Full Refinement with Distortion)
        # -------------------
        print("\n" + "=" * 80)
        print(">>> STAGE 3: BA on Real World (Full Refinement with Distortion)")
        print("=" * 80)

        success_s3, final_results = bundle_adjustment.run_bundle_adjustment(
            K_s3_init, D_s3_init, cam_r_s3_init, cam_t_s3_init, board_r_s3_init, board_t_s3_init,
            pts2d_buf, vis_buf, self._object_points, self.images_sizes_wh,

            radial_penalty=4.0,     # now we kinda want to ignore the points far from the working volume

            shared_intrinsics=False,
            fix_aspect_ratio=False,
            # distortion_model='standard',  # Use the 5-parameter model
            distortion_model='full',  # Use the 8-parameter model

            fix_focal_principal=False,
            fix_distortion=False,
            fix_extrinsics=False,
            fix_board_poses=False,

            # priors_weight=0.1
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
            self._points2d, self._visibility_mask = pts2d_buf.copy(), vis_buf.copy()

            self._refined = True
            self.ba_samples.clear()

            volume_of_trust = compute_volume_of_trust(
                *self._refined_intrinsics,
                *self._refined_extrinsics,
                *self._refined_board_poses,
                self._points2d, self._visibility_mask,
                self._object_points,
                error_threshold_px=1.5,
                percentile=1.0      # 99th percentile
            )

            if volume_of_trust:
                print("--- Volume of Trust ---")
                print(f"X range: {volume_of_trust['x'][0]:.2f} to {volume_of_trust['x'][1]:.2f} mm")
                print(f"Y range: {volume_of_trust['y'][0]:.2f} to {volume_of_trust['y'][1]:.2f} mm")
                print(f"Z range: {volume_of_trust['z'][0]:.2f} to {volume_of_trust['z'][1]:.2f} mm")

            return True
        else:
            print("[ERROR] Final BA Stage 3 failed. Aborting calibration.")
            return False

    @property
    def initial_extrinsics(self):
        return np.array(self._rvecs_cam2world), np.array(self._tvecs_cam2world)

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



# TODO: This should be using the new JAXed functions
def compute_volume_of_trust(

        K_opt: np.ndarray,
        D_opt: np.ndarray,
        cam_r_opt: np.ndarray,
        cam_t_opt: np.ndarray,
        board_r_opt: np.ndarray,
        board_t_opt: np.ndarray,

        pts2d_buffer: np.ndarray,
        vis_mask_buffer: np.ndarray,
        object_points_3d: np.ndarray,

        error_threshold_px: float = 1.0,
        percentile: float = 1.0
) -> Optional[Dict[str, Tuple[float, float]]]:

    C, P, N, _ = pts2d_buffer.shape

    E_c2w_all = extrinsics_matrix(cam_r_opt, cam_t_opt)
    E_b2w_all = extrinsics_matrix(board_r_opt, board_t_opt)

    all_errors = np.full((C, P, N), np.nan)

    for c_idx in range(C):

        E_c2w = E_c2w_all[c_idx]
        E_w2c = invert_extrinsics_matrix(E_c2w)

        for p_idx in range(P):

            visible_point_ids = np.where(vis_mask_buffer[c_idx, p_idx, :])[0]
            if len(visible_point_ids) == 0:
                continue

            # Get the board pose for this frame
            E_b2w = E_b2w_all[p_idx]
            E_b2c = E_w2c @ E_b2w

            rvec, tvec = extmat_to_rtvecs(E_b2c)
            rvec, tvec = np.asarray(rvec.squeeze()), np.asarray(tvec.squeeze())

            object_pts_to_project = object_points_3d[visible_point_ids]
            observed_pts_2d = pts2d_buffer[c_idx, p_idx, visible_point_ids]

            # Project the points for this single pose
            reprojected_pts, _ = cv2.projectPoints(
                object_pts_to_project,
                rvec,
                tvec,
                K_opt[c_idx],
                D_opt[c_idx]
            )
            reprojected_pts = reprojected_pts.squeeze(axis=1)
            errors = np.linalg.norm(observed_pts_2d - reprojected_pts, axis=1)
            all_errors[c_idx, p_idx, visible_point_ids] = errors


    # For each instance (a point n in a frame p), calculate its mean error across all cameras that saw it
    mean_error_per_instance = np.nanmean(all_errors, axis=0)
    reliable_instance_mask = mean_error_per_instance < error_threshold_px

    # 3D coordinates of all points in all frames
    obj_pts_hom = np.hstack([object_points_3d, np.ones((N, 1))])
    world_pts_all_frames = np.einsum('pij,nj->pni', E_b2w_all, obj_pts_hom)[:, :, :3]

    # cloud of reliable 3D points only
    reliable_world_pts_cloud = world_pts_all_frames[reliable_instance_mask]

    if reliable_world_pts_cloud.shape[0] < 3:
        print("Warning: Not enough reliable point instances found to define a volume of trust.")
        return None

    lower_p, upper_p = percentile, 100 - percentile

    x_min, x_max = np.percentile(reliable_world_pts_cloud[:, 0], [lower_p, upper_p])
    y_min, y_max = np.percentile(reliable_world_pts_cloud[:, 1], [lower_p, upper_p])
    z_min, z_max = np.percentile(reliable_world_pts_cloud[:, 2], [lower_p, upper_p])

    return {
        'x': (x_min, x_max),
        'y': (y_min, y_max),
        'z': (z_min, z_max)
    }