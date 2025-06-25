from pathlib import Path
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Dict, Tuple, List
import numpy as np
from itertools import product
from functools import lru_cache
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

from mokap.utils import fileio
from mokap.utils.geometry.fitting import bundle_intersection_AABB
from mokap.utils.geometry.projective import undistort_points, back_projection, project_points, project_to_multiple_cameras
from mokap.utils.geometry.transforms import extrinsics_matrix, extmat_to_rtvecs
from mokap.utils.visualisation import plot_cameras_3d, plot_points_3d, CUSTOM_COLORS
from mokap.reconstruction.reconstructor import Reconstructor


class ReconstructorVisualizer:
    """ A (kinda terrible) companion class to the Reconstructor for debugging and visualization """

    def __init__(self, reconstructor: Reconstructor):
        self.r = reconstructor

    def plot_epipolar_segments(
            self,
            dets_i: ArrayLike,
            dets_j: ArrayLike,
            img_j: np.ndarray,
            cam_idx_i: int,
            cam_idx_j: int
    ):
        """ Visualizes epipolar segments from cam_i on the image of cam_j """
        if dets_i.shape[0] == 0 or dets_j.shape[0] == 0:
            print(
                f"Skipping viz between {self.r.camera_names[cam_idx_i]} and {self.r.camera_names[cam_idx_j]}: no detections.")
            return

        h, w = img_j.shape[:2]
        K_j, D_j = self.r.all_K[cam_idx_j], self.r.all_D[cam_idx_j]
        new_K_j, _ = cv2.getOptimalNewCameraMatrix(np.asarray(K_j), np.asarray(D_j), (w, h), 1, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(np.asarray(K_j), np.asarray(D_j), None, new_K_j, (w, h), 5)
        ud_img_j = cv2.remap(img_j, map1, map2, cv2.INTER_LINEAR)
        udets_j = undistort_points(dets_j, K_j, D_j, P=new_K_j)

        # Undistort points from camera i before back-projection
        udets_i = undistort_points(dets_i, self.r.all_K[cam_idx_i], self.r.all_D[cam_idx_i])

        E_c2w_i = jnp.linalg.inv(self.r.all_E[cam_idx_i])
        cam_center_i = E_c2w_i[:3, 3]
        # Now back-project the clean points with no further distortion correction
        p_3d_on_ray = back_projection(udets_i, 1.0, self.r.all_K[cam_idx_i], E_c2w_i, dist_coeffs=None)

        ray_dirs = p_3d_on_ray - cam_center_i
        ray_dirs /= jnp.linalg.norm(ray_dirs, axis=-1, keepdims=True)

        p_near_3d, p_far_3d, has_intersection = bundle_intersection_AABB(cam_center_i, ray_dirs, self.r.aabb_min,
                                                                         self.r.aabb_max)

        rvec_w2c_j, tvec_w2c_j = extmat_to_rtvecs(self.r.all_E[cam_idx_j])
        segments_3d = jnp.vstack([p_near_3d, p_far_3d])
        segments_2d, _ = project_points(segments_3d, rvec_w2c_j, tvec_w2c_j, new_K_j, dist_coeffs=jnp.zeros_like(D_j))

        plt.figure(figsize=(12, 9))
        plt.imshow(ud_img_j)
        plt.title(f"Epipolar Segments from {self.r.camera_names[cam_idx_i]} on {self.r.camera_names[cam_idx_j]}")

        for idx, (p_near, p_far) in enumerate(zip(segments_2d[:len(dets_i)], segments_2d[len(dets_i):])):
            if has_intersection[idx]:
                color = CUSTOM_COLORS[idx % len(CUSTOM_COLORS)]
                plt.plot([p_near[0], p_far[0]], [p_near[1], p_far[1]], color=color, linestyle='-', linewidth=2)
                plt.text(p_near[0], p_near[1], str(idx), color='white',
                         bbox=dict(facecolor=color, alpha=0.7, boxstyle='circle,pad=0.1'))

        for idx, p_j in enumerate(udets_j):
            plt.scatter(p_j[0], p_j[1], c='lime', marker='x', s=100, linewidth=2,
                        label=f"Detections in {self.r.camera_names[cam_idx_j]}")
            plt.text(p_j[0] + 5, p_j[1] + 5, str(idx), color='lime', fontweight='bold')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.xlim(0, w)
        plt.ylim(h, 0)
        plt.show()

    def plot_cameras_rays(self, dets_per_cam: List[ArrayLike]):
        """ Shows camera origins, the Volume of Trust, and 3D rays from 2D detections """

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax = plot_cameras_3d(self.r.rvecs_c2w, self.r.tvecs_c2w, self.r.all_K, self.r.all_D,
                             cameras_names=self.r.camera_names, trust_volume=self.r.volume_bounds, ax=ax)

        E_c2w = extrinsics_matrix(self.r.rvecs_c2w, self.r.tvecs_c2w)
        for c, cam_name in enumerate(self.r.camera_names):
            if dets_per_cam[c].shape[0] == 0: continue
            cam_center = self.r.tvecs_c2w[c]
            points_3d = back_projection(dets_per_cam[c], 10.0, self.r.all_K[c], E_c2w[c], self.r.all_D[c])
            for pt_3d in points_3d:
                ray_end = cam_center + (pt_3d - cam_center) * 30
                ax.plot(*np.stack([cam_center, ray_end]).T, color=CUSTOM_COLORS[c], linestyle='-', linewidth=1.0,
                        alpha=0.7)
        ax.set_title("Ray Casting Sanity Check")
        plt.show()

    def plot_reprojection_and_epilines(self, point3d: jnp.ndarray, detections_for_point: np.ndarray,
                                       all_other_detections: list, images: Dict[str, np.ndarray]):
        """ Comprehensive visualisation for a single 3D point """

        C = self.r.num_cams
        fig, axes = plt.subplots(C, 1, figsize=(10, 8 * C), constrained_layout=True)
        fig.suptitle(f"Reprojection Analysis for Point at {np.round(point3d, 2)}", fontsize=16)
        if C == 1: axes = [axes]

        reproj, _ = project_to_multiple_cameras(point3d[None, :],
                                                self.r.rvecs_w2c,
                                                self.r.tvecs_w2c,
                                                self.r.all_K,
                                                self.r.all_D)
        reproj_pts = np.squeeze(np.array(reproj), axis=1)

        F_mats = {}
        for i, j in product(range(C), repeat=2):
            if i == j: continue
            E_i_inv = jnp.linalg.inv(self.r.all_E[i])
            E_i_to_j = self.r.all_E[j] @ E_i_inv
            R, t = E_i_to_j[:3, :3], E_i_to_j[:3, 3]
            t_skew = jnp.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
            Essential = t_skew @ R
            F_mats[(i, j)] = jnp.linalg.inv(self.r.all_K[j]).T @ Essential @ jnp.linalg.inv(self.r.all_K[i])

        handles, labels = [], []
        for j, cam_name in enumerate(self.r.camera_names):
            ax = axes[j]
            img = images.get(cam_name, np.zeros((1080, 1440, 3), dtype=np.uint8))
            h, w = img.shape[:2]
            ax.imshow(img, cmap='gray')
            ax.set_title(f"View from: {cam_name}")
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)

            ax.scatter([], [], c='yellow', marker='x', s=30, label='Other Detections')
            for other_det in all_other_detections[j]:
                ax.scatter(other_det[0], other_det[1], c='yellow', marker='x', s=30)

            for i in range(C):
                if i == j or np.isnan(detections_for_point[i, 0]): continue
                pt_i_hom = jnp.array([*detections_for_point[i], 1.0])
                epiline = F_mats.get((i, j)) @ pt_i_hom
                x_coords = np.array([0, w])
                y_coords = (-epiline[2] - epiline[0] * x_coords) / epiline[1]
                ax.plot(x_coords, y_coords, color=CUSTOM_COLORS[i], linestyle='--',
                        label=f'Epi from {self.r.camera_names[i]}')

            if not np.isnan(detections_for_point[j, 0]):
                ax.scatter(detections_for_point[j, 0], detections_for_point[j, 1], edgecolor='lime', facecolor='none',
                           marker='o', s=80, linewidth=2, label='Actual Detection')
            ax.scatter(reproj_pts[j, 0], reproj_pts[j, 1], c='red', marker='x', s=50, linewidth=2,
                       label='Reprojected Point')

            if not handles:
                h, l = ax.get_legend_handles_labels()
                by_label = dict(zip(l, h))
                handles, labels = list(by_label.values()), list(by_label.keys())
            ax.set_axis_off()

        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 0.95))
        plt.show()

    def plot_reconstructed_skeletons(self, reconstructed_data: Dict, bones: List[Tuple[str, str]], ax: Axes3D,
                                     color_map: str = 'tab20', bone_color: str = 'gray', **kwargs):
        """ Plots the full reconstructed skeletons using a super basic greedy nearest-neighbor approach for bones """

        keypoint_names = list(reconstructed_data.keys())
        colors = plt.get_cmap(color_map, len(keypoint_names))

        for i, kp_name in enumerate(keypoint_names):
            points, _ = reconstructed_data.get(kp_name, (jnp.empty((0, 3)), np.array([])))
            if points.shape[0] > 0:
                plot_points_3d(points3d=points, color=colors(i), label=kp_name, ax=ax, **kwargs)

        plotted_bone_label = False
        for kp_a, kp_b in bones:
            pts_a, _ = reconstructed_data.get(kp_a, (None, None))
            pts_b, _ = reconstructed_data.get(kp_b, (None, None))
            if pts_a is None or pts_b is None or pts_a.shape[0] == 0 or pts_b.shape[0] == 0: continue

            dist_matrix = cdist(pts_a, pts_b)
            closest_b_indices = np.argmin(dist_matrix, axis=1)
            for i, p_a in enumerate(pts_a):
                p_b = pts_b[closest_b_indices[i]]
                label = "Bones (NN)" if not plotted_bone_label else ""
                ax.plot(*np.array([p_a, p_b]).T, color=bone_color, linewidth=1.0, alpha=0.7, label=label)
                plotted_bone_label = True

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')


def get_frames(videos_paths, cameras_names, frame: int) -> Dict[str, np.ndarray]:
    """ mini helper to load one frame from multiple video files """

    images = {}
    for cam_name in cameras_names:
        file = next((f for f in videos_paths if cam_name in f.name), None)
        if not file: continue
        cap = cv2.VideoCapture(file.as_posix())
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        r, img = cap.read()
        cap.release()
        if r: images[cam_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return images


if __name__ == '__main__':

    folder = Path().home() / 'Desktop' / '3d_ant_data'
    prefix = '240905-1616'
    session = 22
    DEBUG_FRAME = 926
    DEBUG_KEYPOINT = 'thorax'
    DEBUG_CAM_I, DEBUG_CAM_J = 0, 3

    df = fileio.load_session(folder / prefix / 'inputs' / 'tracking', session=session)
    df = df.reorder_levels(['camera', 'track', 'frame']).sort_index()
    cal_data = fileio.read_parameters(folder / prefix / 'calibration')
    keypoints, bones = fileio.load_skeleton_SLEAP(folder / prefix / 'inputs' / 'tracking', indices=False)

    volume_bounds = {'x': (-10.5, 13.0), 'y': (-21.0, 11.0), 'z': (180.0, 201.0)}

    cameras_names = sorted(df.index.get_level_values('camera').unique())
    videos_paths = list((folder / prefix / 'sources').glob(f'*session{session}.mp4'))
    images = get_frames(videos_paths, cameras_names, DEBUG_FRAME)

    reconstructor_config = {'repro_thresh': 10.0, 'cluster_radius': 2.0}
    reconstructor = Reconstructor(camera_parameters=cal_data, volume_bounds=volume_bounds, config=reconstructor_config)
    viz = ReconstructorVisualizer(reconstructor)

    df_frame = df.loc[pd.IndexSlice[:, :, DEBUG_FRAME], :]
    detections_by_keypoint = reconstructor._prepare_data(df_frame, [DEBUG_KEYPOINT])
    dets_per_cam = detections_by_keypoint[DEBUG_KEYPOINT]


    # ==========================================================================
    #  LEVEL 1: RAYS CASTING VISUALIZATION IN 3D
    # ==========================================================================
    print("\n--- LEVEL 1: Visualizing 3D Rays ---")
    viz.plot_cameras_rays(dets_per_cam)


    # ==========================================================================
    #  LEVEL 2: EPIPOLAR SEGMENT VISUALIZATION IN 2D
    # ==========================================================================
    print(f"\n--- LEVEL 2: Visualizing Epipolar Segments ---")
    viz.plot_epipolar_segments(
        dets_i=dets_per_cam[DEBUG_CAM_I],
        dets_j=dets_per_cam[DEBUG_CAM_J],
        img_j=images[cameras_names[DEBUG_CAM_J]],
        cam_idx_i=DEBUG_CAM_I,
        cam_idx_j=DEBUG_CAM_J
    )


    # ==========================================================================
    #  LEVEL 3: HYPOTHESIS & REPROJECTION ANALYSIS
    # ==========================================================================
    print("\n--- LEVEL 3.1: Analyzing Initial Hypotheses (Single-Pass) ---")

    # Get the raw, unfiltered hypotheses by calling the internal generation method
    @lru_cache(maxsize=None)
    def get_cost_mat(i, j):
        return reconstructor._compute_cost_matrix(dets_per_cam[i], dets_per_cam[j], i, j)


    raw_pts, raw_groups, _, raw_errors = reconstructor._generate_hypotheses(dets_per_cam, get_cost_mat)
    print(f"Found {raw_pts.shape[0]} initial point hypotheses.")

    if raw_pts.shape[0] > 0:
        POINT_TO_ANALYZE_IDX = np.argmin(raw_errors)  # analyze the point with the best reprojection error
        point_to_analyze = raw_pts[POINT_TO_ANALYZE_IDX]
        group_for_point = raw_groups[POINT_TO_ANALYZE_IDX]

        # Prepare data for the reprojection plot
        detections_for_point = np.full((reconstructor.num_cams, 2), np.nan)
        used_indices = set(group_for_point)
        for cam_idx, det_idx in group_for_point:
            detections_for_point[cam_idx] = dets_per_cam[cam_idx][det_idx]

        other_detections = []
        for c_idx, dets_in_cam in enumerate(dets_per_cam):
            other_dets_in_cam = [d for d_idx, d in enumerate(dets_in_cam) if (c_idx, d_idx) not in used_indices]
            other_detections.append(np.array(other_dets_in_cam) if other_dets_in_cam else np.empty((0, 2)))

        print(
            f"Visualizing reprojection for point {POINT_TO_ANALYZE_IDX} (error: {raw_errors[POINT_TO_ANALYZE_IDX]:.2f}px)")
        viz.plot_reprojection_and_epilines(point_to_analyze, detections_for_point, other_detections, images)


    # ==========================================================================
    #  LEVEL 4: FINAL RECONSTRUCTION & SKELETON PLOTTING
    # ==========================================================================
    print("\n--- LEVEL 4: Visualizing Final Filtered Reconstruction ---")
    reconstructed_3d = reconstructor.reconstruct_frame(df_frame=df_frame, keypoint_names=keypoints)

    total_points = sum(points.shape[0] for points, confs in reconstructed_3d.values())
    print(f"Final reconstruction found {total_points} total points across all keypoints.")

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax = plot_cameras_3d(reconstructor.rvecs_c2w, reconstructor.tvecs_c2w, reconstructor.all_K, reconstructor.all_D,
                         cameras_names=cameras_names,
                         trust_volume=volume_bounds,
                         ax=ax)

    viz.plot_reconstructed_skeletons(reconstructed_3d, bones, ax=ax)
    ax.set_title(f"Final Reconstructed Skeletons for Frame {DEBUG_FRAME}")
    plt.show()