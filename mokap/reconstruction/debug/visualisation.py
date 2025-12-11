import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cv2
from scipy.spatial.distance import cdist
from collections import defaultdict
from typing import List, Tuple
from mokap.reconstruction.datatypes import SoupData
from lucida.geometry.backend import xp
from lucida.geometry import undistort, unproject, project
from lucida.geometry import compose_transform_matrix, decompose_transform_matrix
from lucida.geometry import intersect_aabb
from lucida.visualisation import draw_cameras, CUSTOM_COLORS


class ReconstructorVisualizer:
    """Helper class for visualizing geometric reconstruction steps."""

    def __init__(self, reconstructor):
        self.r = reconstructor

    def plot_cameras_rays(self, dets_per_cam: List[np.ndarray]):
        """Visualises 3D rays cast from 2D detections."""

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        draw_cameras(self.r.rig, depths = 1, ax=ax)

        E_c2w = compose_transform_matrix(self.r.rvecs_c2w, self.r.tvecs_c2w)

        for c, cam_name in enumerate(self.r.camera_names):
            if dets_per_cam[c].shape[0] == 0: continue

            cam_center = self.r.tvecs_c2w[c]

            # Back project to get points on the rays far away (z=1000)
            points_3d = unproject(dets_per_cam[c], 1000.0, self.r.K[c], E_c2w[c], self.r.D[c])

            for pt_3d in points_3d:
                # Draw line from camera center to point
                ax.plot(*np.stack([np.array(cam_center), np.array(pt_3d)]).T, color=CUSTOM_COLORS[c % len(CUSTOM_COLORS)],
                        linestyle='-', linewidth=0.5, alpha=0.5)

        ax.set_title("Ray Casting Sanity Check")
        plt.show()

    def plot_epipolar_segments(self, dets_i, dets_j, img_j, cam_idx_i, cam_idx_j):
        """Visualizes epipolar segments from cam_i projected onto cam_j."""

        if len(dets_i) == 0:
            return

        h, w = img_j.shape[:2]
        K_j, D_j = self.r.rig.K[cam_idx_j], self.r.rig.D[cam_idx_j]

        # Undistort image
        new_K_j, _ = cv2.getOptimalNewCameraMatrix(np.asarray(K_j), np.asarray(D_j), (w, h), 1, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(np.asarray(K_j), np.asarray(D_j), None, new_K_j, (w, h), 5)
        ud_img_j = cv2.remap(img_j, map1, map2, cv2.INTER_LINEAR)

        # Undistort target detections to match the remapped image
        udets_j = None
        if len(dets_j) > 0:
            udets_j = undistort(dets_j, K_j, D_j, P=new_K_j)

        # Backproject rays from i -> project to j
        udets_i = undistort(dets_i, self.r.rig.K[cam_idx_i], self.r.rig.D[cam_idx_i])
        E_c2w_i = xp.linalg.inv(self.r.rig.T_w2c[cam_idx_i])
        cam_center_i = E_c2w_i[:3, 3]

        p_3d_ray = unproject(udets_i, 1.0, self.r.rig.K[cam_idx_i], E_c2w_i, D=None)
        ray_dirs = p_3d_ray - cam_center_i
        ray_dirs /= xp.linalg.norm(ray_dirs, axis=-1, keepdims=True)

        # Intersect with volume AABB
        p_near, p_far, hit = intersect_aabb(cam_center_i, ray_dirs, self.r.aabb_min, self.r.aabb_max)

        # Project segments to cam j
        T_w2c_j = self.r.rig.T_w2c[cam_idx_j]
        segments_3d = xp.vstack([p_near, p_far])
        segments_2d, _ = project(segments_3d, T_w2c_j, new_K_j, D=xp.zeros_like(D_j))

        # Plot
        plt.figure(figsize=(12, 9))
        plt.imshow(ud_img_j)
        plt.title(f"Epipolar Segments: {self.r.rig[cam_idx_i].name} -> {self.r.rig[cam_idx_j].name}")

        # Draw segments (projected rays from camera i)
        n = len(dets_i)
        for idx in range(n):
            if hit[idx]:
                start, end = segments_2d[idx], segments_2d[idx + n]
                color = CUSTOM_COLORS[idx % len(CUSTOM_COLORS)]
                plt.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2)
                plt.text(start[0], start[1], str(idx), color='white',
                         bbox=dict(facecolor=color, alpha=0.7, boxstyle='circle,pad=0.1'))

        # Draw target detections (points in camera j)
        if udets_j is not None:
            plt.scatter(udets_j[:, 0], udets_j[:, 1], c='lime', marker='x', s=80, linewidth=2, label='Detections J')
            for i, pt in enumerate(udets_j):
                plt.text(pt[0] + 5, pt[1] + 5, str(i), color='lime', fontweight='bold')

        plt.legend(loc='upper right')
        plt.xlim(0, w)
        plt.ylim(h, 0)
        plt.show()

    def plot_reprojection(self, point3d, group_indices, all_dets, images):
        """Visualises a specific 3D hypothesis reprojected onto all views."""

        C = self.r.num_cams
        reproj, _ = project_to_multiple_cameras(point3d[None, :], self.r.rvecs_w2c, self.r.tvecs_w2c, self.r.K,
                                                self.r.D)
        reproj_pts = np.squeeze(np.array(reproj), axis=1)

        fig, axes = plt.subplots(1, C, figsize=(5 * C, 5))
        if C == 1: axes = [axes]

        used_map = {cam_idx: det_idx for cam_idx, det_idx in group_indices}

        for j, ax in enumerate(axes):
            cam_name = self.r.camera_names[j]
            img = images.get(cam_name, np.zeros((100, 100, 3), dtype=np.uint8))
            ax.imshow(img)
            ax.set_title(cam_name)

            # Plot all detections faintly
            if len(all_dets[j]) > 0:
                ax.scatter(all_dets[j][:, 0], all_dets[j][:, 1], c='yellow', marker='x', alpha=0.5, label='Other')

            # Plot used detection
            if j in used_map:
                u_idx = used_map[j]
                det = all_dets[j][u_idx]
                ax.scatter(det[0], det[1], facecolors='none', edgecolors='lime', s=80, lw=2, label='Used')

            # Plot reprojection
            ax.scatter(reproj_pts[j, 0], reproj_pts[j, 1], c='red', marker='+', s=100, lw=2, label='Reproj')
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_reconstructed_frame(self, soup: SoupData, bones: List[Tuple[str, str]], ax=None):
        """Plots the final reconstructed soup for a frame."""

        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

        draw_cameras(self.r.rig, depths=1, ax=ax)

        if soup.num_points == 0: return ax

        # Group points by KP
        kp_dict = defaultdict(list)
        for i in range(soup.num_points):
            name = soup.keypoint_names[soup.kp_types[i]]
            kp_dict[name].append(soup.positions[i])

        colors = plt.get_cmap('tab20', len(soup.keypoint_names))

        # Plot points
        for i, name in enumerate(soup.keypoint_names):
            if name in kp_dict:
                pts = np.array(np.stack(kp_dict[name]))
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=colors(i), s=20, label=name)

        # Plot bones (greedy Nearest Neighbor)
        for kp1, kp2 in bones:
            if kp1 in kp_dict and kp2 in kp_dict:
                pts1, pts2 = np.stack(kp_dict[kp1]), np.stack(kp_dict[kp2])

                # Find closest pairs
                dists = cdist(pts1, pts2)
                matches = np.argmin(dists, axis=1)
                for idx1, idx2 in enumerate(matches):
                    p1, p2 = pts1[idx1], pts2[idx2]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray', alpha=0.5)

        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
        return ax


# Tracklet / Sequence viewer

def convert_track_centric_to_frame_centric(track_data: dict) -> list:
    """Converts {track_idx: [skeletons]} -> sorted list of frame dicts."""

    frames = defaultdict(lambda: {'skeletons': []})

    for tid, skels in track_data.items():
        for s in skels:
            s['track_idx'] = tid  # ensure track ID is embedded
            frames[s['frame_idx']]['skeletons'].append(s)
            frames[s['frame_idx']]['frame_idx'] = s['frame_idx']

    if not frames:
        return []
    mn, mx = min(frames), max(frames)
    return [frames.get(i, {'frame_idx': i, 'skeletons': []}) for i in range(mn, mx + 1)]


def draw_skeletons_3d(frame_data, bones, ax):
    """Draws skeletons for the interactive viewer."""

    artists = []
    cmap = plt.get_cmap('tab20', 20)

    for skel in frame_data.get('skeletons', []):
        kps = skel.get('keypoints_smoothed') or skel.get('keypoints')
        if not kps:
            continue

        tid = skel.get('track_idx', -1)
        color = cmap(tid % 20) if tid >= 0 else 'gray'

        # Bones
        for k1, k2 in bones:
            if k1 in kps and k2 in kps:
                p1, p2 = np.array(kps[k1]), np.array(kps[k2])
                l, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=color, lw=2)
                artists.append(l)

        # ID Text
        root = next(iter(kps.values()))
        t = ax.text(root[0], root[1], root[2], str(tid), color=color)
        artists.append(t)

    return artists


def view_soup_frame(soup: SoupData, frame_idx: int):
    f_slice = soup.get_frame_slice(frame_idx)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot real points
    if f_slice.num_points > 0:
        ax.scatter(f_slice.positions[:, 0], f_slice.positions[:, 1], f_slice.positions[:, 2],
                   c='blue', s=20, label='Triangulated')

    # Plot orphan rays
    if len(f_slice.ray_origins) > 0:
        # Just plot the first 50 to avoid clutter
        for i in range(min(50, len(f_slice.ray_origins))):
            o = f_slice.ray_origins[i]
            d = f_slice.ray_directions[i]
            # Draw a line 250 mm long
            end = o + d * 250
            ax.plot([o[0], end[0]], [o[1], end[1]], [o[2], end[2]], 'r-', alpha=0.3)

    ax.set_title(f"Frame {frame_idx} (Blue=Pts, Red=Rays)")
    plt.show()


def run_sequence_viewer(frame_data_list, bones, bounds):
    """Runs the interactive matplotlib 3D slider viewer."""

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_axes((0.0, 0.1, 1.0, 0.9), projection='3d')
    slider_ax = fig.add_axes((0.2, 0.02, 0.6, 0.03))

    ax.set_box_aspect([
        bounds['x'][1] - bounds['x'][0],
        bounds['y'][1] - bounds['y'][0],
        bounds['z'][1] - bounds['z'][0]
    ])
    ax.set_xlim(*bounds['x'])
    ax.set_ylim(*bounds['y'])
    ax.set_zlim(*bounds['z'])

    slider = Slider(slider_ax, 'Frame', 0, len(frame_data_list) - 1, valinit=0, valstep=1)

    current_art = []

    def update(val):
        nonlocal current_art
        for a in current_art:
            a.remove()
        current_art.clear()

        idx = int(slider.val)
        fdata = frame_data_list[idx]
        current_art = draw_skeletons_3d(fdata, bones, ax)
        ax.set_title(f"Frame {fdata['frame_idx']}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()