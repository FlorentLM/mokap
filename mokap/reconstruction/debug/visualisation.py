import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import cv2
from scipy.spatial.distance import cdist
from collections import defaultdict
from typing import List, Tuple
from mokap.reconstruction.datatypes import SoupData
from lucida.geometry.backend import xp
from lucida.geometry import intersect_aabb, project
from lucida.visualisation import _init_plot, draw_cameras, draw_points, CUSTOM_COLORS


def plot_cameras_rays(reconstructor, dets_per_cam: List[np.ndarray]):
    """Visualises 3D rays cast from 2D detections."""

    ax = _init_plot(figsize=(12, 12))

    draw_cameras(reconstructor.rig, ax=ax, title="Ray Casting Sanity Check")

    # Draw rays manually (ragged data structure doesn't fit draw_rays)
    for c, cam in enumerate(reconstructor.rig):
        dets = dets_per_cam[c]
        if len(dets) == 0:
            continue

        origins, directions = cam.raycast(dets)

        # Create segments: Origin -> Origin + direction * length
        starts = np.broadcast_to(origins, directions.shape)
        ends = starts + directions * 1000.0  # 1000 units length

        segments = np.stack([starts, ends], axis=1)

        col = CUSTOM_COLORS[c % len(CUSTOM_COLORS)]
        lc = Line3DCollection(segments, colors=col, linewidths=0.5, alpha=0.5)
        ax.add_collection3d(lc)

    plt.show()


def plot_epipolar_segments(reconstructor, dets_i, dets_j, img_j, cam_idx_i, cam_idx_j):
    """Visualizes epipolar segments from cam_i projected onto cam_j."""

    if len(dets_i) == 0:
        return

    cam_i = reconstructor.rig[cam_idx_i]
    cam_j = reconstructor.rig[cam_idx_j]
    h, w = img_j.shape[:2]

    # Undistort image J
    K_j_np = np.array(cam_j.K)
    D_j_np = np.array(cam_j.D)
    new_K_j, _ = cv2.getOptimalNewCameraMatrix(K_j_np, D_j_np, (w, h), 1, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K_j_np, D_j_np, None, new_K_j, (w, h), 5)
    ud_img_j = cv2.remap(img_j, map1, map2, cv2.INTER_LINEAR)

    # Undistort detections on J to match the remapped image
    udets_j = None
    if len(dets_j) > 0:
        udets_j = cv2.undistortPoints(dets_j.reshape(-1, 1, 2), K_j_np, D_j_np, P=new_K_j).reshape(-1, 2)

    # Raycast from I and Intersect AABB
    origins, directions = cam_i.raycast(dets_i)
    p_near, p_far, hit = intersect_aabb(origins, directions, reconstructor.aabb_min, reconstructor.aabb_max)

    # Project segments to J using the "New K" (undistorted view)
    segments_3d = xp.concatenate([p_near, p_far], axis=0)  # (2N, 3)

    # Using Lucida project with 'none' distortion model since image is rectified
    segments_2d, _ = project(
        segments_3d,
        cam_j.T,
        xp.asarray(new_K_j),
        xp.zeros_like(cam_j.D),
        distortion_model='none'
    )

    # Plot
    plt.figure(figsize=(12, 9))
    plt.imshow(ud_img_j)
    plt.title(f"Epipolar Segments: {cam_i.name} -> {cam_j.name}")

    n = len(dets_i)
    for idx in range(n):
        if hit[idx]:
            start, end = segments_2d[idx], segments_2d[idx + n]
            color = CUSTOM_COLORS[idx % len(CUSTOM_COLORS)]
            plt.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2)
            plt.text(start[0], start[1], str(idx), color='white',
                     bbox=dict(facecolor=color, alpha=0.7, boxstyle='circle,pad=0.1'))

    if udets_j is not None:
        plt.scatter(udets_j[:, 0], udets_j[:, 1], c='lime', marker='x', s=80, linewidth=2, label='Detections J')
        for i, pt in enumerate(udets_j):
            plt.text(pt[0] + 5, pt[1] + 5, str(i), color='lime', fontweight='bold')

    plt.legend(loc='upper right')
    plt.xlim(0, w)
    plt.ylim(h, 0)
    plt.show()

def plot_reprojection(reconstructor, point3d, group_indices, all_dets, images):
    """Visualises a specific 3D hypothesis reprojected onto all views."""

    C = len(reconstructor.rig)

    reproj, mask = reconstructor.rig.project(point3d)
    reproj[~mask.astype(bool)] = np.nan
    reproj_pts = reproj.squeeze()

    fig, axes = plt.subplots(1, C, figsize=(5 * C, 5))
    if C == 1:
        axes = [axes]

    used_map = {cam_idx: det_idx for cam_idx, det_idx in group_indices}

    for j, ax in enumerate(axes):
        cam = reconstructor.rig[j]
        img = images.get(cam.name, np.zeros((100, 100, 3), dtype=np.uint8))
        ax.imshow(img)
        ax.set_title(cam.name)

        if len(all_dets[j]) > 0:
            ax.scatter(all_dets[j][:, 0], all_dets[j][:, 1], c='yellow', marker='x', alpha=0.5, label='Other')

        if j in used_map:
            u_idx = used_map[j]
            det = all_dets[j][u_idx]
            ax.scatter(det[0], det[1], facecolors='none', edgecolors='lime', s=80, lw=2, label='Used')

        ax.scatter(reproj_pts[j, 0], reproj_pts[j, 1], c='red', marker='+', s=100, lw=2, label='Reproj')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_reconstructed_frame(reconstructor, soup: SoupData, bones: List[Tuple[str, str]], ax=None):
    """Plots the final reconstructed soup for a frame."""

    ax = _init_plot(ax, figsize=(10, 10))
    draw_cameras(reconstructor.rig, ax=ax, title="Reconstructed Soup")

    if soup.num_points == 0:
        return ax

    # Group by KP for consistent coloring
    kp_dict = defaultdict(list)
    for i in range(soup.num_points):
        name = soup.keypoint_names[soup.kp_types[i]]
        kp_dict[name].append(soup.positions[i])

    colors = plt.get_cmap('tab20', len(soup.keypoint_names))

    for i, name in enumerate(soup.keypoint_names):
        if name in kp_dict:
            pts = np.array(np.stack(kp_dict[name]))
            draw_points(pts, default_color=colors(i), ax=ax)
            # Proxy artist for legend
            ax.scatter([], [], [], color=colors(i), label=name)

    # Plot bones
    for kp1, kp2 in bones:
        if kp1 in kp_dict and kp2 in kp_dict:
            pts1, pts2 = np.stack(kp_dict[kp1]), np.stack(kp_dict[kp2])
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
            s['track_idx'] = tid
            frames[s['frame_idx']]['skeletons'].append(s)
            frames[s['frame_idx']]['frame_idx'] = s['frame_idx']

    if not frames: return []
    mn, mx = min(frames), max(frames)
    return [frames.get(i, {'frame_idx': i, 'skeletons': []}) for i in range(mn, mx + 1)]


def draw_skeletons_3d(frame_data, bones, ax):
    """Draws skeletons for the interactive viewer."""
    artists = []
    cmap = plt.get_cmap('tab20', 20)

    for skel in frame_data.get('skeletons', []):
        kps = skel.get('keypoints_smoothed') or skel.get('keypoints')
        if not kps: continue

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


def view_soup_frame(soup: SoupData, frame_idx: int, rig=None):
    f_slice = soup.get_frame_slice(frame_idx)

    ax = _init_plot(figsize=(10, 10))
    if rig:
        draw_cameras(rig, ax=ax)

    # Plot real points
    if f_slice.num_points > 0:
        draw_points(f_slice.positions, default_color='blue', ax=ax, title=f"Frame {frame_idx}")

    # Plot orphan rays
    if len(f_slice.ray_origins) > 0:
        n_rays = min(50, len(f_slice.ray_origins))
        starts = f_slice.ray_origins[:n_rays]
        directions = f_slice.ray_directions[:n_rays]

        ends = starts + directions * 250.0

        segments = np.stack([starts, ends], axis=1)
        lc = Line3DCollection(segments, colors='red', linewidths=0.5, alpha=0.3)
        ax.add_collection3d(lc)

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
        for a in current_art: a.remove()
        current_art.clear()
        idx = int(slider.val)
        fdata = frame_data_list[idx]
        current_art = draw_skeletons_3d(fdata, bones, ax)
        ax.set_title(f"Frame {fdata['frame_idx']}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()