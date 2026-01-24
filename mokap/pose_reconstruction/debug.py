from typing import Dict, List, Union, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mokap.pose_reconstruction.skeleton import Skeleton, Bone
from mokap.pose_reconstruction.datatypes import Pose3D, PointSoup
from mpl_toolkits.mplot3d import Axes3D


class SkeletonViewer:
    """
    Interactive 3D viewer for visualising assembled poses.

    Displays point soup and skeleton poses frame-by-frame with a slider.

    Args:
        soup: PointSoup containing the raw 3D points
        poses_by_frame: Dict mapping frame index to list of Pose3D (or pose-like objects)
        skeleton: Skeleton topology (used for drawing bones)
        view_radius: Radius around centroid for axis limits (default: 15)
    """

    def __init__(
            self,
            soup: PointSoup,
            poses_by_frame: Dict[int, List[Union[Pose3D, object]]],
            skeleton: Skeleton,
            view_radius: float = 15.0
    ):
        self.soup = soup
        self.skeleton = skeleton
        self.poses_by_frame = poses_by_frame
        self.view_radius = view_radius

        self.frames = sorted(self.poses_by_frame.keys())

        if not self.frames:
            print("No poses to show.")
            return

        # Create figure
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.1)

        # Frame slider
        slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])
        self.slider = Slider(
            slider_ax,
            'Frame',
            0,
            len(self.frames) - 1,
            valinit=0,
            valfmt='%d'
        )
        self.slider.on_changed(self._on_slider_change)

        # Initial draw
        self._draw_frame(0)
        plt.show()

    def _on_slider_change(self, val):
        """Handle slider value change."""
        self._draw_frame(int(self.slider.val))

    def _draw_frame(self, frame_list_idx: int):
        """Draw a single frame."""
        self.ax.clear()

        frame_idx = self.frames[frame_list_idx]
        self.ax.set_title(f"Frame {frame_idx}")

        # Draw soup points
        self._draw_soup(frame_idx)

        # Draw poses
        poses = self.poses_by_frame[frame_idx]
        self._draw_poses(poses)

        # Update view limits
        if poses:
            self._set_view_limits(poses[0])

        self.fig.canvas.draw_idle()

    def _draw_soup(self, frame_idx: int):
        """Draw point soup for the given frame."""
        try:
            frame_soup = self.soup[frame_idx]
            if frame_soup.nb_points > 0:
                pos = frame_soup.positions
                self.ax.scatter(
                    pos[:, 0], pos[:, 1], pos[:, 2],
                    c='k', alpha=0.1, s=5
                )
        except Exception:
            pass

    def _draw_poses(self, poses: List[Union[Pose3D, object]]):
        """Draw all poses for the current frame."""
        if not poses:
            return

        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(poses))))

        for i, pose in enumerate(poses):
            keypoints = self._get_keypoints(pose)
            if not keypoints:
                continue

            color = colors[i]

            # Draw keypoints
            xyz = np.array(list(keypoints.values()))
            self.ax.scatter(
                xyz[:, 0], xyz[:, 1], xyz[:, 2],
                color=color, s=50, edgecolors='w'
            )

            # Draw bones
            for bone in self.skeleton.bones:
                if bone.k1 in keypoints and bone.k2 in keypoints:
                    p1, p2 = keypoints[bone.k1], keypoints[bone.k2]
                    self.ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        color=color, lw=2
                    )

            # Draw scale label
            scale = self._get_scale(pose)
            centroid = np.nanmean(xyz, axis=0)
            self.ax.text(
                centroid[0], centroid[1], centroid[2],
                f"S:{scale:.2f}",
                color='k', fontsize=8, weight='bold'
            )

    def _get_keypoints(self, pose: Union[Pose3D, object]) -> Optional[Dict[str, np.ndarray]]:
        """Extract keypoints dict from pose or pose-like object."""
        if hasattr(pose, 'keypoints'):
            return pose.keypoints
        return None

    def _get_scale(self, pose: Union[Pose3D, object]) -> float:
        """Extract scale from pose or pose-like object."""
        if hasattr(pose, 'scale'):
            return pose.scale
        return 1.0

    def _set_view_limits(self, pose: Union[Pose3D, object]):
        """Set axis limits centered on the first pose."""
        keypoints = self._get_keypoints(pose)
        if not keypoints:
            return

        xyz = np.array(list(keypoints.values()))
        centroid = np.mean(xyz, axis=0)
        r = self.view_radius

        self.ax.set_xlim(centroid[0] - r, centroid[0] + r)
        self.ax.set_ylim(centroid[1] - r, centroid[1] + r)
        self.ax.set_zlim(centroid[2] - r, centroid[2] + r)


class TrackletViewer:
    """
    Interactive viewer for visualizing tracklets over time.

    Shows trajectory trails and current pose for each tracked individual.

    Args:
        soup: PointSoup containing the raw 3D points
        tracklets_by_id: Dict mapping track_id to list of pose dicts
        skeleton: Skeleton topology
        trail_length: Number of frames to show in trajectory trail
        view_radius: Radius around centroid for axis limits
    """

    def __init__(
            self,
            soup: PointSoup,
            tracklets_by_id: Dict[int, List[dict]],
            skeleton: Skeleton,
            trail_length: int = 30,
            view_radius: float = 20.0
    ):
        self.soup = soup
        self.skeleton = skeleton
        self.tracklets_by_id = tracklets_by_id
        self.trail_length = trail_length
        self.view_radius = view_radius

        # Build frame index
        self.poses_by_frame: Dict[int, List[dict]] = {}
        for track_id, history in tracklets_by_id.items():
            for pose_dict in history:
                frame_idx = pose_dict.get('frame_idx', -1)
                if frame_idx >= 0:
                    if frame_idx not in self.poses_by_frame:
                        self.poses_by_frame[frame_idx] = []
                    pose_dict['_track_id'] = track_id
                    self.poses_by_frame[frame_idx].append(pose_dict)

        self.frames = sorted(self.poses_by_frame.keys())

        if not self.frames:
            print("No tracklets to show.")
            return

        # Create figure
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.1)

        # Frame slider
        slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])
        self.slider = Slider(
            slider_ax,
            'Frame',
            0,
            len(self.frames) - 1,
            valinit=0,
            valfmt='%d'
        )
        self.slider.on_changed(self._on_slider_change)

        # Color map for tracks
        all_track_ids = sorted(tracklets_by_id.keys())
        self.track_colors = {
            tid: plt.cm.tab20(i % 20)
            for i, tid in enumerate(all_track_ids)
        }

        self._draw_frame(0)
        plt.show()

    def _on_slider_change(self, val):
        self._draw_frame(int(self.slider.val))

    def _draw_frame(self, frame_list_idx: int):
        self.ax.clear()

        frame_idx = self.frames[frame_list_idx]
        self.ax.set_title(f"Frame {frame_idx}")

        # Draw soup
        try:
            frame_soup = self.soup[frame_idx]
            if frame_soup.nb_points > 0:
                pos = frame_soup.positions
                self.ax.scatter(
                    pos[:, 0], pos[:, 1], pos[:, 2],
                    c='lightgray', alpha=0.3, s=3
                )
        except Exception:
            pass

        # Draw tracklet trails and current poses
        centroids = []
        for track_id, history in self.tracklets_by_id.items():
            color = self.track_colors.get(track_id, 'blue')

            # Find poses up to current frame
            relevant = [p for p in history if p.get('frame_idx', -1) <= frame_idx]
            if not relevant:
                continue

            # Draw trail (centroids of recent poses)
            trail_poses = relevant[-self.trail_length:]
            trail_points = []
            for pose_dict in trail_poses:
                kps = pose_dict.get('keypoints', {})
                if kps:
                    cent = np.mean(list(kps.values()), axis=0)
                    trail_points.append(cent)

            if len(trail_points) > 1:
                trail_arr = np.array(trail_points)
                self.ax.plot(
                    trail_arr[:, 0], trail_arr[:, 1], trail_arr[:, 2],
                    color=color, alpha=0.5, lw=1
                )

            # Draw current pose if it's this frame
            current = relevant[-1]
            if current.get('frame_idx') == frame_idx:
                kps = current.get('keypoints', {})
                if kps:
                    xyz = np.array(list(kps.values()))
                    centroids.append(np.mean(xyz, axis=0))

                    # Draw keypoints
                    self.ax.scatter(
                        xyz[:, 0], xyz[:, 1], xyz[:, 2],
                        color=color, s=40, edgecolors='w', linewidths=0.5
                    )

                    # Draw bones
                    for bone in self.skeleton.bones:
                        if bone.k1 in kps and bone.k2 in kps:
                            p1, p2 = kps[bone.k1], kps[bone.k2]
                            self.ax.plot(
                                [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                color=color, lw=2
                            )

                    # Label
                    cent = np.mean(xyz, axis=0)
                    self.ax.text(
                        cent[0], cent[1], cent[2] + 1,
                        f"T{track_id}",
                        color=color, fontsize=8, weight='bold'
                    )

        # Set view limits
        if centroids:
            center = np.mean(centroids, axis=0)
            r = self.view_radius
            self.ax.set_xlim(center[0] - r, center[0] + r)
            self.ax.set_ylim(center[1] - r, center[1] + r)
            self.ax.set_zlim(center[2] - r, center[2] + r)

        self.fig.canvas.draw_idle()