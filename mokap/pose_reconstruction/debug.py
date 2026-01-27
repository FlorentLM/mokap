from collections import defaultdict
from typing import Dict, List, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

from mokap.pose_reconstruction.skeleton import SkeletonTopology
from mokap.pose_reconstruction.datatypes import PointSoup, SkeletonHypothesis, Tracklet


class TrackletViewer:
    """
    Interactive 3D viewer for visualising assembled poses and tracklets.

    Args:
        soup: PointSoup containing the raw 3D points
        data: Can be one of two formats:
              1. A Dict[int, List[dict]] mapping track_id -> list of records
              2. A Dict[int, List[SkeletonHypothesis]] mapping frame_idx -> list of hypotheses
        skeleton: Skeleton topology (used for drawing bones)
        view_radius: Radius around centroid for axis limits (default: 15)
    """

    def __init__(
            self,
            soup: PointSoup,
            data: Union[Dict[int, List[dict]], Dict[int, List[SkeletonHypothesis]]],
            skeleton: SkeletonTopology,
            view_radius: float = 15.0
    ):
        self.soup = soup
        self.skeleton = skeleton
        self.view_radius = view_radius

        self.objects_by_frame = self._organize_data_by_frame(data)

        # Get sorted frames that actually exist in the data or soup
        soup_frames = set(self.soup.frame_indices)
        data_frames = set(self.objects_by_frame.keys())
        self.frames = sorted(list(soup_frames | data_frames))

        if not self.frames:
            print("No data to show.")
            return

        # Setup
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

    def _organize_data_by_frame(self, data: Any) -> Dict[int, List[Any]]:
        """
        Pivots the input data to be Dict[frame_idx, List[Object]].
        """
        organized = defaultdict(list)

        if not data:
            return organized

        # Check the type of the first value to determine structure
        first_val = next(iter(data.values()))
        if not first_val:
            return organized

        # Case: input is records_by_track (Dict[track_id, List[dict]])
        if isinstance(first_val[0], dict):
            for track_id, records in data.items():
                for rec in records:
                    frame = rec['frame_idx']
                    organized[frame].append(rec)

        # Case: input is already frame-based (Dict[frame, List[Hypothesis/Tracklet]])
        else:
            return data

        return organized

    def _on_slider_change(self, val):
        frame_idx = int(self.slider.val)
        self._draw_frame(frame_idx)

    def _draw_frame(self, frame_list_idx: int):

        self.ax.clear()

        # Handle out of bounds slider (though unlikely)
        if frame_list_idx >= len(self.frames):
            return

        frame_idx = self.frames[frame_list_idx]
        self.ax.set_title(f"Frame {frame_idx}")

        # Draw soup points
        self._draw_soup(frame_idx)

        # Draw assembled skeletons/tracklets
        objects = self.objects_by_frame.get(frame_idx, [])
        self._draw_all_objects(objects)

        # Update view limits based on the first object found
        if objects:
            self._set_view_limits(objects[0])
        else:
            # fallback to soup centroid if no objects
            try:
                frame_soup = self.soup[frame_idx]
                if frame_soup.nb_points > 0:
                    centroid = np.mean(frame_soup.positions, axis=0)
                    self._set_limits_around(centroid)
            except Exception:
                pass

        self.fig.canvas.draw_idle()

    def _draw_soup(self, frame_idx: int):

        try:
            frame_soup = self.soup[frame_idx]

            if frame_soup.nb_points > 0:
                pos = frame_soup.positions
                self.ax.scatter(
                    pos[:, 0], pos[:, 1], pos[:, 2],
                    c='k', alpha=0.075, s=15, linewidths=0, depthshade=False
                )
        except Exception:
            pass

    def _draw_all_objects(self, objects: List[Any]):

        if not objects:
            return

        cmap = plt.cm.tab10

        for obj in objects:
            data = self._extract_pose_data(obj)
            if not data:
                continue

            kps, scale, track_id, point_indices = data

            # Colour based on track ID (or random if no ID)
            color_idx = track_id % 10
            color = cmap(color_idx)

            self._draw_skeleton(kps, scale, point_indices, color, track_id)

    def _draw_skeleton(self, keypoints, scale, point_indices, color, label_id):

        # Draw nodes
        xyz = np.array(list(keypoints.values()))

        # Split into real and virtual for styling
        real_kps = []
        virt_kps = []

        for name, pos in keypoints.items():
            idx = point_indices.get(name, 0)
            if idx < 0:
                virt_kps.append(pos)
            else:
                real_kps.append(pos)

        if real_kps:
            rk = np.array(real_kps)
            # Real: round markers
            self.ax.scatter(rk[:, 0], rk[:, 1], rk[:, 2], color=color, s=40, edgecolors='w', alpha=1.0)

        if virt_kps:
            vk = np.array(virt_kps)
            # Virtual: triangles
            self.ax.scatter(vk[:, 0], vk[:, 1], vk[:, 2], color=color, s=30, marker='^', edgecolors='w', alpha=0.7)

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

        # Label (centroid)
        # TODO: (re)add scale maybe
        centroid = np.mean(xyz, axis=0)
        label_text = f"ID:{label_id}"
        if scale != 1.0:
            label_text += f"\nS:{scale:.2f}"

        self.ax.text(
            centroid[0], centroid[1], centroid[2],
            label_text,
            color='k', fontsize=9, weight='bold',
            bbox=dict(facecolor='w', alpha=0.5, edgecolor='none', pad=1)
        )

    def _extract_pose_data(self, obj: Any):

        # Record Dictionary (from tracker.collect_records)
        if isinstance(obj, dict):
            return (
                obj.get('keypoints', {}),
                obj.get('scale', 1.0),
                obj.get('track_idx', 0),
                obj.get('point_indices', {})
            )

        # Tracklet
        if isinstance(obj, Tracklet):
            return (
                obj.hypothesis.positions,
                obj.estimated_scale,
                obj.track_idx,
                {n.name: n.idx for n in obj.hypothesis}
            )

        # SkeletonHypothesis (no ID)
        if isinstance(obj, SkeletonHypothesis):
            return (
                obj.positions,
                obj.scale,
                0,  # Default ID
                {n.name: n.idx for n in obj.nodes}
            )

        return None

    def _set_view_limits(self, obj: Any):

        data = self._extract_pose_data(obj)
        if not data:
            return

        kps = data[0]
        if not kps:
            return

        xyz = np.array(list(kps.values()))
        centroid = np.mean(xyz, axis=0)
        self._set_limits_around(centroid)

    def _set_limits_around(self, center: np.ndarray):
        r = self.view_radius
        self.ax.set_xlim(center[0] - r, center[0] + r)
        self.ax.set_ylim(center[1] - r, center[1] + r)
        self.ax.set_zlim(center[2] - r, center[2] + r)