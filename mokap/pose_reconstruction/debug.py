from collections import defaultdict
from typing import Dict, List, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import polars as pl
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass

from mokap.pose_reconstruction.skeleton import Skeleton
from mokap.pose_reconstruction.datatypes import PointSoup, Pose3D, Tracklet


class TrackletViewer:
    """
    Interactive 3D viewer.

    Args:
        soup: PointSoup containing the raw 3D points
        data: Can be:
              1. Tracks3D Polars DataFrame
              2. Tracker history: Dict[int, List[Tuple[int, dict]]]
              3. Frame dictionary: Dict[int, List[Pose3D]]
        skeleton: Skeleton topology
        view_radius: Radius around centroid for axis limits
    """

    def __init__(
            self,
            soup: PointSoup,
            data: Union[pl.DataFrame, Dict[int, Any]],
            skeleton: Skeleton,
            view_radius: float = 15.0
    ):
        self.soup = soup
        self.skeleton = skeleton
        self.view_radius = view_radius

        self.objects_by_frame = self._organize_data_by_frame(data)

        soup_frames = set(self.soup.frame_indices)
        data_frames = set(self.objects_by_frame.keys())
        self.frames = sorted(list(soup_frames | data_frames))

        if not self.frames:
            print("No data to show.")
            return

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

        if data is None:
            return organized

        # Polars
        if isinstance(data, pl.DataFrame):
            if data.is_empty():
                return organized

            if "frame" not in data.columns or "track_id" not in data.columns:
                print("Warning: DataFrame missing 'frame' or 'track_id' columns")
                return organized

            try:
                grouped = data.group_by(["frame", "track_id"]).agg([
                    pl.col("scale").first(),
                    pl.col("keypoint"),
                    pl.col("x"),
                    pl.col("y"),
                    pl.col("z")
                ]).sort("frame")

                # Iterate the grouped objects
                for row in grouped.iter_rows(named=True):
                    frame = row['frame']

                    # Reconstruct keypoints dict {name: [x,y,z]}
                    kps = {}
                    for name, x, y, z in zip(row['keypoint'], row['x'], row['y'], row['z']):
                        kps[name] = [x, y, z]

                    obj = {
                        'track_idx': row['track_id'],
                        'scale': row['scale'],
                        'keypoints': kps,
                        'point_indices': {} # Indices lost in DF export
                    }
                    organized[frame].append(obj)
            except Exception as e:
                print(f"Error parsing DataFrame: {e}")

            return organized

        if not data:
            return organized

        # Check type of first value to determine structure
        first_val_list = next(iter(data.values()))
        if not first_val_list:
            return organized

        first_item = first_val_list[0]

        # Tracker history (Dict[track_id, List[(frame, dict)]])
        if isinstance(first_item, tuple) or (isinstance(first_item, dict) and 'frame_idx' in first_item):
            for track_id, history in data.items():
                for item in history:
                    if isinstance(item, tuple):
                        _, state = item
                    else:
                        state = item

                    frame = state.get('frame_idx')
                    if frame is not None:
                        organized[frame].append(state)

        # Frame-based hypotheses (Dict[frame, List[Pose3D]])
        else:
            return data

        return organized

    def _on_slider_change(self, val):
        frame_list_idx = int(self.slider.val)
        self._draw_frame(frame_list_idx)

    def _draw_frame(self, frame_list_idx: int):
        self.ax.clear()

        if frame_list_idx >= len(self.frames):
            return

        frame_idx = self.frames[frame_list_idx]
        self.ax.set_title(f"Frame {frame_idx}")

        self._draw_soup(frame_idx)

        objects = self.objects_by_frame.get(frame_idx, [])
        self._draw_all_objects(objects)

        if objects:
            self._set_view_limits(objects[0])
        else:
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
                    c='k', alpha=0.05, s=10, linewidths=0, depthshade=False
                )
        except Exception:
            pass

    def _draw_all_objects(self, objects: List[Any]):
        if not objects:
            return

        cmap = plt.cm.tab10

        for obj in objects:
            extracted = self._extract_pose_data(obj)
            if not extracted:
                continue

            kps, scale, track_id, point_indices = extracted

            color_idx = track_id % 10
            color = cmap(color_idx)

            self._draw_skeleton(kps, scale, point_indices, color, track_id)

    def _draw_skeleton(self, keypoints: Dict[str, Any], scale: float, point_indices: Dict[str, int], color, label_id):

        def to_arr(v):
            return np.array(v) if not isinstance(v, np.ndarray) else v

        real_kps = []
        virt_kps = []

        for name, pos_raw in keypoints.items():
            pos = to_arr(pos_raw)
            idx = point_indices.get(name, 0)

            if idx < 0:
                virt_kps.append(pos)
            else:
                real_kps.append(pos)

        # Draw nodes
        if real_kps:
            rk = np.array(real_kps)
            self.ax.scatter(rk[:, 0], rk[:, 1], rk[:, 2],
                           color=color, s=40, edgecolors='w', alpha=1.0)

        if virt_kps:
            vk = np.array(virt_kps)
            self.ax.scatter(vk[:, 0], vk[:, 1], vk[:, 2],
                           color=color, s=30, marker='^', edgecolors='w', alpha=0.7)

        # Draw bones
        for bone in self.skeleton.bones:
            if bone.k1 in keypoints and bone.k2 in keypoints:
                p1 = to_arr(keypoints[bone.k1])
                p2 = to_arr(keypoints[bone.k2])

                self.ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=color, lw=2
                )

        # Label (centroid)
        all_vals = [to_arr(p) for p in keypoints.values()]
        if all_vals:
            centroid = np.mean(all_vals, axis=0)
            label_text = f"ID:{label_id} (s:{scale:.2f})"

            self.ax.text(
                centroid[0], centroid[1], centroid[2],
                label_text,
                color='k', fontsize=8, weight='bold',
                bbox=dict(facecolor='w', alpha=0.6, edgecolor='none', pad=0.5)
            )

    def _extract_pose_data(self, obj: Any):

        # serialised dict (from tracker history or dataframe export)
        if isinstance(obj, dict):
            return (
                obj.get('keypoints', {}),
                obj.get('scale', 1.0),
                obj.get('track_idx', 0),
                obj.get('point_indices', {})
            )

        # Tracklet object
        if isinstance(obj, Tracklet):
            return (
                obj.hypothesis.positions,
                obj.estimated_scale,
                obj.track_idx,
                {n.name: n.idx for n in obj.hypothesis.nodes}
            )

        # Pose3D object (raw assembly)
        if isinstance(obj, Pose3D):
            return (
                obj.positions,
                obj.scale,
                getattr(obj, 'track_affinity', 0) or 0,
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

        xyz = np.array([np.array(p) for p in kps.values()])
        centroid = np.mean(xyz, axis=0)
        self._set_limits_around(centroid)

    def _set_limits_around(self, center: np.ndarray):
        r = self.view_radius
        self.ax.set_xlim(center[0] - r, center[0] + r)
        self.ax.set_ylim(center[1] - r, center[1] + r)
        self.ax.set_zlim(center[2] - r, center[2] + r)