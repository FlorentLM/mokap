from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, FrozenSet, Optional, Union, Sequence, Iterator, Set
import numpy as np
import pandas as pd
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from scipy.spatial import cKDTree

from lucida.geometry import align_rigid, intersect_ray_sphere
from mokap.pose_reconstruction.configs import TrackerConfig


@dataclass(frozen=True, slots=True)
class Node:
    """
    A keypoint observation in a frame.
    Immutable and hashable for comparisons.
    Negative indices indicate virtual points.
    """
    name: str
    idx: int
    position: np.ndarray
    confidence: float
    ray_idx: int = -1  # source ray index if virtual, -1 otherwise

    def __hash__(self):
        return hash((self.name, self.idx))

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.name == other.name and self.idx == other.idx

    @property
    def is_virtual(self) -> bool:
        return self.idx < 0


@dataclass
class SkeletonHypothesis:
    """
    A candidate skeleton during assembly.
    Nodes are stored as a frozenset (for hashing/comparison) and indexed by name (for O(1) lookup).
    """
    _nodes: FrozenSet[Node]
    scale: float
    competition_score: float
    anatomical_score: float
    constituent_indices: Optional[FrozenSet[int]] = None  # for tracking merge provenance

    # Cached lookups
    _by_name: Dict[str, Node] = field(init=False, repr=False, compare=False)
    _point_indices: FrozenSet[int] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, '_by_name', {n.name: n for n in self._nodes})
        object.__setattr__(self, '_point_indices', frozenset(n.idx for n in self._nodes))

    def __getitem__(self, name: str) -> Node:
        """Get node by keypoint name. Raises KeyError if not found."""
        return self._by_name[name]

    def __contains__(self, item: Union[str, Node]) -> bool:
        """Check if keypoint name or node is in this hypothesis."""
        if isinstance(item, str):
            return item in self._by_name
        return item in self._nodes

    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def get(self, name: str, default: Optional[Node] = None) -> Optional[Node]:
        """Get node by name, return default if not found."""
        return self._by_name.get(name, default)

    @property
    def names(self) -> FrozenSet[str]:
        """Set of keypoint names in this hypothesis."""
        return frozenset(self._by_name.keys())

    @property
    def positions(self) -> Dict[str, np.ndarray]:
        """Dict mapping keypoint names to positions."""
        return {n.name: n.position for n in self._nodes}

    @property
    def centroid(self) -> np.ndarray:
        """Mean position of all nodes."""
        return np.mean([n.position for n in self._nodes], axis=0)

    @property
    def point_indices(self) -> FrozenSet[int]:
        """Set of point indices (soup indices) in this hypothesis."""
        return self._point_indices

    @property
    def ray_indices(self) -> Set[int]:
        """Set of source ray indices for virtual points in this hypothesis."""
        return {n.ray_idx for n in self._nodes if n.is_virtual}

    def shares_points_with(self, other: 'SkeletonHypothesis') -> bool:
        """Check if two hypotheses share any point indices."""
        return not self._point_indices.isdisjoint(other._point_indices)

    def shares_rays_with(self, other: 'SkeletonHypothesis') -> bool:
        """Check if two hypotheses share any source rays."""
        return not self.ray_indices.isdisjoint(other.ray_indices)

    def is_related(self, other: 'SkeletonHypothesis') -> bool:
        """Check if one hypothesis is a constituent of the other (merge provenance)."""
        if self.constituent_indices and other.constituent_indices:
            return (self.constituent_indices.issubset(other.constituent_indices) or
                    other.constituent_indices.issubset(self.constituent_indices))
        return False

    def to_pose(self, track_idx: int = -1) -> 'Pose3D':
        """Convert to a resolved Pose3D for output."""
        return Pose3D(
            keypoints=self.positions,
            scale=self.scale,
            score=self.anatomical_score,
            soup_point_indices={n.name: n.idx for n in self._nodes},
            track_idx=track_idx
        )


class PointSoup:
    """
    Soup of 3D points from multi-view reconstruction (SoA).
    """

    def __init__(
            self,
            positions: Optional[np.ndarray] = None,
            confidences: Optional[np.ndarray] = None,
            reprojection_errors: Optional[np.ndarray] = None,
            keypoint_indices: Optional[np.ndarray] = None,
            frame_indices: Optional[np.ndarray] = None,
            camera_masks: Optional[np.ndarray] = None,
            ray_origins: Optional[np.ndarray] = None,
            ray_directions: Optional[np.ndarray] = None,
            ray_confidences: Optional[np.ndarray] = None,
            ray_keypoint_indices: Optional[np.ndarray] = None,
            ray_frame_indices: Optional[np.ndarray] = None,
            keypoint_names: Optional[List[str]] = None,
            camera_names: Optional[List[str]] = None,
            sort: bool = True
    ):
        # Point data (N, ...)
        self.positions = self._as_array(positions, (0, 3), np.float32)
        self.confidences = self._as_array(confidences, (0,), np.float32)
        self.reprojection_errors = self._as_array(reprojection_errors, (0,), np.float32)
        self.keypoint_indices = self._as_array(keypoint_indices, (0,), np.int16)
        self.frame_indices = self._as_array(frame_indices, (0,), np.int32)

        # Ray data (M, ...)
        self.ray_origins = self._as_array(ray_origins, (0, 3), np.float32)
        self.ray_directions = self._as_array(ray_directions, (0, 3), np.float32)
        self.ray_confidences = self._as_array(ray_confidences, (0,), np.float32)
        self.ray_keypoint_indices = self._as_array(ray_keypoint_indices, (0,), np.int16)
        self.ray_frame_indices = self._as_array(ray_frame_indices, (0,), np.int32)

        self.camera_masks = self._as_array(camera_masks, (0,), dtype=np.uint64) # bitmask of contributing cameras

        # Metadata
        self.keypoint_names = keypoint_names or []
        self.camera_names = camera_names or []

        # Sort by frame index to enable O(log N) slicing
        if sort:
            self._sort_inplace()

    @staticmethod
    def _as_array(data, shape, dtype):
        if not bool(np.any(data)):
            return np.empty(shape, dtype=dtype)
        return np.asarray(data, dtype=dtype)

    def _sort_inplace(self):
        """Sorts all point and ray arrays by their respective frame indices."""

        if len(self.frame_indices) > 1 and np.all(self.frame_indices[:-1] <= self.frame_indices[1:]):
            return

        if len(self.frame_indices) > 0:
            p_idx = np.argsort(self.frame_indices)

            self.positions = self.positions[p_idx]
            self.confidences = self.confidences[p_idx]
            self.reprojection_errors = self.reprojection_errors[p_idx]
            self.keypoint_indices = self.keypoint_indices[p_idx]
            self.frame_indices = self.frame_indices[p_idx]
            self.camera_masks = self.camera_masks[p_idx]

        if len(self.ray_frame_indices) > 0:
            r_idx = np.argsort(self.ray_frame_indices)

            self.ray_origins = self.ray_origins[r_idx]
            self.ray_directions = self.ray_directions[r_idx]
            self.ray_confidences = self.ray_confidences[r_idx]
            self.ray_keypoint_indices = self.ray_keypoint_indices[r_idx]
            self.ray_frame_indices = self.ray_frame_indices[r_idx]

    @property
    def nb_points(self) -> int:
        return len(self.positions)

    @property
    def nb_rays(self) -> int:
        return len(self.ray_origins)

    @cached_property
    def tree(self) -> Optional[cKDTree]:
        """Builds a KDTree of all 3D points in this soup (lazy)."""
        if self.nb_points > 0:
            return cKDTree(self.positions)
        return None

    @cached_property
    def points_by_name(self) -> Dict[str, np.ndarray]:
        """Returns a map of keypoint_name -> array of point indices (lazy & vectorised)."""
        return self._group_indices(self.keypoint_indices)

    @cached_property
    def rays_by_name(self) -> Dict[str, np.ndarray]:
        """Returns a map of keypoint_name -> array of ray indices (lazy & vectorised)."""
        return self._group_indices(self.ray_keypoint_indices)

    def _group_indices(self, keypoints: Sequence) -> Dict[str, np.ndarray]:
        result = {name: np.array([], dtype=np.int32) for name in self.keypoint_names}
        if len(keypoints) == 0:
            return result

        order = np.argsort(keypoints)
        sorted_kp = np.asarray(keypoints)[order]
        diff = np.diff(sorted_kp, append=sorted_kp[-1] + 1)
        splits = np.where(diff != 0)[0] + 1

        groups = np.split(order, splits[:-1])
        unique_ids = sorted_kp[splits[:-1]]
        for kp_id, g in zip(unique_ids, groups):
            result[self.keypoint_names[kp_id]] = g
        return result

    @property
    def frame_range(self):
        """Returns (min_frame, max_frame) present in the data."""

        all_frames = []
        if self.nb_points > 0:
            all_frames.extend([self.frame_indices[0], self.frame_indices[-1]])
        if self.nb_rays > 0:
            all_frames.extend([self.ray_frame_indices[0], self.ray_frame_indices[-1]])

        if not all_frames:
            return 0, 0

        return min(all_frames), max(all_frames)

    def __len__(self) -> int:
        """Returns the number of points (standard for SoA objects)."""
        return self.nb_points

    def __repr__(self) -> str:
        f0, f1 = self.frame_range
        return f"PointSoup(points={self.nb_points}, rays={self.nb_rays}, frames={f0}-{f1})"

    def __getitem__(self, key: Union[int, slice]) -> 'PointSoup':
        """
        Slice the soup by frame index.
        Returns an empty soup if no data exists for those frames.
        """
        if isinstance(key, int):
            start_f, stop_f = key, key + 1
        elif isinstance(key, slice):
            start_f = key.start if key.start is not None else -np.inf
            stop_f = key.stop if key.stop is not None else np.inf
        else:
            raise TypeError("Index must be int or slice")

        # Slice points
        p_start = np.searchsorted(self.frame_indices, start_f, side='left')
        p_stop = np.searchsorted(self.frame_indices, stop_f, side='left')

        # Slice rays
        r_start = np.searchsorted(self.ray_frame_indices, start_f, side='left')
        r_stop = np.searchsorted(self.ray_frame_indices, stop_f, side='left')

        return PointSoup(
            positions=self.positions[p_start:p_stop],
            confidences=self.confidences[p_start:p_stop],
            reprojection_errors=self.reprojection_errors[p_start:p_stop],
            keypoint_indices=self.keypoint_indices[p_start:p_stop],
            frame_indices=self.frame_indices[p_start:p_stop],
            camera_masks=self.camera_masks[p_start:p_stop],
            ray_origins=self.ray_origins[r_start:r_stop],
            ray_directions=self.ray_directions[r_start:r_stop],
            ray_confidences=self.ray_confidences[r_start:r_stop],
            ray_keypoint_indices=self.ray_keypoint_indices[r_start:r_stop],
            ray_frame_indices=self.ray_frame_indices[r_start:r_stop],
            keypoint_names=self.keypoint_names,
            camera_names=self.camera_names,
            sort=False  # subsets are already sorted
        )

    @classmethod
    def concatenate(cls, soups: Sequence['PointSoup']) -> 'PointSoup':
        if not soups:
            return cls()
        if len(soups) == 1:
            return soups[0]

        # Combine all unique names from all soups while preserving order
        kp_names_union = []
        seen_kp = set()
        for s in soups:
            for name in s.keypoint_names:
                if name not in seen_kp:
                    kp_names_union.append(name)
                    seen_kp.add(name)

        cam_names_union = []
        seen_cam = set()
        for s in soups:
            for name in s.camera_names:
                if name not in seen_cam:
                    cam_names_union.append(name)
                    seen_cam.add(name)

        # Prepare concatenated arrays
        concatenated_data = {}
        simple_attrs = [
            'positions', 'confidences', 'reprojection_errors',
            'frame_indices', 'camera_masks', 'ray_origins',
            'ray_directions', 'ray_confidences', 'ray_frame_indices'
        ]

        for attr in simple_attrs:
            concatenated_data[attr] = np.concatenate([getattr(s, attr) for s in soups])

        # Re-indexing of keypoints
        def remap_indices(soup, attr, global_names):
            local_indices = getattr(soup, attr)
            if len(local_indices) == 0:
                return local_indices

            lookup = np.array([global_names.index(name) for name in soup.keypoint_names], dtype=np.int16)
            return lookup[local_indices]

        concatenated_data['keypoint_indices'] = np.concatenate([
            remap_indices(s, 'keypoint_indices', kp_names_union) for s in soups
        ])
        concatenated_data['ray_keypoint_indices'] = np.concatenate([
            remap_indices(s, 'ray_keypoint_indices', kp_names_union) for s in soups
        ])

        return cls(
            **concatenated_data,
            keypoint_names=kp_names_union,
            camera_names=cam_names_union,
            sort=True
        )

    def to_df(self, keypoint_names: Optional[List[str]] = None) -> 'pd.DataFrame':
        """Convert the point data to a pandas DataFrame for tracking/analysis."""

        if keypoint_names is not None:
            valid_ids = [self.keypoint_names.index(n) for n in keypoint_names if n in self.keypoint_names]
            mask = np.isin(self.keypoint_indices, valid_ids)
        else:
            mask = np.ones(self.nb_points, dtype=bool)

        return pd.DataFrame({
            "frame": self.frame_indices[mask],
            "x": self.positions[mask, 0],
            "y": self.positions[mask, 1],
            "z": self.positions[mask, 2],
            "keypoint": [self.keypoint_names[i] for i in self.keypoint_indices[mask]],
            "confidence": self.confidences[mask]
        })


class FrameData:
    """
    Single-frame view of a PointSoup with virtual point management.
    Provides access to real points (from triangulation) and virtual points (from ray-sphere intersection).
    """

    VIRTUAL_CONFIDENCE_PENALTY = 0.8  # TODO: Move this to config?

    def __init__(self, soup_slice: PointSoup):
        self.soup = soup_slice
        self._virtual_nodes: Dict[int, Node] = {}
        self._next_virt_id = -1

    def __bool__(self) -> bool:
        return self.soup.nb_points > 0 or self.soup.nb_rays > 0

    def get_node(self, name: str, idx: int) -> Node:
        """
        Instantiate a Node for the given keypoint name and index.
        """
        if idx < 0:
            return self._virtual_nodes[idx]

        return Node(
            name=name,
            idx=idx,
            position=self.soup.positions[idx],
            confidence=self.soup.confidences[idx],
            ray_idx=-1
        )

    def get_indices(self, keypoint_name: str) -> np.ndarray:
        """Get indices of real 3D points for a specific keypoint type."""
        return self.soup.points_by_name.get(keypoint_name, np.array([], dtype=np.int32))

    def nearby(self, center: np.ndarray, radius: float) -> List[int]:
        """Find all real point indices within radius of center."""
        if not self.soup.tree:
            return []
        return list(self.soup.tree.query_ball_point(center, r=radius))

    def intersect_rays(
            self,
            keypoint_name: str,
            center: np.ndarray,
            radius: float
    ) -> List[Node]:
        """
        Intersect rays of type `keypoint_name` with a sphere (center, radius).
        Creates and registers virtual Nodes for intersection points.
        Returns list of new virtual Nodes.
        """

        ray_indices = self.soup.rays_by_name.get(keypoint_name, np.array([]))

        if len(ray_indices) == 0:
            return []

        origins = self.soup.ray_origins[ray_indices]
        dirs = self.soup.ray_directions[ray_indices]

        p1, p2, valid_mask = intersect_ray_sphere(origins, dirs, center, radius)

        if not np.any(valid_mask):
            return []

        valid_locs = np.where(valid_mask)[0]
        valid_ray_indices = ray_indices[valid_locs]
        valid_confs = self.soup.ray_confidences[valid_ray_indices]

        p1_valid = p1[valid_locs]
        p2_valid = p2[valid_locs]

        new_nodes = []

        # Register both intersection solutions
        for i in range(len(valid_locs)):
            conf = valid_confs[i] * self.VIRTUAL_CONFIDENCE_PENALTY
            ray_idx = int(valid_ray_indices[i])

            for pos in (p1_valid[i], p2_valid[i]):
                node = Node(
                    name=keypoint_name,
                    idx=self._next_virt_id,
                    position=pos,
                    confidence=conf,
                    ray_idx=ray_idx
                )
                self._virtual_nodes[self._next_virt_id] = node
                self._next_virt_id -= 1
                new_nodes.append(node)

        return new_nodes


@dataclass
class Pose3D:
    """
    A resolved 3D pose for a single frame.
    This is the output format containing actual positions (not indices).
    # TODO: This class should be replaced / removed
    """
    keypoints: Dict[str, np.ndarray]  # keypoint name -> (3,) position
    scale: float
    score: float
    soup_point_indices: Dict[str, int] = field(default_factory=dict)  # provenance
    track_idx: int = -1

    def to_dict(self) -> dict:
        return {
            'keypoints': self.keypoints,
            'score': self.score,
            'scale': self.scale,
            'soup_points_indices': self.soup_point_indices,
            'track_idx': self.track_idx
        }


class Tracklet:
    """
    Stateful class for a single skeleton in a tracklet.
    Manages state estimation (position, velocity, scale) with a Kalman Filter.
    """

    def __init__(
            self,
            track_idx: int,
            initial_pose: Pose3D,
            frame_idx: int,
            central_kp: str,
            config: TrackerConfig
    ):

        self.config = config
        self.track_idx = track_idx
        self.age = 0
        self.time_since_update = 0
        self.last_update_frame = frame_idx

        self.pose: Pose3D = initial_pose
        self.central_kp = central_kp

        # Tracklet health and score metrics
        self.health = 1.0
        self.anatomical_integrity = self.pose.score

        # Kalman Filter for 3D position (x, y, z), 3D velocity (vx, vy, vz), and scale (s)
        # State vector (dim_x = 7): [x, y, z, vx, vy, vz, s]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        dt = 1.0

        self.kf.F = np.array([[1.0, 0.0, 0.0, dt, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0, dt, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0, 0.0, dt, 0.0],
                              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        # Measurement function: we measure position (x, y, z) and scale (s)
        self.kf.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        # Process noise
        pos_vel_q = Q_discrete_white_noise(dim=2, dt=dt, var=self.config.kf_process_noise_pos, block_size=3)
        scale_q = np.array([[self.config.kf_process_noise_scale]])
        self.kf.Q = block_diag(pos_vel_q, scale_q)

        # Measurement noise
        self.kf.R = np.diag([self.config.kf_measurement_noise_pos, self.config.kf_measurement_noise_pos,
                             self.config.kf_measurement_noise_pos, self.config.kf_measurement_noise_scale])

        # Initial state covariance
        self.kf.P[3:6, 3:6] *= 1.0
        self.kf.P[6, 6] = 1.0

        # Initial state
        self.kf.x[:3] = self.pose.keypoints[self.central_kp].reshape(3, 1)
        self.kf.x[6] = self.pose.scale

    def predict(self, current_frame_idx: int):
        """Predicts the state of the tracklet for the current frame."""

        steps_to_predict = current_frame_idx - self.last_update_frame

        for _ in range(steps_to_predict):
            self.kf.predict()
            self.age += 1
            self.time_since_update += 1
            self.health *= self.config.health_decay_rate

    def update(self, pose: Pose3D, frame_idx: int):
        """Updates the tracklet's state with a new pose."""

        inferred = False

        # If the primary keypoint is missing, try to infer it
        if self.central_kp not in pose.keypoints:
            inferred_pose = self._infer_missing_central_kp(pose)

            if inferred_pose:
                pose = inferred_pose
                inferred = True
            else:
                self.pose = pose
                self.time_since_update = 0
                self.last_update_frame = frame_idx
                return

        self.pose = pose
        self.time_since_update = 0
        self.last_update_frame = frame_idx

        measurement = np.array([*pose.keypoints[self.central_kp], pose.scale])

        if inferred:
            original_R = self.kf.R.copy()
            self.kf.R[:3, :3] *= self.config.kf_inference_uncertainty_factor
            self.kf.update(measurement)
            self.kf.R = original_R
        else:
            self.kf.update(measurement)

        # Update health metrics
        self.anatomical_integrity = self.config.anatomical_score_alpha * pose.score + (
                1 - self.config.anatomical_score_alpha) * self.anatomical_integrity

        if inferred:
            self.health = 1.0 - self.config.inferred_health_penalty
        else:
            self.health = 1.0

    def _infer_missing_central_kp(self, fragment: Pose3D) -> Optional[Pose3D]:
        prev_kps, curr_kps = self.pose.keypoints, fragment.keypoints
        common_names = list(prev_kps.keys() & curr_kps.keys())

        if len(common_names) < self.config.min_kps_for_inference:
            return None

        points_A = np.array([prev_kps[name] for name in common_names])
        points_B = np.array([curr_kps[name] for name in common_names])

        R_mat, t_vec = align_rigid(points_A, points_B)
        inferred_pos = np.array(R_mat) @ prev_kps[self.central_kp] + np.array(t_vec)

        completed_skeleton = Pose3D(
            keypoints=fragment.keypoints.copy(),
            score=fragment.score,
            scale=fragment.scale,
            soup_point_indices=fragment.soup_point_indices
        )
        completed_skeleton.keypoints[self.central_kp] = inferred_pos
        return completed_skeleton

    @property
    def predicted_pose(self) -> Optional[Dict[str, np.ndarray]]:
        if self.central_kp not in self.pose.keypoints:
            return None
        translation = self.predicted_position - self.pose.keypoints[self.central_kp]
        return {kp_name: pos + translation for kp_name, pos in self.pose.keypoints.items()}

    @property
    def predicted_position(self) -> np.ndarray:
        return self.kf.x[:3].flatten()

    @property
    def predicted_scale(self) -> float:
        return self.kf.x[6, 0]

    @property
    def uncertainty(self) -> Dict[str, np.ndarray]:
        diag_P = self.kf.P.diagonal()
        return {
            'position': diag_P[0:3],
            'velocity': diag_P[3:6],
            'scale': diag_P[6]
        }