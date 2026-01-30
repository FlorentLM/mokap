from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, FrozenSet, Optional, Union, Sequence, Iterator, Set
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy.spatial import cKDTree

from lucida.geometry import intersect_ray_sphere
from mokap.pose_reconstruction.skeleton import SkeletonStats, Skeleton
from mokap.pose_reconstruction.utils import ema_update

if TYPE_CHECKING:
    import polars as pl
    from mokap.pose_reconstruction.configs import TrackletConfig


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

        self.camera_masks = self._as_array(camera_masks, (0,), dtype=np.uint64)  # bitmask of contributing cameras

        # Metadata
        self.keypoint_names = keypoint_names or []
        self.camera_names = camera_names or []

        # Sort by frame index to enable O(log N) slicing
        if sort:
            self._sort_inplace()

    def __len__(self) -> int:
        return self.nb_points

    def __repr__(self) -> str:
        f0, f1 = self.frame_range
        return f"PointSoup(points={self.nb_points}, rays={self.nb_rays}, frames={f0}-{f1})"

    def __getitem__(self, key: Union[int, slice]) -> 'PointSoup':
        """
        Slice the soup by frame index.
        Returns an empty soup if no data exists for those frames.
        """
        if isinstance(key, (int, np.integer)):
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

    @staticmethod
    def _as_array(data, shape, dtype):
        if data is None or len(data) == 0:
            return np.empty(shape, dtype=dtype)
        return np.asarray(data, dtype=dtype)

    def _sort_inplace(self):
        """Sorts all point and ray arrays by their respective frame indices."""

        # check if already sorted
        if len(self.frame_indices) > 1 and np.all(self.frame_indices[:-1] <= self.frame_indices[1:]):
            pass

        elif len(self.frame_indices) > 0:
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
            # Safe check to prevent wrapping indices
            if 0 <= kp_id < len(self.keypoint_names):
                result[self.keypoint_names[kp_id]] = g

        return result

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
        """Map of keypoint_name -> array of point indices (lazy)."""
        return self._group_indices(self.keypoint_indices)

    @cached_property
    def rays_by_name(self) -> Dict[str, np.ndarray]:
        """Map of keypoint_name -> array of ray indices (lazy)."""
        return self._group_indices(self.ray_keypoint_indices)

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

    # Serialisation methods

    def to_dataframe(self) -> 'pl.DataFrame':
        """
        Convert to a Points3D Polars DataFrame.
        """
        from mokap.mokap_io import soup_to_dataframe
        return soup_to_dataframe(self)

    @classmethod
    def from_dataframe(
        cls,
        df: 'pl.DataFrame',
        keypoint_names: Sequence[str],
        camera_names: Sequence[str],
    ) -> 'PointSoup':
        """
        Create a PointSoup from a Points3D DataFrame.

        Args:
            df: DataFrame with Points3D schema
            keypoint_names: Ordered keypoint names (for index mapping)
            camera_names: Ordered camera names (for mask decoding)
        """
        from mokap.mokap_io import dataframe_to_soup
        return dataframe_to_soup(df, keypoint_names, camera_names)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save to file (parquet recommended, pickle supported).
        """
        from mokap.mokap_io import soup_to_dataframe, save_dataframe
        df = soup_to_dataframe(self)
        save_dataframe(df, path, schema_name='Points3D', validate=True)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        keypoints_order: Sequence[str],
        cameras_order: Sequence[str],
    ) -> 'PointSoup':
        """
        Load from file.

        Args:
            path: Path to data file (.parquet, .csv, or .pkl)
            keypoints_order: Ordered keypoint names for index mapping
            cameras_order: Ordered camera names for mask decoding
        """
        from mokap.mokap_io import load_point_soup
        return load_point_soup(path, keypoints_order, cameras_order)

    def to_pandas(self, keypoint_filter: Optional[List[str]] = None):
        """
        Returns a pandas DataFrame.
        """
        import polars as pl

        df = self.to_dataframe()

        # Filter for reconstructed points only
        # TODO: maybe this filtering should go in the bootstrap
        if "status" in df.columns:
            df = df.filter(pl.col("status") == "reconstructed")

        if keypoint_filter is not None:
            df = df.filter(pl.col('keypoint').is_in(keypoint_filter))

        return df.to_pandas()


class TimestepData:
    """
    Single timestep view of the data with virtual point management.
    Provides access to real points (from triangulation) and virtual points (from ray-sphere intersection).
    """

    VIRTUAL_CONFIDENCE_PENALTY = 0.8  # TODO: Move this to config?

    def __init__(self, soup_slice: PointSoup):
        self.soup = soup_slice
        self._virtual_nodes: Dict[int, 'Node3D'] = {}
        self._next_virt_id = -1

    def __bool__(self) -> bool:
        return self.soup.nb_points > 0 or self.soup.nb_rays > 0

    def get_node(self, name: str, idx: int) -> 'Node3D':
        """
        Instantiate a Node for the given keypoint name and index.
        """
        if idx < 0:
            return self._virtual_nodes[idx]

        return Node3D(
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
    ) -> List['Node3D']:
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
                node = Node3D(
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


@dataclass(frozen=True, slots=True)
class Node3D:
    """
    A keypoint observation in a given time step.
    Immutable and hashable. Negative indices indicate virtual points.
    """
    name: str
    idx: int
    position: np.ndarray
    confidence: float
    ray_idx: int = -1  # source ray index if virtual, -1 otherwise

    def __hash__(self):
        return hash((self.name, self.idx))

    def __eq__(self, other):
        if not isinstance(other, Node3D):
            return NotImplemented
        return self.name == other.name and self.idx == other.idx

    @property
    def is_virtual(self) -> bool:
        return self.idx < 0


@dataclass
class Pose3D:
    """
    A candidate skeleton (a pose in 3D) during assembly.
    """
    nodes: FrozenSet[Node3D]
    scale: float
    competition_score: float
    anatomical_score: float
    constituent_indices: Optional[FrozenSet[int]] = None  # for tracking merge provenance
    track_affinity: Optional[int] = None  # track ID if guided, None if blind

    # Cached lookups
    _by_name: Dict[str, Node3D] = field(init=False, repr=False, compare=False)
    _point_indices: FrozenSet[int] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, '_by_name', {n.name: n for n in self.nodes})
        object.__setattr__(self, '_point_indices', frozenset(n.idx for n in self.nodes))

    def __getitem__(self, name: str) -> Node3D:
        """Get node by keypoint name."""
        return self._by_name[name]

    def __contains__(self, item: Union[str, Node3D]) -> bool:
        if isinstance(item, str):
            return item in self._by_name
        return item in self.nodes

    def __iter__(self) -> Iterator[Node3D]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def get(self, name: str) -> Optional[Node3D]:
        """Get node by name."""
        return self._by_name.get(name)

    @property
    def nodes_by_name(self) -> Dict[str, Node3D]:
        return self._by_name

    @property
    def names(self) -> FrozenSet[str]:
        """Set of keypoint names in this hypothesis."""
        return frozenset(self._by_name.keys())

    @property
    def positions(self) -> Dict[str, np.ndarray]:
        """Dict mapping keypoint names to positions."""
        return {n.name: n.position for n in self.nodes}

    @property
    def centroid(self) -> np.ndarray:
        """Mean position of all nodes."""
        return np.mean([n.position for n in self.nodes], axis=0)

    @property
    def point_indices(self) -> FrozenSet[int]:
        """Set of point indices (soup indices) in this hypothesis."""
        return self._point_indices

    @property
    def ray_indices(self) -> Set[int]:
        """Set of source ray indices for virtual points in this hypothesis."""
        return {n.ray_idx for n in self.nodes if n.is_virtual}

    def shares_points_with(self, other: 'Pose3D') -> bool:
        """Check if two hypotheses share any point indices."""
        return not self._point_indices.isdisjoint(other._point_indices)

    def shares_rays_with(self, other: 'Pose3D') -> bool:
        """Check if two hypotheses share any source rays."""
        return not self.ray_indices.isdisjoint(other.ray_indices)

    def is_related(self, other: 'Pose3D') -> bool:
        """Check if one hypothesis is a constituent of the other (merge provenance)."""
        if self.constituent_indices and other.constituent_indices:
            return (self.constituent_indices.issubset(other.constituent_indices) or
                    other.constituent_indices.issubset(self.constituent_indices))
        return False


class Tracklet:
    """
    Hierarchical state representation for articulated skeleton tracking.
    """
    def __init__(
            self,
            track_idx: int,
            initial_hypothesis: 'Pose3D',
            frame_idx: int,
            skeleton: 'Skeleton',
            stats: 'SkeletonStats',
            config: 'TrackletConfig'
    ):
        self.track_idx = track_idx
        self.skeleton = skeleton
        self.stats = stats
        self.config = config
        self.central_kp = skeleton.central_keypoint

        # Temporal state
        self.age = 0
        self.time_since_update = 0
        self.last_update_frame = frame_idx

        # Current hypothesis
        self.hypothesis: Pose3D = initial_hypothesis

        # Health metrics
        self.health = 1.0
        self.anatomical_integrity = initial_hypothesis.anatomical_score

        # Body frame orientation (world -> body rotation matrix)
        self.body_rotation = np.eye(3)

        # Rest pose offsets (learned over time)
        self.rest_offsets: Dict[str, np.ndarray] = {}

        self.central_kf = self._init_central_kf(initial_hypothesis)
        self.offset_kfs: Dict[str, 'KalmanFilter'] = {}
        self._init_offset_kfs(initial_hypothesis)

        self._cached_predictions: Optional[Dict[str, np.ndarray]] = None

    def _init_central_kf(self, hypothesis: 'Pose3D') -> 'KalmanFilter':
        """
        Central KF: state = [x, y, z, vx, vy, vz, scale]
        """
        kf = KalmanFilter(dim_x=7, dim_z=4)
        dt = 1.0

        # State transition (constant velocity model)
        kf.F = np.array([
            [1,  0,  0, dt,  0,  0,  0],
            [0,  1,  0,  0, dt,  0,  0],
            [0,  0,  1,  0,  0, dt,  0],
            [0,  0,  0,  1,  0,  0,  0],
            [0,  0,  0,  0,  1,  0,  0],
            [0,  0,  0,  0,  0,  1,  0],
            [0,  0,  0,  0,  0,  0,  1],
        ], dtype=float)

        # Observation: position + scale
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Process noise
        q_pos = Q_discrete_white_noise(
            dim=2, dt=dt,
            var=self.config.central_process_noise_pos,
            block_size=3
        )
        q_scale = np.array([[self.config.central_process_noise_scale]])
        kf.Q = np.zeros((7, 7))
        kf.Q[:6, :6] = q_pos
        kf.Q[6, 6] = q_scale[0, 0]

        # Measurement noise
        kf.R = np.diag([
            self.config.central_measurement_noise_pos,
            self.config.central_measurement_noise_pos,
            self.config.central_measurement_noise_pos,
            self.config.central_measurement_noise_scale
        ])

        # Initial covariance
        kf.P = np.eye(7)
        kf.P[3:6, 3:6] *= 1.0  # velocity uncertainty
        kf.P[6, 6] = 0.1  # scale uncertainty

        # Initial state
        if self.central_kp in hypothesis:
            kf.x[:3, 0] = hypothesis[self.central_kp].position
        else:
            kf.x[:3, 0] = hypothesis.centroid
        kf.x[6, 0] = hypothesis.scale

        return kf

    def _init_offset_kfs(self, hypothesis: 'Pose3D'):
        """
        Initialise per-keypoint offset KFs.
        State = [dx, dy, dz, d_vx, d_vy, d_vz] (offset position and offset velocity in body frame).
        """
        central_pos = self.central_kf.x[:3, 0]
        scale = self.central_kf.x[6, 0]

        self.body_rotation = np.eye(3)

        for kp in self.skeleton.keypoints:
            if kp == self.central_kp:
                continue

            dynamics = self.stats.get_dynamics(kp)

            kf = KalmanFilter(dim_x=6, dim_z=3)  # state = [dx, dy, dz, dvx, dvy, dvz]
            dt = 1.0

            # State transition with velocity damping
            damp = self.config.offset_velocity_damping
            kf.F = np.array([
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, damp, 0, 0],
                [0, 0, 0, 0, damp, 0],
                [0, 0, 0, 0, 0, damp],
            ], dtype=float)

            # Observation = offset position only
            kf.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ], dtype=float)

            # Process noise (from learned dynamics)
            # TODO: Should probably update online
            q = Q_discrete_white_noise(dim=2, dt=dt, var=dynamics.process_noise, block_size=3)
            kf.Q = q

            # Measurement noise
            kf.R = np.eye(3) * dynamics.measurement_noise

            # Initial covariance
            kf.P = np.eye(6) * 0.5

            # Initial state
            if kp in hypothesis:
                world_offset = hypothesis[kp].position - central_pos
                local_offset = self.body_rotation @ world_offset / max(scale, 0.1)
            else:
                local_offset = np.zeros(3)

            kf.x[:3, 0] = local_offset
            kf.x[3:6, 0] = 0.0  # zero initial velocity

            self.offset_kfs[kp] = kf
            self.rest_offsets[kp] = local_offset.copy()

    # Properties and accessors

    @property
    def predicted_keypoints(self) -> Dict[str, np.ndarray]:
        """Current predicted positions (without advancing time)."""

        if self._cached_predictions is not None:
            return self._cached_predictions

        central_pos = self.central_kf.x[:3, 0]
        scale = self.central_kf.x[6, 0]

        predictions = {self.central_kp: central_pos.copy()}

        for kp, offset_kf in self.offset_kfs.items():
            local_offset = offset_kf.x[:3, 0]
            rest = self.rest_offsets.get(kp, local_offset)

            blended_offset = (
                    (1 - self.config.rigidity_factor) * local_offset + self.config.rigidity_factor * rest
            )
            world_offset = self.body_rotation.T @ (blended_offset * scale)
            predictions[kp] = central_pos + world_offset

        self._cached_predictions = predictions
        return predictions

    @property
    def position(self) -> np.ndarray:
        """Central position estimate."""
        return self.central_kf.x[:3, 0].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Central velocity estimate."""
        return self.central_kf.x[3:6, 0].copy()

    @property
    def estimated_scale(self) -> float:
        """Current scale estimate."""
        return float(self.central_kf.x[6, 0])

    @property
    def position_uncertainty(self) -> np.ndarray:
        """Position uncertainty (diagonal of covariance)."""
        return self.central_kf.P.diagonal()[:3]

    @property
    def is_active(self) -> bool:
        """Was this tracklet updated this frame?"""
        return self.time_since_update == 0

    def get_offset_uncertainty(self, keypoint: str) -> Optional[np.ndarray]:
        """Get position uncertainty for a specific keypoint's offset."""
        if keypoint in self.offset_kfs:
            return self.offset_kfs[keypoint].P.diagonal()[:3]
        return None

    def get_world_uncertainty(self, keypoint: str) -> Optional[np.ndarray]:
        """
        Get position uncertainty for a keypoint in world frame.

        For the central keypoint, returns the central KF position uncertainty.
        For other keypoints, transforms the offset uncertainty to world frame
        and adds the central position uncertainty.
        """

        if keypoint == self.central_kp:
            return self.central_kf.P.diagonal()[:3]

        if keypoint not in self.offset_kfs:
            return None

        # Offset uncertainty is in body frame, scaled
        local_var = self.offset_kfs[keypoint].P.diagonal()[:3]
        scale = self.estimated_scale

        # Transform variance to world frame
        R = self.body_rotation.T  # body -> world
        world_var = (scale ** 2) * np.sum((R ** 2) * local_var, axis=1)

        # Add central position uncertainty (offsets are relative to center)
        world_var += self.central_kf.P.diagonal()[:3]

        return world_var

    # Public interface

    def predict(self, current_frame_idx: int) -> Dict[str, np.ndarray]:
        """
        Predict all keypoint positions forward to current frame.
        Returns dict of keypoint_name -> predicted world position.
        """
        steps = current_frame_idx - self.last_update_frame

        # Predict central state
        for _ in range(steps):
            self.central_kf.predict()
            self.age += 1
            self.time_since_update += 1
            self.health *= self.config.health_decay_rate

        central_pos = self.central_kf.x[:3, 0]
        scale = self.central_kf.x[6, 0]

        predictions = {self.central_kp: central_pos.copy()}

        # Predict each offset KF
        for kp, offset_kf in self.offset_kfs.items():
            for _ in range(steps):
                offset_kf.predict()

            # Rigidity constraint: pull toward rest offset
            local_offset = offset_kf.x[:3, 0].copy()
            rest = self.rest_offsets.get(kp, local_offset)

            blended_offset = (
                    (1 - self.config.rigidity_factor) * local_offset +
                    self.config.rigidity_factor * rest
            )

            # Transform to world frame
            world_offset = self.body_rotation.T @ (blended_offset * scale)
            predictions[kp] = central_pos + world_offset

        self._cached_predictions = predictions
        return predictions

    def update(self, hypothesis: 'Pose3D', frame_idx: int):
        """
        Update tracklet with new observation.
        """

        self._cached_predictions = None

        self.time_since_update = 0
        self.last_update_frame = frame_idx
        self.hypothesis = hypothesis

        self._update_body_orientation(hypothesis.positions, alpha=0.3)

        # Handle missing central keypoint
        if self.central_kp not in hypothesis:
            inferred_pos = self._infer_central_position(hypothesis)
            if inferred_pos is None:
                self.health = max(0.1, self.health - self.config.inferred_health_penalty)
                return
            central_obs = inferred_pos
            inferred = True
        else:
            central_obs = hypothesis[self.central_kp].position
            inferred = False

        scale_obs = hypothesis.scale

        # Update central KF
        measurement = np.array([*central_obs, scale_obs])

        if inferred:
            original_R = self.central_kf.R.copy()
            self.central_kf.R[:3, :3] *= self.config.centre_inference_uncertainty_factor  # increase uncertainty
            self.central_kf.update(measurement)
            self.central_kf.R = original_R
            self.health = max(0.1, 1.0 - self.config.inferred_health_penalty)
        else:
            self.central_kf.update(measurement)
            self.health = 1.0

        # Update offset KFs for observed keypoints
        central_pos = self.central_kf.x[:3, 0]
        scale = max(self.central_kf.x[6, 0], 0.1)

        for kp, node in hypothesis.nodes_by_name.items():

            if kp == self.central_kp or kp not in self.offset_kfs:
                continue

            # Compute observed offset in body frame
            world_offset = node.position - central_pos
            local_offset = self.body_rotation @ world_offset / scale

            self.offset_kfs[kp].update(local_offset)

            # Update rest offset
            # alpha = 0.02
            alpha = 0.1  # TODO: Not sure how slow this EMA should be
            self.rest_offsets[kp] = ema_update(self.rest_offsets[kp], local_offset, alpha=alpha)

        self.anatomical_integrity = ema_update(self.anatomical_integrity, hypothesis.anatomical_score, alpha=0.1)

    # State inference

    def _infer_central_position(self, hypothesis: 'Pose3D') -> Optional[np.ndarray]:
        """
        Infer central keypoint position from other observed keypoints.
        """
        observed_kps = [kp for kp in hypothesis.names if kp in self.offset_kfs]

        if len(observed_kps) < self.config.min_kps_for_inference:
            return None

        scale = max(self.central_kf.x[6, 0], 0.1)

        # For each observed keypoint, estimate where central should be
        central_estimates = []
        weights = []

        for kp in observed_kps:
            obs_pos = hypothesis[kp].position

            # Expected offset for this keypoint (in body frame)
            expected_local_offset = self.offset_kfs[kp].x[:3, 0]
            expected_world_offset = self.body_rotation.T @ (expected_local_offset * scale)

            # Estimated central = observed - expected_offset
            estimated_central = obs_pos - expected_world_offset

            # Weight by association weight (stable keypoints contribute more)
            dynamics = self.stats.get_dynamics(kp)

            central_estimates.append(estimated_central)
            weights.append(dynamics.association_weight)

        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()

        return np.average(central_estimates, axis=0, weights=weights)

    def _estimate_body_orientation(self, keypoints: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimate body frame orientation from observed keypoints.
        """
        positions = np.array(list(keypoints.values()))

        if len(positions) < 3:
            return self.body_rotation  # keep previous estimate

        # Center the points
        centroid = positions.mean(axis=0)
        centered = positions - centroid

        # PCA via SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Principal axis (longest extent): this is (hopefully) the anterior-posterior axis
        principal_axis = Vt[0]  # first row of Vt

        # Disambiguate direction using velocity (head points in movement direction)
        velocity = self.velocity
        speed = np.linalg.norm(velocity)

        if speed > 0.05 * self.stats.reference_length_world:  # moving
            # Flip if principal axis points opposite to velocity
            if np.dot(principal_axis, velocity) < 0:
                principal_axis = -principal_axis
        else:
            # Stationary: use continuity with previous frame
            prev_axis = self.body_rotation[0, :]  # previous x-axis (forward)
            if np.dot(principal_axis, prev_axis) < 0:
                principal_axis = -principal_axis

        # Secondary axis: try to use skeleton plane normal, else use world Z
        if len(positions) >= 3:
            plane_normal = Vt[2]
            if plane_normal[2] < 0:
                plane_normal = -plane_normal
        else:
            plane_normal = np.array([0., 0., 1.])

        # Construct frame: X=forward, Z=up, Y=right (right-handed)
        x_axis = principal_axis / np.linalg.norm(principal_axis)
        z_axis = plane_normal - np.dot(plane_normal, x_axis) * x_axis
        z_norm = np.linalg.norm(z_axis)

        if z_norm < 1e-6:
            # Degenerate case: principal axis aligned with up vector
            z_axis = np.array([0., 0., 1.])
            if abs(np.dot(x_axis, z_axis)) > 0.99:
                z_axis = np.array([0., 1., 0.])
            z_axis = z_axis - np.dot(z_axis, x_axis) * x_axis
            z_axis = z_axis / np.linalg.norm(z_axis)
        else:
            z_axis = z_axis / z_norm

        y_axis = np.cross(z_axis, x_axis)

        # Rotation matrix: rows are the body axes in world coordinates
        # (this transforms world -> body: body_coords = R @ world_coords)
        R = np.array([x_axis, y_axis, z_axis])

        return R

    def _update_body_orientation(self, keypoints: Dict[str, np.ndarray], alpha: float = 0.3):
        """
        Update body orientation with temporal smoothing.
        """
        new_R = self._estimate_body_orientation(keypoints)

        # Blend with previous orientation (simple linear blend + re-orthogonalise)
        # (for small rotations this approximates SLERP)
        blended = (1 - alpha) * self.body_rotation + alpha * new_R

        # Re-orthogonalise with SVD (closest orthogonal matrix)
        U, _, Vt = np.linalg.svd(blended)
        self.body_rotation = U @ Vt

    # Serialisation

    def to_dict(self, frame_idx: int) -> dict:

        return {
            'frame_idx': frame_idx,
            'track_idx': self.track_idx,
            'keypoints': {k: v.tolist() for k, v in self.hypothesis.positions.items()},
            'predicted_keypoints': {k: v.tolist() for k, v in self.predicted_keypoints.items()},
            'scale': self.estimated_scale,
            'score': self.hypothesis.anatomical_score,
            'point_indices': {n.name: n.idx for n in self.hypothesis},
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'position_uncertainty': self.position_uncertainty.tolist(),
            'body_rotation': self.body_rotation.tolist(),
            'health': self.health,
            'anatomical_integrity': self.anatomical_integrity,
            'age': self.age,
            'time_since_update': self.time_since_update,
        }