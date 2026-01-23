from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Set, Tuple, FrozenSet, Optional, Union, Sequence
import numpy as np
from scipy.spatial import cKDTree


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
            return (0, 0)

        return (min(all_frames), max(all_frames))

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

        # Combine all unique names from all soups while preserving order as much as possible
        all_kp_names = []
        seen_kp = set()
        for s in soups:
            for name in s.keypoint_names:
                if name not in seen_kp:
                    all_kp_names.append(name)
                    seen_kp.add(name)

        all_cam_names = []
        seen_cam = set()
        for s in soups:
            for name in s.camera_names:
                if name not in seen_cam:
                    all_cam_names.append(name)
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

            # Create a translation table: map[local_id] = global_id
            lookup = np.array([global_names.index(name) for name in soup.keypoints], dtype=np.int16)
            return lookup[local_indices]

        concatenated_data['keypoint_indices'] = np.concatenate([
            remap_indices(s, 'keypoint_indices', all_kp_names) for s in soups
        ])
        concatenated_data['ray_keypoint_indices'] = np.concatenate([
            remap_indices(s, 'ray_keypoint_indices', all_kp_names) for s in soups
        ])

        return cls(
            **concatenated_data,
            keypoint_names=all_kp_names,
            camera_names=all_cam_names,
            sort=True
        )

    def to_df(self, keypoint_names: Optional[List[str]] = None) -> 'pd.DataFrame':
        """Convert the point data to a pandas DataFrame for tracking/analysis."""

        import pandas as pd

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


##

# TODO: Improve the datatypes below

@dataclass
class AssemblyNode:
    """A node in a candidate skeleton."""
    kp_name: str
    point_idx: int  # >=0 for real, <0 for virtual
    position: np.ndarray
    confidence: float

    def __eq__(self, other):
        if not isinstance(other, AssemblyNode):
            return NotImplemented
        # Nodes are equal if they refer to the same source point and same role
        return self.point_idx == other.point_idx and self.kp_name == other.kp_name

    def __hash__(self):
        # Hash based on unique ID of the point and the role name
        return hash((self.point_idx, self.kp_name))


@dataclass
class CandidateSkeleton:
    """Output from assembly stage."""

    nodes: List[AssemblyNode]
    score: float
    scale: float

    # For lineage tracking during merges
    constituent_ids: Tuple[int, ...] = field(default_factory=tuple)

    @property
    def keypoints(self) -> Dict[str, np.ndarray]:
        return {n.kp_name: n.position for n in self.nodes}

    @property
    def point_indices(self) -> Set[int]:
        return {n.point_idx for n in self.nodes}


Bone = FrozenSet[str]


@dataclass
class CandidateSkeleton:
    """Assembler's internal representation for a potential skeleton during the assembly process."""

    nodes: FrozenSet[Tuple[str, int]]
    scale: float
    competition_score: float
    anatomical_score: float


@dataclass
class AssembledSkeleton:
    """Represents a final assembled skeleton for a frame."""

    keypoints: Dict[str, np.ndarray]
    score: float
    scale: float
    point_indices: Dict[str, int] = field(default_factory=dict)
    track_idx: int = -1

    def to_dict(self) -> dict:
        return {
            'keypoints': self.keypoints,
            'score': self.score,
            'scale': self.scale,
            'point_indices': self.point_indices,
            'track_idx': self.track_idx
        }
