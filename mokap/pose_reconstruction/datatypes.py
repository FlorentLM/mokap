from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, FrozenSet
import numpy as np


# TODO: Improve these datatypes


@dataclass
class SoupData:
    """Soup of 3D points from multi-view reconstruction (as SoA)."""

    positions: np.ndarray  # (N, 3)
    confidences: np.ndarray  # (N,)
    reprojection_errors: np.ndarray  # (N,)
    kp_types: np.ndarray  # (N,) int16
    frame_indices: np.ndarray  # (N,) int32
    camera_masks: np.ndarray  # (N,) uint64 bitmask of contributing cameras

    # Orphan rays (single-view detections)
    ray_origins: np.ndarray  # (M, 3)
    ray_directions: np.ndarray  # (M, 3)
    ray_confidences: np.ndarray  # (M,)
    ray_kp_types: np.ndarray  # (M,) int16
    ray_frame_indices: np.ndarray  # (M,) int32

    # Metadata
    keypoint_names: List[str] = field(default_factory=list)
    camera_names: List[str] = field(default_factory=list)

    @property
    def num_points(self) -> int:
        return len(self.positions)

    @property
    def num_rays(self) -> int:
        return len(self.ray_origins)

    def get_frame(self, frame_idx: int) -> 'SoupData':
        """Slice view into the soup."""

        pt_mask = self.frame_indices == frame_idx
        ray_mask = self.ray_frame_indices == frame_idx

        return SoupData(
            positions=self.positions[pt_mask],
            confidences=self.confidences[pt_mask],
            reprojection_errors=self.reprojection_errors[pt_mask],
            kp_types=self.kp_types[pt_mask],
            frame_indices=self.frame_indices[pt_mask],
            camera_masks=self.camera_masks[pt_mask],
            ray_origins=self.ray_origins[ray_mask],
            ray_directions=self.ray_directions[ray_mask],
            ray_confidences=self.ray_confidences[ray_mask],
            ray_kp_types=self.ray_kp_types[ray_mask],
            ray_frame_indices=self.ray_frame_indices[ray_mask],
            keypoint_names=self.keypoint_names,
            camera_names=self.camera_names
        )

    @staticmethod
    def concatenate(soups: List['SoupData']) -> 'SoupData':
        """Combine multiple SoupData objects."""
        if not soups:
            raise ValueError("Empty list")
        return SoupData(
            positions=np.vstack([s.positions for s in soups]),
            confidences=np.concatenate([s.confidences for s in soups]),
            reprojection_errors=np.concatenate([s.reprojection_errors for s in soups]),
            kp_types=np.concatenate([s.kp_types for s in soups]),
            frame_indices=np.concatenate([s.frame_indices for s in soups]),
            camera_masks=np.concatenate([s.camera_masks for s in soups]),
            ray_origins=np.vstack([s.ray_origins for s in soups]) if any(s.num_rays > 0 for s in soups) else np.empty((0, 3)),
            ray_directions=np.vstack([s.ray_directions for s in soups]) if any(s.num_rays > 0 for s in soups) else np.empty((0, 3)),
            ray_confidences=np.concatenate([s.ray_confidences for s in soups]),
            ray_kp_types=np.concatenate([s.ray_kp_types for s in soups]),
            ray_frame_indices=np.concatenate([s.ray_frame_indices for s in soups]),
            keypoint_names=soups[0].keypoint_names,
            camera_names=soups[0].camera_names
        )

    @staticmethod
    def empty(keypoint_names: List[str], camera_names: List[str]) -> 'SoupData':
        return SoupData(
            positions=np.empty((0, 3), dtype=np.float32),
            confidences=np.array([], dtype=np.float32),
            reprojection_errors=np.array([], dtype=np.float32),
            kp_types=np.array([], dtype=np.int16),
            frame_indices=np.array([], dtype=np.int32),
            camera_masks=np.array([], dtype=np.uint64),
            ray_origins=np.empty((0, 3), dtype=np.float32),
            ray_directions=np.empty((0, 3), dtype=np.float32),
            ray_confidences=np.array([], dtype=np.float32),
            ray_kp_types=np.array([], dtype=np.int16),
            ray_frame_indices=np.array([], dtype=np.int32),
            keypoint_names=keypoint_names,
            camera_names=camera_names
        )


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
