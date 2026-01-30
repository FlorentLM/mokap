"""
Skeleton topology and statistics definitions.

- Bone: Immutable undirected edge between two keypoints
- Skeleton: Immutable skeleton topology definition
- SkeletonStats: Learned anatomical and dynamics statistics
"""
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple, Optional, Sequence, List, Union
import hashlib
import networkx as nx
import numpy as np

from lucida.geometry.backend import ArrayLike

from mokap.pose_reconstruction.configs import MIN_PROCESS_NOISE, MAX_PROCESS_NOISE
from mokap.pose_reconstruction.utils import ema_update
from mokap.utils import common_prefix_suffix

if TYPE_CHECKING:
    from mokap.pose_reconstruction.datatypes import Node3D, Pose3D


@dataclass(frozen=True, order=True)
class Bone:
    """
    A bone definition: immutable, undirected edge between two keypoints.
    """
    k1: str
    k2: str
    _sep: str = ';'

    def __post_init__(self):
        # Enforce lexicographic order
        if self.k1 > self.k2:
            k1, k2 = self.k2, self.k1
            # __setattr__ because dataclass is frozen
            object.__setattr__(self, 'k1', k1)
            object.__setattr__(self, 'k2', k2)

    def __contains__(self, kp: str) -> bool:
        return kp == self.k1 or kp == self.k2

    def __iter__(self):
        yield self.k1
        yield self.k2

    def __len__(self):
        return 2

    def __str__(self):
        return f"{self.k1}{self._sep}{self.k2}"

    def __repr__(self):
        return f'Bone({self.k1}, {self.k2})'

    def to_key(self) -> str:
        return str(self)

    @classmethod
    def from_key(cls, key: str) -> 'Bone':
        return cls(*key.split(cls._sep))


def _bone_or_key(item: Bone | str | Sequence[str]):
    if isinstance(item, Bone):
        return item
    elif isinstance(item, Sequence) and len(item) == 2:
        return Bone(*item)
    elif isinstance(item, str) and len(item) >= 3:
        if Bone._sep in item and item[0] != Bone._sep and item[-1] != Bone._sep:
            return Bone.from_key(item)
    raise KeyError(item)


@dataclass
class BoneStats:
    """
    Statistics for a single bone (relative to a reference).
    """
    ratio_length: float     # length relative to reference bone
    variability: float      # intra-individual variation (MAD of the ratio length, pooled)
    count: int = 0          # number of observations
    pairs: int = 0          # number of tracklet pairs used

    # Optional absolute measurements
    length_world: Optional[float] = None

    def update(self, current_length: float, reference_length: float, alpha: float = 0.01):
        """Update stats with a new observation."""

        ratio_obs = current_length / reference_length
        self.ratio_length = ema_update(self.ratio_length, ratio_obs, alpha)
        self.length_world = self.ratio_length * reference_length
        self.count += 1

    def to_dict(self) -> dict:
        d = {
            'ratio_length': self.ratio_length,
            'variability': self.variability,
            'count': self.count,
            'pairs': self.pairs,
        }
        if self.length_world is not None:
            d['length_world'] = self.length_world
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'BoneStats':
        return cls(
            ratio_length=d['ratio_length'],
            variability=d['variability'],
            count=d.get('count', 0),
            pairs=d.get('pairs', 0),
            length_world=d.get('length_world')
        )


@dataclass
class KeypointDynamics:
    """
    Per-keypoint dynamics parameters for Kalman filtering.
    """
    process_noise: float  # Q (how erratic is this keypoint's motion)
    measurement_noise: float  # R (observation uncertainty)
    association_weight: float  # How much to trust this keypoint in association (0-1)
    source: str = "default"  # Where these params came from: "data", "topology_prior", "default"

    def update(self, observed_process_noise: Optional[float] = None, alpha: float = 0.01):
        """Update dynamics parameters (for example adapt Q based on observed velocity)."""

        if observed_process_noise is not None:
            new_val = np.clip(observed_process_noise, MIN_PROCESS_NOISE, MAX_PROCESS_NOISE)
            self.process_noise = ema_update(self.process_noise, float(new_val), alpha)

    def to_dict(self) -> dict:
        return {
            'process_noise': self.process_noise,
            'measurement_noise': self.measurement_noise,
            'association_weight': self.association_weight,
            'source': self.source
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'KeypointDynamics':
        return cls(
            process_noise=d['process_noise'],
            measurement_noise=d['measurement_noise'],
            association_weight=d['association_weight'],
            source=d.get('source', 'loaded')
        )


@dataclass
class SkeletonMetadata:
    """
    Metadata for a skeleton definition.

    species: Scientific or common species name
    common_name: Display name for the skeleton
    skeleton_type: 'articulated' or 'rigid' (fixed shape)
    reference_bone: Tuple of (kp1, kp2) defining the reference bone for scaling
    units: World coordinate units ('mm', 'cm', 'px', etc)
    notes: Free-form notes about the skeleton
    version: Schema version for forward compatibility
    created_date: ISO date string when skeleton was created
    """
    species: str = 'unknown'
    common_name: str = ''
    skeleton_type: str = 'articulated'
    reference_bone: Optional[Tuple[str, str]] = None
    units: str = 'mm'
    notes: str = ''
    version: str = '1.0'
    created_date: str = ''


# TODO: Consider adding this:
#     - [bones] section: optional prior statistics (mean length, std) treated as ground truth
#     - [bones] section could also be extended with angle affordances, joint limits
#     - Real-life measurements could be added as ground truth for validation
#     - Deformable skeleton types could include mesh/surface definitions


class Skeleton:
    """
    An immutable definition of a Skeleton topology.

    - Graph structure (keypoints, bones, connectivity)
    - Topological classifications (leaves, anchors, central keypoint)
    - Canonical name mapping for symmetry pooling
    - Graph distances for dynamics priors
    """

    def __init__(
            self,
            keypoints: Sequence[str],
            bones: Sequence[Tuple[str, str]],
            symmetry: Optional[Sequence[Tuple[str, str]]] = None,
            name: str = "default",
            metadata: Optional[SkeletonMetadata] = None
    ):
        self.name = name
        self.keypoints = tuple(keypoints)
        self.bones = tuple([Bone(u, v) for u, v in bones])
        self.symmetry = tuple(symmetry) if symmetry else ()

        self.metadata = metadata

        self._bone_set = set(self.bones)
        self._keypoint_set = set(self.keypoints)

        # Topological classification
        self._graph = nx.Graph()
        self._graph.add_nodes_from(self.keypoints)
        self._graph.add_edges_from(self.bones)
        self._degrees = dict(self._graph.degree())

        leaves, anchors = set(), set()
        for kp, deg in self._degrees.items():
            if deg == 1:
                leaves.add(kp)
            if deg > 1:
                anchors.add(kp)

        self._leaf_keypoints = tuple(leaves)
        self._anchor_keypoints = tuple(sorted(anchors, key=lambda k: self._degrees[k], reverse=True))
        self._central_keypoint: str = max(self._degrees, key=self._degrees.get)

        self._canonical_map = self._build_canonical_map()
        self._graph_distances = self._compute_graph_distances()

    def __contains__(self, item: Bone | str | Sequence[str]) -> bool:
        try:
            return _bone_or_key(item) in self._bone_set
        except KeyError:
            return item in self._keypoint_set

    def __repr__(self):
        return f"Skeleton('{self.name}', {len(self.keypoints)} keypoints, {len(self.bones)} bones)"

    def _build_canonical_map(self) -> Dict[str, str]:
        """Build a mapping from each keypoint to its canonical name."""
        mapping = {kp: kp for kp in self.keypoints}

        for left, right in self.symmetry:
            if left in mapping and right in mapping:
                prefix, suffix = common_prefix_suffix(left, right)
                canonical = prefix + suffix if prefix or suffix else left
                mapping[left] = canonical
                mapping[right] = canonical

        return mapping

    def _compute_graph_distances(self) -> Dict[str, int]:
        """Compute shortest path distances from central keypoint."""
        central = self.central_keypoint
        distances = {central: 0}

        if not nx.is_connected(self._graph):
            for kp in self.keypoints:
                if kp not in distances:
                    distances[kp] = 0
            return distances

        lengths = nx.single_source_shortest_path_length(self._graph, central)
        distances.update(lengths)
        return distances

    # Public properties and accessors

    @property
    def hash(self) -> str:
        """
        Short hash identifying this skeleton topology.
        (useful for tagging stats files to their skeleton definition)
        """
        content = f"{self.name}|{','.join(sorted(self.keypoints))}|{','.join(sorted(str(b) for b in self.bones))}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    @property
    def leaf_keypoints(self) -> Tuple[str, ...]:
        """
        Keypoints with degree 1 (extremities).
        """
        return self._leaf_keypoints

    @property
    def anchor_keypoints(self) -> Tuple[str, ...]:
        """
        Keypoints with degree > 1 (junctions).
        """
        return self._anchor_keypoints

    @property
    def central_keypoint(self) -> str:
        """
        Most connected keypoint.
        """
        return self._anchor_keypoints[0] if self._anchor_keypoints else self.keypoints[0]

    @property
    def central_bone(self) -> Bone:
        """Most connected bone."""
        return max(self.bones, key=lambda b: self._degrees[b.k1] + self._degrees[b.k2])

    @property
    def canonical_map(self) -> Dict[str, str]:
        """Mapping from keypoint names to their canonical names."""
        return self._canonical_map

    def degree(self, keypoint: str) -> int:
        return self._degrees[keypoint]

    def neighbours(self, keypoint: str) -> List[str]:
        return list(self._graph.neighbors(keypoint))

    def canonical(self, item: Bone | str | Sequence[str]) -> str:
        """Get canonical name from a keypoint or bone name."""
        try:
            bone = _bone_or_key(item)
            return Bone._sep.join(sorted([self.canonical(bone.k1), self.canonical(bone.k2)]))
        except KeyError:
            return self.canonical_map[item]

    def graph_distance(self, keypoint: str) -> int:
        """
        Return graph distance from keypoint to central.
        """
        return self._graph_distances.get(keypoint, len(self.keypoints))

    def adjacent_bones(self, keypoint: str) -> List[Bone]:
        """
        Return all bones connected to a keypoint.
        """
        return [b for b in self.bones if keypoint in b]

    def draw(self):
        # TODO: This needs to be symmetry and stats aware
        pos = nx.kamada_kawai_layout(self._graph)
        nx.draw(self._graph, pos=pos, with_labels=True)

    # I/O methods

    def save(self, path: Union[Path, str]) -> None:
        """
        Save skeleton definition to TOML file.

        Args:
            path: Output path (.toml)
        """
        from mokap.mokap_io import save_skeleton
        save_skeleton(self, path)

    @classmethod
    def load(cls, path: Union[Path, str]) -> 'Skeleton':
        """
        Load skeleton from file.

        Supports TOML (.toml) and SLEAP (.slp) formats.
        If a directory is provided, searches for .slp files.

        Args:
            path: Path to skeleton file or directory

        Returns:
            Skeleton instance
        """
        from mokap.mokap_io import load_skeleton, load_skeleton_sleap

        path = Path(path)

        if path.suffix == '.toml':
            return load_skeleton(path)
        elif path.suffix == '.slp':
            return load_skeleton_sleap(path)
        else:
            # Default to SLEAP for directories or unknown extensions
            return load_skeleton_sleap(path)


class SkeletonStats:
    """
    Learned statistics for a skeleton definition.

    - Bone length ratios and variabilities (anatomy)
    - Per-keypoint dynamics (process noise, association weights)
    """

    def __init__(self, skeleton: Skeleton):
        self.skeleton = skeleton

        # Anatomy
        self.reference_bone: Optional[Bone] = None
        self.reference_length_world: float = 1.0  # length of the reference bone in world units

        self.bone_stats: Dict[Bone, BoneStats] = {}

        # Dynamics
        self.keypoint_dynamics: Dict[str, KeypointDynamics] = {}

    def __repr__(self):
        return (f"SkeletonStats(skeleton='{self.skeleton.name}', "
                f"bones={len(self.bone_stats)}, dynamics={len(self.keypoint_dynamics)})")

    @property
    def skeleton_hash(self) -> str:
        """Skeleton ID for file naming."""
        return self.skeleton.hash

    # Anatomy accessors

    def expected_ratio(self, bone: Bone | str | Sequence[str]) -> float:
        """
        Expected length in relation to reference bone.
        """
        return self.bone_stats[_bone_or_key(bone)].ratio_length

    def expected_length(self, bone: Bone | str | Sequence[str]) -> float:
        """
        Absolute expected length in world units.
        """
        return self.expected_ratio(bone) * self.reference_length_world

    def ratio_variability(self, bone: Bone | str | Sequence[str]) -> float:
        return self.bone_stats[_bone_or_key(bone)].variability

    def length_variability(self, bone: Bone | str | Sequence[str]) -> float:
        return self.ratio_variability(bone) * self.reference_length_world

    def ratio_bounds(self, bone: Bone | str | Sequence[str], n_sigma: float = 3.0) -> Tuple[float, float]:
        """
        Acceptable ratio range for validation.
        """
        expected = self.expected_ratio(bone)
        tolerance = n_sigma * self.ratio_variability(bone)
        return expected - tolerance, expected + tolerance

    def length_bounds(self, bone: Bone | str | Sequence[str], n_sigma: float = 3.0) -> Tuple[float, float]:
        """
        Acceptable length range for validation.
        """
        lower, upper = self.ratio_bounds(bone, n_sigma)
        return lower * self.reference_length_world, upper * self.reference_length_world

    def estimate_scale(self,
                       hypothesis: Union['Pose3D', Dict[str, 'Node3D'], Dict[str, ArrayLike]],
                       min_scale: float = 0.3,
                       max_scale: float = 4.0
                       ) -> float:
        """
        Estimate skeleton scale from observed keypoint positions.

        Args:
            hypothesis: A SkeletonHypothesis, Dict[str, Node], or Dict[str, position]
            min_scale: Minimum acceptable scale
            max_scale: Maximum acceptable scale
        """

        if hasattr(hypothesis, 'positions'):
            keypoints = hypothesis.positions
        elif hypothesis and hasattr(next(iter(hypothesis.values())), 'position'):
            keypoints = {name: node.position for name, node in hypothesis.items()}
        else:
            keypoints = hypothesis

        bones_scales = []

        for bone, stats in self.bone_stats.items():
            if bone.k1 in keypoints and bone.k2 in keypoints:

                length_expected = self.reference_length_world * stats.ratio_length
                length_observed = float(np.linalg.norm(keypoints[bone.k1] - keypoints[bone.k2]))

                bone_scale = length_observed / length_expected

                if min_scale <= bone_scale <= max_scale:
                    bones_scales.append(bone_scale)

        if not bones_scales:
            return 1.0

        return float(np.nanmedian(bones_scales))

    def score_bone(
            self,
            bone: Bone | str | Sequence[str],
            node1: 'Node3D',
            node2: 'Node3D',
            scale: float = 1.0,
            MAD_threshold: float = 5.0,
            MAD_floor: float = 0.05
    ) -> float:
        """
        Score a proposed bone based on consistency with learned statistics.

        Args:
            bone: The bone being scored
            node1: First endpoint node
            node2: Second endpoint node
            scale: Current skeleton scale estimate
            MAD_threshold: Number of MADs beyond which to reject
            MAD_floor: Minimum variability floor
        """

        if bone not in self.bone_stats:
            return 0.0

        stats = self.bone_stats[bone]
        proposed = float(np.linalg.norm(node1.position - node2.position))

        expected = stats.ratio_length * self.reference_length_world * scale
        variability = stats.variability * self.reference_length_world * scale + MAD_floor

        n_mads = abs(proposed - expected) / max(1e-6, variability)
        if n_mads > MAD_threshold:
            return -1000.0

        length_score = np.exp(-0.5 * n_mads ** 2)
        confidence_score = (node1.confidence + node2.confidence) / 2.0
        return length_score * confidence_score

    def update_anatomy(self, keypoints: Dict[str, np.ndarray]) -> bool:
        """
        Update anatomy statistics from a high-quality pose observation.
        Returns True if the sample was accepted.
        """
        # TODO: This method should probably be scale-aware..?

        # Only accept if reference bone is present
        if not (self.reference_bone and self.reference_bone.k1 in keypoints and self.reference_bone.k2 in keypoints):
            return False

        ref_obs = float(np.linalg.norm(keypoints[self.reference_bone.k1] - keypoints[self.reference_bone.k2]))

        self.reference_length_world = ema_update(self.reference_length_world, ref_obs)

        for bone, stats in self.bone_stats.items():
            if bone.k1 in keypoints and bone.k2 in keypoints:
                dist = float(np.linalg.norm(keypoints[bone.k1] - keypoints[bone.k2]))
                stats.update(dist, self.reference_length_world)

        return True

    # Dynamics accessors

    def get_dynamics(self, keypoint: str) -> 'KeypointDynamics':
        """Get dynamics for a keypoint."""

        if keypoint in self.keypoint_dynamics:
            return self.keypoint_dynamics[keypoint]

        # Fallback based on graph distance (extremities are more erratic)
        graph_dist = self.skeleton.graph_distance(keypoint)
        return KeypointDynamics(
            process_noise=0.1 * (1 + graph_dist),
            measurement_noise=0.1,
            association_weight=1.0 / (1 + graph_dist),
            source='fallback'
        )

    def update_dynamics(self, keypoint: str, observed_metric: float, alpha: float = 0.01):
        """Update dynamics stats."""

        if keypoint in self.keypoint_dynamics:
            self.keypoint_dynamics[keypoint].update(observed_metric, alpha)

    # Serialisation

    def to_dict(self) -> dict:
        """Serialise to dictionary."""
        data = {
            'skeleton_hash': self.skeleton_hash,
            'skeleton_name': self.skeleton.name,
            'anatomy': {},
            'dynamics': {}
        }

        if self.reference_bone:
            data['anatomy'] = {
                'reference_bone': self.reference_bone.to_key(),
                'reference_length_world': self.reference_length_world,
                'bones': {b.to_key(): s.to_dict() for b, s in self.bone_stats.items()}
            }

        if self.keypoint_dynamics:
            data['dynamics'] = {k: d.to_dict() for k, d in self.keypoint_dynamics.items()}

        return data

    @classmethod
    def from_dict(cls, data: dict, skeleton: Skeleton) -> 'SkeletonStats':
        """Deserialise from dictionary."""
        stats = cls(skeleton)

        # Verify skeleton compatibility
        stored_hash = data.get('skeleton_hash')

        if stored_hash and stored_hash != skeleton.hash:
            warnings.warn(
                f"Stats skeleton_hash '{stored_hash}' doesn't match skeleton '{skeleton.hash}'. "
                "Stats may not be compatible with this skeleton definition.",
                UserWarning
            )

        if 'anatomy' in data and data['anatomy']:
            anat = data['anatomy']
            if 'reference_bone' in anat:
                stats.reference_bone = _bone_or_key(anat['reference_bone'])
                stats.reference_length_world = anat.get('reference_length_world', 1.0)

                for k, v in anat.get('bones', {}).items():
                    try:
                        bone = Bone.from_key(k)
                        if bone in skeleton:
                            bs = BoneStats.from_dict(v)

                            # fill absolute length if missing
                            if bs.length_world is None:
                                bs.length_world = bs.ratio_length * stats.reference_length_world
                            stats.bone_stats[bone] = bs

                    except (KeyError, ValueError):
                        pass

        if 'dynamics' in data and data['dynamics']:
            for k, v in data['dynamics'].items():
                if k in skeleton.keypoints:
                    stats.keypoint_dynamics[k] = KeypointDynamics.from_dict(v)

        return stats

    # I/O methods

    def save(self, path: Union[Path, str], merge_existing: bool = True) -> None:
        """
        Save statistics to JSON file.

        Args:
            path: Output path (.json)
            merge_existing: If True, merge with existing file
        """
        from mokap.mokap_io import save_skeleton_stats
        save_skeleton_stats(self, path, merge_existing=merge_existing)

    @classmethod
    def load(cls, path: Union[Path, str], skeleton: Skeleton) -> 'SkeletonStats':
        """
        Load statistics from JSON file.

        Args:
            path: Path to stats.json
            skeleton: Skeleton instance (required for reconstruction)
        """
        from mokap.mokap_io import load_skeleton_stats
        return load_skeleton_stats(path, skeleton)