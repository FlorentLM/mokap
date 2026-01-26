import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple, Optional, Sequence, List, Union
import networkx as nx
import numpy as np

from lucida.geometry.backend import ArrayLike
from mokap.utils import common_prefix_suffix

if TYPE_CHECKING:
    from mokap.pose_reconstruction.datatypes import Node3D, SkeletonHypothesis


def _ema_update(existing: float, new: float, alpha: float = 0.01):
    return (1 - alpha) * existing + alpha * new


@dataclass(frozen=True, order=True)
class BoneDefinition:
    """
    A bone definition: immutable, undirected edge between two keypoints.
    """
    k1: str
    k2: str
    _sep: str = ';'

    def __post_init__(self):
        # Enforce order
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
    def from_key(cls, key: str) -> 'BoneDefinition':
        return cls(*key.split(cls._sep))


def _bone_or_key(item: BoneDefinition | str | Sequence[str]):
    if isinstance(item, BoneDefinition):
        return item
    if isinstance(item, str) and len(item) >= 3:
        if BoneDefinition._sep in item and item[0] != BoneDefinition._sep and item[-1] != BoneDefinition._sep:
            return BoneDefinition.from_key(item)
    if isinstance(item, Sequence) and len(item) == 2:
        return BoneDefinition(*item)
    raise KeyError(item)


@dataclass
class BoneStats:
    """
    Statistics for a single bone (relative to a reference).
    """
    ratio_length: float  # length relative to reference bone
    variability: float  # intra-individual variation (MAD of the ratio length, pooled)
    count: int = 0  # number of observations
    pairs: int = 0  # number of tracklet pairs used

    # Optional absolute measurements (if known/calibrated)
    length_world: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            'ratio_length': self.ratio_length,
            'variability': self.variability,
            'count': self.count,
            'pairs': self.pairs,
            'length_world': self.length_world
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'BoneStats':
        return cls(
            ratio_length=d['ratio_length'],
            variability=d['variability'],
            count=d.get('count', 0),
            pairs=d.get('pairs', 0),
            length_world=d.get('length_world')
        )


class SkeletonTopology:
    """
    An immutable definition of a Skeleton topology.

    Provides:
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
            name: str = "default"
    ):
        self.name = name
        self.keypoints = tuple(keypoints)
        self.bones = tuple([BoneDefinition(u, v) for u, v in bones])
        self.symmetry = tuple(symmetry) if symmetry else ()

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

        # Graph distances from central keypoint
        self._graph_distances: Dict[str, int] = nx.single_source_shortest_path_length(
            self._graph, self._central_keypoint
        )

        # Canonical map for symmetry pooling
        self.canonical_map = self._build_canonical_map()

    @classmethod
    def from_sleap(cls, slp_path: str | Path):
        """Factory method to load topology and symmetry from SLEAP files."""
        import sleap_io

        slp_path = Path(slp_path)

        if slp_path.is_dir():
            slp_path = next(slp_path.glob('*.slp'))

        slp_content = sleap_io.load_file(slp_path)
        return cls(
            keypoints=slp_content.skeleton.node_names,
            bones=slp_content.skeleton.edge_names,
            symmetry=slp_content.skeleton.symmetry_names
        )

    def _build_canonical_map(self) -> Dict[str, str]:
        """Maps specific keypoints to pooled types ('left_hand' -> 'hand')."""
        canonical_map = {}
        delim = re.compile(r'[-_. ;,]')

        if self.symmetry:
            for name1, name2 in self.symmetry:
                prefix, suffix = common_prefix_suffix(name1, name2)
                side_part = name1[len(prefix):len(name1) - len(suffix)]
                # Remove side and clean
                canon = name1.replace(side_part, '')
                canon = delim.sub('', canon).lower()
                canonical_map[name1] = canonical_map[name2] = canon

        for kp in self.keypoints:
            if kp not in canonical_map:
                canonical_map[kp] = delim.sub('', kp).lower()
        return canonical_map

    def __contains__(self, item: BoneDefinition | str | Sequence[str]) -> bool:
        try:
            bone = _bone_or_key(item)
            return bone in self._bone_set
        except KeyError:
            return item in self._keypoint_set

    @property
    def leaf_keypoints(self) -> Tuple[str]:
        """Keypoints with degree 1 (extremities)."""
        return self._leaf_keypoints

    @property
    def anchor_keypoints(self) -> Tuple[str]:
        """Keypoints with degree > 1 (junctions)."""
        return self._anchor_keypoints

    @property
    def central_keypoint(self) -> str:
        """Most connected keypoint."""
        return self._central_keypoint

    @property
    def central_bone(self) -> BoneDefinition:
        """Most connected bone (stable for scale estimation)."""
        return max(self.bones, key=lambda b: self._degrees[b.k1] + self._degrees[b.k2])

    def degree(self, keypoint: str) -> int:
        return self._degrees[keypoint]

    def neighbours(self, keypoint: str) -> List[str]:
        return list(self._graph.neighbors(keypoint))

    def canonical(self, item: BoneDefinition | str | Sequence[str]) -> str:
        """Get canonical name from a keypoint or bone name."""
        try:
            bone = _bone_or_key(item)
            canon1 = self.canonical(bone.k1)
            canon2 = self.canonical(bone.k2)
            return BoneDefinition._sep.join(sorted([canon1, canon2]))

        except KeyError:
            return self.canonical_map[item]

    def graph_distance(self, keypoint: str) -> int:
        """Distance from central keypoint in graph hops."""
        return self._graph_distances.get(keypoint, len(self.keypoints))

    def draw(self):
        # TODO: This needs to be symmetry and stats aware
        pos = nx.kamada_kawai_layout(self._graph)
        nx.draw(self._graph, pos=pos, with_labels=True)


class SkeletonStats:
    """
    Learned statistics for a skeleton topology.

    Separate from Skeleton class because:
    - Skeleton topology is immutable (from annotation)
    - Stats are learned/updated during bootstrap and tracking
    - Stats may be serialized/loaded independently
    """

    def __init__(self, skeleton: SkeletonTopology):
        self.skeleton = skeleton

        self.reference_bone: Optional[BoneDefinition] = None
        self.reference_length_world = 1.0  # length of the reference bone in world units

        self.bone_stats: Dict[BoneDefinition, BoneStats] = {}

    def expected_ratio(self, bone: BoneDefinition | str | Sequence[str]) -> float:
        """Expected length in relation to reference bone."""
        return self.bone_stats[_bone_or_key(bone)].ratio_length

    def expected_length(self, bone: BoneDefinition | str | Sequence[str]) -> float:
        """Absolute expected length in world units."""
        return self.expected_ratio(bone) * self.reference_length_world

    def ratio_variability(self, bone: BoneDefinition | str | Sequence[str]) -> float:
        return self.bone_stats[_bone_or_key(bone)].variability

    def length_variability(self, bone: BoneDefinition | str | Sequence[str]) -> float:
        return self.ratio_variability(bone) * self.reference_length_world

    def ratio_bounds(self, bone: BoneDefinition | str | Sequence[str], n_sigma: float = 3.0) -> Tuple[float, float]:
        """Acceptable ratio range for validation."""
        expected = self.expected_ratio(bone)
        tolerance = n_sigma * self.ratio_variability(bone)
        return expected - tolerance, expected + tolerance

    def length_bounds(self, bone: BoneDefinition | str | Sequence[str], n_sigma: float = 3.0) -> Tuple[float, float]:
        """Acceptable length range for validation."""
        lower, upper = self.ratio_bounds(bone, n_sigma)
        return lower * self.reference_length_world, upper * self.reference_length_world

    def score_bone(
            self,
            bone: BoneDefinition | str | Sequence[str],
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
        if bone not in self.skeleton:
            return 0.0

        proposed_length = float(np.linalg.norm(node1.position - node2.position))
        expected = self.expected_length(bone) * scale

        variability = self.length_variability(bone) * scale + MAD_floor
        n_mads = abs(proposed_length - expected) / max(1e-6, variability)

        if n_mads > MAD_threshold:
            return -1000.0

        length_score = np.exp(-0.5 * n_mads ** 2)
        confidence_score = (node1.confidence + node2.confidence) / 2.0
        return length_score * confidence_score

    def estimate_scale(self,
                       hypothesis: Union['SkeletonHypothesis', Dict[str, 'Node3D'], Dict[str, ArrayLike]],
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

    def update(self, keypoints: Dict[str, np.ndarray]) -> bool:
        """
        Update statistics from a high-quality pose observation.
        Returns True if the sample was accepted.
        """

        # TODO: This method should probably be scale-aware..?

        # Only accept if reference bone is present
        if not (self.reference_bone.k1 in keypoints and self.reference_bone.k2 in keypoints):
            return False

        ref_bone_length_obs = float(
            np.linalg.norm(keypoints[self.reference_bone.k1] - keypoints[self.reference_bone.k2]))

        # Update reference length with exponential moving average
        self.reference_length_world = _ema_update(self.reference_length_world, ref_bone_length_obs)

        for bone, stats in self.bone_stats.items():
            if bone.k1 in keypoints and bone.k2 in keypoints:
                length_observed = float(np.linalg.norm(keypoints[bone.k1] - keypoints[bone.k2]))
                ratio_obs = length_observed / ref_bone_length_obs

                # Update ratio length with exponential moving average
                stats.ratio_length = _ema_update(stats.ratio_length, ratio_obs)
                stats.count += 1

        return True

    # Serialization

    def to_dict(self) -> dict:
        return {
            'reference_bone': self.reference_bone.to_key(),
            'reference_length_world': self.reference_length_world,
            'bones': {
                bone.to_key(): stats.to_dict()
                for bone, stats in self.bone_stats.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict, skeleton: SkeletonTopology) -> 'SkeletonStats':
        skel_stats = data['anatomy']

        ref_bone = _bone_or_key(skel_stats.get('reference_bone'))
        ref_len = skel_stats.get('reference_length_world', 1.0)

        stats = cls(skeleton)
        stats.reference_bone = ref_bone
        stats.reference_length_world = ref_len

        for key, bone_data in skel_stats['bones'].items():
            bone = BoneDefinition.from_key(key)

            if bone in stats.skeleton:
                bone_data = BoneStats.from_dict(bone_data)
                if bone_data.length_world is None:
                    bone_data.length_world = ref_len * bone_data.ratio_length

                stats.bone_stats[bone] = bone_data
        return stats

    def to_json(self, path: Path | str):
        path = Path(path)

        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        if not path.is_file():
            data = {'anatomy': self.to_dict()}
        else:
            data = json.loads(path.read_text())
            data['anatomy'] = self.to_dict()

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: Path, skeleton: SkeletonTopology) -> 'SkeletonStats':
        return cls.from_dict(json.loads(path.read_text()), skeleton)