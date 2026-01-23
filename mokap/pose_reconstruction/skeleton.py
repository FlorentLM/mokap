import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence
import networkx as nx

from mokap.utils import common_prefix_suffix


@dataclass(frozen=True, order=True)
class Bone:
    """
    An immutable, undirected edge between two keypoints.
    """
    k1: str
    k2: str

    def __post_init__(self):
        # Enforce order
        if self.k1 > self.k2:
            # __setattr__ because dataclass is frozen
            object.__setattr__(self, 'k1', self.k2)
            object.__setattr__(self, 'k2', self.k1)

    def __contains__(self, kp: str) -> bool:
        return kp == self.k1 or kp == self.k2

    def __iter__(self):
        yield self.k1
        yield self.k2

    def __len__(self):
        return 2


class Skeleton:

    def __init__(
            self,
            keypoints: Sequence[str],
            bones: Sequence[Tuple[str, str]],
            symmetry: Optional[Sequence[Tuple[str, str]]] = None,
            name: str = "default"
    ):
        self.name = name
        self.keypoints = keypoints
        self.symmetry = symmetry or []

        # Topology
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.keypoints)
        self.graph.add_edges_from(bones)

        # Internal collections
        self._degrees = dict(self.graph.degree())
        self._bones = [Bone(u, v) for u, v in self.graph.edges]
        self.canonical_map = self._build_canonical_map()

        # Statistics
        self.reference_bone: Optional[Bone] = None
        self.median_reference_length: float = 1.0
        self.bone_stats: Dict[Bone, Dict[str, float]] = {}

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

    @property
    def bones(self) -> List[Bone]:
        return self._bones