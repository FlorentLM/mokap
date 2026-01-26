"""
Skeleton assembly and multi-individual tracking.
"""
import logging
from typing import Tuple, Optional, Dict, List, Set
from itertools import combinations
from collections import defaultdict

import numpy as np
import networkx as nx
from alive_progress import alive_bar
from scipy.optimize import linear_sum_assignment

from mokap.pose_reconstruction.skeleton import BoneDefinition, SkeletonTopology, SkeletonStats
from mokap.pose_reconstruction.datatypes import Node3D, SkeletonHypothesis, PointSoup, Tracklet, TimestepData
from mokap.pose_reconstruction.configs import AssemblerConfig, TrackerConfig
from mokap.pose_reconstruction.utils import solve_mwis

logger = logging.getLogger(__name__)


class SkeletonAssembler:
    """
    Assembles skeleton hypotheses in a frame view.

    Uses a greedy growth strategy starting from anchor keypoints,
    with orphan detection rescue via ray-sphere intersection.
    """

    def __init__(
            self,
            skeleton: SkeletonTopology,
            skeleton_stats: SkeletonStats,
            config: AssemblerConfig,
    ):
        self.skeleton = skeleton
        self.skeleton_stats = skeleton_stats
        self.config = config

    def assemble(self, frame_data: TimestepData) -> List[SkeletonHypothesis]:
        """
        Main assembly entry point.
        Returns a list of skeleton hypotheses (both initial and merged).
        """
        if not frame_data:
            return []

        # Generate initial fragments (including orphan rescue)
        initial_fragments = self._generate_hypotheses(frame_data)
        if not initial_fragments:
            return []

        # Generate merge hypotheses
        merge_hypotheses = self._generate_merge_hypotheses(initial_fragments)

        return initial_fragments + merge_hypotheses

    def _generate_hypotheses(self, frame_data: TimestepData) -> List[SkeletonHypothesis]:
        """Generate skeleton hypotheses by seeding from anchors and leaves."""

        candidate_skeletons = []
        used_indices: Set[int] = set()

        # Seed from anchors (full skeleton growth)
        for anchor_kp in self.skeleton.anchor_keypoints:
            for seed_idx in frame_data.get_indices(anchor_kp):

                if seed_idx in used_indices:
                    continue

                seed_node = frame_data.get_node(anchor_kp, seed_idx)
                candidate = self._grow_skeleton(frame_data, seed_nodes={anchor_kp: seed_node})

                if candidate:
                    candidate_skeletons.append(candidate)
                    used_indices.update(idx for idx in candidate.point_indices if idx >= 0)

        # Seed from leaves (limited growth)
        for leaf_kp in self.skeleton.leaf_keypoints:
            for seed_idx in frame_data.get_indices(leaf_kp):

                if seed_idx in used_indices:
                    continue

                seed_node = frame_data.get_node(leaf_kp, seed_idx)
                fragment = self._grow_skeleton(
                    frame_data,
                    seed_nodes={leaf_kp: seed_node},
                    max_iterations=1,
                    min_score_threshold=self.config.min_bone_score_for_fragment
                )

                if fragment:
                    candidate_skeletons.append(fragment)
                    used_indices.update(idx for idx in fragment.point_indices if idx >= 0)

        return candidate_skeletons

    def _generate_merge_hypotheses(
            self,
            fragments: List[SkeletonHypothesis],
    ) -> List[SkeletonHypothesis]:
        """Generate hypotheses by merging compatible fragments."""

        if len(fragments) < 2:
            return []

        # Tag fragments with their index for merge provenance
        for i, frag in enumerate(fragments):
            frag.constituent_indices = frozenset([i])

        merge_hypotheses = []

        for i, j in combinations(range(len(fragments)), 2):
            merged = self._try_merge(fragments[i], fragments[j])

            if merged:
                merged.constituent_indices = (fragments[i].constituent_indices | fragments[j].constituent_indices)
                merge_hypotheses.append(merged)

        return merge_hypotheses

    def _grow_skeleton(
            self,
            frame_data: TimestepData,
            seed_nodes: Dict[str, Node3D],
            max_iterations: Optional[int] = None,
            min_score_threshold: float = 0.0
    ) -> Optional[SkeletonHypothesis]:
        """
        Grow a skeleton from seed node(s) using greedy extension.

        Args:
            frame_data: Current frame's point data
            seed_nodes: Initial nodes to grow from (name -> Node)
            max_iterations: Maximum growth steps (None = unlimited)
            min_score_threshold: Minimum final score to accept result
        """
        current_nodes = dict(seed_nodes)
        total_bone_score = 0.0
        num_bones = 0
        iterations = 0

        max_search_radius = self.skeleton_stats.reference_length_world * self.config.max_bone_len

        while max_iterations is None or iterations < max_iterations:
            # Find candidate extensions
            candidates = self._find_extension_candidates(frame_data, current_nodes, max_search_radius)

            if not candidates:
                break

            # Check scale sanity
            current_scale = self.skeleton_stats.estimate_scale(
                current_nodes,
                min_scale=self.config.min_sane_scale,
                max_scale=self.config.max_sane_scale
            )
            if not (self.config.min_sane_scale < current_scale < self.config.max_sane_scale):
                break

            # Calculate current growth score for comparison
            current_growth_score = self._compute_growth_score(
                total_bone_score, num_bones, len(current_nodes)
            )

            # Evaluate each candidate and pick the best
            best_extension = None
            best_growth_score = -float('inf')

            for cand_node in candidates:
                extension_result = self._evaluate_extension(current_nodes, cand_node, current_scale)

                if extension_result is None:
                    continue

                new_bone_score, new_bone_count = extension_result

                new_growth_score = self._compute_growth_score(
                    total_bone_score + new_bone_score,
                    num_bones + new_bone_count,
                    len(current_nodes) + 1
                )

                if new_growth_score > best_growth_score:
                    best_growth_score = new_growth_score
                    best_extension = (cand_node, new_bone_score, new_bone_count)

            # Decision: extend or stop
            if best_extension and best_growth_score > (current_growth_score - self.config.score_debt_tolerance):
                node, score_contrib, bone_count = best_extension
                current_nodes[node.name] = node
                total_bone_score += score_contrib
                num_bones += bone_count
                iterations += 1
            else:
                break

        # Validate result
        if len(current_nodes) < self.config.min_kps_for_skeleton:
            return None

        final_avg_score = (total_bone_score / num_bones) if num_bones > 0 else 0.0
        if final_avg_score <= min_score_threshold:
            return None

        final_scale = self.skeleton_stats.estimate_scale(
            current_nodes,
            min_scale=self.config.min_sane_scale,
            max_scale=self.config.max_sane_scale
        )

        return self._create_hypothesis(
            frozenset(current_nodes.values()), final_avg_score, final_scale
        )

    def _find_extension_candidates(
            self,
            frame_data: TimestepData,
            current_nodes: Dict[str, Node3D],
            max_search_radius: float
    ) -> List[Node3D]:
        """
        Find all candidate nodes that could extend the current skeleton.
        (real 3D points and virtual points from single-view rescue).
        """

        current_kp_names = set(current_nodes.keys())
        candidates_by_type: Dict[str, List[Node3D]] = defaultdict(list)

        # For each existing node find potential neighbours
        for node_kp, node in current_nodes.items():
            neighbor_types = {
                n for n in self.skeleton.neighbours(node_kp)
                if n not in current_kp_names
            }
            if not neighbor_types:
                continue

            # Search for real 3D points nearby
            nearby_indices = frame_data.nearby(node.position, max_search_radius)

            for idx in nearby_indices:
                cand_type_idx = frame_data.soup.keypoint_indices[idx]
                cand_type = frame_data.soup.keypoint_names[cand_type_idx]

                if cand_type in neighbor_types:
                    cand_node = frame_data.get_node(cand_type, int(idx))
                    candidates_by_type[cand_type].append(cand_node)

            # Orphan rescue for types without real candidates
            for target_type in neighbor_types:
                if candidates_by_type[target_type]:
                    continue  # already have real candidates

                bone = BoneDefinition(node_kp, target_type)
                if bone in self.skeleton:
                    expected_len = self.skeleton_stats.expected_length(bone)
                    virtual_nodes = frame_data.intersect_rays(
                        target_type, node.position, expected_len
                    )
                    candidates_by_type[target_type].extend(virtual_nodes)

        # Flatten and deduplicate
        seen = set()
        candidates = []
        for node_list in candidates_by_type.values():
            for node in node_list:
                key = (node.name, node.idx)
                if key not in seen:
                    seen.add(key)
                    candidates.append(node)

        return candidates

    def _evaluate_extension(
            self,
            current_nodes: Dict[str, Node3D],
            candidate: Node3D,
            current_scale: float,
    ) -> Optional[Tuple[float, int]]:
        """
        Evaluate adding a candidate node to the current skeleton.
        Returns (total_bone_score, bone_count) or None if invalid.
        """
        total_score = 0.0
        bone_count = 0

        for existing_name, existing_node in current_nodes.items():
            bone = BoneDefinition(candidate.name, existing_name)

            if bone not in self.skeleton:
                continue

            score = self.skeleton_stats.score_bone(
                bone, candidate, existing_node,
                scale=current_scale,
                MAD_threshold=self.config.MAD_threshold
            )

            if score < -500:  # Hard rejection threshold
                return None

            total_score += score
            bone_count += 1

        if bone_count == 0:
            return None

        return total_score, bone_count

    def _compute_growth_score(
            self,
            total_bone_score: float,
            num_bones: int,
            num_nodes: int
    ) -> float:
        """Compute the growth score for skeleton extension decisions."""

        if num_bones == 0:
            return 0.0

        avg_score = total_bone_score / num_bones
        base_score = avg_score * num_nodes

        # Quality bonus for high-scoring skeletons
        quality_bonus = 0.0
        if avg_score > self.config.high_quality_threshold:
            bonus_factor = self.config.quality_bonus_factor - 1.0
            normalized_quality = max(0.0, (avg_score - 75.0) / 25.0)
            quality_bonus = base_score * bonus_factor * normalized_quality

        return base_score + quality_bonus

    def _try_merge(
            self,
            skel_A: SkeletonHypothesis,
            skel_B: SkeletonHypothesis,
    ) -> Optional[SkeletonHypothesis]:
        """Attempt to merge two skeleton hypotheses."""

        # Check disjointness
        if skel_A.shares_points_with(skel_B):
            return None
        if not skel_A.names.isdisjoint(skel_B.names):
            return None

        # Scale consistency
        if abs(skel_A.scale - skel_B.scale) > self.config.merge_scale_tolerance:
            return None
        combined_scale = (skel_A.scale + skel_B.scale) / 2.0

        # Find linking bone between the two skeletons
        best_link_score = -1.0

        for kp_a in skel_A.names:
            for kp_b in skel_B.names:
                bone = BoneDefinition(kp_a, kp_b)
                if bone not in self.skeleton:
                    continue

                score = self.skeleton_stats.score_bone(
                    bone, skel_A[kp_a], skel_B[kp_b],
                    scale=combined_scale,
                    MAD_threshold=self.config.MAD_threshold
                )
                if score > best_link_score:
                    best_link_score = score

        if best_link_score < self.config.merge_linking_bone_threshold:
            return None

        # Construct merged hypothesis
        num_nodes_A, num_nodes_B = len(skel_A), len(skel_B)

        avg_score_A = skel_A.anatomical_score / num_nodes_A
        avg_score_B = skel_B.anatomical_score / num_nodes_B

        num_bones_A = max(1, num_nodes_A - 1)
        num_bones_B = max(1, num_nodes_B - 1)

        new_total_score = (avg_score_A * num_bones_A) + (avg_score_B * num_bones_B) + best_link_score
        new_num_bones = num_bones_A + num_bones_B + 1
        new_avg_score = new_total_score / new_num_bones

        combined_nodes = skel_A.nodes | skel_B.nodes
        return self._create_hypothesis(combined_nodes, new_avg_score, combined_scale)

    def _create_hypothesis(
            self,
            nodes: frozenset,
            avg_score: float,
            scale: float
    ) -> SkeletonHypothesis:
        """Create a SkeletonHypothesis with computed scores."""

        if avg_score <= 0:
            return SkeletonHypothesis(
                nodes=nodes,
                competition_score=0.0,
                anatomical_score=0.0,
                scale=scale
            )

        num_nodes = len(nodes)
        base_score = avg_score * num_nodes

        quality_bonus = 0.0
        if avg_score > self.config.high_quality_threshold:
            quality_bonus = base_score * (self.config.quality_bonus_factor - 1.0)

        return SkeletonHypothesis(
            nodes=nodes,
            competition_score=base_score + quality_bonus,
            anatomical_score=avg_score * num_nodes,
            scale=scale
        )


class MultiObjectTracker:
    """
    Tracks multiple skeletons over time.
    Does skeleton assembly and temporal association.
    """

    def __init__(
            self,
            soup: PointSoup,
            skeleton: SkeletonTopology,
            assembler: SkeletonAssembler,
            config: TrackerConfig
    ):
        self.skeleton = skeleton
        self.assembler = assembler
        self.config = config
        self.soup = soup

        self._active_tracklets: Dict[int, Tracklet] = {}   # track_idx -> Tracklet (updated this frame)
        self._pending_tracklets: Dict[int, Tracklet] = {}  # track_idx -> Tracklet (coasting)
        # TODO: Maybe store terminated tracklets instead of pruning them?

        self._next_track_idx = 0
        self._current_frame = -1

    def _poses_mse(self,
             kps_a: Dict[str, np.ndarray],
             kps_b: Dict[str, np.ndarray]
         ) -> Optional[float]:
        """
        Computes Mean Squared Error between two sets of keypoints.
        Returns None if overlap is insufficient.
        """
        common_kps = kps_a.keys() & kps_b.keys()

        if len(common_kps) < self.config.association_min_kps:
            return None

        total_dist_sq = sum(
            np.sum((kps_a[kp] - kps_b[kp]) ** 2) for kp in common_kps
        )

        return total_dist_sq / len(common_kps)

    @property
    def active_tracklets(self) -> List[Tracklet]:
        """Tracklets that were updated this frame."""
        return list(self._active_tracklets.values())

    @property
    def pending_tracklets(self) -> List[Tracklet]:
        """Tracklets that are coasting (not updated this frame)."""
        return list(self._pending_tracklets.values())

    @property
    def tracklets(self) -> List[Tracklet]:
        """All tracklets (active + pending)."""
        return self.active_tracklets + self.pending_tracklets

    @property
    def current_frame(self) -> int:
        return self._current_frame

    def update(self, frame_idx: int):
        """
        Process a new frame and update tracklets.
        (after calling this access results via .active and .pending properties)
        """
        self._current_frame = frame_idx

        # Move all active tracklets to pending at start of frame
        self._pending_tracklets.update(self._active_tracklets)
        self._active_tracklets.clear()

        # Predict all tracklets forward
        for tracklet in self._pending_tracklets.values():
            tracklet.predict(frame_idx)

        # Generate and resolve hypotheses
        frame_data = TimestepData(self.soup[frame_idx])
        frame_candidates = self.assembler.assemble(frame_data)

        if not frame_candidates:
            self._prune_pending()
            return

        # Association bonus for temporal continuity
        bonuses = self._association_bonuses(frame_candidates)
        for i, cand in enumerate(frame_candidates):
            cand.competition_score += bonuses[i] * self.config.continuity_bonus

        # Solve conflicts
        remaining_hypotheses = self._resolve_conflicts(frame_candidates)

        # Commit surviving hypotheses: either extend existing tracklets or create new ones
        self._extend_tracklets(remaining_hypotheses, frame_idx)

        self._prune_pending()

    def _extend_tracklets(self,
            hypotheses: List[SkeletonHypothesis],
            frame_idx: int
        ):
        """Match hypotheses to existing tracklets, and create new tracklets from unmatched hypotheses."""

        if not hypotheses:
            return

        # Assign to existing tracklets and get unmatched hypotheses
        unmatched = self._assign_hypotheses(hypotheses, frame_idx)

        # Create new tracklets for unmatched hypotheses that have the central keypoint
        for hyp in unmatched:

            if self.skeleton.central_keypoint not in hyp:
                continue

            tracklet = Tracklet(
                track_idx=self._next_track_idx,
                initial_hypothesis=hyp,
                frame_idx=frame_idx,
                central_kp=self.skeleton.central_keypoint,
                config=self.config
            )
            self._active_tracklets[tracklet.track_idx] = tracklet
            self._next_track_idx += 1

    def _assign_hypotheses(self,
            hypotheses: List[SkeletonHypothesis],
            frame_idx: int
    ) -> List[SkeletonHypothesis]:
        """
        Assign hypotheses to pending tracklets using Hungarian algorithm.
        Returns list of unmatched hypotheses.
        """

        if not self._pending_tracklets:
            return hypotheses

        pending_tracklets = list(self._pending_tracklets.values())

        # Build cost matrix for linear assignment
        cost_matrix = np.full((len(pending_tracklets), len(hypotheses)), 1e9)

        for i, tracklet in enumerate(pending_tracklets):

            if not tracklet.predicted_keypoints:
                continue

            for j, hyp in enumerate(hypotheses):

                mean_dist_sq = self._poses_mse(tracklet.predicted_keypoints, hyp.positions)
                if mean_dist_sq is None:
                    continue

                if mean_dist_sq > self.config.association_radius ** 2:
                    continue

                cost = (self.config.cost_pose_distance_weight * mean_dist_sq +
                        self.config.cost_skeleton_score_weight * hyp.anatomical_score)
                cost_matrix[i, j] = cost

        # Solve assignment
        tracklet_indices, hypotheses_indices = linear_sum_assignment(cost_matrix)

        # Update pending tracklets with assigned hypotheses
        assigned_indices = set()
        for t_idx, h_idx in zip(tracklet_indices, hypotheses_indices):

            if cost_matrix[t_idx, h_idx] >= 1e9:
                continue

            tracklet = pending_tracklets[t_idx]
            tracklet.update(hypothesis=hypotheses[h_idx], frame_idx=frame_idx)

            # Move from pending to active
            del self._pending_tracklets[tracklet.track_idx]
            self._active_tracklets[tracklet.track_idx] = tracklet

            assigned_indices.add(h_idx)

        # Return unmatched hypotheses
        unmatched_indices = set(hypotheses_indices).difference(assigned_indices)

        return [hypotheses[i] for i in unmatched_indices]

    def _prune_pending(self):
        """Remove stale or uncertain tracklets from pending."""
        # TODO: Maybe should be removed, finished tracklets should be stored

        to_remove = []

        for track_idx, t in self._pending_tracklets.items():

            if t.time_since_update > self.config.max_tracklet_age:
                to_remove.append(track_idx)

            elif np.sum(t.position_uncertainty) > self.config.uncertainty_threshold:
                to_remove.append(track_idx)

        for track_idx in to_remove:
            del self._pending_tracklets[track_idx]

    def _resolve_conflicts(self, candidates: List[SkeletonHypothesis]) -> List[SkeletonHypothesis]:
        """Build graph where edges represent mutually exclusive hypotheses, and solve with MWIS."""

        conflict_graph = nx.Graph()

        # Add nodes with weights
        for i, cand in enumerate(candidates):
            weight = max(0, int(cand.competition_score * 100))
            conflict_graph.add_node(i, weight=weight)

        centroids = np.array([cand.centroid for cand in candidates])

        # Check conflicts for all pairs
        for i, j in combinations(range(len(candidates)), 2):

            cand_i = candidates[i]
            cand_j = candidates[j]

            # Hierarchical conflict (merge provenance)
            if cand_i.is_related(cand_j):
                conflict_graph.add_edge(i, j)
                continue

            # Spatial conflict (shared point indices)
            if cand_i.shares_points_with(cand_j):
                conflict_graph.add_edge(i, j)
                continue

            # Ray source conflict
            if cand_i.shares_rays_with(cand_j):
                conflict_graph.add_edge(i, j)
                continue

            # Spatial proximity (clone detection)
            dist_sq = np.sum((centroids[i] - centroids[j]) ** 2)

            if dist_sq < self.config.conflict_solver_broad_radius ** 2:
                kps_i = cand_i.positions
                kps_j = cand_j.positions

                common = kps_i.keys() & kps_j.keys()
                union = kps_i.keys() | kps_j.keys()

                if union:
                    proximal_count = sum(
                        1 for name in common
                        if np.linalg.norm(kps_i[name] - kps_j[name]) < self.config.conflict_solver_proximity_radius
                    )
                    if proximal_count / len(union) > self.config.conflict_solver_jaccard_threshold:
                        conflict_graph.add_edge(i, j)

        # Solve assignment
        winner_indices = solve_mwis(conflict_graph, method='networkx')
        winning_hypotheses = [candidates[i] for i in winner_indices]

        return winning_hypotheses

    def _association_bonuses(self, candidates: List[SkeletonHypothesis]) -> np.ndarray:
        """Calculate temporal association bonuses based on tracklet predictions."""

        all_tracklets = self.tracklets
        if not all_tracklets or not candidates:
            return np.zeros(len(candidates))

        bonuses = np.zeros(len(candidates))

        for j, cand in enumerate(candidates):

            if not cand.positions:
                continue

            max_bonus = 0.0
            for t in all_tracklets:

                if not t.predicted_keypoints:
                    continue

                mean_dist_sq = self._poses_mse(t.predicted_keypoints, cand.positions)
                if mean_dist_sq is None:
                    continue

                # Gaussian falloff based on squared distance
                bonus = np.exp(-0.5 * mean_dist_sq / (self.config.association_radius ** 2))
                max_bonus = max(max_bonus, bonus)

            bonuses[j] = max_bonus

        return bonuses

    def collect_records(self, frame_idx: int) -> List[dict]:
        """
        Collect serialisable records for all active tracklets.
        Use this for building output data structures.
        """
        return [t.to_record(frame_idx) for t in self.active_tracklets]


##

if __name__ == '__main__':
    import pickle
    from pathlib import Path

    BASE_DIR = Path.home() / 'Desktop' / '3d_ant_data'
    PREFIX = '240905-1616'
    SESSION = 22
    DEBUG_PLOT = False

    input_dir = BASE_DIR / PREFIX / 'inputs' / 'tracking'
    output_dir = BASE_DIR / PREFIX / 'outputs'

    soup_file = output_dir / f"soup_session{SESSION}.pkl"
    stats_file = output_dir / "skeleton_stats.json"
    tracklets_file = output_dir / f'tracklets_session{SESSION}.pkl'

    # Load stuff
    soup = PointSoup.from_file(soup_file)
    skeleton = SkeletonTopology.from_sleap(input_dir)
    stats = SkeletonStats.from_json(stats_file, skeleton)

    print(f"Loaded point soup from {soup_file}")
    print(f"Loaded skeleton with {len(skeleton.keypoints)} keypoints, {len(skeleton.bones)} bones")
    print(f"Loaded stats from {stats_file}")

    # Initialise pipeline

    # TODO: The config defaults may need tuning
    assembler_cfg = AssemblerConfig()
    tracker_cfg = TrackerConfig()

    assembler = SkeletonAssembler(
        skeleton=skeleton,
        skeleton_stats=stats,
        config=assembler_cfg,
    )

    tracker = MultiObjectTracker(
        soup=soup,
        skeleton=skeleton,
        assembler=assembler,
        config=tracker_cfg
    )

    # Run tracking
    unique_frames = np.unique(soup.frame_indices)
    min_frame, max_frame = int(unique_frames[0]), int(unique_frames[-1])

    records_by_track: Dict[int, List[dict]] = defaultdict(list)

    print(f"Tracking frames {min_frame} to {max_frame}...")

    with alive_bar(total=(max_frame - min_frame + 1), length=20, force_tty=True) as bar:

        for frame_idx in range(min_frame, max_frame + 1):
            tracker.update(frame_idx)

            # Update skeleton stats from high-quality observations
            for tracklet in tracker.active_tracklets:

                if tracklet.last_update_frame == frame_idx:
                    stats.update(tracklet.keypoints)

            # Collect records
            for record in tracker.collect_records(frame_idx):
                records_by_track[record['track_idx']].append(record)

            bar()

    print(f"Tracking complete. Generated {len(records_by_track)} tracklets.")

    # Save tracking results
    tracklets_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tracklets_file, 'wb') as f:
        pickle.dump(dict(records_by_track), f)
    print(f"Saved to '{tracklets_file}'")

    # Save updated stats
    stats.to_json(stats_file)
    print(f"Updated stats saved to '{stats_file}'")

    if DEBUG_PLOT:
        from mokap.pose_reconstruction.debug import TrackletViewer
        viewer = TrackletViewer(soup, records_by_track, skeleton)