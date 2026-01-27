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
from mokap.pose_reconstruction.configs import AssemblerConfig, TrackerConfig, TrackletConfig
from mokap.pose_reconstruction.utils import solve_mwis

logger = logging.getLogger(__name__)


class SkeletonAssembler:
    """
    Assembles skeleton hypotheses in a frame view.
    """
    def __init__(self,
        skeleton: SkeletonTopology,
        skeleton_stats: SkeletonStats,
        config: AssemblerConfig,
    ):
        self.skeleton = skeleton
        self.skeleton_stats = skeleton_stats
        self.config = config

    # ──── Public interface ────

    def assemble(self,
            frame_data: TimestepData,
            predictions: Optional[Dict[int, Dict[str, np.ndarray]]] = None
        ) -> List[SkeletonHypothesis]:
        """
        Main assembly entry point.

        Args:
            frame_data: Current frame's point data
            predictions: Optional dict of {track_idx: {keypoint_name: predicted_position}}
                        from existing tracklets. Used to guide assembly.

        Returns:
            List of skeleton hypotheses.
        """
        if not frame_data:
            return []

        used_indices: Set[int] = set()

        # Generate primary hypotheses
        all_fragments = self._generate_hypotheses(frame_data, predictions, used_indices)

        if not all_fragments:
            return []

        # Generate merge hypotheses for fragmented detections
        merge_hypotheses = self._generate_merge_hypotheses(all_fragments)

        return all_fragments + merge_hypotheses

    # ──── Score and bonus helpers ────

    def _calc_prediction_bonus(self, node: Node3D, predicted_pos: np.ndarray) -> float:
        """Calculates a Gaussian bonus based on proximity to prediction."""

        dist = np.linalg.norm(node.position - predicted_pos)
        sigma = self.config.prediction_bonus_sigma * self.skeleton_stats.reference_length_world
        bonus = self.config.prediction_bonus_weight * np.exp(-0.5 * (dist / sigma) ** 2)

        return bonus

    def _calc_growth_score(self,
        total_bone_score: float,
        num_bones: int,
        num_nodes: int
    ) -> float:
        """Calculates the growth score for skeleton extension decisions."""

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

    def _evaluate_extension(self,
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

            observed_length = float(np.linalg.norm(candidate.position - existing_node.position))
            expected_length = self.skeleton_stats.expected_length(bone) * current_scale
            bone_scale_ratio = observed_length / max(expected_length, 1e-6)

            if not (
                    1.0 / self.config.scale_consistency_factor < bone_scale_ratio < self.config.scale_consistency_factor):
                return None  # this bone implies incompatible scale -> reject

            score = self.skeleton_stats.score_bone(
                bone, candidate, existing_node,
                scale=current_scale,
                MAD_threshold=self.config.MAD_threshold
            )

            if score < -500:  # hard rejection threshold
                return None

            total_score += score
            bone_count += 1

        if bone_count == 0:
            return None

        return total_score, bone_count

    # ──── Hypotheses generation ────

    def _generate_hypotheses(self,
        frame_data: TimestepData,
        predictions: Dict[int, Dict[str, np.ndarray]],
        used_indices: Set[int]
    ) -> List[SkeletonHypothesis]:
        """
        Generates skeleton hypotheses in two (well, three actually) passes:

        Guided: Seeded from points near tracklet predictions.
        Blind: Seeded from remaining anchor/leaf keypoints.
        """
        hypotheses = []

        # Pass 1: Guided assembly from predictions
        if predictions:
            central_kp = self.skeleton.central_keypoint
            for track_idx, pred_kps in predictions.items():
                if central_kp not in pred_kps:
                    continue

                seed_node = self._find_node_guided(
                    frame_data, central_kp, pred_kps[central_kp], used_indices
                )

                if seed_node:
                    hyp = self._grow_skeleton(frame_data, {central_kp: seed_node}, predictions=pred_kps)
                    if hyp:
                        hyp.track_affinity = track_idx  # tag for association boost
                        hypotheses.append(hyp)
                        used_indices.update(idx for idx in hyp.point_indices if idx >= 0)

        # Pass 2: Blind assembly for new individuals (anchors)
        for anchor_kp in self.skeleton.anchor_keypoints:
            for seed_idx in frame_data.get_indices(anchor_kp):
                if seed_idx in used_indices:
                    continue

                seed_node = frame_data.get_node(anchor_kp, seed_idx)
                hyp = self._grow_skeleton(frame_data, {anchor_kp: seed_node})
                if hyp:
                    hypotheses.append(hyp)
                    used_indices.update(idx for idx in hyp.point_indices if idx >= 0)

        # Pass 3: Fragment rescue (leaf nodes)
        for leaf_kp in self.skeleton.leaf_keypoints:
            for seed_idx in frame_data.get_indices(leaf_kp):
                if seed_idx in used_indices:
                    continue

                seed_node = frame_data.get_node(leaf_kp, seed_idx)
                hyp = self._grow_skeleton(
                    frame_data, {leaf_kp: seed_node},
                    max_iterations=1,
                    min_score_threshold=self.config.min_bone_score_for_fragment
                )
                if hyp:
                    hypotheses.append(hyp)
                    used_indices.update(idx for idx in hyp.point_indices if idx >= 0)

        return hypotheses

    def _generate_merge_hypotheses(self, fragments: List[SkeletonHypothesis]) -> List[SkeletonHypothesis]:
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

    def _try_merge(self,
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

    def _create_hypothesis(self,
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

    # ──── Core skeleton growth algorithm ────

    def _grow_skeleton(self,
        frame_data: TimestepData,
        seed_nodes: Dict[str, Node3D],
        predictions: Optional[Dict[str, np.ndarray]] = None,
        max_iterations: Optional[int] = None,
        min_score_threshold: float = 0.0
    ) -> Optional[SkeletonHypothesis]:
        """
        Generic greedy growth algorithm.

        Args:
            frame_data: Current frame's point data
            seed_nodes: Initial nodes to grow from (name -> Node)
            predictions: If provided, apply a spatial bonus to candidate scores
            max_iterations: Maximum growth steps (None = unlimited)
            min_score_threshold: Minimum final score to accept result
        """

        current_nodes = dict(seed_nodes)
        total_bone_score = 0.0
        num_bones = 0
        iterations = 0

        max_search_radius = self.skeleton_stats.reference_length_world * self.config.max_bone_len

        while max_iterations is None or iterations < max_iterations:
            candidates = self._find_neighbouring_nodes(frame_data, current_nodes, max_search_radius)
            if not candidates:
                break

            current_scale = self.skeleton_stats.estimate_scale(
                current_nodes,
                min_scale=self.config.min_sane_scale,
                max_scale=self.config.max_sane_scale
            )

            # Evaluate growth from current state
            current_growth_score = self._calc_growth_score(total_bone_score, num_bones, len(current_nodes))
            best_ext = None
            best_ext_score = -float('inf')

            for cand_node in candidates:
                eval_result = self._evaluate_extension(current_nodes, cand_node, current_scale)
                if eval_result is None:
                    continue

                new_score, new_count = eval_result

                # Apply prediction guidance bonus if available
                if predictions and cand_node.name in predictions:
                    new_score += self._calc_prediction_bonus(cand_node, predictions[cand_node.name])

                new_growth_score = self._calc_growth_score(
                    total_bone_score + new_score,
                    num_bones + new_count,
                    len(current_nodes) + 1
                )

                if new_growth_score > best_ext_score:
                    best_ext_score = new_growth_score
                    best_ext = (cand_node, new_score, new_count)

            # Check if best extension is worth the debt
            if best_ext and best_ext_score > (current_growth_score - self.config.score_debt_tolerance):
                node, score_contrib, bone_count = best_ext
                current_nodes[node.name] = node
                total_bone_score += score_contrib
                num_bones += bone_count
                iterations += 1
            else:
                break

        # Validation
        if len(current_nodes) < self.config.min_kps_for_skeleton:
            return None

        final_avg = (total_bone_score / num_bones) if num_bones > 0 else 0.0
        if final_avg <= min_score_threshold:
            return None

        return self._create_hypothesis(
            frozenset(current_nodes.values()),
            final_avg,
            self.skeleton_stats.estimate_scale(current_nodes)
        )

    # ──── Point discovery ────

    def _find_node_guided(self,
            frame_data: TimestepData,
            kp_type: str,
            predicted_pos: np.ndarray,
            used_indices: Set[int]
        ) -> Optional[Node3D]:
        """
        Finds the best Node3D for a specific keypoint type near a predicted position.
        """

        search_radius = self.config.guided_search_radius * self.skeleton_stats.reference_length_world
        nearby_indices = frame_data.nearby(predicted_pos, search_radius)

        # Filter indices by type and availability
        valid_indices = [i for i in nearby_indices
            if i not in used_indices and frame_data.soup.keypoint_names[frame_data.soup.keypoint_indices[i]] == kp_type
        ]

        if not valid_indices:
            return None

        # Select best index based on distance and confidence
        best_idx = min(
            valid_indices,
            key=lambda i: (np.linalg.norm(frame_data.soup.positions[i] - predicted_pos)
                           - 0.1 * frame_data.soup.confidences[i])
        )

        return frame_data.get_node(kp_type, best_idx)

    def _find_neighbouring_nodes(self,
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


class MultiObjectTracker:
    """
    Tracks multiple skeletons over time.
    Does skeleton assembly and temporal association.
    """

    def __init__(
            self,
            soup: PointSoup,
            skeleton: SkeletonTopology,
            stats: SkeletonStats,
            assembler: SkeletonAssembler,
            config: TrackerConfig
    ):
        self.skeleton = skeleton
        self.stats = stats
        self.assembler = assembler
        self.config = config
        self.soup = soup

        self._active_tracklets: Dict[int, Tracklet] = {}   # track_idx -> Tracklet (updated this frame)
        self._pending_tracklets: Dict[int, Tracklet] = {}  # track_idx -> Tracklet (coasting)
        # TODO: Maybe store terminated tracklets instead of pruning them?

        self._next_track_idx = 0
        self._current_frame = -1

        self._tracklet_config = TrackletConfig()

    # ──── Properties ────

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

    # ──── Scoring helpers ────

    def _calc_pose_distance(
            self,
            kps_a: Dict[str, np.ndarray],
            kps_b: Dict[str, np.ndarray]
    ) -> Optional[float]:
        """
        Weighted Mean Squared Error using per-keypoint association weights.
        Returns None if overlap is insufficient.
        """

        common_kps = kps_a.keys() & kps_b.keys()

        if len(common_kps) < self.config.association_min_kps:
            return None

        total_weighted_dist_sq = 0.0
        total_weight = 0.0

        for kp in common_kps:
            dist_sq = float(np.sum((kps_a[kp] - kps_b[kp]) ** 2))
            weight = self.stats.get_dynamics(kp).association_weight

            total_weighted_dist_sq += weight * dist_sq
            total_weight += weight

        if total_weight < 1e-6:
            return None

        return total_weighted_dist_sq / total_weight

    def _calc_continuity_bonuses(self, candidates: List[SkeletonHypothesis]):
        """
        Adds score bonuses to hypotheses that align well with existing tracklet predictions.
        """

        with_predictions = [t for t in self.tracklets if t.predicted_keypoints]

        if not with_predictions:
            return

        for cand in candidates:
            max_bonus = 0.0

            for t in with_predictions:
                dist_sq = self._calc_pose_distance(t.predicted_keypoints, cand.positions)
                if dist_sq is not None:

                    # Gaussian falloff based on squared distance
                    bonus = np.exp(-0.5 * dist_sq / (self.config.association_radius ** 2))
                    max_bonus = max(max_bonus, bonus)

            cand.competition_score += max_bonus * self.config.continuity_bonus

            # Additional bonus for assembly guided by a specific tracklet
            if cand.track_affinity is not None:
                cand.competition_score += self.config.guided_affinity_bonus

    # ──── Public interface ────

    def update(self, frame_idx: int):
        """Process a single frame."""

        self._current_frame = frame_idx

        # Move all previously active to pending
        self._pending_tracklets.update(self._active_tracklets)
        self._active_tracklets.clear()

        # Predict all tracklets forward
        for t in self._pending_tracklets.values():
            t.predict(frame_idx)

        # Build predictions dict for guided assembly
        predictions = {
            t.track_idx: t.predicted_keypoints
            for t in self._pending_tracklets.values() if t.predicted_keypoints
        }

        # Generate and resolve hypotheses
        frame_data = TimestepData(self.soup[frame_idx])
        frame_candidates = self.assembler.assemble(frame_data, predictions=predictions)

        if not frame_candidates:
            self._prune_pending()
            return

        # Association bonus for temporal continuity
        self._calc_continuity_bonuses(frame_candidates)

        # Solve conflicts
        remaining_hypotheses = self._resolve_conflicts(frame_candidates)

        # Commit surviving hypotheses (extends existing, and initialises new ones)
        self._commit_tracklet_hypotheses(remaining_hypotheses, frame_idx)

        self._prune_pending()

    # ──── Association and matching ────

    def _commit_tracklet_hypotheses(self,
            hypotheses: List[SkeletonHypothesis],
            frame_idx: int
        ):
        """
        Matches hypotheses to existing tracklets, and creates new tracklets from unmatched hypotheses.
        """

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
                skeleton=self.skeleton,
                stats=self.stats,
                config=self._tracklet_config
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

            predicted_keypoints = tracklet.predicted_keypoints
            if not predicted_keypoints:
                continue

            for j, hyp in enumerate(hypotheses):
                # Scale gate
                scale_ratio = abs(hyp.scale - tracklet.estimated_scale) / max(tracklet.estimated_scale, 0.1)
                if scale_ratio > self.config.scale_gate_hard_threshold:
                    continue  # leave at 1e9

                scale_penalty = self.config.scale_gate_soft_weight * (scale_ratio ** 2)

                # Weighted pose distance
                mean_dist_sq = self._calc_pose_distance(predicted_keypoints, hyp.positions)
                if mean_dist_sq is None:
                    continue

                if mean_dist_sq > self.config.association_radius ** 2:
                    continue

                cost = (
                        self.config.cost_pose_distance_weight * mean_dist_sq
                        - self.config.skeleton_score_bonus_weight * hyp.anatomical_score
                        + scale_penalty
                )
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
        return [hypotheses[i] for i in range(len(hypotheses)) if i not in assigned_indices]

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

    # ──── Lifecycle and serialisation ────

    # TODO: Should not prune at all, should just move them to a third internal store

    def _prune_pending(self):
        """Remove stale or uncertain tracklets from pending."""

        to_remove = []

        for track_idx, t in self._pending_tracklets.items():

            if t.time_since_update > self.config.max_tracklet_age:
                to_remove.append(track_idx)

            elif np.sum(t.position_uncertainty) > self.config.uncertainty_threshold:
                to_remove.append(track_idx)

        for track_idx in to_remove:
            del self._pending_tracklets[track_idx]

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
    DEBUG_PLOT = True

    input_dir = BASE_DIR / PREFIX / 'inputs' / 'tracking'
    output_dir = BASE_DIR / PREFIX / 'outputs'

    # soup_file = output_dir / f"soup_session{SESSION}.pkl"
    soup_file = output_dir / f"soup2_session{SESSION}.pkl"
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
        stats=stats,
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
                    stats.update_anatomy(tracklet.hypothesis.positions)

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