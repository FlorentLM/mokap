"""
Skeleton assembly and multi-object tracking pipeline.
"""
import logging
from typing import Tuple, Optional, Dict, List, FrozenSet
from itertools import combinations
from collections import defaultdict

import numpy as np
import networkx as nx
from alive_progress import alive_bar
from scipy.optimize import linear_sum_assignment

from mokap.pose_reconstruction.skeleton import Bone, Skeleton, SkeletonStats
from mokap.pose_reconstruction.datatypes import Pose3D, SkeletonHypothesis, PointSoup, Tracklet, FrameData
from mokap.pose_reconstruction.configs import AssemblerConfig, TrackerConfig
from mokap.pose_reconstruction.utils import solve_mwis


logger = logging.getLogger(__name__)


class SkeletonAssembler:
    """
    Assembles skeleton hypotheses in a frame view.
    """

    def __init__(
            self,
            skeleton: Skeleton,
            skeleton_stats: SkeletonStats,
            config: AssemblerConfig,
    ):
        self.skeleton = skeleton
        self.skeleton_stats = skeleton_stats
        self.config = config

    def assemble(self, frame_data: FrameData) -> List[SkeletonHypothesis]:
        """
        Main assembly entry point.
        """

        if not frame_data:
            return []

        # Generate initial fragments (including orphan rescue)
        initial_fragments = self._generate_hypotheses(frame_data)
        if not initial_fragments:
            return []

        # Generate merge hypotheses
        merge_hypotheses = self._generate_merge_hypotheses(initial_fragments, frame_data)

        return initial_fragments + merge_hypotheses

    def _generate_hypotheses(self, frame_data: FrameData) -> List[SkeletonHypothesis]:
        """
        Generate skeleton hypotheses.
        """
        candidate_skeletons = []
        used_as_seed_indices = set()

        # Seed from anchors
        for anchor_type in self.skeleton.anchor_keypoints:
            # Only seed from real points
            for seed_idx in frame_data.get_indices(anchor_type):
                if seed_idx in used_as_seed_indices:
                    continue

                candidate = self._grow_skeleton(frame_data, anchor_kp=anchor_type, anchor_idx=seed_idx)

                if candidate:
                    candidate_skeletons.append(candidate)
                    for _, idx in candidate.nodes:
                        if idx >= 0:
                            used_as_seed_indices.add(idx)

        # Then seed from leaves
        for leaf_type in self.skeleton.leaf_keypoints:
            for seed_idx in frame_data.get_indices(leaf_type):
                if seed_idx in used_as_seed_indices:
                    continue

                fragment = self._find_leaf_fragment(frame_data, leaf_kp=leaf_type, leaf_idx=seed_idx)

                if fragment:
                    candidate_skeletons.append(fragment)
                    for _, idx in fragment.nodes:
                        if idx >= 0:
                            used_as_seed_indices.add(idx)

        return candidate_skeletons

    def _generate_merge_hypotheses(
            self,
            fragments: List[SkeletonHypothesis],
            frame_data: FrameData
    ) -> List[SkeletonHypothesis]:
        """
        Generate hypotheses by merging compatible fragments.
        """
        if len(fragments) < 2:
            return []

        for i, frag in enumerate(fragments):
            frag.constituent_indices = frozenset([i])

        merge_hypotheses = []

        for i, j in combinations(range(len(fragments)), 2):
            skel_A, skel_B = fragments[i], fragments[j]

            merged = self._try_merge(skel_A, skel_B, frame_data)

            if merged:
                merged.constituent_indices = skel_A.constituent_indices | skel_B.constituent_indices
                merge_hypotheses.append(merged)

        return merge_hypotheses

    def _grow_skeleton(self, frame_data: FrameData, anchor_kp: str, anchor_idx: int) -> Optional[SkeletonHypothesis]:
        """
        Grow a skeleton from an anchor using FrameData for geometric lookups.
        """
        anchor_pos = frame_data.position(anchor_idx)

        current_nodes = {(anchor_kp, anchor_idx)}
        current_kps = {anchor_kp: anchor_pos}
        total_bone_score_sum = 0.0
        num_bones = 0

        # Search radius heuristic
        max_search_radius = self.skeleton_stats.reference_length_world * self.config.max_bone_len

        while True:
            current_kp_names = {node[0] for node in current_nodes}
            nodes_to_evaluate = set()

            # Find candidates via graph topology + spatial proximity
            for node_kp, node_idx in current_nodes:
                node_pos = frame_data.position(node_idx)

                neighbor_kp_types = {
                    n for n in self.skeleton.neighbours(node_kp)
                    if n not in current_kp_names
                }
                if not neighbor_kp_types:
                    continue

                # Standard 3D search (real points)
                nearby_indices = frame_data.nearby(node_pos, max_search_radius)

                for idx in nearby_indices:
                    cand_type_idx = frame_data.soup.keypoint_indices[idx]
                    cand_type = frame_data.soup.keypoint_names[cand_type_idx]

                    if cand_type in neighbor_kp_types:
                        nodes_to_evaluate.add((cand_type, int(idx)))

                # Orphan ray rescue (single view observations)
                for target_type in neighbor_kp_types:
                    # Check if we already found real candidates for this type
                    has_3d_candidates = any(n[0] == target_type for n in nodes_to_evaluate)

                    if not has_3d_candidates:
                        # Determine expected bone length
                        bone = Bone(node_kp, target_type)
                        if bone in self.skeleton:
                            expected_len = self.skeleton_stats.expected_length(bone)

                            virtual_indices = frame_data.intersect_rays(
                                keypoint_name=target_type,
                                center=node_pos,
                                radius=expected_len
                            )

                            for vp_idx in virtual_indices:
                                nodes_to_evaluate.add((target_type, vp_idx))

            if not nodes_to_evaluate:
                break

            # Check scale sanity
            current_step_scale = self.skeleton_stats.estimate_scale(
                current_kps,
                min_scale=self.config.min_sane_scale,
                max_scale=self.config.max_sane_scale
            )
            if not (self.config.min_sane_scale < current_step_scale < self.config.max_sane_scale):
                break

            # Baseline scores
            current_avg_score = (total_bone_score_sum / num_bones) if num_bones > 0 else 0
            current_base_score = current_avg_score * len(current_nodes)
            current_quality_bonus = 0.0

            if current_avg_score > self.config.high_quality_threshold:
                current_quality_bonus = current_base_score * (self.config.quality_bonus_factor - 1.0)
            current_growth_score = current_base_score + current_quality_bonus

            best_extension = None
            best_new_growth_score = -float('inf')

            # Evaluate extensions
            for cand_node in nodes_to_evaluate:

                cand_kp_name, cand_idx = cand_node
                cand_pos = frame_data.position(cand_idx)

                temp_kps = current_kps.copy()
                temp_kps[cand_kp_name] = cand_pos

                new_bone_score_sum = 0
                new_bone_count = 0

                for existing_node in current_nodes:
                    existing_kp = existing_node[0]
                    bone = Bone(cand_kp_name, existing_kp)

                    if bone in self.skeleton:
                        temp_nodes = frozenset(current_nodes | {cand_node})
                        score = self._score_bone(bone, temp_kps, temp_nodes, current_step_scale, frame_data)

                        if score < -500:
                            new_bone_score_sum = -float('inf')
                            break

                        new_bone_score_sum += score
                        new_bone_count += 1

                if new_bone_count == 0 or new_bone_score_sum == -float('inf'):
                    continue

                new_total_bones = num_bones + new_bone_count
                new_total_score_sum = total_bone_score_sum + new_bone_score_sum
                new_avg_score = new_total_score_sum / new_total_bones
                new_num_nodes = len(current_nodes) + 1
                new_base_score = new_avg_score * new_num_nodes

                # Quality bonus
                bonus_factor = self.config.quality_bonus_factor - 1.0
                normalized_quality = max(0.0, (new_avg_score - 75.0) / 25.0)
                new_quality_bonus = new_base_score * bonus_factor * normalized_quality

                new_growth_score = new_base_score + new_quality_bonus

                if new_growth_score > best_new_growth_score:
                    best_new_growth_score = new_growth_score
                    best_extension = {
                        "node": cand_node,
                        "position": cand_pos,
                        "score_contribution": new_bone_score_sum,
                        "bone_count_increase": new_bone_count
                    }

            # Decision: extend or stop
            if best_extension and best_new_growth_score > (current_growth_score - self.config.score_debt_tolerance):
                best_node = best_extension["node"]
                current_nodes.add(best_node)
                current_kps[best_node[0]] = best_extension["position"]
                total_bone_score_sum += best_extension["score_contribution"]
                num_bones += best_extension["bone_count_increase"]
            else:
                break

        # Finish
        if len(current_nodes) < self.config.min_kps_for_skeleton:
            return None

        final_avg_score = (total_bone_score_sum / num_bones) if num_bones > 0 else 0.0
        if final_avg_score <= 0:
            return None

        final_scale = self.skeleton_stats.estimate_scale(
            current_kps,
            min_scale=self.config.min_sane_scale,
            max_scale=self.config.max_sane_scale
        )

        return self._create_candidate(frozenset(current_nodes), final_avg_score, final_scale)

    def _find_leaf_fragment(
            self,
            frame_data: FrameData,
            leaf_kp: str,
            leaf_idx: int
    ) -> Optional[SkeletonHypothesis]:
        """
        Find a two-node fragment starting from a leaf keypoint.
        """

        neighbours = self.skeleton.neighbours(leaf_kp)
        if not neighbours:
            return None

        parent_kp = neighbours[0]
        leaf_pos = frame_data.position(leaf_idx)
        best_score = -1.0
        best_cand_data = None

        # Check real points
        parent_indices = frame_data.get_indices(parent_kp)

        for p_idx in parent_indices:
            p_pos = frame_data.position(p_idx)
            kps = {leaf_kp: leaf_pos, parent_kp: p_pos}

            scale = self.skeleton_stats.estimate_scale(kps, self.config.min_sane_scale, self.config.max_sane_scale)
            if not (self.config.min_sane_scale < scale < self.config.max_sane_scale):
                continue

            nodes = frozenset([(leaf_kp, leaf_idx), (parent_kp, p_idx)])
            bone = Bone(leaf_kp, parent_kp)
            score = self._score_bone(bone, kps, nodes, scale, frame_data)

            if score > best_score:
                best_score = score
                best_cand_data = (p_idx, p_pos)

        # Try orphan rescue if no good real connection
        if best_score < self.config.min_bone_score_for_fragment:
            bone = Bone(leaf_kp, parent_kp)
            expected_len = self.skeleton_stats.expected_length(bone)

            vp_indices = frame_data.intersect_rays(parent_kp, leaf_pos, expected_len)

            for vp_idx in vp_indices:
                vp_pos = frame_data.position(vp_idx)

                kps = {leaf_kp: leaf_pos, parent_kp: vp_pos}
                scale = self.skeleton_stats.estimate_scale(
                    kps,
                    self.config.min_sane_scale,
                    self.config.max_sane_scale
                )
                if not (self.config.min_sane_scale < scale < self.config.max_sane_scale):
                    continue

                nodes = frozenset([(leaf_kp, leaf_idx), (parent_kp, vp_idx)])
                score = self._score_bone(bone, kps, nodes, scale, frame_data)

                if score > best_score:
                    best_score = score
                    best_cand_data = (vp_idx, vp_pos)

        if best_cand_data and best_score > self.config.min_bone_score_for_fragment:
            p_idx, p_pos = best_cand_data
            nodes = frozenset([(leaf_kp, leaf_idx), (parent_kp, p_idx)])

            # Re-estimate final scale
            final_kps = {leaf_kp: leaf_pos, parent_kp: p_pos}
            final_scale = self.skeleton_stats.estimate_scale(
                final_kps,
                self.config.min_sane_scale,
                self.config.max_sane_scale
            )
            return self._create_candidate(nodes, best_score, final_scale)

        return None

    def _try_merge(
            self,
            skel_A: SkeletonHypothesis,
            skel_B: SkeletonHypothesis,
            frame_data: FrameData
    ) -> Optional[SkeletonHypothesis]:
        """
        Attempt to merge two skeleton hypotheses.
        """

        # Disjoint check (indices)
        nodes_A = {n[1] for n in skel_A.nodes}
        nodes_B = {n[1] for n in skel_B.nodes}
        if not nodes_A.isdisjoint(nodes_B):
            return None

        # Disjoint check (keypoint types)
        kps_A = {n[0] for n in skel_A.nodes}
        kps_B = {n[0] for n in skel_B.nodes}
        if not kps_A.isdisjoint(kps_B):
            return None

        # Scale consistency
        if abs(skel_A.scale - skel_B.scale) > self.config.merge_scale_tolerance:
            return None
        combined_scale = (skel_A.scale + skel_B.scale) / 2.0

        # Find linking bone
        best_link_score = -1.0

        kps_map_A = {n[0]: frame_data.position(n[1]) for n in skel_A.nodes}
        kps_map_B = {n[0]: frame_data.position(n[1]) for n in skel_B.nodes}

        combined_kps_map = {**kps_map_A, **kps_map_B}
        combined_nodes = skel_A.nodes | skel_B.nodes

        for kp_a in kps_A:
            for kp_b in kps_B:
                bone = Bone(kp_a, kp_b)
                if bone in self.skeleton:
                    score = self._score_bone(bone, combined_kps_map, combined_nodes, combined_scale, frame_data)
                    if score > best_link_score:
                        best_link_score = score

        if best_link_score < self.config.merge_linking_bone_threshold:
            return None

        # Construct merged hypothesis
        avg_score_A = skel_A.anatomical_score / len(skel_A.nodes)
        avg_score_B = skel_B.anatomical_score / len(skel_B.nodes)
        num_bones_A = max(1, len(skel_A.nodes) - 1)
        num_bones_B = max(1, len(skel_B.nodes) - 1)

        new_total_score = (avg_score_A * num_bones_A) + (avg_score_B * num_bones_B) + best_link_score
        new_num_bones = num_bones_A + num_bones_B + 1
        new_avg_score = new_total_score / new_num_bones

        return self._create_candidate(combined_nodes, new_avg_score, combined_scale)

    def _create_candidate(
            self,
            nodes: FrozenSet[Tuple[str, int]],
            avg_score: float,
            scale: float
    ) -> SkeletonHypothesis:
        """
        Create a SkeletonHypothesis with computed scores.
        """

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

        competition_score = base_score + quality_bonus
        anatomical_score = avg_score * num_nodes

        return SkeletonHypothesis(
            nodes=nodes,
            competition_score=competition_score,
            anatomical_score=anatomical_score,
            scale=scale
        )

    def _score_bone(
            self,
            bone: Bone,
            keypoints: Dict[str, np.ndarray],
            nodes: FrozenSet[Tuple[str, int]],
            scale: float,
            frame_data: FrameData
    ) -> float:
        """
        Score a bone using SkeletonStats, incorporating point confidences.
        """

        node1 = next(n for n in nodes if n[0] == bone.k1)
        node2 = next(n for n in nodes if n[0] == bone.k2)

        return self.skeleton_stats.score_bone(
            bone,
            coords_p1=keypoints[bone.k1],
            coords_p2=keypoints[bone.k2],
            conf_p1=frame_data.confidence(node1[1]),
            conf_p2=frame_data.confidence(node2[1]),
            scale=scale,
            MAD_threshold=self.config.MAD_threshold
        )


class MultiObjectTracker:
    """
    Tracks multiple skeletons over time.
    Combines skeleton assembly with temporal association using Kalman filtering.
    """

    def __init__(
            self,
            soup: PointSoup,
            skeleton: Skeleton,
            assembler: SkeletonAssembler,
            config: TrackerConfig
    ):
        self.skeleton = skeleton
        self.assembler = assembler
        self.config = config
        self.soup = soup

        self.tracklets: List[Tracklet] = []
        self.next_track_idx = 0

    def update(self, frame_idx: int) -> List[Tracklet]:
        """Process a new frame and update tracklets."""

        frame_data = FrameData(self.soup[frame_idx])

        for tracklet in self.tracklets:
            tracklet.predict(frame_idx)

        # Generate hypotheses
        frame_candidates = self.assembler.assemble(frame_data)

        if not frame_candidates:
            self._prune_tracklets()
            return self.get_active_tracklets()

        # Association bonus for temporal continuity
        bonuses = self._association_bonuses(frame_candidates, frame_data)
        for i, cand in enumerate(frame_candidates):
            cand.competition_score += bonuses[i] * self.config.continuity_bonus

        # Solve conflicts
        conflict_graph = self._build_conflict_graph(frame_candidates, frame_data)
        winner_indices = solve_mwis(conflict_graph, method='networkx')

        # Reify winners into Pose3D
        winning_poses = []
        for i in winner_indices:
            cand = frame_candidates[i]

            kps = {node[0]: frame_data.position(node[1]) for node in cand.nodes}
            pt_indices = {node[0]: node[1] for node in cand.nodes}

            winning_poses.append(Pose3D(
                keypoints=kps,
                soup_point_indices=pt_indices,
                score=cand.anatomical_score,
                scale=cand.scale
            ))

        # Update existing tracklets
        matched_winner_indices = set()
        if self.tracklets and winning_poses:
            cost_matrix = self._build_assignment_cost_matrix(self.tracklets, winning_poses)
            tracklet_inds, winner_inds = linear_sum_assignment(cost_matrix)

            for t_idx, w_idx in zip(tracklet_inds, winner_inds):
                if cost_matrix[t_idx, w_idx] < 1e9:
                    self.tracklets[t_idx].update(
                        pose=winning_poses[w_idx],
                        frame_idx=frame_idx
                    )
                    matched_winner_indices.add(w_idx)

        # Create new tracklets for unmatched poses
        for i, pose in enumerate(winning_poses):
            if i not in matched_winner_indices and self.skeleton.central_keypoint in pose.keypoints:
                new_tracklet = Tracklet(
                    track_idx=self.next_track_idx,
                    initial_pose=pose,
                    frame_idx=frame_idx,
                    central_kp=self.skeleton.central_keypoint,
                    config=self.config
                )
                self.tracklets.append(new_tracklet)
                self.next_track_idx += 1

        self._prune_tracklets()
        return self.get_active_tracklets()

    def get_active_tracklets(self) -> List[Tracklet]:
        """Get tracklets that were updated this frame."""
        return [t for t in self.tracklets if t.time_since_update == 0]

    def _prune_tracklets(self):
        """Remove stale or uncertain tracklets."""
        self.tracklets = [
            t for t in self.tracklets
            if (t.time_since_update <= self.config.max_tracklet_age and
                np.sum(t.uncertainty['position']) <= self.config.uncertainty_threshold)
        ]

    def _build_conflict_graph(
            self,
            candidates: List[SkeletonHypothesis],
            frame_data: FrameData
    ) -> nx.Graph:
        """Build graph where edges represent mutually exclusive hypotheses."""

        conflict_graph = nx.Graph()
        num_candidates = len(candidates)

        for i, cand in enumerate(candidates):
            weight = max(0, int(cand.competition_score * 100))
            conflict_graph.add_node(i, weight=weight)

        centroids = np.array([
            np.mean([frame_data.position(node[1]) for node in cand.nodes], axis=0)
            if cand.nodes else np.array([np.nan] * 3)
            for cand in candidates
        ])

        # Collect ray sources for each candidate
        cand_ray_sources = []
        for cand in candidates:
            rays = set()
            for kp_name, idx in cand.nodes:
                ray_idx = frame_data.get_ray_index(idx)
                if ray_idx >= 0:
                    rays.add(ray_idx)
            cand_ray_sources.append(rays)

        for i, j in combinations(range(num_candidates), 2):
            cand_i, cand_j = candidates[i], candidates[j]

            # Hierarchical conflict
            if cand_i.constituent_indices and cand_j.constituent_indices:
                if (cand_j.constituent_indices.issubset(cand_i.constituent_indices) or
                        cand_i.constituent_indices.issubset(cand_j.constituent_indices)):
                    conflict_graph.add_edge(i, j)
                    continue

            # Spatial conflict (shared point indices)
            nodes_i = {n[1] for n in cand_i.nodes}
            nodes_j = {n[1] for n in cand_j.nodes}
            if not nodes_i.isdisjoint(nodes_j):
                conflict_graph.add_edge(i, j)
                continue

            # Ray source conflict
            if not cand_ray_sources[i].isdisjoint(cand_ray_sources[j]):
                conflict_graph.add_edge(i, j)
                continue

            # Spatial proximity (clone detection)
            dist_sq = np.sum((centroids[i] - centroids[j]) ** 2)

            if dist_sq < self.config.conflict_solver_broad_radius ** 2:
                kps_i = {node[0]: frame_data.position(node[1]) for node in cand_i.nodes}
                kps_j = {node[0]: frame_data.position(node[1]) for node in cand_j.nodes}

                common = kps_i.keys() & kps_j.keys()
                union = kps_i.keys() | kps_j.keys()

                if union:
                    proximal_count = sum(
                        1 for name in common
                        if np.linalg.norm(kps_i[name] - kps_j[name]) < self.config.conflict_solver_proximity_radius
                    )

                    if proximal_count / len(union) > self.config.conflict_solver_jaccard_threshold:
                        conflict_graph.add_edge(i, j)

        return conflict_graph

    def _association_bonuses(
            self,
            candidates: List[SkeletonHypothesis],
            frame_data: FrameData
    ) -> np.ndarray:
        """
        Calculate temporal association bonuses.
        """

        if not self.tracklets or not candidates:
            return np.zeros(len(candidates))

        bonuses = np.zeros(len(candidates))

        for j, cand_skel in enumerate(candidates):
            skel_kps = {node[0]: frame_data.position(node[1]) for node in cand_skel.nodes}

            if not skel_kps:
                continue

            max_bonus = 0.0

            for t in self.tracklets:
                pred_pose = t.predicted_pose

                if not pred_pose:
                    continue

                common_kps = pred_pose.keys() & skel_kps.keys()
                if len(common_kps) < self.config.association_min_kps:
                    continue

                mean_dist_sq = sum(np.sum((pred_pose[kp] - skel_kps[kp]) ** 2) for kp in common_kps) / len(common_kps)

                bonus = np.exp(-0.5 * mean_dist_sq / (self.config.association_radius ** 2))
                if bonus > max_bonus:
                    max_bonus = bonus

            bonuses[j] = max_bonus

        return bonuses

    def _build_assignment_cost_matrix(
            self,
            tracklets: List[Tracklet],
            poses: List[Pose3D]
    ) -> np.ndarray:
        """
        Build cost matrix for tracklet-pose assignment.
        """

        cost_matrix = np.full((len(tracklets), len(poses)), 1e9)

        for i, tracklet in enumerate(tracklets):
            pred_pose = tracklet.predicted_pose

            if not pred_pose:
                continue

            for j, pose in enumerate(poses):
                common_kps = pred_pose.keys() & pose.keypoints.keys()

                if len(common_kps) < self.config.association_min_kps:
                    continue

                mean_dist_sq = sum(
                    np.sum((pred_pose[kp] - pose.keypoints[kp]) ** 2)
                    for kp in common_kps
                ) / len(common_kps)

                if mean_dist_sq > self.config.association_radius ** 2:
                    continue

                cost = (self.config.cost_pose_distance_weight * mean_dist_sq +
                        self.config.cost_skeleton_score_weight * pose.score)
                cost_matrix[i, j] = cost

        return cost_matrix


if __name__ == '__main__':
    import pickle
    from pathlib import Path

    BASE_DIR = Path.home() / 'Desktop' / '3d_ant_data'
    PREFIX = '240905-1616'
    SESSION = 22
    DEBUG_PLOT = True

    input_dir = BASE_DIR / PREFIX / 'inputs' / 'tracking'
    output_dir = BASE_DIR / PREFIX / 'outputs'

    soup_file = output_dir / f"soup_session{SESSION}.pkl"
    stats_file = output_dir / "skeleton_stats.json"
    tracklets_file = output_dir / f'tracklets_session{SESSION}.pkl'

    assembler_cfg = AssemblerConfig()
    tracker_cfg = TrackerConfig()

    # Load stuff
    print(f"Loading soup from {soup_file}...")
    with open(soup_file, 'rb') as f:
        soup = pickle.load(f)

    skeleton = Skeleton.from_sleap(input_dir)
    print(f"Loaded skeleton with {len(skeleton.keypoints)} keypoints, {len(skeleton.bones)} bones")

    print(f"Loading stats from {stats_file}...")
    stats = SkeletonStats.from_json(stats_file, skeleton)

    # Initialize pipeline
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
    tracklets_by_id = defaultdict(list)

    print(f"Tracking from frame {min_frame} to {max_frame}...")
    with alive_bar(total=(max_frame - min_frame + 1), length=20, force_tty=True) as bar:

        for frame_idx in range(min_frame, max_frame + 1):
            active_tracklets = tracker.update(frame_idx)

            for tracklet in active_tracklets:
                if tracklet.last_update_frame == frame_idx:
                    stats.update(tracklet.pose.keypoints)

                pose_dict = tracklet.pose.to_dict()
                pose_dict.update({
                    'track_idx': tracklet.track_idx,
                    'track_health': tracklet.health,
                    'track_anatomical_integrity': tracklet.anatomical_integrity,
                    'track_uncertainty_pos': tracklet.uncertainty['position'].tolist(),
                    'track_velocity': tracklet.kf.x[3:6].flatten().tolist(),
                    'track_predicted_pos': tracklet.predicted_position.tolist(),
                    'time_since_update': tracklet.time_since_update,
                    'frame_idx': frame_idx
                })
                tracklets_by_id[tracklet.track_idx].append(pose_dict)
            bar()

    print("Tracking complete.")
    print(f"Generated {len(tracklets_by_id)} unique tracklets.")

    # Save tracking results
    tracklets_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tracklets_file, 'wb') as f:
        pickle.dump(dict(tracklets_by_id), f)
    print(f"Tracklet results saved to '{tracklets_file}'")

    # Save updated stats
    stats.to_json(stats_file)
    print(f"Updated stats saved to '{stats_file}'")

    if DEBUG_PLOT:
        from mokap.pose_reconstruction.debug import TrackletViewer

        viewer = TrackletViewer(soup, tracklets_by_id, skeleton)