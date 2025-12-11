import logging
from typing import Tuple, Optional, Dict, List, FrozenSet
from itertools import combinations
from collections import defaultdict
import networkx as nx

import numpy as np

from alive_progress import alive_bar
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from mokap.utils import fileio
from lucida.geometry import align_rigid

from mokap.reconstruction.config import AssemblerConfig, TrackerConfig, AnatomyConfig
from mokap.reconstruction.datatypes import Bone, AssembledSkeleton, CandidateSkeleton, SoupData
from mokap.reconstruction.utils import solve_mwis_networkx


logger = logging.getLogger(__name__)


class Tracklet:
    """
    Stateful class that represents a single skeleton in a tracklet
    Manages state estimation (position, velocity, scale) using a Kalman Filter
    """

    def __init__(self,
                 track_idx: int,
                 initial_skeleton: AssembledSkeleton,
                 frame_idx: int,
                 central_kp: str,
                 config: TrackerConfig
                 ):

        self.config = config
        self.track_idx = track_idx
        self.age = 0
        self.time_since_update = 0
        self.last_update_frame = frame_idx

        # Tracklet health and score metrics
        self.health = 1.0
        self.anatomical_integrity = initial_skeleton.score

        self.skeleton: AssembledSkeleton = initial_skeleton
        self.central_kp = central_kp

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
        pos_vel_q = Q_discrete_white_noise(dim=2, dt=dt, var=self.config.KF_PROCESS_NOISE_POS, block_size=3)
        scale_q = np.array([[self.config.KF_PROCESS_NOISE_SCALE]])
        self.kf.Q = block_diag(pos_vel_q, scale_q)

        # Measurement noise
        self.kf.R = np.diag([self.config.KF_MEASUREMENT_NOISE_POS, self.config.KF_MEASUREMENT_NOISE_POS,
                             self.config.KF_MEASUREMENT_NOISE_POS, self.config.KF_MEASUREMENT_NOISE_SCALE])

        # Initial state covariance
        self.kf.P[3:6, 3:6] *= 1.0
        self.kf.P[6, 6] = 1.0

        # Initial state
        self.kf.x[:3] = self.skeleton.keypoints[self.central_kp].reshape(3, 1)
        self.kf.x[6] = self.skeleton.scale

    def predict(self, current_frame_idx: int):
        """Predicts the state of the tracklet for the current frame."""

        steps_to_predict = current_frame_idx - self.last_update_frame

        for _ in range(steps_to_predict):
            self.kf.predict()
            self.age += 1
            self.time_since_update += 1
            self.health *= self.config.HEALTH_DECAY_RATE

    def update(self, skeleton: AssembledSkeleton, frame_idx: int):
        """Updates the tracklet's state with a new skeleton measurement."""

        update_skeleton = skeleton
        inferred = False

        # If the primary keypoint is missing, try to infer it
        if self.central_kp not in skeleton.keypoints:
            inferred_skeleton = self._infer_missing_central_kp(skeleton)
            if inferred_skeleton:
                update_skeleton = inferred_skeleton
                inferred = True
            else:
                self.skeleton = skeleton
                self.score = skeleton.score
                self.time_since_update = 0
                self.last_update_frame = frame_idx
                return

        self.skeleton = update_skeleton
        self.time_since_update = 0
        self.last_update_frame = frame_idx

        measurement = np.array([*update_skeleton.keypoints[self.central_kp], update_skeleton.scale])

        if inferred:
            original_R = self.kf.R.copy()
            self.kf.R[:3, :3] *= self.config.KF_INFERENCE_UNCERTAINTY_FACTOR
            self.kf.update(measurement)
            self.kf.R = original_R
        else:
            self.kf.update(measurement)

        # Update health metrics
        self.anatomical_integrity = self.config.ANATOMICAL_SCORE_ALPHA * skeleton.score + (
                1 - self.config.ANATOMICAL_SCORE_ALPHA) * self.anatomical_integrity

        if inferred:
            self.health = 1.0 - self.config.INFERRED_HEALTH_PENALTY
        else:
            self.health = 1.0

    def _infer_missing_central_kp(self, fragment: AssembledSkeleton) -> Optional[AssembledSkeleton]:
        prev_kps, curr_kps = self.skeleton.keypoints, fragment.keypoints
        common_names = list(prev_kps.keys() & curr_kps.keys())

        if len(common_names) < self.config.MIN_KPS_FOR_INFERENCE:
            return None

        points_A = np.array([prev_kps[name] for name in common_names])
        points_B = np.array([curr_kps[name] for name in common_names])

        R_mat, t_vec = align_rigid(points_A, points_B)
        inferred_pos = np.array(R_mat) @ prev_kps[self.central_kp] + np.array(t_vec)

        completed_skeleton = AssembledSkeleton(
            keypoints=fragment.keypoints.copy(),
            score=fragment.score,
            scale=fragment.scale,
            point_indices=fragment.point_indices
        )
        completed_skeleton.keypoints[self.central_kp] = inferred_pos
        return completed_skeleton

    @property
    def predicted_pose(self) -> Optional[Dict[str, np.ndarray]]:
        if self.central_kp not in self.skeleton.keypoints:
            return None
        translation = self.predicted_position - self.skeleton.keypoints[self.central_kp]
        return {kp_name: pos + translation for kp_name, pos in self.skeleton.keypoints.items()}

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


class SkeletonAssembler:
    """Assembles skeletons from a SoupData frame slice."""

    def __init__(self,
                 bones_list: List[Bone],
                 bone_stats: Dict,
                 assembler_config: AssemblerConfig,
                 tracker_config: TrackerConfig
                 ):

        self.config = assembler_config
        self.tracker_config = tracker_config

        self.bones_list = [frozenset(bone) for bone in bones_list]
        self.update_bone_stats(bone_stats)

        skeleton_graph = nx.Graph([tuple(b) for b in self.bones_list])
        if not skeleton_graph.nodes:
            raise ValueError("Cannot assemble skeletons from an empty bone list.")

        self._skeleton_graph = skeleton_graph

        # Topology analysis
        degrees = dict(self._skeleton_graph.degree())
        self.leaf_nodes = {node for node, degree in degrees.items() if degree == 1}

        non_leaf_nodes = set(degrees.keys()) - self.leaf_nodes
        if non_leaf_nodes:
            sorted_anchors = sorted(list(non_leaf_nodes), key=lambda node: degrees[node], reverse=True)
            self.central_anchors = set(sorted_anchors[:self.config.MIN_CENTRAL_ANCHORS])
            self.secondary_anchors = non_leaf_nodes - self.central_anchors
        else:
            self.central_anchors = set()
            self.secondary_anchors = set()

        self.central_kp = max(degrees, key=degrees.get)

        # Temporary storage for the current frame
        self._current_soup: Optional[SoupData] = None
        self._current_virtual_points: Dict[int, Dict] = {}  # Negative indices -> {pos, conf, kp_type}
        self._current_kdtree: Optional[cKDTree] = None
        self._kp_to_indices: Dict[str, List[int]] = defaultdict(list)
        self._kp_to_rays: Dict[str, List[int]] = defaultdict(list)

    def update_bone_stats(self, new_stats: Dict):
        self.reference_bone: Bone = frozenset(new_stats['reference_bone'])
        self.median_ref_len = new_stats['median_reference_length']
        self.bones_ratios = {frozenset(k.split(';')): v for k, v in new_stats['bones_ratios'].items()}

    def _reset_frame_context(self, soup: SoupData):
        """Prepares lookup structures for the current frame."""
        self._current_soup = soup
        self._current_virtual_points = {}

        # Index map for 3D points
        self._kp_to_indices.clear()
        for i in range(soup.num_points):
            kp_name = soup.keypoint_names[soup.kp_types[i]]
            self._kp_to_indices[kp_name].append(i)

        # Index map for Rays (Orphan views)
        self._kp_to_rays.clear()
        num_rays = len(soup.ray_origins)
        for i in range(num_rays):
            kp_name = soup.keypoint_names[soup.ray_kp_types[i]]
            self._kp_to_rays[kp_name].append(i)

        # KDTree
        if soup.num_points > 0:
            self._current_kdtree = cKDTree(soup.positions)
        else:
            self._current_kdtree = None

    def _get_pos_conf(self, idx: int) -> Tuple[np.ndarray, float]:
        """Abstracts retrieval of position/confidence for Real (>= 0) vs Virtual (< 0) points."""
        if idx >= 0:
            return self._current_soup.positions[idx], self._current_soup.confidences[idx]
        else:
            vp = self._current_virtual_points[idx]
            return vp['pos'], vp['conf']

    def assemble_frame(self, soup: SoupData) -> Tuple[List[CandidateSkeleton], Dict[int, Dict]]:
        """
        Main assembly entry point.
        Returns candidates and the registry of virtual points created during rescue.
        """
        self._reset_frame_context(soup)

        # Generate initial fragments (including orphan rescue)
        initial_fragments = self._generate_candidates()
        if not initial_fragments:
            return [], self._current_virtual_points

        # Generate merges
        merge_hypotheses = self._generate_merge_hypotheses(initial_fragments)

        all_hypotheses = initial_fragments + merge_hypotheses

        return all_hypotheses, self._current_virtual_points

    def _generate_candidates(self) -> List[CandidateSkeleton]:
        candidate_skeletons = []
        used_as_seed_indices = set()

        # Seed from anchors
        all_anchor_types = self.central_anchors.union(self.secondary_anchors)

        for anchor_type in all_anchor_types:
            # Iterate over all real 3D points of this type
            for seed_idx in self._kp_to_indices[anchor_type]:
                if seed_idx in used_as_seed_indices:
                    continue

                candidate = self._grow_skeleton(
                    (anchor_type, seed_idx),
                    score_debt_tol=self.config.SCORE_DEBT_TOLERANCE
                )

                if candidate:
                    candidate_skeletons.append(candidate)
                    for _, idx in candidate.nodes:
                        # Only mark real points as used seeds
                        if idx >= 0:
                            used_as_seed_indices.add(idx)

        # Seed from leaves (cleanup)
        for leaf_type in self.leaf_nodes:
            for seed_idx in self._kp_to_indices[leaf_type]:
                if seed_idx in used_as_seed_indices:
                    continue

                fragment = self._find_leaf_fragment(leaf_type, seed_idx)
                if fragment:
                    candidate_skeletons.append(fragment)
                    for _, idx in fragment.nodes:
                        if idx >= 0:
                            used_as_seed_indices.add(idx)

        return candidate_skeletons

    def _grow_skeleton(self,
            anchor_node: Tuple[str, int],
            score_debt_tol: float
        ) -> Optional[CandidateSkeleton]:
        """
        Grows a skeleton from an anchor and tries to rescue single view observations.
        """

        # anchor_node is (kp_name, index)
        anchor_idx = anchor_node[1]
        anchor_pos, _ = self._get_pos_conf(anchor_idx)

        # Initialise
        current_nodes = {anchor_node}
        current_kps = {anchor_node[0]: anchor_pos}
        total_bone_score_sum = 0.0
        num_bones = 0

        # Heuristic: don't search infinitely far
        max_search_radius = self.median_ref_len * self.config.MAX_BONE_LEN

        while True:
            current_kp_names = {node[0] for node in current_nodes}
            nodes_to_evaluate = set()

            # Find candidates via Graph topology + spatial proximity
            for node_in_skel in current_nodes:
                node_kp, node_idx = node_in_skel
                node_pos, _ = self._get_pos_conf(node_idx)

                neighbor_kp_types = {n for n in self._skeleton_graph.neighbors(node_kp) if n not in current_kp_names}
                if not neighbor_kp_types: continue

                # Standard 3D search
                if self._current_kdtree:
                    nearby_indices = self._current_kdtree.query_ball_point(node_pos, r=max_search_radius)
                    for idx in nearby_indices:
                        cand_type_idx = self._current_soup.kp_types[idx]
                        cand_type = self._current_soup.keypoint_names[cand_type_idx]

                        if cand_type in neighbor_kp_types:
                            nodes_to_evaluate.add((cand_type, int(idx)))

                # Orphan Ray rescue (single view observations)

                # Try to intersect rays
                for target_type in neighbor_kp_types:
                    # Only rescue if we don't have strong 3D candidates for this type already
                    # (to avoid solving geometry for everything)
                    has_3d_candidates = any(n[0] == target_type for n in nodes_to_evaluate)

                    if not has_3d_candidates:
                        orphans = self._single_view_rescue(node_pos, node_kp, target_type)
                        for vp_idx, vp_type in orphans:
                            nodes_to_evaluate.add((vp_type, vp_idx))

            if not nodes_to_evaluate:
                break

            # Scale
            current_step_scale = self._get_skeleton_scale(current_kps)
            if not (self.config.MIN_SANE_SCALE < current_step_scale < self.config.MAX_SANE_SCALE):
                break

            # Baseline scores
            current_avg_score = (total_bone_score_sum / num_bones) if num_bones > 0 else 0
            current_base_score = current_avg_score * len(current_nodes)
            current_quality_bonus = 0.0
            if current_avg_score > self.config.HIGH_QUALITY_THRESHOLD:
                current_quality_bonus = current_base_score * (self.config.QUALITY_BONUS_FACTOR - 1.0)
            current_growth_score = current_base_score + current_quality_bonus

            best_extension = None
            best_new_growth_score = -float('inf')

            # Evaluate extensions
            for cand_node in nodes_to_evaluate:
                cand_kp_name, cand_idx = cand_node
                cand_pos, _ = self._get_pos_conf(cand_idx)

                temp_kps = current_kps.copy()
                temp_kps[cand_kp_name] = cand_pos

                new_bone_score_sum = 0
                new_bone_count = 0

                for existing_node in current_nodes:
                    bone = frozenset((cand_kp_name, existing_node[0]))

                    if bone in self.bones_ratios:

                        temp_nodes = frozenset(current_nodes | {cand_node})
                        score = self._score_bone(bone, temp_kps, temp_nodes, current_step_scale)

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
                bonus_factor = self.config.QUALITY_BONUS_FACTOR - 1.0
                normalized_quality = max(0, (new_avg_score - 75.0) / 25.0)
                new_quality_bonus = new_base_score * bonus_factor * normalized_quality

                new_growth_score = new_base_score + new_quality_bonus

                if new_growth_score > best_new_growth_score:
                    best_new_growth_score = new_growth_score

                    best_extension = {
                        "node": cand_node,
                        "pos": cand_pos,
                        "score_contribution": new_bone_score_sum,
                        "bone_count_increase": new_bone_count
                    }

            # Decision
            if best_extension and best_new_growth_score > (current_growth_score - score_debt_tol):
                best_node = best_extension["node"]
                current_nodes.add(best_node)
                current_kps[best_node[0]] = best_extension["pos"]
                total_bone_score_sum += best_extension["score_contribution"]
                num_bones += best_extension["bone_count_increase"]

            else:
                break

        # Finish
        if len(current_nodes) < self.config.MIN_KPS_FOR_SKELETON:
            return None

        final_avg_score = (total_bone_score_sum / num_bones) if num_bones > 0 else 0.0
        if final_avg_score <= 0:
            return None

        final_scale = self._get_skeleton_scale(current_kps)

        return self._create_candidate(frozenset(current_nodes), final_avg_score, final_scale)

    def _single_view_rescue(self,
            anchor_pos: np.ndarray,
            anchor_kp: str,
            target_kp: str
        ) -> List[Tuple[int, str]]:
        """
        Intersects rays for target_kp with sphere around anchor_pos.
        Returns list of (virtual_idx, kp_type) tuples.
        """
        bone = frozenset((anchor_kp, target_kp))
        if bone not in self.bones_ratios:
            return []

        # Expected radius
        r = self.median_ref_len * self.bones_ratios[bone]['median_ratio']

        # Get orphan rays
        ray_indices = self._kp_to_rays[target_kp]
        if not ray_indices:
            return []

        ray_indices_arr = np.array(ray_indices)
        origins = self._current_soup.ray_origins[ray_indices_arr]
        dirs = self._current_soup.ray_directions[ray_indices_arr]

        # Ray-sphere intersection
        # TODO: Maybe make a jitted version
        V = origins - anchor_pos
        b = 2.0 * np.einsum('ij,ij->i', V, dirs)
        c = np.einsum('ij,ij->i', V, V) - (r ** 2)
        discriminant = b ** 2 - 4 * c

        # Filter valid intersections
        valid_mask = discriminant >= 0
        if not np.any(valid_mask):
            return []

        valid_indices = np.where(valid_mask)[0]

        # Calculate points for valid rays
        sqrt_delta = np.sqrt(discriminant[valid_indices])
        b_valid = b[valid_indices]

        t1 = (-b_valid - sqrt_delta) / 2.0
        t2 = (-b_valid + sqrt_delta) / 2.0

        orig_valid = origins[valid_indices]
        dirs_valid = dirs[valid_indices]

        p1s = orig_valid + t1[:, np.newaxis] * dirs_valid
        p2s = orig_valid + t2[:, np.newaxis] * dirs_valid

        # Register virtual points
        result = []

        # Original indices into the SoupData ray arrays
        source_ray_ids = ray_indices_arr[valid_indices]

        for i in range(len(valid_indices)):
            ray_idx = source_ray_ids[i]

            conf = self._current_soup.ray_confidences[ray_idx]

            # Add both solutions as candidates
            # The conflict solver should trash the wrong one
            vid1 = self._register_virtual_point(p1s[i], conf, target_kp, ray_idx)
            result.append((vid1, target_kp))

            vid2 = self._register_virtual_point(p2s[i], conf, target_kp, ray_idx)
            result.append((vid2, target_kp))

        return result

    def _register_virtual_point(self, pos: np.ndarray, conf: float, kp_type: str, source_ray_idx: int) -> int:
        """
        Stores a computed 3D point in the temporary registry and returns a unique negative index.
        """

        next_idx = -1 * (len(self._current_virtual_points) + 1)
        self._current_virtual_points[next_idx] = {
            'pos': pos,
            'conf': conf * 0.8,  # penalty for being inferred from single view
            'kp_type': kp_type,
            'source_ray_idx': source_ray_idx    # ray source index (needed to make the candidates mutually exclusive)
        }
        return next_idx

    def _find_leaf_fragment(self, leaf_kp: str, leaf_idx: int) -> Optional[CandidateSkeleton]:

        # Simple leaf finder logic
        try:
            parent_kp = next(self._skeleton_graph.neighbors(leaf_kp))
        except StopIteration:
            return None

        leaf_pos, _ = self._get_pos_conf(leaf_idx)
        best_score = -1.0
        best_cand_data = None  # stores (idx, pos, is_virtual)

        # Check for real parent points
        parent_indices = self._kp_to_indices[parent_kp]
        for p_idx in parent_indices:
            p_pos, _ = self._get_pos_conf(p_idx)

            kps = {leaf_kp: leaf_pos, parent_kp: p_pos}
            scale = self._get_skeleton_scale(kps)
            if not (self.config.MIN_SANE_SCALE < scale < self.config.MAX_SANE_SCALE):
                continue

            # Score it
            nodes = frozenset([(leaf_kp, leaf_idx), (parent_kp, p_idx)])
            bone = frozenset((leaf_kp, parent_kp))
            score = self._score_bone(bone, kps, nodes, scale)

            if score > best_score:
                best_score = score
                best_cand_data = (p_idx, p_pos)

        # Check for orphan parent points
        # (only if we didn't find a high-quality real connection)
        if best_score < self.config.MIN_BONE_SCORE_FOR_FRAGMENT:
            orphans = self._single_view_rescue(leaf_pos, leaf_kp, parent_kp)

            for vp_idx, _ in orphans:
                vp_pos, _ = self._get_pos_conf(vp_idx)

                kps = {leaf_kp: leaf_pos, parent_kp: vp_pos}
                scale = self._get_skeleton_scale(kps)  # estimate scale
                if not (self.config.MIN_SANE_SCALE < scale < self.config.MAX_SANE_SCALE):
                    continue

                nodes = frozenset([(leaf_kp, leaf_idx), (parent_kp, vp_idx)])
                bone = frozenset((leaf_kp, parent_kp))
                score = self._score_bone(bone, kps, nodes, scale)

                if score > best_score:
                    best_score = score
                    best_cand_data = (vp_idx, vp_pos)

        if best_cand_data and best_score > self.config.MIN_BONE_SCORE_FOR_FRAGMENT:
            p_idx, p_pos = best_cand_data
            nodes = frozenset([(leaf_kp, leaf_idx), (parent_kp, p_idx)])

            # Recompute scale
            final_kps = {leaf_kp: leaf_pos, parent_kp: p_pos}
            final_scale = self._get_skeleton_scale(final_kps)
            return self._create_candidate(nodes, best_score, final_scale)

        return None

    def _generate_merge_hypotheses(self, fragments: List[CandidateSkeleton]) -> List[CandidateSkeleton]:

        if len(fragments) < 2:
            return []

        for i, frag in enumerate(fragments):
            frag.constituent_indices = frozenset([i])

        merge_hypotheses = []
        for i, j in combinations(range(len(fragments)), 2):
            skel_A, skel_B = fragments[i], fragments[j]
            merged = self._attempt_merge(skel_A, skel_B)
            if merged:
                merged.constituent_indices = skel_A.constituent_indices.union(skel_B.constituent_indices)
                merge_hypotheses.append(merged)

        return merge_hypotheses

    def _attempt_merge(self, skel_A: CandidateSkeleton, skel_B: CandidateSkeleton) -> Optional[CandidateSkeleton]:

        # Disjoint check
        nodes_A = {n[1] for n in skel_A.nodes}  # indices only
        nodes_B = {n[1] for n in skel_B.nodes}
        if not nodes_A.isdisjoint(nodes_B):
            return None

        kps_A = {n[0] for n in skel_A.nodes}
        kps_B = {n[0] for n in skel_B.nodes}
        if not kps_A.isdisjoint(kps_B):
            return None

        # Scale consistency
        if abs(skel_A.scale - skel_B.scale) > self.config.MERGE_SCALE_TOLERANCE:
            return None
        combined_scale = (skel_A.scale + skel_B.scale) / 2.0

        # Find linking bone
        best_link_score = -1.0

        # Collect positions
        kps_map_A = {n[0]: self._get_pos_conf(n[1])[0] for n in skel_A.nodes}
        kps_map_B = {n[0]: self._get_pos_conf(n[1])[0] for n in skel_B.nodes}

        combined_kps_map = {**kps_map_A, **kps_map_B}
        combined_nodes = skel_A.nodes.union(skel_B.nodes)

        for kp_a in kps_A:
            for kp_b in kps_B:
                bone = frozenset((kp_a, kp_b))
                if bone in self.bones_ratios:
                    score = self._score_bone(bone, combined_kps_map, combined_nodes, combined_scale)
                    if score > best_link_score:
                        best_link_score = score

        if best_link_score < self.config.MERGE_LINKING_BONE_THRESHOLD:
            return None

        # Construct merged
        avg_score_A = skel_A.anatomical_score / len(skel_A.nodes)
        avg_score_B = skel_B.anatomical_score / len(skel_B.nodes)
        num_bones_A = max(1, len(skel_A.nodes) - 1)
        num_bones_B = max(1, len(skel_B.nodes) - 1)

        new_total_score = (avg_score_A * num_bones_A) + (avg_score_B * num_bones_B) + best_link_score
        new_num_bones = num_bones_A + num_bones_B + 1
        new_avg_score = new_total_score / new_num_bones

        return self._create_candidate(combined_nodes, new_avg_score, combined_scale)

    def _create_candidate(self,
            nodes: FrozenSet[Tuple[str, int]],
            avg_score: float,
            scale: float
         ) -> CandidateSkeleton:

        if avg_score <= 0:
            return CandidateSkeleton(nodes=nodes, competition_score=0.0, anatomical_score=0.0, scale=scale)

        num_nodes = len(nodes)
        base_score = avg_score * num_nodes

        quality_bonus = 0.0
        if avg_score > self.config.HIGH_QUALITY_THRESHOLD:
            quality_bonus = base_score * (self.config.QUALITY_BONUS_FACTOR - 1.0)

        competition_score = base_score + quality_bonus
        anatomical_score = avg_score * num_nodes

        return CandidateSkeleton(
            nodes=nodes,
            competition_score=competition_score,
            anatomical_score=anatomical_score,
            scale=scale
        )

    def _get_skeleton_scale(self, keypoints: Dict[str, np.ndarray]) -> float:

        scales = []

        if self.reference_bone.issubset(keypoints) and self.median_ref_len > 1e-6:
            kp1, kp2 = tuple(self.reference_bone)
            scales.append(np.linalg.norm(keypoints[kp1] - keypoints[kp2]) / self.median_ref_len)

        for bone_type, stats in self.bones_ratios.items():
            kp1, kp2 = tuple(bone_type)
            if kp1 in keypoints and kp2 in keypoints:
                expected_len = self.median_ref_len * stats['median_ratio']
                if expected_len > 1e-6:
                    scales.append(np.linalg.norm(keypoints[kp1] - keypoints[kp2]) / expected_len)

        if not scales:
            return 1.0

        sane_scales = [s for s in scales if self.config.MIN_SANE_SCALE <= s <= self.config.MAX_SANE_SCALE]

        return float(np.median(sane_scales)) if sane_scales else 1.0

    def _score_bone(self,
            bone: Bone,
            keypoints: Dict[str, np.ndarray],
            nodes: FrozenSet[Tuple[str, int]],
            scale: float
        ) -> float:

        if bone not in self.bones_ratios:
            return 0.0

        stats = self.bones_ratios[bone]
        kp1_name, kp2_name = tuple(bone)
        p1, p2 = keypoints[kp1_name], keypoints[kp2_name]

        # Retrieve confidences
        node1 = next(n for n in nodes if n[0] == kp1_name)
        node2 = next(n for n in nodes if n[0] == kp2_name)

        _, conf1 = self._get_pos_conf(node1[1])
        _, conf2 = self._get_pos_conf(node2[1])

        expected_len = self.median_ref_len * stats['median_ratio'] * scale
        expected_mad = self.median_ref_len * stats['mad_ratio'] * scale + self.config.BONE_SCORE_MAD_EPSILON
        distance = np.linalg.norm(p1 - p2)

        num_mads_away = abs(distance - expected_len) / (expected_mad + 1e-6)

        if num_mads_away > self.config.BONE_SCORE_MAD_THRESH:
            return -1000.0

        length_score = np.exp(-0.5 * num_mads_away ** 2)
        confidence_score = (conf1 + conf2) / 2.0

        return length_score * confidence_score


class MultiObjectTracker:
    """Main class for tracking multiple skeletons over time."""

    def __init__(self, assembler: SkeletonAssembler, config: TrackerConfig = TrackerConfig()):
        self.assembler = assembler
        self.config = config
        self.frame_idx = -1
        self.tracklets: List[Tracklet] = []
        self.next_track_idx = 0

    def update(self, soup: SoupData, frame_idx: int) -> List[Tracklet]:
        self.frame_idx = frame_idx

        for tracklet in self.tracklets:
            tracklet.predict(self.frame_idx)

        # Generate Hypotheses
        all_candidates, virtual_registry = self.assembler.assemble_frame(soup)

        if not all_candidates:
            self.prune_tracklets()
            return self.get_active_tracklets()

        # mini helper to resolve positions for conflict graph and association
        def get_pos(idx):
            if idx >= 0:
                return soup.positions[idx]
            return virtual_registry[idx]['pos']

        # Association bonus
        bonuses = self._calculate_association_bonuses(all_candidates, get_pos)
        for i, cand in enumerate(all_candidates):
            cand.competition_score += bonuses[i] * self.config.CONTINUITY_BONUS

        # Conflict solver
        conflict_graph = self._build_conflict_graph(all_candidates, get_pos, virtual_registry)
        winner_indices = solve_mwis_networkx(conflict_graph)

        # Reify winners into AssembledSkeletons
        winning_skeletons = []

        for i in winner_indices:
            cand = all_candidates[i]
            kps = {node[0]: get_pos(node[1]) for node in cand.nodes}
            pt_indices = {node[0]: node[1] for node in cand.nodes}

            winning_skeletons.append(
                AssembledSkeleton(keypoints=kps, point_indices=pt_indices, score=cand.anatomical_score,
                                  scale=cand.scale)
            )

        # Update tracklets
        matched_winner_indices = set()
        if self.tracklets and winning_skeletons:
            cost_matrix = self._build_final_assignment_cost_matrix(self.tracklets, winning_skeletons)
            tracklet_inds, winner_inds = linear_sum_assignment(cost_matrix)

            for t_idx, w_idx in zip(tracklet_inds, winner_inds):
                if cost_matrix[t_idx, w_idx] < 1e9:
                    self.tracklets[t_idx].update(skeleton=winning_skeletons[w_idx], frame_idx=self.frame_idx)
                    matched_winner_indices.add(w_idx)

        # Create new tracklets
        for i, skel in enumerate(winning_skeletons):
            if i not in matched_winner_indices and self.assembler.central_kp in skel.keypoints:
                new_tracklet = Tracklet(self.next_track_idx, skel, self.frame_idx, self.assembler.central_kp,
                                        self.config)
                self.tracklets.append(new_tracklet)
                self.next_track_idx += 1

        self.prune_tracklets()

        return self.get_active_tracklets()

    def predict_only(self, frame_idx: int) -> List[Tracklet]:
        self.frame_idx = frame_idx

        for tracklet in self.tracklets:
            tracklet.predict(self.frame_idx)

        self.prune_tracklets()

        return self.get_active_tracklets()

    def get_active_tracklets(self) -> List[Tracklet]:
        return [t for t in self.tracklets if t.time_since_update == 0]

    def prune_tracklets(self):
        self.tracklets = [
            t for t in self.tracklets
            if t.time_since_update <= self.config.MAX_TRACKLET_AGE and not np.sum(
                t.uncertainty['position']) > self.config.UNCERTAINTY_THRESHOLD
        ]

    def _build_conflict_graph(self,
            candidates: List[CandidateSkeleton],
            pos_lookup,
            virtual_registry: Dict[int, Dict]
        ) -> nx.Graph:

        conflict_graph = nx.Graph()
        num_candidates = len(candidates)

        for i, cand in enumerate(candidates):
            weight = max(0, int(cand.competition_score * 100))
            conflict_graph.add_node(i, weight=weight)

        centroids = np.array([
            np.mean([pos_lookup(node[1]) for node in cand.nodes], axis=0) if cand.nodes else np.array([np.nan] * 3)
            for cand in candidates
        ])

        # Grab the set of source ray IDs for each candidate
        # For real points ray_id is None (or -1)
        # For virtual points it's the index in SoupData.rays
        cand_ray_sources = []
        for cand in candidates:
            rays = set()
            for kp_name, idx in cand.nodes:
                if idx < 0:  # virtual point
                    source_ray = virtual_registry[idx]['source_ray_idx']
                    rays.add(source_ray)
            cand_ray_sources.append(rays)

        for i, j in combinations(range(num_candidates), 2):
            cand_i, cand_j = candidates[i], candidates[j]

            # Hierarchical conflict
            if cand_i.constituent_indices and cand_j.constituent_indices:
                if cand_j.constituent_indices.issubset(cand_i.constituent_indices) or \
                        cand_i.constituent_indices.issubset(cand_j.constituent_indices):
                    conflict_graph.add_edge(i, j)
                    continue

            # Spatial conflict (shared point indices)
            nodes_i = {n[1] for n in cand_i.nodes}
            nodes_j = {n[1] for n in cand_j.nodes}
            if not nodes_i.isdisjoint(nodes_j):
                conflict_graph.add_edge(i, j)
                continue

            # Ray source conflict
            # If both candidates come from the same 2D detection ray, they are mutually exclusive
            # (even if the 3D points are different)
            if not cand_ray_sources[i].isdisjoint(cand_ray_sources[j]):
                conflict_graph.add_edge(i, j)
                continue

            # Spatial proximity (clones)
            # Jaccard similarity: if most of the points are in the same positions then they're likely clones
            dist_sq = np.sum((centroids[i] - centroids[j]) ** 2)

            if dist_sq < self.config.CONFLICT_SOLVER_BROAD_RADIUS ** 2:
                kps_i = {node[0]: pos_lookup(node[1]) for node in cand_i.nodes}
                kps_j = {node[0]: pos_lookup(node[1]) for node in cand_j.nodes}

                common = kps_i.keys() & kps_j.keys()
                union = kps_i.keys() | kps_j.keys()

                if not union:
                    continue

                proximal_count = sum(
                    1 for name in common if
                    np.linalg.norm(kps_i[name] - kps_j[name]) < self.config.CONFLICT_SOLVER_PROXIMITY_RADIUS
                )

                if proximal_count / len(union) > self.config.CONFLICT_SOLVER_JACCARD_THRESHOLD:
                    conflict_graph.add_edge(i, j)

        return conflict_graph

    def _calculate_association_bonuses(self, candidates: List[CandidateSkeleton], pos_lookup) -> np.ndarray:

        if not self.tracklets or not candidates:
            return np.zeros(len(candidates))

        bonuses = np.zeros(len(candidates))

        for j, cand_skel in enumerate(candidates):
            skel_kps = {node[0]: pos_lookup(node[1]) for node in cand_skel.nodes}

            if not skel_kps:
                continue

            max_bonus = 0.0

            for tracklet in self.tracklets:
                pred_pose = tracklet.predicted_pose

                if not pred_pose:
                    continue

                common_kps = pred_pose.keys() & skel_kps.keys()
                if len(common_kps) < self.config.ASSOCIATION_MIN_KPS:
                    continue

                mean_dist_sq = sum(np.sum((pred_pose[kp] - skel_kps[kp]) ** 2) for kp in common_kps) / len(common_kps)
                bonus = np.exp(-0.5 * mean_dist_sq / (self.config.ASSOCIATION_RADIUS ** 2))

                if bonus > max_bonus:
                    max_bonus = bonus

            bonuses[j] = max_bonus

        return bonuses

    def _build_final_assignment_cost_matrix(self,
            tracklets: List[Tracklet],
            skeletons: List[AssembledSkeleton]
        ) -> np.ndarray:

        cost_matrix = np.full((len(tracklets), len(skeletons)), 1e9)

        for i, tracklet in enumerate(tracklets):
            pred_pose = tracklet.predicted_pose

            if not pred_pose:
                continue

            for j, skel in enumerate(skeletons):
                common_kps = pred_pose.keys() & skel.keypoints.keys()

                if len(common_kps) < self.config.ASSOCIATION_MIN_KPS:
                    continue

                mean_dist_sq = sum(np.sum((pred_pose[kp] - skel.keypoints[kp]) ** 2) for kp in common_kps) / len(
                    common_kps)

                if mean_dist_sq > self.config.ASSOCIATION_RADIUS ** 2:
                    continue

                cost = (self.config.COST_POSE_DISTANCE_WEIGHT * mean_dist_sq +
                        self.config.COST_SKELETON_SCORE_WEIGHT * skel.score)
                cost_matrix[i, j] = cost

        return cost_matrix


if __name__ == '__main__':
    import pickle
    import json
    from pathlib import Path
    import numpy as np

    from mokap.reconstruction.anatomy import StatsBootstrapper, AnatomyLearner
    from mokap.reconstruction.datatypes import SoupData

    # Configuration
    folder = Path().home() / 'Desktop' / '3d_ant_data'
    prefix = '240905-1616'
    session = 22

    anatomy_cfg = AnatomyConfig()
    assembler_cfg = AssemblerConfig()
    tracker_cfg = TrackerConfig()

    stats_output_file = folder / prefix / 'outputs' / f'bone_stats_session{session}.json'
    points_soup_file = folder / prefix / 'outputs' / f'points_soup_session{session}.pkl'
    skeleton_input_file = folder / prefix / 'inputs' / 'tracking'
    tracklets_output_file = folder / prefix / 'outputs' / f'tracklets_session{session}.pkl'

    # Load data
    print(f"Loading soup from {points_soup_file}...")
    with open(points_soup_file, 'rb') as f:
        soup: SoupData = pickle.load(f)

    if soup.num_points == 0:
        print("Soup is empty (no 3D points). Exiting.")
        exit()

    # Load metadata
    keypoints, bones, symmetry = fileio.load_skeleton_SLEAP(skeleton_input_file, symmetry=True)

    # Bootstrap anatomy
    bootstrapper = StatsBootstrapper(
        output_path=stats_output_file,
        bones_list=bones,
        symmetry_map=symmetry,
        prior_stats_path=None,
        bootstrap_data=soup,
        config=anatomy_cfg
    )
    bone_stats = bootstrapper.get_initial_stats()

    # Initialise pipeline
    anatomy_learner = AnatomyLearner(initial_stats=bone_stats, config=anatomy_cfg)

    assembler = SkeletonAssembler(
        bones_list=bones,
        bone_stats=bone_stats,
        assembler_config=assembler_cfg,
        tracker_config=tracker_cfg
    )

    tracker = MultiObjectTracker(assembler=assembler, config=tracker_cfg)

    # Run Tracking

    unique_frames = np.unique(soup.frame_indices)
    min_frame, max_frame = int(unique_frames[0]), int(unique_frames[-1])

    tracklets_by_id = defaultdict(list)

    print(f"Tracking from frame {min_frame} to {max_frame}...")

    with alive_bar(total=(max_frame - min_frame + 1), force_tty=True) as bar:
        for frame_idx in range(min_frame, max_frame + 1):

            # Update anatomical model with learning
            current_stats = anatomy_learner.get_stats()
            assembler.update_bone_stats(current_stats)

            frame_soup = soup.get_frame_slice(frame_idx)

            # Check if we have any data (Points or Rays) to process
            if frame_soup.num_points > 0 or len(frame_soup.ray_origins) > 0:
                active_tracklets = tracker.update(frame_soup, frame_idx)
            else:
                # Coasting / Prediction only
                active_tracklets = tracker.predict_only(frame_idx)

            # Store results
            for tracklet in active_tracklets:
                # Feed higher quality skeletons to the learner
                if tracklet.last_update_frame == frame_idx:
                    anatomy_learner.add_sample(tracklet.skeleton)

                # Serialize to dictionary for Stage 3 (Linking)
                skel_dict = tracklet.skeleton.to_dict()
                skel_dict.update({
                    'track_idx': tracklet.track_idx,
                    'track_health': tracklet.health,
                    'track_anatomical_integrity': tracklet.anatomical_integrity,
                    'track_uncertainty_pos': tracklet.uncertainty['position'].tolist(),
                    'track_velocity': tracklet.kf.x[3:6].flatten().tolist(),
                    'track_predicted_pos': tracklet.predicted_position.tolist(),
                    'time_since_update': tracklet.time_since_update,
                    'frame_idx': frame_idx
                })
                tracklets_by_id[tracklet.track_idx].append(skel_dict)

            bar()

    print("Tracking complete.")
    print(f"Generated {len(tracklets_by_id)} unique tracklets.")

    # Save Tracklets
    tracklets_output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tracklets_output_file, 'wb') as f:
        pickle.dump(dict(tracklets_by_id), f)
    print(f"Tracklet results saved to '{tracklets_output_file}'")

    # Save updated anatomy stats
    with open(stats_output_file, 'w') as f:
        json.dump(anatomy_learner.get_stats(), f, indent=2)
    print(f"Refined bone stats saved to '{stats_output_file}'")

