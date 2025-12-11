import logging
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
from itertools import combinations
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from alive_progress import alive_bar
from lucida.geometry.backend import xp
from lucida.geometry import align_rigid
from mokap.reconstruction.config import MergerConfig, LinkerConfig
from mokap.reconstruction.datatypes import TrackletData

logger = logging.getLogger(__name__)


class FragmentMerger:
    """
    Merges tracklet fragments that occur in the *same timeframe*
    (uses anatomical plausibility and proximity as evidence)
    """

    def __init__(self, tracklets: Dict[int, 'TrackletData'], bone_stats: Dict, bones_list: List, config: MergerConfig):
        
        self.config = config
        self.tracklets = tracklets
        self.bone_stats = bone_stats
        self.bones_set = {frozenset(b) for b in bones_list}

    def merge_fragments(self) -> Dict[int, 'TrackletData']:
        """
        Builds a merge graph, finds components, validates and splits them based
        on internal conflicts, and then merges the final valid groups.
        """

        tracklet_ids = list(self.tracklets.keys())
        n = len(tracklet_ids)
        adj, merge_evidence = defaultdict(list), {}

        for i in range(n):
            for j in range(i + 1, n):

                track_a = self.tracklets[tracklet_ids[i]]
                track_b = self.tracklets[tracklet_ids[j]]

                if not (track_a.start <= track_b.end and track_b.start <= track_a.end):
                    continue

                if self.config.DEBUG:
                    print(f"\n[ Checking pair ({track_a.idx}, {track_b.idx}) ]")

                if not self._check_motion_consistency(track_a, track_b):
                    if self.config.DEBUG:
                        print(f"  -> Merge REJECTED by motion inconsistency.")
                    continue

                mean_fit, p90_fit = self._calculate_anatomical_fit_score(track_a, track_b)

                if (mean_fit > self.config.ANATOMY_MEAN_THRESHOLD and
                        p90_fit > self.config.ANATOMY_P90_THRESHOLD):

                    adj[track_a.idx].append(track_b.idx)
                    adj[track_b.idx].append(track_a.idx)

                    merge_evidence[tuple(
                        sorted((track_a.idx, track_b.idx)))] = f"Anatomy (Mean: {mean_fit:.1f}, P90: {p90_fit:.1f})"

                    if self.config.DEBUG:
                        print(f"  -> Anatomical match FOUND.")

                    continue

                if self.config.DEBUG:
                    print(f"  -> Anatomical match FAILED (Mean: {mean_fit:.1f}, P90: {p90_fit:.1f}).")

                is_proximal, min_dist = self._check_proximity_and_complementarity(track_a, track_b)

                if is_proximal:
                    adj[track_a.idx].append(track_b.idx)
                    adj[track_b.idx].append(track_a.idx)
                    merge_evidence[
                        tuple(sorted((track_a.idx, track_b.idx)))] = f"Proximity (Avg Min Dist: {min_dist:.2f}mm)"

                    if self.config.DEBUG:
                        print(f"  -> Proximity match FOUND.")

        # Find initial connected components
        visited, initial_components = set(), []

        for track_idx in tracklet_ids:

            if track_idx not in visited:
                component, q = [], [track_idx]
                visited.add(track_idx)
                head = 0

                while head < len(q):
                    curr = q[head]
                    head += 1
                    component.append(curr)

                    for neighbor in adj.get(curr, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append(neighbor)

                initial_components.append(component)

        # Validate and split components, then merge
        new_tracklets = {}
        processed_ids = set()
        print(f"Found {len(initial_components)} components to consolidate. Validating and splitting...")

        for component in initial_components:
            # Get the final, valid, potentially smaller subcomponents
            final_sub_components = self._validate_and_split_component(component, adj)

            for sub_comp in final_sub_components:
                if len(sub_comp) > 1:
                    # This is a valid group to be merged
                    tracklets_to_merge = [self.tracklets[tid] for tid in sub_comp]
                    merged_tracklet = self._perform_multi_merge(tracklets_to_merge)
                    new_tracklets[merged_tracklet.idx] = merged_tracklet

                    # Mark all original IDs as processed
                    for tid in sub_comp:
                        processed_ids.add(tid)

                elif len(sub_comp) == 1:
                    # This is a singleton, just copy it over if not already part of a merge
                    tid = sub_comp[0]
                    if tid not in processed_ids:
                        new_tracklets[tid] = self.tracklets[tid]
                        processed_ids.add(tid)

        # Ensure any tracklets not in any component are carried over
        for tid in tracklet_ids:
            if tid not in processed_ids:
                new_tracklets[tid] = self.tracklets[tid]

        print(f"Fragment merging complete. {len(self.tracklets)} tracklets -> {len(new_tracklets)} tracklets.")

        return new_tracklets

    def _validate_and_split_component(self, component: List[int], merge_adj: Dict[int, List[int]]) -> List[
        List[int]]:
        """
        Validates a merge component for internal conflicts and splits it using an
        agglomerative clustering approach based on compatibility

        Args:
            component: A list of tracklet IDs forming a connected component
            merge_adj: The original merge adjacency list

        Returns:
            A list of valid subcomponents that are safe to merge
        """

        if len(component) <= 1:
            return [component]

        # 1. Pre-compute all pairs that are in "strong conflict". This is our ground truth.
        conflicts = {
            frozenset((id_a, id_b))
            for id_a, id_b in combinations(component, 2)
            if self._check_strong_conflict(self.tracklets[id_a], self.tracklets[id_b])
        }

        if not conflicts:
            # if there are no conflicts, the entire component is valid
            print(f"  - Component {component} is valid, no internal conflicts.")
            return [component]
        else:
            print(f"  - Component {component} has conflicts: {list(conflicts)}. Running agglomerative splitting.")

        # Start with each tracklet as its own group
        groups = [[tid] for tid in component]

        def _check_compatibility(group1, group2):
            """Check if merging two groups would introduce a conflict."""
            for id1 in group1:
                for id2 in group2:
                    if frozenset((id1, id2)) in conflicts:
                        return False
            return True

        def _check_connectivity(group1, group2):
            """Check if there's a merge link between any members of the two groups."""
            for id1 in group1:
                for id2 in group2:
                    if id2 in merge_adj.get(id1, []):
                        return True
            return False

        # iteratively merge compatible and connected groups
        while True:
            merged_in_this_pass = False
            i = 0
            while i < len(groups):
                j = i + 1
                while j < len(groups):
                    group_i = groups[i]
                    group_j = groups[j]

                    # Check for both compatibility and a direct merge link
                    if _check_compatibility(group_i, group_j) and _check_connectivity(group_i, group_j):
                        # Merge group_j into group_i
                        groups[i].extend(group_j)
                        groups.pop(j)
                        merged_in_this_pass = True
                        # restart the pass
                        break
                    else:
                        j += 1
                if merged_in_this_pass:
                    break
                else:
                    i += 1

            # if a full pass completes with no merges, all done
            if not merged_in_this_pass:
                break

        print(f"  - Split component {component} into {len(groups)} conflict-free groups: {groups}")
        return groups

    def _check_strong_conflict(self, track_a: 'TrackletData', track_b: 'TrackletData') -> bool:
        """
        Determines if two tracklets are in 'strong conflict'
        This prevents merging of two distinct, otherwise well-tracked animals

        A 'strong conflict' is defined as two high-quality tracklets that consistently
        share the same keypoint identities over their overlapping timeframe
        """

        # Check 1: Do the tracklets overlap in time?
        frames_a, frames_b = set(track_a.frames), set(track_b.frames)
        overlap = sorted(list(frames_a.intersection(frames_b)))
        if not overlap:
            return False  # No temporal overlap means no conflict.

        # Check 2: Are both tracklets of high quality?
        # We only want to prevent merges between two 'good' tracks
        # (a conflict with a low-quality fragment is less meaningful and might be noise)
        avg_integrity_a = np.mean(track_a.integrities)
        avg_integrity_b = np.mean(track_b.integrities)
        min_integrity = self.config.VALIDATION_MIN_INTEGRITY

        if avg_integrity_a < min_integrity or avg_integrity_b < min_integrity:
            return False

        # Check 3: Do they consistently share keypoints?
        frame_to_skel_a = {s['frame_idx']: s for s in track_a.skeletons}
        frame_to_skel_b = {s['frame_idx']: s for s in track_b.skeletons}

        shared_kp_counts = []

        for frame in overlap:
            kps_a = frame_to_skel_a[frame]['keypoints']
            kps_b = frame_to_skel_b[frame]['keypoints']

            if not kps_a or not kps_b:
                continue

            num_shared_kps = len(set(kps_a.keys()).intersection(set(kps_b.keys())))
            shared_kp_counts.append(num_shared_kps)

        if not shared_kp_counts:
            return False

        # median for robustness against a few outlier frames
        median_shared_kps = np.median(shared_kp_counts)
        kp_thresh = self.config.VALIDATION_SHARED_KP_THRESH

        if median_shared_kps >= kp_thresh:
            if self.config.DEBUG:
                print(f"      -> VALIDATION: Strong conflict detected between "
                      f"({track_a.idx}, {track_b.idx}). "
                      f"Median shared KPs: {median_shared_kps:.1f} >= {kp_thresh}. "
                      f"Integrities: ({avg_integrity_a:.1f}, {avg_integrity_b:.1f})")
            return True

        return False

    def _check_motion_consistency(self, track_a: 'TrackletData', track_b: 'TrackletData') -> bool:
        """
        Checks if two fragments have consistent velocities over their overlapping frames
        (a large difference in velocity is strong evidence they are separate objects)
        """

        # If a tracklet is very short, its velocity estimate is pure noise
        # so we skip the motion check and rely only on anatomy/proximity
        if len(track_a.frames) < 4 or len(track_b.frames) < 4:
            return True

        frames_a = {f: i for i, f in enumerate(track_a.frames)}
        frames_b = {f: i for i, f in enumerate(track_b.frames)}

        common_frames = sorted(list(frames_a.keys() & frames_b.keys()))

        if len(common_frames) < 3:  # not enough data for a reliable check
            return True

        velocity_diffs = []
        for frame in common_frames:
            idx_a = frames_a[frame]
            idx_b = frames_b[frame]
            vel_a = track_a.velocities[idx_a]
            vel_b = track_b.velocities[idx_b]

            # magnitude of the difference vector
            diff = np.linalg.norm(vel_a - vel_b)
            velocity_diffs.append(diff)

        if not velocity_diffs:
            return True

        # median for robustness against outlier frames
        median_diff = np.median(velocity_diffs)

        is_consistent = median_diff < self.config.MOTION_VELOCITY_THRESH_MM_S

        if self.config.DEBUG:
            status = "CONSISTENT" if is_consistent else "INCONSISTENT"
            print(f"    - Motion check: Median velocity diff = {median_diff:.2f} mm/s. ({status})")

        return is_consistent

    def _check_proximity_and_complementarity(self, track_a: 'TrackletData', track_b: 'TrackletData') -> Tuple[bool, float]:
        """
        Checks if two fragments are consistently close to each other *and* complementary
        (they don't represent the same body parts)
        """

        # At least one of the tracklets must have high average anatomical integrity

        # This prevents two low-quality nonsense tracklets from merging, and prevents a high-quality
        # tracklet from being merged with a low-quality one based on proximity alone

        avg_integrity_a = np.mean(track_a.integrities)
        avg_integrity_b = np.mean(track_b.integrities)

        min_integrity_threshold = self.config.PROXIMITY_MIN_INTEGRITY

        if avg_integrity_a < min_integrity_threshold and avg_integrity_b < min_integrity_threshold:
            if self.config.DEBUG: print(
                f"    - Proximity check failed: Both fragments have low integrity ({avg_integrity_a:.1f}, {avg_integrity_b:.1f}). Thresh: >{min_integrity_threshold}")
            return False, -1.0

        frames_a, frames_b = set(track_a.frames), set(track_b.frames)
        overlap = sorted(list(frames_a.intersection(frames_b)))
        if not overlap:
            return False, -1.0

        frame_to_skel_a = {s['frame_idx']: s for s in track_a.skeletons}
        frame_to_skel_b = {s['frame_idx']: s for s in track_b.skeletons}

        min_distances = []
        conflicting_frames = 0
        for frame in overlap:
            kps_a = frame_to_skel_a[frame]['keypoints']
            kps_b = frame_to_skel_b[frame]['keypoints']

            if not kps_a or not kps_b:
                continue

            # Check for keypoint overlap (conflict) in this frame
            num_shared_kps = len(set(kps_a.keys()).intersection(set(kps_b.keys())))
            if num_shared_kps > self.config.PROXIMITY_CONFLICT_THRESHOLD:
                conflicting_frames += 1
                continue  # skip distance calculation for conflicting frames

            dist_matrix = cdist(np.array(list(kps_a.values())), np.array(list(kps_b.values())))
            if dist_matrix.size > 0:
                min_distances.append(np.min(dist_matrix))

        # if too many overlapping frames are in conflict, they are likely duplicates, not merge candidates
        if len(overlap) > 0 and (conflicting_frames / len(overlap)) > self.config.CONFLICTING_FRAME_RATIO:
            if self.config.DEBUG:
                print(
                f"    - Proximity check failed: Too many conflicting frames ({conflicting_frames}/{len(overlap)}).")
            return False, -1.0

        if not min_distances:
            if self.config.DEBUG:
                print(f"    - Proximity check failed: No valid non-conflicting frames found.")
            return False, -1.0

        avg_min_dist = np.mean(min_distances)
        if self.config.DEBUG:
            print(
            f"    - Proximity check: Avg min_dist = {avg_min_dist:.2f}mm. (Thresh: <{self.config.PROXIMITY_DIST_THRESH_MM}mm)"
            )

        is_proximal = avg_min_dist < self.config.PROXIMITY_DIST_THRESH_MM
        
        return is_proximal, avg_min_dist

    def _calculate_anatomical_fit_score(self, track_a: 'TrackletData', track_b: 'TrackletData') -> Tuple[float, float]:
        """Calculates how well two fragments fit together anatomically by scoring all possible inter-fragment bones."""

        frames_a, frames_b = set(track_a.frames), set(track_b.frames)

        overlap = sorted(list(frames_a.intersection(frames_b)))
        if not overlap:
            return -1.0, -1.0

        frame_to_skel_a = {s['frame_idx']: s for s in track_a.skeletons}
        frame_to_skel_b = {s['frame_idx']: s for s in track_b.skeletons}

        all_bone_scores = []
        for frame in overlap:
            skel_a, skel_b = frame_to_skel_a[frame], frame_to_skel_b[frame]
            kps_a, kps_b = skel_a['keypoints'], skel_b['keypoints']

            # Skip frames with too much keypoint overlap
            if len(set(kps_a.keys()).intersection(set(kps_b.keys()))) > self.config.ANATOMY_CONFLICT_THRESHOLD:
                continue

            # Skip frames with inconsistent scale
            if abs(skel_a.get('scale', 1.0) - skel_b.get('scale', 1.0)) > self.config.BONE_SCALE_TOLERANCE:
                continue

            # Velocity consistency
            v_a = np.array(skel_a.get('track_velocity', [0, 0, 0]))
            v_b = np.array(skel_b.get('track_velocity', [0, 0, 0]))
            norm_a = np.linalg.norm(v_a)
            norm_b = np.linalg.norm(v_b)

            # Only check if both tracklets are actually moving
            velocity_sim_thresh = self.config.VELOCITY_COSINE_SIMILARITY_THRESHOLD

            if norm_a > 1e-3 and norm_b > 1e-3:
                cosine_similarity = np.dot(v_a, v_b) / (norm_a * norm_b)
                if cosine_similarity < velocity_sim_thresh:
                    if self.config.DEBUG:
                        print(
                        f"    - Velocity check failed for frame {frame}: similarity {cosine_similarity:.2f} < {velocity_sim_thresh}"
                        )
                    continue  # velocities are too different, skip this frame's evidence

            # Score all bones that can be formed between the two fragments
            for kp1, p1 in kps_a.items():
                for kp2, p2 in kps_b.items():
                    bone = frozenset((kp1, kp2))
                    if bone in self.bones_set:
                        # Use an average scale from both fragments for the expectation
                        scale = (skel_a.get('scale', 1.0) + skel_b.get('scale', 1.0)) / 2.0
                        all_bone_scores.append(self._score_single_bone(bone, p1, p2, scale))

        if not all_bone_scores:
            return -1.0, -1.0

        return np.mean(all_bone_scores), np.percentile(all_bone_scores, 90)

    def _score_single_bone(self, bone: frozenset, p1: np.ndarray, p2: np.ndarray, scale: float) -> float:
        """Scores a single bone based on its length conformity, returning a value from 0 to 100."""

        bone_str = ';'.join(sorted(tuple(bone)))
        if bone_str not in self.bone_stats['bones_ratios']:
            return 0.0

        stats = self.bone_stats['bones_ratios'][bone_str]
        ref_len = self.bone_stats['median_reference_length']

        expected_len = ref_len * stats['median_ratio'] * scale

        # scaled Median Absolute Deviation (MAD) for robust error measurement
        expected_mad = ref_len * stats['mad_ratio'] * scale + 1e-6
        dist = np.linalg.norm(np.array(p1) - np.array(p2))

        mad_away = abs(dist - expected_len) / expected_mad

        if mad_away >= self.config.ANATOMY_BONE_MAD_THRESH:
            return 0.0

        # Linearly scale score from 100 (perfect fit) to 0 (at the MAD threshold)
        return 100 * (1.0 - (mad_away / self.config.ANATOMY_BONE_MAD_THRESH))

    def _perform_multi_merge(self, tracklets_to_merge: List['TrackletData']) -> 'TrackletData':
        """Combines multiple tracklets from a connected component into a single new tracklet."""

        # The new tracklet inherits the ID of the longest original tracklet
        tracklets_to_merge.sort(key=lambda t: len(t.frames), reverse=True)
        new_id = tracklets_to_merge[0].idx

        frame_to_skeletons = defaultdict(list)
        all_frame_indices = set()
        for t in tracklets_to_merge:
            for s in t.skeletons:
                frame_idx = s['frame_idx']
                frame_to_skeletons[frame_idx].append(s)
                all_frame_indices.add(frame_idx)

        final_skeletons = []
        for frame in sorted(list(all_frame_indices)):
            skeletons_in_frame = frame_to_skeletons.get(frame, [])
            if not skeletons_in_frame: continue

            # Combine all keypoints from the fragments in this frame
            merged_kps = {}
            total_kp_count, weighted_scale_sum = 0, 0

            # Prepare to collect and average all relevant properties
            uncertainties, predicted_pos_list, velocities_in_frame = [], [], []

            for skel in skeletons_in_frame:
                merged_kps.update(skel['keypoints'])
                kp_count = len(skel['keypoints'])
                total_kp_count += kp_count
                weighted_scale_sum += skel.get('scale', 1.0) * kp_count

                # Collect vector properties to be averaged
                uncertainties.append(np.array(skel.get('track_uncertainty_pos', [0, 0, 0])))
                predicted_pos_list.append(np.array(skel.get('track_predicted_pos', [0, 0, 0])))
                velocities_in_frame.append(np.array(skel.get('track_velocity', [0, 0, 0])))

            if total_kp_count == 0: continue

            final_scale = weighted_scale_sum / total_kp_count

            # Calculate the average of the vector properties for the new merged skeleton
            final_uncertainty = np.mean(uncertainties, axis=0)
            final_predicted_pos = np.mean(predicted_pos_list, axis=0)
            final_velocity = np.mean(velocities_in_frame, axis=0)

            # Create the new merged skeleton, inheriting metadata from the first fragment
            new_skel = skeletons_in_frame[0].copy()
            new_skel['keypoints'], new_skel['scale'] = merged_kps, final_scale

            # Add the newly calculated properties to the skeleton dictionary
            new_skel['track_uncertainty_pos'] = final_uncertainty.tolist()
            new_skel['track_predicted_pos'] = final_predicted_pos.tolist()
            new_skel['track_velocity'] = final_velocity.tolist()

            # TODO: 'health' and 'integrity' are left as is from the base skeleton for simplicity, which is not ideal

            final_skeletons.append(new_skel)

        if not final_skeletons:
            # Fallback in case merge fails, return the original longest tracklet
            return tracklets_to_merge[0]

        new_frames = np.array([s['frame_idx'] for s in final_skeletons])

        # extract the newly created velocities from the final skeletons list
        new_velocities = np.array([s['track_velocity'] for s in final_skeletons])

        return TrackletData(
            idx=new_id,
            frames=new_frames,
            skeletons=final_skeletons,
            healths=np.array([s['track_health'] for s in final_skeletons]),
            integrities=np.array([s['track_anatomical_integrity'] for s in final_skeletons]),
            uncertainties=np.array([s['track_uncertainty_pos'] for s in final_skeletons]),
            velocities=new_velocities,
            end_state_prediction=np.array(final_skeletons[-1]['track_predicted_pos'])
        )

class TrackletLinker:
    """Links fragmented tracklets that are separated *in time* using a globally optimal assignment."""

    def __init__(self, tracklets: Dict[int, TrackletData], canonical_map: Dict, config: LinkerConfig):
        self.config = config
        self.tracklets = tracklets
        self.canonical_map = canonical_map
        logger.debug(f"Initialized TrackletLinker with {len(self.tracklets)} tracklets.")

    def link_tracklets(self) -> Dict[int, List[int]]:
        tracklet_list = list(self.tracklets.values())
        n = len(tracklet_list)
        
        if n < 2:
            return {t.idx: [t.idx] for t in tracklet_list}

        cost_matrix = np.full((n, n), np.inf)
        
        with alive_bar(n * n, length=20, title="Cost Matrix", force_tty=True) as bar:

            for i in range(n):
                for j in range(n):
                    # Skip diagonal (self-linking)
                    if i == j:
                        bar()
                        continue

                    track_a = tracklet_list[i]
                    track_b = tracklet_list[j]

                    # Check temporal ordering and gap
                    frame_gap = track_b.start - track_a.end

                    if 1 <= frame_gap <= self.config.MAX_FRAME_GAP:
                        cost = self._calculate_link_cost(track_a, track_b, frame_gap)

                        if cost < self.config.COST_THRESHOLD:
                            cost_matrix[i, j] = cost
                    bar()

        penalized_cost_matrix = self._penalize_ambiguous_links(cost_matrix)

        # Having rows/cols of all infs is normal (start/end of tracks)
        # but linear_sum_assignment crashes on infs so we use a huge cost
        # and we filter out the garbage later using COST_THRESHOLD
        safe_max_cost = 1e6     # value larger than any possible valid cost + penalty
        solver_matrix = np.where(np.isinf(penalized_cost_matrix), safe_max_cost, penalized_cost_matrix)

        row_ind, col_ind = linear_sum_assignment(solver_matrix)

        final_links = {}
        for r, c in zip(row_ind, col_ind):
            # Check against the original cost matrix (or penalized one)
            # to ensure we don't accept the dummy safe_max_cost links
            real_cost = penalized_cost_matrix[r, c]

            if real_cost < self.config.COST_THRESHOLD:
                id_a, id_b = tracklet_list[r].idx, tracklet_list[c].idx
                final_links[id_a] = id_b
                logger.debug(f"  - Link: {id_a} -> {id_b} (Cost: {real_cost:.2f})")

        return self._rebuild_chains(final_links)

    def _penalize_ambiguous_links(self, cost_matrix: np.ndarray) -> np.ndarray:
        """
        Identifies and penalizes ambiguous links in the cost matrix.

        An ambiguous link occurs when a tracklet has more than one high-quality
        candidate to link to (either forward or backward in time)
        """

        penalized_cost_matrix = cost_matrix.copy()
        n, m = cost_matrix.shape
        ambiguity_threshold_ratio = self.config.AMBIGUITY_THRESHOLD_RATIO
        ambiguity_penalty = self.config.AMBIGUITY_PENALTY
        cost_threshold = self.config.COST_THRESHOLD

        # Check for forward ambiguity (one ender -> multiple starters)
        for i in range(n):
            row_costs = cost_matrix[i, :]
            valid_indices = np.where(row_costs < cost_threshold)[0]

            if len(valid_indices) > 1:
                valid_costs = row_costs[valid_indices]

                top_2 = np.partition(valid_costs, 1)[:2]
                best_cost = min(top_2)
                second_best_cost = max(top_2)

                if (second_best_cost - best_cost) / (best_cost + 1e-6) < ambiguity_threshold_ratio:
                    penalized_cost_matrix[i, valid_indices] += ambiguity_penalty

        # Check for backward ambiguity (multiple enders -> one starter)
        for j in range(m):
            col_costs = cost_matrix[:, j]
            valid_indices = np.where(col_costs < cost_threshold)[0]

            if len(valid_indices) > 1:
                valid_costs = col_costs[valid_indices]

                top_2 = np.partition(valid_costs, 1)[:2]
                best_cost = min(top_2)
                second_best_cost = max(top_2)

                if (second_best_cost - best_cost) / (best_cost + 1e-6) < ambiguity_threshold_ratio:
                    penalized_cost_matrix[valid_indices, j] += ambiguity_penalty

        return penalized_cost_matrix

    def _calculate_link_cost(self, track_a: TrackletData, track_b: TrackletData, frame_gap: int) -> float:
        """Calculates a cost for linking tracklet A to B."""
        # TODO: This should be revamped

        # Get template poses and check for overlap
        kps_a = track_a.template_pose(end=True)
        kps_b = track_b.template_pose(end=False)
        common_kps = list(kps_a.keys() & kps_b.keys())

        if len(common_kps) < self.config.MIN_COMMON_KPS:
            return 1e9  # not enough shared geometry for a stable cost

        # Shape cost
        pts_a_common = xp.array([kps_a[name] for name in common_kps])
        pts_b_common = xp.array([kps_b[name] for name in common_kps])

        R, t = align_rigid(pts_a_common, pts_b_common)
        aligned_pts_a = (R @ pts_a_common.T).T + t
        shape_dist_sq = np.mean(np.sum((aligned_pts_a - pts_b_common) ** 2, axis=1))
        shape_cost = np.sqrt(shape_dist_sq)

        # Shape-informed motion cost (Mahalanobis distance)
        pred_pos_a = track_a.end_state_prediction
        common_centroid_a = np.mean(pts_a_common, axis=0)
        common_centroid_b = np.mean(pts_b_common, axis=0)
        vec_common_to_center_a = pred_pos_a - common_centroid_a
        inferred_center_b = common_centroid_b + vec_common_to_center_a
        delta = inferred_center_b - pred_pos_a

        # Handle cases where uncertainties might be 0
        cov_a = np.diag(track_a.uncertainties[-1] + 1e-6)

        central_kp_name = min(kps_a.keys(), key=lambda k: np.linalg.norm(kps_a[k] - pred_pos_a))
        p_noise_var = self.config.SMOOTHER_KF_PROCESS_NOISE.get(
            self.canonical_map.get(central_kp_name, 'default'),
            self.config.SMOOTHER_KF_PROCESS_NOISE['default']
        )
        propagated_cov = cov_a + np.eye(3) * p_noise_var * frame_gap

        if np.linalg.det(propagated_cov) < 1e-9:
            inv_cov = np.eye(3)
        else:
            inv_cov = np.linalg.inv(propagated_cov)

        mahalanobis_dist_sq = delta.T @ inv_cov @ delta
        motion_cost = np.sqrt(mahalanobis_dist_sq)

        # Velocity continuity cost
        # if tracklets are very short, velocity is noisy so the weight of this cost is reduced
        # TODO: more magic numbers, I know
        VERY_SHORT_TRACKLET = 3
        VERY_SHORT_TRACKLET_UNRELIABILITY = 0.1
        
        if len(track_a.frames) < VERY_SHORT_TRACKLET or len(track_b.frames) < VERY_SHORT_TRACKLET:
            vel_reliability = VERY_SHORT_TRACKLET_UNRELIABILITY
        else:
            vel_reliability = 1.0

        vel_a_end = track_a.template_velocity(end=True)
        vel_b_start = track_b.template_velocity(end=False)
        velocity_cost = np.linalg.norm(vel_a_end - vel_b_start)

        # Quality Score
        quality_a = np.mean(track_a.healths[-3:]) * np.mean(track_a.integrities[-3:])
        quality_b = np.mean(track_b.healths[:3]) * np.mean(track_b.integrities[:3])
        quality_score = np.sqrt(quality_a * quality_b)

        # Stability bias
        # TODO: too many magic numbers, this should be reworked
        QUITE_SHORT_TRACKLET = 5   # haha this is terrible
        PENALTY_SHORT_SHORT = 10.0
        PENALTY_LONG_SHORT = 12.0
        
        is_a_short = len(track_a.frames) < QUITE_SHORT_TRACKLET
        is_b_short = len(track_b.frames) < QUITE_SHORT_TRACKLET
        if is_a_short and is_b_short:
            # Two quite short tracklets connecting: high risk, but allowed if otherwise a good link
            risk_penalty = PENALTY_SHORT_SHORT
        elif is_a_short or is_b_short:
            # One long one short: Dangerous case
            risk_penalty = PENALTY_LONG_SHORT
        else:
            risk_penalty = 0.0

        # Weighted Cost
        total_cost = (self.config.COST_W_MOTION * motion_cost +
                      self.config.COST_W_SHAPE * shape_cost +
                      self.config.COST_W_VELOCITY * velocity_cost * vel_reliability +
                      self.config.COST_W_QUALITY * quality_score +
                      risk_penalty)

        return total_cost


    def _rebuild_chains(self, links: Dict[int, int]) -> Dict[int, List[int]]:
        """
        Follows the links from the assignment to construct the full tracklet chains.
        """
        all_nodes = set(self.tracklets.keys())
        linked_destinations = set(links.values())

        # Start nodes are those that are not destinations of any link
        start_nodes = all_nodes - linked_destinations

        chains = {}
        new_track_idx_counter = 0

        for start_node in start_nodes:
            chain = [start_node]
            curr = start_node

            while curr in links:
                curr = links[curr]
                chain.append(curr)

            chains[new_track_idx_counter] = chain
            new_track_idx_counter += 1

        return chains


def load_tracklets(track_data: Dict[int, List[Dict]], config: LinkerConfig) -> Dict[int, TrackletData]:
    """Parses the raw tracked data from a file into Tracklet objects."""

    tracklets = {}
    for track_idx, skeletons_list in track_data.items():

        if not skeletons_list or len(skeletons_list) < config.MIN_TRACKLET_LEN:
            continue

        skeletons_list.sort(key=lambda x: x['frame_idx'])

        frames = np.array([s['frame_idx'] for s in skeletons_list])

        tracklets[track_idx] = TrackletData(
            idx=track_idx,
            frames=frames,
            skeletons=skeletons_list,
            healths=np.array([s['track_health'] for s in skeletons_list]),
            integrities=np.array([s['track_anatomical_integrity'] for s in skeletons_list]),
            velocities=np.array([s.get('track_velocity', [0, 0, 0]) for s in skeletons_list]),
            end_state_prediction=np.array(skeletons_list[-1]['track_predicted_pos']),
            uncertainties=np.array([s.get('track_uncertainty_pos', [0, 0, 0]) for s in skeletons_list])
        )

    return tracklets


def combine_chains(
        chains: Dict[int, List[int]],
        tracklet_pool: Dict[int, 'TrackletData']
    ) -> Dict[int, List[Dict]]:
    """Combines the skeletons from linked and merged tracklets into final, full tracks."""

    final_tracks = {}
    print("\nCombining linked chains into final tracks...")

    for final_track_idx, original_tracklet_ids in chains.items():
        all_skeletons_for_track = []

        for tid in original_tracklet_ids:
            if tid in tracklet_pool:
                skeletons_to_add = [s.copy() for s in tracklet_pool[tid].skeletons]
                for skel in skeletons_to_add:
                    skel['track_idx'] = final_track_idx
                all_skeletons_for_track.extend(skeletons_to_add)

            else:
                print(f"[Warning] Tracklet ID {tid} from a chain was not found in the tracklet pool. Skipping.")

        if not all_skeletons_for_track:
            continue

        all_skeletons_for_track.sort(key=lambda s: s['frame_idx'])
        final_tracks[final_track_idx] = all_skeletons_for_track

    print(f"Created {len(final_tracks)} final tracks.")
    return final_tracks


if __name__ == '__main__':
    import pickle
    import json
    import numpy as np
    from pathlib import Path

    from mokap.utils import fileio
    from mokap.reconstruction.utils import create_canonical_map

    # Configuration
    folder = Path().home() / 'Desktop' / '3d_ant_data'
    prefix = '240905-1616'
    session = 22

    linker_cfg = LinkerConfig()
    merger_cfg = MergerConfig()

    input_dir = folder / prefix / 'inputs' / 'tracking'
    output_dir = folder / prefix / 'outputs'

    # Inputs
    tracklets_file = output_dir / f'tracklets_session{session}.pkl'
    stats_file = output_dir / f'bone_stats_session{session}.json'

    # Output
    final_output_file = output_dir / f'linked_tracks_session{session}.pkl'

    # Load data
    print(f"Loading tracklets from {tracklets_file}...")
    if not tracklets_file.exists():
        print("Tracklets file not found. Run tracking step first.")
        exit()

    with open(tracklets_file, 'rb') as f:
        all_tracked_data = pickle.load(f)

    if not all_tracked_data:
        print("Tracklet dictionary is empty. Exiting.")
        exit()

    print(f"Loaded {len(all_tracked_data)} raw tracklet fragments.")

    # Load metadata
    keypoints, bones_list, symmetry = fileio.load_skeleton_SLEAP(input_dir, symmetry=True)
    canonical_map = create_canonical_map(keypoints, symmetry)

    with open(stats_file, 'r') as f:
        bone_stats = json.load(f)

    # Merge overlapping fragments (stage 3a)
    print("\n[ Stage 3a: fragment merging ]")

    # Convert raw dicts to TrackletData objects
    initial_tracklets = load_tracklets(all_tracked_data, config=linker_cfg)
    print(f"  -> Filtered to {len(initial_tracklets)} valid tracklets (min length > {linker_cfg.MIN_TRACKLET_LEN}).")

    merger = FragmentMerger(initial_tracklets, bone_stats, bones_list, config=merger_cfg)
    merged_tracklets = merger.merge_fragments()

    reduction = 100 * (1 - len(merged_tracklets) / len(initial_tracklets))
    print(
        f"  -> Merging complete: {len(initial_tracklets)} -> {len(merged_tracklets)} tracklets ({reduction:.1f}% reduction).")

    # Link temporal gaps (stage 3b)
    print("\n[ Stage 3b: Temporal linking ]")
    linker = TrackletLinker(merged_tracklets, canonical_map, config=linker_cfg)
    chains = linker.link_tracklets()

    # Combine and summarize (stage 3c)
    print("\n[  Stage 3c: Final assembly ]")
    final_linked_tracks = combine_chains(chains, merged_tracklets)

    # Some statistics
    num_tracks = len(final_linked_tracks)
    if num_tracks > 0:
        lengths = [len(t) for t in final_linked_tracks.values()]
        total_skeletons = sum(lengths)
        max_len = max(lengths)
        median_len = int(np.median(lengths))

        print(f"\n[Summary]")
        print(f"  Total unique tracks: {num_tracks}")
        print(f"  Total skeletons:     {total_skeletons}")
        print(f"  Max track length:    {max_len} frames")
        print(f"  Median track length: {median_len} frames")

        # Print top 5 longest tracks
        sorted_ids = sorted(final_linked_tracks.keys(), key=lambda k: len(final_linked_tracks[k]), reverse=True)
        print(f"  Top 5 longest tracks: {[len(final_linked_tracks[i]) for i in sorted_ids[:5]]}")
    else:
        print("\n[Summary] No tracks generated.")

    # Save
    final_output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(final_output_file, 'wb') as f:
        pickle.dump(final_linked_tracks, f)

    print(f"\nSaved final tracks to '{final_output_file}'")
