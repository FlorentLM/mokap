import logging
import time
from functools import partial
from typing import Dict, List, Set, Tuple
import itertools
import networkx as nx
from networkx.algorithms.clique import find_cliques
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN
import numpy as np

from lucida import CameraRig
from lucida.geometry.backend import xp, jit, set_at
from lucida.geometry import intersect_aabb

from mokap.reconstruction.config import ReconstructorConfig
from mokap.reconstruction.datatypes import SoupData
from mokap.reconstruction.utils import solve_mwis_networkx, prepare_reconstruction_input

logger = logging.getLogger(__name__)


class Reconstructor:
    """
    Performs robust 3D reconstruction of keypoints from multiple camera views.
    Uses a multi-stage evidence-based pipeline to handle ambiguities, occlusions and duplicate detections.

    1. All geometrically plausible 3D points are generated using a graph-based approach on epipolar constraints.
    2. Point candidates are filtered using a conflict graph and Maximum Weight Independent Set algorithm to select
     the most likely non-conflicting set.
    3. Then high-confidence candidates are merged.
    """

    def __init__(self,
                 rig: CameraRig,
                 volume_bounds: Dict,
                 config: ReconstructorConfig = ReconstructorConfig()
                 ):

        self.config = config
        self.rig = rig
        self.volume_bounds = volume_bounds
        self.n_cams = len(rig)

        # Setup bounding box for intersection checks
        self.aabb_min = xp.array([self.volume_bounds[axis][0] for axis in ['x', 'y', 'z']])
        self.aabb_max = xp.array([self.volume_bounds[axis][1] for axis in ['x', 'y', 'z']])

        self._init_empty_arrays()

    def _init_empty_arrays(self):
        """Initialise reusable empty arrays to reduce allocation overhead."""
        self.EMPTY_F32_NP = np.array([], dtype=np.float32)
        self.EMPTY_U32_NP = np.array([], dtype=np.uint32)
        self.EMPTY_I16_NP = np.array([], dtype=np.int16)
        self.EMPTY_I32_NP = np.array([], dtype=np.int32)
        self.NULL_POINT3D_NP = np.empty((0, 3), dtype=np.float32)

        # Standard empty return tuple for _reconstruct_keypoint
        self.EMPTY_RESULT = (
            self.NULL_POINT3D_NP,
            self.EMPTY_F32_NP,
            self.EMPTY_F32_NP,
            [],  # indices list
            self.EMPTY_U32_NP
        )

    @property
    def max_point_score(self) -> float:
        """
        Returns the theoretical maximum score a single 3D point can achieve.
        Assumes visibility in all cameras with perfect confidence and zero error.
        """
        return (self.n_cams * self.config.view_count_weight) + \
            (self.n_cams * 1.0 * self.config.detection_confidence_weight)

    def reconstruct_batch(self, inputs: Dict[str, np.ndarray], keypoint_names: List[str]) -> SoupData:
        """
        Reconstructs 3D points from flattened inputs.
        Also computes 'Orphan Rays' for single-view detections.
        """
        total_detections = len(inputs['kp_type_ids'])
        is_used = np.zeros(total_detections, dtype=bool)

        out_positions = []
        out_confs = []
        out_errs = []
        out_kp_types = []
        out_frame_indices = []
        out_cam_masks = []

        unique_frames = np.unique(inputs['frame_indices'])

        for frame_idx in unique_frames:
            start = np.searchsorted(inputs['frame_indices'], frame_idx, side='left')
            end = np.searchsorted(inputs['frame_indices'], frame_idx, side='right')

            f_kp_ids = inputs['kp_type_ids'][start:end]
            f_cam_ids = inputs['cam_ids'][start:end]
            f_coords = inputs['coords'][start:end]
            f_scores = inputs['scores'][start:end]
            f_global_indices = np.arange(start, end)

            for kp_id in np.unique(f_kp_ids):
                kp_mask = (f_kp_ids == kp_id)
                if np.sum(kp_mask) < self.config.min_views:
                    continue

                curr_cam_ids = f_cam_ids[kp_mask]
                curr_coords = f_coords[kp_mask]
                curr_scores = f_scores[kp_mask]
                curr_indices = f_global_indices[kp_mask]

                final_pts, final_confs, final_errors, used_indices_list, cam_masks = self._reconstruct_keypoint(
                    curr_coords, curr_cam_ids, curr_indices, curr_scores
                )

                if final_pts.shape[0] > 0:
                    n_pts = len(final_pts)
                    out_positions.append(final_pts)
                    out_confs.append(final_confs)
                    out_errs.append(final_errors)
                    out_kp_types.append(np.full(n_pts, kp_id, dtype=np.int16))
                    out_frame_indices.append(np.full(n_pts, frame_idx, dtype=np.int32))
                    out_cam_masks.append(cam_masks)

                    for idx_group in used_indices_list:
                        is_used[idx_group] = True

        # Build 3D point soup arrays
        if out_positions:
            soup_pos = np.vstack(out_positions)
            soup_conf = np.concatenate(out_confs)
            soup_errs = np.concatenate(out_errs)
            soup_kp = np.concatenate(out_kp_types)
            soup_frame = np.concatenate(out_frame_indices)
            soup_mask = np.concatenate(out_cam_masks)
        else:
            soup_pos = self.NULL_POINT3D_NP
            soup_conf = self.EMPTY_F32_NP
            soup_errs = self.EMPTY_F32_NP
            soup_kp = self.EMPTY_I16_NP
            soup_frame = self.EMPTY_I32_NP
            soup_mask = self.EMPTY_U32_NP

        # Handle Orphan Rays (unused detections)
        has_valid_coords = ~np.isnan(inputs['coords'][:, 0])
        orphan_mask = (~is_used) & has_valid_coords

        if np.any(orphan_mask):
            ray_origins, ray_dirs = self._compute_orphan_rays(
                xp.asarray(inputs['cam_ids'][orphan_mask]),
                xp.asarray(inputs['coords'][orphan_mask])
            )
            ray_origins = np.asarray(ray_origins)
            ray_dirs = np.asarray(ray_dirs)
            ray_confs = inputs['scores'][orphan_mask]
            ray_kp = inputs['kp_type_ids'][orphan_mask]
            ray_frame = inputs['frame_indices'][orphan_mask]
        else:
            ray_origins = self.NULL_POINT3D_NP
            ray_dirs = self.NULL_POINT3D_NP
            ray_confs = self.EMPTY_F32_NP
            ray_kp = self.EMPTY_I16_NP
            ray_frame = self.EMPTY_I32_NP

        return SoupData(
            positions=soup_pos.astype(np.float32),
            confidences=soup_conf.astype(np.float32),
            reprojection_errors=soup_errs.astype(np.float32),
            kp_types=soup_kp.astype(np.int16),
            frame_indices=soup_frame.astype(np.int32),
            camera_masks=soup_mask.astype(np.uint32),

            ray_origins=ray_origins.astype(np.float32),
            ray_directions=ray_dirs.astype(np.float32),
            ray_confidences=ray_confs.astype(np.float32),
            ray_kp_types=ray_kp.astype(np.int16),
            ray_frame_indices=ray_frame.astype(np.int32),

            keypoint_names=keypoint_names,
            camera_names=[c.name for c in self.rig]
        )

    @partial(jit, static_argnums=(0,))
    def _compute_orphan_rays(self, cam_ids: xp.ndarray, coords: xp.ndarray) -> Tuple[xp.ndarray, xp.ndarray]:
        """Computes 3D rays for orphan (single-view) detections."""
        N = coords.shape[0]

        # Build dense (C, N, 2) array and raycast through rig
        dense_uvs = xp.full((self.n_cams, N, 2), xp.nan, dtype=xp.float32)
        point_indices = xp.arange(N)
        dense_uvs = set_at(dense_uvs, (cam_ids, point_indices), coords)

        origins_all, dirs_all = self.rig.raycast(dense_uvs)  # (C, 3), (C, N, 3)

        # Extract per-point: origin and direction from the camera that saw it
        out_origins = origins_all[cam_ids]  # (N, 3)
        out_dirs = dirs_all[cam_ids, point_indices]  # (N, 3)

        return out_origins, out_dirs

    def _reconstruct_keypoint(self,
                              coords: np.ndarray,
                              cam_ids: np.ndarray,
                              indices: np.ndarray,
                              scores: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[int]], np.ndarray]:
        """Runs the full reconstruction pipeline for a single keypoint type."""

        # Generate hypothesis groups via epipolar matching
        groups = self._group_detections(coords, cam_ids)
        if not groups:
            return self.EMPTY_RESULT

        # Triangulate all hypotheses and compute reprojection errors
        points3d, view_counts, summed_confs, reproj_errors, valid_mask = self._triangulate_hypotheses(
            coords, cam_ids, scores, groups
        )

        # Filter to valid hypotheses
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            return self.EMPTY_RESULT

        points3d = points3d[valid_idx]
        view_counts = view_counts[valid_idx]
        summed_confs = summed_confs[valid_idx]
        reproj_errors = reproj_errors[valid_idx]
        groups = [groups[i] for i in valid_idx]

        # Filter conflicts and merge duplicates
        return self._filter_and_merge(
            points3d, view_counts, summed_confs, reproj_errors, groups, cam_ids, indices
        )

    def _group_detections(self, coords: np.ndarray, cam_ids: np.ndarray) -> List[List[int]]:
        """
        Groups 2D detections across cameras based on epipolar constraints.
        Returns list of groups, where each group is a list of detection indices.
        """
        n_dets = len(coords)
        if n_dets < self.config.min_views:
            return []

        # Index detections by camera
        cam_indices = [np.where(cam_ids == c)[0] for c in range(self.n_cams)]

        # Find epipolar matches between all camera pairs
        source_indices, target_indices = [], []

        for i in range(self.n_cams):
            if len(cam_indices[i]) == 0:
                continue

            coords_i = coords[cam_indices[i]]

            for j in range(i + 1, self.n_cams):
                if len(cam_indices[j]) == 0:
                    continue

                coords_j = coords[cam_indices[j]]
                cost_mat = self._epipolar_cost(coords_i, coords_j, i, j)

                match_rows, match_cols = np.where(cost_mat < self.config.T_epi)
                source_indices.extend(cam_indices[i][match_rows])
                target_indices.extend(cam_indices[j][match_cols])

        if not source_indices:
            return []

        return self._resolve_groups(source_indices, target_indices, n_dets, cam_ids)

    def _resolve_groups(self, source_indices, target_indices, n_dets, cam_ids) -> List[List[int]]:
        """Resolves epipolar matches into consistent detection groups via graph analysis."""

        adj_matrix = csr_matrix(
            (np.ones(len(source_indices)), (source_indices, target_indices)),
            shape=(n_dets, n_dets)
        )
        n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)

        all_groups = []
        seen_groups = set()

        for comp_id in range(n_components):
            component_indices = np.where(labels == comp_id)[0]
            if len(component_indices) < self.config.min_views:
                continue

            # Build subgraph and find cliques
            subgraph_adj = adj_matrix[component_indices, :][:, component_indices]
            component_graph = nx.from_scipy_sparse_array(subgraph_adj)
            mapping = {local: global_idx for local, global_idx in enumerate(component_indices)}
            nx.relabel_nodes(component_graph, mapping, copy=False)

            for clique in find_cliques(component_graph):
                if len(clique) < self.config.min_views:
                    continue

                clique_cams = cam_ids[clique]
                if len(np.unique(clique_cams)) < self.config.min_views:
                    continue

                # Build conflict graph (same-camera detections conflict)
                conflict_graph = nx.Graph()
                conflict_graph.add_nodes_from(clique)

                for idx_a, node_a in enumerate(clique):
                    for node_b in clique[idx_a + 1:]:
                        if cam_ids[node_a] == cam_ids[node_b]:
                            conflict_graph.add_edge(node_a, node_b)

                # Find maximum independent sets (non-conflicting groups)
                complement_g = nx.complement(conflict_graph)
                for group in find_cliques(complement_g):
                    if len(group) >= self.config.min_views:
                        sorted_group = tuple(sorted(group))
                        if sorted_group not in seen_groups:
                            all_groups.append(list(sorted_group))
                            seen_groups.add(sorted_group)

        return all_groups

    @partial(jit, static_argnums=(0, 3, 4))
    def _epipolar_distances(self, coords_i: xp.ndarray, coords_j: xp.ndarray, cam_i: int, cam_j: int) -> xp.ndarray:
        """
        Returns cost matrix (Ni, Nj) of distances from points in j to epipolar lines from i.
        """
        Ni = coords_i.shape[0]

        # Undistort target points (camera j) for comparison in linear space
        udets_j = self.rig.undistort(coords_j[None, :, :], cameras=[cam_j])[0]  # (Nj, 2)

        # Get rays from camera i
        origins_i, dirs_i = self.rig.raycast(coords_i[None, :, :], cameras=[cam_i])
        origin_i = origins_i[0]  # (3,) - single camera origin
        dirs_i = dirs_i[0]  # (Ni, 3)

        # Intersect rays with scene bounding box
        origins_broadcast = xp.broadcast_to(origin_i, (Ni, 3))
        p_near, p_far, has_intersection = intersect_aabb(
            origins_broadcast, dirs_i, self.aabb_min, self.aabb_max
        )

        # Stack near/far for batch projection
        segments_3d = xp.vstack([p_near, p_far])  # (2*Ni, 3)
        segments_3d = xp.nan_to_num(segments_3d)

        # Project to camera j and undistort for straight epipolar lines
        segments_2d = self.rig.project(segments_3d, cameras=[cam_j])[0]  # (2*Ni, 2)
        segments_2d_undist = self.rig.undistort(segments_2d[None, :, :], cameras=[cam_j])[0]

        a_pts = segments_2d_undist[:Ni]  # near points
        b_pts = segments_2d_undist[Ni:]  # far points

        # Compute point-to-segment distances
        p = udets_j[None, :, :]  # (1, Nj, 2)
        a = a_pts[:, None, :]  # (Ni, 1, 2)
        b = b_pts[:, None, :]

        ab = b - a
        ap = p - a
        t = xp.sum(ap * ab, axis=-1) / (xp.sum(ab * ab, axis=-1) + 1e-6)
        t_clamped = xp.clip(t, 0.0, 1.0)
        closest = a + t_clamped[..., None] * ab
        dists = xp.linalg.norm(p - closest, axis=-1)

        # Mask invalid rays
        has_intersection = xp.atleast_1d(has_intersection)
        costs = xp.where(has_intersection[:, None], dists, 1e6)
        costs = xp.nan_to_num(costs, nan=1e6)

        return costs

    def _epipolar_cost(self, coords_i: np.ndarray, coords_j: np.ndarray, cam_i: int, cam_j: int) -> np.ndarray:
        """
        Computes epipolar distance cost matrix between detections in two cameras.
        """
        costs = self._epipolar_distances(
            xp.asarray(coords_i),
            xp.asarray(coords_j),
            cam_i,
            cam_j
        )
        return np.asarray(costs)

    def _triangulate_hypotheses(self,
                                coords: np.ndarray,
                                cam_ids: np.ndarray,
                                scores: np.ndarray,
                                groups: List[List[int]]
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Triangulates all hypothesis groups and computes quality metrics."""

        M = len(groups)
        coords_xp = xp.asarray(coords)
        scores_xp = xp.asarray(scores)

        # Build dense (C, M, 2) observation array (Python loop, can't JIT)
        # TODO: Maybe improve this
        group_lengths = [len(g) for g in groups]
        idx_group = np.repeat(np.arange(M), group_lengths)
        idx_val = np.fromiter(itertools.chain.from_iterable(groups), dtype=np.int32)
        idx_cam = cam_ids[idx_val]

        matched_uvs = xp.full((self.n_cams, M, 2), xp.nan, dtype=xp.float32)
        matched_uvs = set_at(matched_uvs, (idx_cam, idx_group), coords_xp[idx_val])

        weights = xp.zeros((self.n_cams, M), dtype=xp.float32)
        weights = set_at(weights, (idx_cam, idx_group), scores_xp[idx_val])

        points3d, view_counts, summed_confs, reproj_errors, valid_mask = self._triangulate_and_validate(
            matched_uvs, weights
        )

        return (
            np.asarray(points3d),
            np.asarray(view_counts),
            np.asarray(summed_confs),
            np.asarray(reproj_errors),
            np.asarray(valid_mask)
        )

    @partial(jit, static_argnums=(0,))
    def _triangulate_and_validate(self, matched_uvs: xp.ndarray, weights: xp.ndarray):
        """JIT-compiled triangulation and validation core."""

        points3d = self.rig.triangulate(matched_uvs, weights=weights)  # (M, 3)
        valid_triangulation = ~xp.any(xp.isnan(points3d), axis=1)

        reprojected = self.rig.project(points3d)  # (C, M, 2)

        obs_valid = ~xp.isnan(matched_uvs[:, :, 0])  # (C, M)
        reproj_valid = ~xp.isnan(reprojected[:, :, 0])  # (C, M)
        both_valid = obs_valid & reproj_valid

        diffs = reprojected - matched_uvs
        diffs = xp.where(both_valid[..., None], diffs, 0.0)
        distances = xp.linalg.norm(diffs, axis=-1)  # (C, M)

        n_valid_views = xp.sum(both_valid, axis=0)  # (M,)
        reproj_errors = xp.sum(distances, axis=0) / xp.maximum(n_valid_views, 1)  # (M,)

        view_counts = xp.sum(obs_valid, axis=0)  # (M,)
        summed_confs = xp.sum(xp.where(obs_valid, weights, 0), axis=0)  # (M,)

        valid_mask = valid_triangulation & (reproj_errors < self.config.repro_thresh) & (n_valid_views > 0)

        return points3d, view_counts, summed_confs, reproj_errors, valid_mask

    def _filter_and_merge(self,
                          points3d: np.ndarray,
                          view_counts: np.ndarray,
                          summed_confs: np.ndarray,
                          reproj_errors: np.ndarray,
                          groups: List[List[int]],
                          cam_ids: np.ndarray,
                          indices: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[int]], np.ndarray]:
        """Filters conflicting hypotheses and merges near-duplicates."""

        num_points = len(points3d)
        if num_points == 0:
            return self.EMPTY_RESULT

        # Compute hypothesis scores
        scores = (
            view_counts * self.config.view_count_weight +
            summed_confs * self.config.detection_confidence_weight +
            reproj_errors * self.config.repro_error_weight
        )

        # Build conflict graph and solve MWIS
        conflict_graph = self._build_conflict_graph(num_points, groups, scores)
        winner_indices = np.array(solve_mwis_networkx(conflict_graph))

        if winner_indices.size == 0:
            return self.EMPTY_RESULT

        winner_points = points3d[winner_indices]
        winner_scores = scores[winner_indices]
        winner_groups = [groups[i] for i in winner_indices]

        # Cluster nearby winners for potential merging
        clustering = DBSCAN(eps=self.config.cluster_radius, min_samples=1).fit(winner_points)
        labels = clustering.labels_

        final_points = []
        final_scores = []
        final_errors = []
        final_indices_list = []
        final_cam_masks = []

        def process_group(group_idxs):
            """Convert local indices to global indices and compute camera bitmask."""
            g_idxs = [indices[idx] for idx in group_idxs]
            cam_bitmask = sum(1 << cam_ids[idx] for idx in group_idxs)
            return g_idxs, cam_bitmask

        for label in np.unique(labels):
            cluster_idx = np.where(labels == label)[0]
            merged = False

            # Try merging if multiple hypotheses in cluster
            if len(cluster_idx) > 1 and self.config.filter_method == 'average':
                cluster_groups = [set(winner_groups[i]) for i in cluster_idx]
                avg_jaccard = self._calculate_average_jaccard(cluster_groups)

                if avg_jaccard > self.config.jaccard_threshold_for_merge:
                    # High overlap -> merge via weighted average
                    cluster_pts = winner_points[cluster_idx]
                    cluster_scores = winner_scores[cluster_idx]
                    w = self._softmax_weights(cluster_scores, self.config.softmax_temperature)

                    merged_point = np.sum(cluster_pts * w[:, None], axis=0)
                    merged_score = np.sum(cluster_scores * w)
                    merged_error = np.sum(reproj_errors[winner_indices[cluster_idx]] * w)

                    merged_indices = []
                    merged_mask = 0
                    for idx in cluster_idx:
                        g_idxs, mask = process_group(winner_groups[idx])
                        merged_indices.extend(g_idxs)
                        merged_mask |= mask

                    final_points.append(merged_point)
                    final_scores.append(merged_score)
                    final_errors.append(merged_error)
                    final_indices_list.append(merged_indices)
                    final_cam_masks.append(merged_mask)
                    merged = True

            if not merged:
                for idx in cluster_idx:
                    final_points.append(winner_points[idx])
                    final_scores.append(winner_scores[idx])
                    final_errors.append(reproj_errors[winner_indices[idx]])
                    g_idxs, mask = process_group(winner_groups[idx])
                    final_indices_list.append(g_idxs)
                    final_cam_masks.append(mask)

        return (
            np.array(final_points, dtype=np.float32),
            np.array(final_scores, dtype=np.float32),
            np.array(final_errors, dtype=np.float32),
            final_indices_list,
            np.array(final_cam_masks, dtype=np.uint32)
        )

    def _build_conflict_graph(self, num_points: int, groups: List[List[int]], scores: np.ndarray) -> nx.Graph:
        """Builds conflict graph where edges represent mutually exclusive hypotheses."""
        conflict_graph = nx.Graph()
        groups_as_sets = [set(g) for g in groups]

        # Normalise scores to non-negative integers for MWIS solver
        min_score = np.min(scores) if scores.size > 0 else 0
        scores_shifted = scores - min_score if min_score < 0 else scores
        int_scores = (scores_shifted * 1000).astype(int)

        for i in range(num_points):
            conflict_graph.add_node(i, weight=int(int_scores[i]))

        for i in range(num_points):
            for j in range(i + 1, num_points):
                # A conflict exists if two hypotheses share a 2D detection
                if not groups_as_sets[i].isdisjoint(groups_as_sets[j]):
                    conflict_graph.add_edge(i, j)

        return conflict_graph

    @staticmethod
    def _softmax_weights(scores: np.ndarray, temperature: float) -> np.ndarray:

        if temperature <= 1e-6:
            weights = np.zeros_like(scores, dtype=float)
            weights[np.argmax(scores)] = 1.0
            return weights

        scores_temp = scores / temperature
        e_scores = np.exp(scores_temp - np.max(scores_temp))

        return e_scores / (e_scores.sum() + 1e-9)

    @staticmethod
    def _calculate_average_jaccard(sets: List[Set]) -> float:
        if len(sets) < 2:
            return 0.0

        jaccard_sum = 0.0
        pair_count = 0

        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                intersection = len(sets[i] & sets[j])
                union = len(sets[i] | sets[j])
                jaccard_sum += intersection / union if union > 0 else 0
                pair_count += 1

        return jaccard_sum / pair_count if pair_count > 0 else 0


if __name__ == '__main__':
    import pickle
    from pathlib import Path
    import polars as pl
    from mokap.utils import fileio

    # Config
    folder = Path().home() / 'Desktop' / '3d_ant_data'
    prefix = '240905-1616'
    session = 22
    BATCH_SIZE = 100  # nb of frames per batch

    input_dir = folder / prefix / 'inputs' / 'tracking'
    output_file = folder / prefix / 'outputs' / f'points_soup_session{session}.pkl'
    rig_file = folder / prefix / 'calibration' / 'camera_rig.toml'

    print("Loading metadata...")
    rig = CameraRig.load(rig_file)
    print(f"Loaded rig with {len(rig)} cameras.")

    keypoints, _ = fileio.load_skeleton_SLEAP(input_dir, indices=False)
    camera_names = [c.name for c in rig]

    volume_bounds = {'x': (-10.5, 13.0), 'y': (-21.0, 11.0), 'z': (180.0, 201.0)}

    print("Loading 2D detections...")
    df = fileio.load_session(input_dir, session=session, use_polars=True)

    reconstructor = Reconstructor(
        rig=rig,
        volume_bounds=volume_bounds,
        config=ReconstructorConfig(
            repro_thresh=10.0,
            cluster_radius=2.0,
            view_count_weight=10.0,
            repro_error_weight=1.0,
            min_views=2,
            enable_disjoint_merge=True
        )
    )

    # Batch processing loop
    all_frame_indices = df["frame"].unique().sort()
    total_frames = len(all_frame_indices)
    batch_results = []
    total_points_found = 0

    print(f"Starting reconstruction of {total_frames} frames...")
    start_time = time.time()

    for i in range(0, total_frames, BATCH_SIZE):
        batch_frames = all_frame_indices[i: i + BATCH_SIZE]
        min_f, max_f = batch_frames[0], batch_frames[-1]

        df_batch = df.filter((pl.col("frame") >= min_f) & (pl.col("frame") <= max_f))

        if df_batch.is_empty():
            continue

        inputs = prepare_reconstruction_input(df_batch, camera_names, keypoints)
        batch_soup = reconstructor.reconstruct_batch(inputs, keypoints)

        nb_new_points = batch_soup.num_points
        total_points_found += nb_new_points

        if nb_new_points > 0 or len(batch_soup.ray_origins) > 0:
            batch_results.append(batch_soup)

        frames_done = min(i + BATCH_SIZE, total_frames)
        curr_time = time.time() - start_time
        print(f"  Processed {frames_done}/{total_frames} frames in {curr_time:.2f}s ({total_points_found} points)")

    total_time = time.time() - start_time
    print(f"Reconstruction finished in {total_time:.2f} seconds.")
    print(f"Average FPS: {total_frames / total_time:.2f}")

    # Merge, save
    if batch_results:
        print("Concatenating batches...")
        full_soup = SoupData.concatenate(batch_results)

        print(f"Saving {full_soup.num_points} points and {len(full_soup.ray_origins)} orphan rays to {output_file}...")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'wb') as f:
            pickle.dump(full_soup, f)

        print("Done.")
    else:
        print("No points reconstructed.")