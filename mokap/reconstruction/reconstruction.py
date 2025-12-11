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
from lucida.geometry import (unproject, triangulate_from_projections, project_to_cameras, undistort, project,
                             invert_transform, intersect_aabb)

from mokap.reconstruction.config import ReconstructorConfig
from mokap.reconstruction.datatypes import SoupData
from mokap.reconstruction.utils import solve_mwis_networkx, prepare_reconstruction_input

logger = logging.getLogger(__name__)

# JAX padding settings
USE_PADDING = True
MAX_DETS_PER_CAM = 32    # Max detections per view to consider for grouping
PAD_BLOCK_SIZE = 64    # Pad hypothesis count to multiples of this

# TODO: Time these two versions better


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

        # Setup bounding box for intersection checks
        self.aabb_min = xp.array([self.volume_bounds[axis][0] for axis in ['x', 'y', 'z']])
        self.aabb_max = xp.array([self.volume_bounds[axis][1] for axis in ['x', 'y', 'z']])

        # Pre-allocate empty arrays
        self._init_emptys()

    def _init_emptys(self):
        """Initialise reusable empty arrays/tuples to reduce garbage collection overhead."""

        self.EMPTY_F32_XP = xp.array([], dtype=xp.float32)

        self.NULL_POINT2D_XP = xp.empty((0, 2), dtype=xp.float32)
        self.NULL_POINT3D_XP = xp.empty((0, 3), dtype=xp.float32)

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
        # Score = (Views * W_views) + (Sum_Conf * W_conf) + (Error * W_error)
        return (self.num_cams * self.config.view_count_weight) + \
            (self.num_cams * 1.0 * self.config.detection_confidence_weight)

    def reconstruct_batch(self, inputs: Dict[str, np.ndarray], keypoint_names: List[str]) -> SoupData:
        """
        Reconstructs 3D points from flattened inputs.
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
            # Searchsorted to slice this frame (input data must be sorted ofc!)
            start = np.searchsorted(inputs['frame_indices'], frame_idx, side='left')
            end = np.searchsorted(inputs['frame_indices'], frame_idx, side='right')

            # Views into the large arrays for this frame
            f_kp_ids = inputs['kp_type_ids'][start:end]
            f_cam_ids = inputs['cam_ids'][start:end]
            f_coords = inputs['coords'][start:end]
            f_scores = inputs['scores'][start:end]
            f_global_indices = np.arange(start, end)

            present_kps = np.unique(f_kp_ids)

            for kp_id in present_kps:
                kp_mask = (f_kp_ids == kp_id)
                if np.sum(kp_mask) < self.config.min_views:
                    continue

                # Data for this specific keypoint
                curr_cam_ids_np = f_cam_ids[kp_mask]
                curr_coords_np = f_coords[kp_mask]
                curr_scores_np = f_scores[kp_mask]
                curr_indices_np = f_global_indices[kp_mask]

                # Convert to JAX once
                curr_coords_xp = xp.array(curr_coords_np)
                curr_scores_xp = xp.array(curr_scores_np)

                # Core reconstruction
                final_pts, final_confs, final_errors, used_indices_list, cam_masks = self._reconstruct_keypoint(
                    curr_coords_np, curr_cam_ids_np, curr_indices_np,
                    curr_coords_xp, curr_scores_xp
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

        # Handle Orphan Rays (Unused detections)
        has_valid_coords = ~np.isnan(inputs['coords'][:, 0])
        orphan_mask = (~is_used) & has_valid_coords

        if np.any(orphan_mask):
            # Batch compute rays
            ray_origins, ray_dirs = self._compute_rays(
                inputs['cam_ids'][orphan_mask],
                inputs['coords'][orphan_mask]
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
    def _compute_rays(self, cam_ids, coords):
        """
        Computes 3D rays.
        """
        # Access parameters via the Rig
        Ks_batch = self.rig.K[cam_ids]
        Ts_c2w_batch = self.rig.T_c2w[cam_ids]
        Ds_batch = self.rig.D[cam_ids]

        pts_uv = xp.array(coords)

        # Unproject points at depth=1.0 to get world coordinates
        world_pts = unproject(
            points2d=pts_uv,
            depth=xp.ones(len(cam_ids)),
            K=Ks_batch,
            T_c2w=Ts_c2w_batch,
            D=Ds_batch,
            distortion_model=self.rig.distortion_model
        )

        # Compute directions
        origins = Ts_c2w_batch[:, :3, 3]  # Camera centers
        ray_vecs = world_pts - origins
        ray_dirs = ray_vecs / (xp.linalg.norm(ray_vecs, axis=1, keepdims=True) + 1e-8)

        return origins, ray_dirs

    def _reconstruct_keypoint(self,
                              coords_np: np.ndarray,
                              cam_ids_np: np.ndarray,
                              indices_np: np.ndarray,
                              coords_xp: xp.ndarray,
                              scores_xp: xp.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[int]], np.ndarray]:
        """
        Runs reconstruction pipeline for a keypoint.
        Accepts both numpy (for Graph logic) and JAX (for geometry logic).
        """

        all_pts_xp, all_groups, view_counts, summed_confs, all_errors = self._generate_hypotheses(
            coords_np, cam_ids_np, coords_xp, scores_xp
        )

        if all_pts_xp.shape[0] == 0:
            return self.EMPTY_RESULT

        # Filter hypotheses and merge redundancies
        final_pts, final_confs, final_errors, final_indices, final_masks = self._filter_and_merge(
            all_pts_xp, view_counts, summed_confs, all_errors, all_groups,
            cam_ids_np, indices_np
        )

        return final_pts, final_confs, final_errors, final_indices, final_masks

    def _generate_hypotheses(self,
                             coords_np: np.ndarray,
                             cam_ids_np: np.ndarray,
                             coords_xp: xp.ndarray,
                             scores_xp: xp.ndarray
                             ) -> Tuple[xp.ndarray, List[List[int]], xp.ndarray, xp.ndarray, xp.ndarray]:
        """
        Generates all plausible 3D points hypotheses from 2D detections.
        """

        if USE_PADDING:
            groups = self._group_points_pad(coords_np, cam_ids_np, coords_xp)
        else:
            groups = self._group_points(coords_np, cam_ids_np, coords_xp)

        M = len(groups)
        if M == 0:
            return self.NULL_POINT3D_XP, [], self.EMPTY_F32_XP, self.EMPTY_F32_XP, self.EMPTY_F32_XP

        # Flatten groups for vectorized triangulation
        group_lengths = [len(g) for g in groups]
        idx_group = np.repeat(np.arange(M), group_lengths)
        idx_val = np.fromiter(itertools.chain.from_iterable(groups), dtype=np.int32)
        idx_cam = cam_ids_np[idx_val]

        M_padded = ((M + PAD_BLOCK_SIZE - 1) // PAD_BLOCK_SIZE) * PAD_BLOCK_SIZE

        matched_uvs = xp.full((len(self.rig), M_padded, 2), xp.nan, dtype=xp.float32)
        matched_uvs = set_at(matched_uvs, (idx_cam, idx_group), coords_xp[idx_val])

        tri_weights = xp.zeros((len(self.rig), M_padded), dtype=xp.float32)
        tri_weights = set_at(tri_weights, (idx_cam, idx_group), scores_xp[idx_val])

        points3d, view_counts, summed_confs, reproj_errors, valid_mask = self._triangulate_and_check(
            matched_uvs, tri_weights
        )

        # Slice back to original M size and filter valid
        points3d = points3d[:M]
        view_counts = view_counts[:M]
        summed_confs = summed_confs[:M]
        reproj_errors = reproj_errors[:M]
        valid_mask = valid_mask[:M]

        valid_indices_xp = xp.where(valid_mask)[0]
        valid_indices_np = np.array(valid_indices_xp)

        if valid_indices_np.size == 0:
            return self.NULL_POINT3D_XP, [], self.EMPTY_F32_XP, self.EMPTY_F32_XP, self.EMPTY_F32_XP

        valid_groups = [groups[i] for i in valid_indices_np]

        return (points3d[valid_indices_xp], valid_groups, view_counts[valid_indices_xp],
                summed_confs[valid_indices_xp], reproj_errors[valid_indices_xp])

    @partial(jit, static_argnums=(0,))
    def _triangulate_and_check(self, matched_uvs, tri_weights):

        undistorted_uvs = undistort(
            matched_uvs,
            self.rig.K,
            self.rig.D,
            distortion_model=self.rig.distortion_model
        )

        points3d = triangulate_from_projections(
            undistorted_uvs,
            self.rig.P,
            weights=tri_weights
        )

        valid_triangulation = ~xp.any(xp.isnan(points3d), axis=1)

        all_reprojected, proj_validity = project_to_cameras(
            points3d,
            self.rig.T_w2c,
            self.rig.K,
            self.rig.D,
            distortion_model=self.rig.distortion_model
        )

        # Errors
        orig_vis_mask = ~xp.isnan(matched_uvs[:, :, 0])
        combined_mask = orig_vis_mask.astype(xp.float32) * proj_validity

        diffs = all_reprojected - matched_uvs
        valid_diffs = xp.where(combined_mask[..., None], diffs, 0.0)
        distances = xp.linalg.norm(valid_diffs, axis=-1)

        num_views = xp.sum(combined_mask, axis=0)
        reproj_errors = xp.sum(distances, axis=0) / xp.maximum(num_views, 1)

        # Aggregates
        view_counts = xp.sum(orig_vis_mask, axis=0)
        summed_confs = xp.sum(xp.where(orig_vis_mask, tri_weights, 0), axis=0)

        final_mask = valid_triangulation & (reproj_errors < self.config.repro_thresh) & (num_views > 0)

        return points3d, view_counts, summed_confs, reproj_errors, final_mask

    def _group_points(self, coords_np, cam_ids_np, coords_xp):
        """CPU-friendly points grouping (no padding)."""

        total_dets = len(coords_np)
        if total_dets < self.config.min_views:
            return []

        cam_indices_map = [np.where(cam_ids_np == i)[0] for i in range(len(self.rig))]
        source_indices, target_indices = [], []

        for i in range(len(self.rig)):
            idxs_i = cam_indices_map[i]
            n_i = len(idxs_i)

            if n_i == 0:
                continue

            d_i = coords_xp[idxs_i]

            for j in range(i + 1, len(self.rig)):
                idxs_j = cam_indices_map[j]
                n_j = len(idxs_j)

                if n_j == 0:
                    continue
                d_j = coords_xp[idxs_j]

                cost_mat = self._compute_cost_matrix(d_i, d_j, i, j)

                if cost_mat.size == 0:
                    continue

                match_rows, match_cols = np.where(np.array(cost_mat) < self.config.T_epi)
                source_indices.extend(idxs_i[match_rows])
                target_indices.extend(idxs_j[match_cols])

        return self._resolve_graph(source_indices, target_indices, total_dets, cam_ids_np)

    def _group_points_pad(self, coords_np, cam_ids_np, coords_xp):
        """GPU-friendly points grouping (with padding)."""

        total_dets = len(coords_np)
        if total_dets < self.config.min_views:
            return []

        cam_indices_map = [np.where(cam_ids_np == i)[0] for i in range(len(self.rig))]
        source_indices, target_indices = [], []

        for i in range(len(self.rig)):
            idxs_i = cam_indices_map[i]

            if len(idxs_i) == 0:
                continue

            n_i = min(len(idxs_i), MAX_DETS_PER_CAM)
            idxs_i = idxs_i[:n_i]  # slice to max bucket
            pad_i = MAX_DETS_PER_CAM - n_i

            d_i = coords_xp[idxs_i]
            if pad_i > 0:
                d_i = xp.pad(d_i, ((0, pad_i), (0, 0)), constant_values=xp.nan)

            for j in range(i + 1, len(self.rig)):

                idxs_j = cam_indices_map[j]
                if len(idxs_j) == 0:
                    continue

                n_j = min(len(idxs_j), MAX_DETS_PER_CAM)
                idxs_j = idxs_j[:n_j]
                pad_j = MAX_DETS_PER_CAM - n_j

                d_j = coords_xp[idxs_j]
                if pad_j > 0: d_j = xp.pad(d_j, ((0, pad_j), (0, 0)), constant_values=xp.nan)

                cost_mat_padded = self._compute_cost_matrix_pad(d_i, d_j, i, j)

                # Unpad and check
                cost_mat = np.asarray(cost_mat_padded)[:n_i, :n_j]

                if cost_mat.size == 0:
                    continue

                match_rows, match_cols = np.where(cost_mat < self.config.T_epi)
                source_indices.extend(idxs_i[match_rows])
                target_indices.extend(idxs_j[match_cols])

        return self._resolve_graph(source_indices, target_indices, total_dets, cam_ids_np)

    def _resolve_graph(self, source_indices, target_indices, total_dets, cam_ids_np):
        """Common graph logic for both grouping methods."""
        if not source_indices:
            return []

        adj_matrix = csr_matrix((np.ones(len(source_indices)), (source_indices, target_indices)),
                                shape=(total_dets, total_dets))
        n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)

        all_final_groups = []
        processed_groups = set()

        for i in range(n_components):
            component_indices = np.where(labels == i)[0]
            if len(component_indices) < self.config.min_views: continue

            subgraph_adj = adj_matrix[component_indices, :][:, component_indices]
            component_graph = nx.from_scipy_sparse_array(subgraph_adj)
            mapping = {local: global_idx for local, global_idx in enumerate(component_indices)}
            nx.relabel_nodes(component_graph, mapping, copy=False)

            # Find cliques (ambiguity handling)
            cliques = find_cliques(component_graph)

            for clique_indices in cliques:
                if len(clique_indices) < self.config.min_views: continue
                if len(np.unique(cam_ids_np[clique_indices])) < self.config.min_views: continue

                clique_cams = cam_ids_np[clique_indices]
                if len(np.unique(clique_cams)) < self.config.min_views: continue

                # Conflict graph
                conflict_graph = nx.Graph()
                conflict_graph.add_nodes_from(clique_indices)

                # Connect same-camera detections (mutually exclusive)
                # (this nested loop should be fine for small cliques < 10 nodes)
                for idx_a in range(len(clique_indices)):
                    node_a = clique_indices[idx_a]
                    cam_a = cam_ids_np[node_a]
                    for idx_b in range(idx_a + 1, len(clique_indices)):
                        node_b = clique_indices[idx_b]
                        if cam_a == cam_ids_np[node_b]:
                            conflict_graph.add_edge(node_a, node_b)

                # Solve independent sets on the clique
                complement_g = nx.complement(conflict_graph)
                for group in find_cliques(complement_g):
                    if len(group) >= self.config.min_views:
                        sorted_group = sorted(group)
                        frozen_group = frozenset(sorted_group)
                        if frozen_group not in processed_groups:
                            all_final_groups.append(sorted_group)
                            processed_groups.add(frozen_group)

        return all_final_groups

    @partial(jit, static_argnums=(0, 3, 4))
    def _compute_cost_matrix(self, dets_i, dets_j, i, j):
        """Dynamic shape cost matrix using Rig parameters."""
        # Extract specific camera parameters from Rig
        K_i, D_i, T_i = self.rig.K[i], self.rig.D[i], self.rig.T_w2c[i]
        K_j, D_j, T_j = self.rig.K[j], self.rig.D[j], self.rig.T_w2c[j]

        # Use helper
        return self._epipolar_cost(dets_i, dets_j, K_i, D_i, T_i, K_j, D_j, T_j)

    @partial(jit, static_argnums=(0, 3, 4))
    def _compute_cost_matrix_pad(self, dets_i, dets_j, i, j):
        """Padded shape cost matrix using Rig parameters."""
        K_i, D_i, T_i = self.rig.K[i], self.rig.D[i], self.rig.T_w2c[i]
        K_j, D_j, T_j = self.rig.K[j], self.rig.D[j], self.rig.T_w2c[j]

        cost = self._epipolar_cost(dets_i, dets_j, K_i, D_i, T_i, K_j, D_j, T_j)
        return xp.nan_to_num(cost, nan=1e6)

    def _epipolar_cost(self, dets_i, dets_j, K_i, D_i, T_i, K_j, D_j, T_j):
        """Geometry logic shared between padded and unpadded calls."""
        Ni, Nj = dets_i.shape[0], dets_j.shape[0]

        udets_j = undistort(dets_j, K_j, D_j, distortion_model=self.rig.distortion_model)
        E_c2w_i = invert_transform(T_i)
        cam_center_i = E_c2w_i[:3, 3]

        # Back project
        p_3d_on_ray = unproject(
            dets_i,
            xp.ones(Ni),
            xp.stack([K_i] * Ni),
            xp.stack([E_c2w_i] * Ni),
            xp.stack([D_i] * Ni),
            distortion_model=self.rig.distortion_model
        )

        ray_dirs = p_3d_on_ray - cam_center_i
        ray_dirs /= (xp.linalg.norm(ray_dirs, axis=-1, keepdims=True) + 1e-8)

        p_near_3d, p_far_3d, has_intersection = intersect_aabb(
            cam_center_i, ray_dirs, self.aabb_min, self.aabb_max
        )
        segments_3d = xp.vstack([p_near_3d, p_far_3d])
        segments_3d = xp.nan_to_num(segments_3d)

        # Project segments to camera J
        segments_2d, _ = project(
            segments_3d, T_j, K_j, xp.zeros_like(D_j), 'none'
        )

        a_pts = segments_2d[:Ni]
        b_pts = segments_2d[Ni:]

        p = udets_j[None, :, :]  # (1, Nj, 2)
        a = a_pts[:, None, :]    # (Ni, 1, 2)
        b = b_pts[:, None, :]
        ab = b - a
        ap = p - a

        denom = xp.sum(ab * ab, axis=-1)
        numer = xp.sum(ap * ab, axis=-1)
        t = numer / (denom + 1e-6)
        t_clamped = xp.clip(t, 0.0, 1.0)
        closest_points = a + t_clamped[..., None] * ab
        dists = xp.linalg.norm(p - closest_points, axis=-1)

        # Filter rays that didn't intersect AABB or were padded nans
        final_costs = xp.where(has_intersection[:, None], dists, 1e6)

        # Also ensure nan inputs (padding) result in high cost
        final_costs = xp.nan_to_num(final_costs, nan=1e6)

        return final_costs

    def _filter_and_merge(self, points3d, view_counts, summed_confs, errors, groups, cam_ids_np, indices_np):
        points3d_np = np.asarray(points3d)
        view_counts_np = np.asarray(view_counts)
        summed_confs_np = np.asarray(summed_confs)
        errors_np = np.asarray(errors)

        num_points = points3d_np.shape[0]
        if num_points == 0:
            return self.EMPTY_RESULT

        float_scores = (
                (view_counts_np * self.config.view_count_weight) +
                (summed_confs_np * self.config.detection_confidence_weight) +
                (errors_np * self.config.repro_error_weight)
        )

        # Build conflict graph and solve MWIS to get the best non-conflicting set
        conflict_graph = self._build_conflict_graph(num_points, groups, float_scores)
        winner_indices = np.array(solve_mwis_networkx(conflict_graph))

        if winner_indices.size == 0:
            return self.EMPTY_RESULT

        # Cluster the winning points by proximity to find candidates for merging
        winner_points_3d = points3d_np[winner_indices]
        winner_scores = float_scores[winner_indices]
        winner_groups = [groups[i] for i in winner_indices]

        clustering = DBSCAN(eps=self.config.cluster_radius, min_samples=1).fit(winner_points_3d)
        labels = clustering.labels_

        final_points = []
        final_scores = []
        final_errors = []
        final_indices_list = []
        final_cam_masks = []

        def process_group(group_idxs):
            """Helper to get global indices and mask from a local group list"""
            g_idxs = []
            cam_bitmask = 0
            for idx in group_idxs:
                # idx is the local index in the SoA batch
                g_idxs.append(indices_np[idx])
                cam_bitmask |= (1 << cam_ids_np[idx])
            return g_idxs, cam_bitmask

        for label in np.unique(labels):
            local_indices = np.where(labels == label)[0]

            # Merging logic
            merged = False

            # Check Jaccard similarity for merge
            if len(local_indices) > 1 and self.config.filter_method == 'average':
                # Check if points in this cluster should be merged
                cluster_groups_sets = [set(winner_groups[i]) for i in local_indices]
                avg_jaccard = self._calculate_average_jaccard(cluster_groups_sets)

                if avg_jaccard > self.config.jaccard_threshold_for_merge:
                    # High Jaccard similarity -> they are duplicate hypotheses, merge them
                    cluster_pts = winner_points_3d[local_indices]
                    cluster_scores = winner_scores[local_indices]

                    weights = self._softmax_weights(cluster_scores, self.config.softmax_temperature)

                    averaged_point = np.sum(cluster_pts * weights[:, np.newaxis], axis=0)
                    averaged_score = np.sum(cluster_scores * weights)
                    averaged_error = np.sum(errors_np[winner_indices[local_indices]] * weights)

                    merged_global_indices = []
                    merged_mask = 0

                    # Merge indices and masks
                    for idx in local_indices:
                        g_idxs, mask = process_group(winner_groups[idx])
                        merged_global_indices.extend(g_idxs)
                        merged_mask |= mask

                    final_points.append(averaged_point)
                    final_scores.append(averaged_score)
                    final_errors.append(averaged_error)
                    final_indices_list.append(merged_global_indices)
                    final_cam_masks.append(merged_mask)
                    merged = True

            if not merged:
                for idx in local_indices:
                    final_points.append(winner_points_3d[idx])
                    final_scores.append(winner_scores[idx])
                    final_errors.append(errors_np[winner_indices[idx]])
                    g_idxs, mask = process_group(winner_groups[idx])
                    final_indices_list.append(g_idxs)
                    final_cam_masks.append(mask)

        return (np.array(final_points, dtype=np.float32),
                np.array(final_scores, dtype=np.float32),
                np.array(final_errors, dtype=np.float32),
                final_indices_list,
                np.array(final_cam_masks, dtype=np.uint32))

    def _build_conflict_graph(self, num_points, groups, scores):
        """Builds conflict graph where an edge represents a conflict between two hypotheses."""

        conflict_graph = nx.Graph()
        groups_as_sets = [set(g) for g in groups]
        min_score = np.min(scores) if scores.size > 0 else 0
        scores_non_negative = scores - min_score if min_score < 0 else scores
        integer_scores = (scores_non_negative * 1000).astype(int)

        for i in range(num_points):
            conflict_graph.add_node(i, weight=int(integer_scores[i]))

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
                intersection = len(sets[i].intersection(sets[j]))
                union = len(sets[i].union(sets[j]))
                jaccard_sum += intersection / union if union > 0 else 0
                pair_count += 1

        return jaccard_sum / pair_count if pair_count > 0 else 0

    @staticmethod
    def _proximity_merging(
            points: np.ndarray,
            scores: np.ndarray,
            indices: List[List[int]],
            masks: np.ndarray,
            radius: float
    ) -> Tuple[np.ndarray, np.ndarray, List[List[int]], np.ndarray]:
        """
        Merges (aggressively) geometrically close points, keeping the best one.
        """
        if points.shape[0] < 2:
            return points, scores, indices, masks

        clustering = DBSCAN(eps=radius, min_samples=1).fit(points)
        labels = clustering.labels_

        final_points = []
        final_scores = []
        final_indices = []
        final_masks = []

        for label in np.unique(labels):
            cluster_idxs = np.where(labels == label)[0]

            # Pick the best point in the cluster (hard max)
            best_local_idx = np.argmax(scores[cluster_idxs])
            best_global_idx = cluster_idxs[best_local_idx]

            final_points.append(points[best_global_idx])
            final_scores.append(scores[best_global_idx])
            final_indices.append(indices[best_global_idx])
            final_masks.append(masks[best_global_idx])

        return (np.asarray(final_points, dtype=np.float32),
                np.asarray(final_scores, dtype=np.float32),
                final_indices,
                np.asarray(final_masks, dtype=np.uint32))


if __name__ == '__main__':
    import pickle
    from pathlib import Path
    import polars as pl
    from mokap.utils import fileio

    # Config
    folder = Path().home() / 'Desktop' / '3d_ant_data'
    prefix = '240905-1616'
    session = 22
    BATCH_SIZE = 100

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

    # Initialise Reconstructor
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

    # Create batch ranges
    for i in range(0, total_frames, BATCH_SIZE):
        batch_frames = all_frame_indices[i: i + BATCH_SIZE]
        min_f, max_f = batch_frames[0], batch_frames[-1]

        df_batch = df.filter((pl.col("frame") >= min_f) & (pl.col("frame") <= max_f))

        if df_batch.is_empty():
            continue

        # Convert to SoA inputs and reconstruct batch
        inputs = prepare_reconstruction_input(df_batch, camera_names, keypoints)
        batch_soup = reconstructor.reconstruct_batch(inputs, keypoints)

        nb_new_points = batch_soup.num_points
        total_points_found += nb_new_points

        if nb_new_points > 0 or len(batch_soup.ray_origins) > 0:
            batch_results.append(batch_soup)

        frames_done = min(i + BATCH_SIZE, total_frames)
        curr_time = time.time() - start_time
        print(
            f"  Processed {frames_done}/{total_frames} frames in {curr_time:.2f} seconds... ({total_points_found} points)")

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