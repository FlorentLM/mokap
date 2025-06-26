import logging
import pickle
from pathlib import Path
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Optional, Dict, Tuple, List, Callable
import numpy as np
from itertools import product
import networkx as nx
from networkx.algorithms.clique import find_cliques
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from functools import lru_cache
from sklearn.cluster import DBSCAN
import pandas as pd
from alive_progress import alive_bar
from mokap.utils import fileio
from mokap.utils.geometry.fitting import bundle_intersection_AABB
from mokap.utils.geometry.projective import (
    undistort_points, back_projection, triangulate_points_from_projections, project_points, project_to_multiple_cameras
)
from mokap.utils.geometry.transforms import (
    extrinsics_matrix, projection_matrix, invert_rtvecs, extmat_to_rtvecs
)

logger = logging.getLogger(__name__)


class Reconstructor:
    """
    A class to perform robust 3D reconstruction of keypoints from multiple camera views

    It uses a multi-stage, evidence-based pipeline to handle ambiguities, occlusions,
    and duplicate detections common in multi-animal tracking scenarios

    The pipeline consists of:
    1. Hypothesis Generation: All geometrically plausible 3D points are generated
       using a graph-based approach on epipolar constraints
    2. Evidence-based Filtering: These candidates are filtered using a conflict graph
       and a Maximum Weight Independent Set (MWIS) algorithm to select the most
       likely, non-conflicting set of points. Redundant, high-confidence candidates
       for the same point are then merged.
    """

    def __init__(self,
            camera_parameters:  Dict,
            volume_bounds:      Dict,
            config:             Optional[Dict] = None
    ):
        """
        Initializes the Reconstructor

        Args:
            camera_parameters: A dictionary where keys are camera names and values are dicts
                               containing 'camera_matrix', 'dist_coeffs', 'rvec', 'tvec'
            volume_bounds: A dictionary defining the 3D volume of interest,
                           e.g., {'x': (min, max), 'y': (min, max), 'z': (min, max)}.
            config: An optional dictionary of tuning parameters
        """
        self.camera_names = sorted(camera_parameters.keys())
        self.num_cams = len(self.camera_names)
        self.volume_bounds = volume_bounds

        # Set configuration with default values
        default_config = {
            'T_epi': 15.0,
            'min_views': 2,
            'repro_thresh': 10.0,
            'filter_method': 'average',
            'cluster_radius': 5.0,
            'view_count_weight': 10.0,
            'repro_error_weight': 1.0,
            'softmax_temperature': 1.0,
            'jaccard_threshold_for_merge': 0.75,
            'enable_disjoint_merge': False,
            'disjoint_merge_radius': 2.0
        }
        # Note:
        # softmax temperature ->   0: The weighting becomes a winner-takes-all (hard-max)
        # softmax temperature -> inf: The weighting becomes a uniform average (all points contribute equally)
        
        if config:
            default_config.update(config)
        self.config = default_config

        # Pre-compute and cache all camera matrices and transforms
        self.all_K = jnp.stack([camera_parameters[name]['camera_matrix'] for name in self.camera_names])
        self.all_D = jnp.stack([camera_parameters[name]['dist_coeffs'] for name in self.camera_names])
        self.rvecs_c2w = jnp.stack([camera_parameters[name]['rvec'] for name in self.camera_names])
        self.tvecs_c2w = jnp.stack([camera_parameters[name]['tvec'] for name in self.camera_names])

        self.rvecs_w2c, self.tvecs_w2c = invert_rtvecs(self.rvecs_c2w, self.tvecs_c2w)
        self.all_E = extrinsics_matrix(self.rvecs_w2c, self.tvecs_w2c)
        self.all_P = projection_matrix(self.all_K, self.all_E)

        self.aabb_min = jnp.array([val[0] for val in self.volume_bounds.values()])
        self.aabb_max = jnp.array([val[1] for val in self.volume_bounds.values()])

    def reconstruct_frame(self,
            df_frame:           pd.DataFrame,
            keypoint_names:     List[str]
    ) -> Dict[str, Tuple[jnp.ndarray, np.ndarray]]:
        """
        Reconstructs all keypoints for a single frame

        Args:
            df_frame: A pandas DataFrame containing 2D detections for a single frame
            keypoint_names: A list of keypoint names (columns in the DataFrame) to reconstruct

        Returns:
            A dictionary where keys are keypoint names and values are a tuple of:
            - (N, 3) JAX array of final 3D points
            - (N,) NumPy array of confidence scores for each point
        """

        reconstructed_data = {}
        detections_by_keypoint = self._prepare_data(df_frame, keypoint_names)

        for kp_name, dets_per_cam in detections_by_keypoint.items():
            logging.debug(f"Reconstructing '{kp_name}'...")
            if sum(d.shape[0] for d in dets_per_cam) < self.config['min_views']:
                logging.debug("  -> Not enough detections to reconstruct")
                reconstructed_data[kp_name] = (jnp.empty((0, 3)), np.array([]))
                continue

            final_pts, final_confs = self._reconstruct_keypoint(dets_per_cam)
            reconstructed_data[kp_name] = (final_pts, final_confs)
            logging.debug(f"  -> Found {final_pts.shape[0]} instances")

        return reconstructed_data

    # --------------------------------------------------------------------------
    # Core reconstruction pipeline
    # --------------------------------------------------------------------------

    def _reconstruct_keypoint(self,
            dets_per_cam: List[jnp.ndarray]
    ) -> Tuple[ArrayLike, np.ndarray]:
        """ Orchestrates the full reconstruction pipeline for a single keypoint """

        @lru_cache(maxsize=None)
        def get_cached_cost_mat(i: int, j: int) -> jnp.ndarray:
            # ensure i < j for cache consistency
            if i > j: i, j = j, i
            return self._compute_cost_matrix(dets_per_cam[i], dets_per_cam[j], i, j)

        # Step 1: Generate all plausible 3D point hypotheses
        all_pts, all_groups, all_confs, all_errors = self._generate_hypotheses(dets_per_cam, get_cached_cost_mat)

        if all_pts.shape[0] == 0:
            return jnp.empty((0, 3)), np.array([])

        # Step 2: Filter hypotheses to resolve conflicts and merge redundancies
        final_pts, _, final_confs = self._filter_hypotheses(
            all_pts, all_confs, all_errors, all_groups
        )

        return jnp.array(final_pts), np.array(final_confs)

    def _generate_hypotheses(self,
            dets_per_cam:       List[jnp.ndarray],
            _cost_mat_getter:   Callable[[int, int], jnp.ndarray]
        ) -> Tuple[jnp.ndarray, list, list, list]:
        """
        First pass of reconstruction: generates all plausible 3D points (hypotheses)
        from the 2D detections without resolving conflicts
        """
        
        groups = self._group_points(dets_per_cam, _cost_mat_getter)
        M = len(groups)
        if M == 0:
            return jnp.empty((0, 3)), [], [], []

        # Triangulate all groups
        matched_uvs = np.full((self.num_cams, M, 2), np.nan, dtype=np.float32)
        for m, group in enumerate(groups):
            for cam_idx, det_idx in group:
                matched_uvs[cam_idx, m] = dets_per_cam[cam_idx][det_idx]

        undistorted_matched_uvs = jnp.full_like(matched_uvs, jnp.nan)
        for c in range(self.num_cams):
            uvs_c = matched_uvs[c, :, :]
            valid_mask = ~np.isnan(uvs_c[:, 0])
            if np.any(valid_mask):
                ud_chunk = undistort_points(jnp.array(uvs_c[valid_mask]), self.all_K[c], self.all_D[c])
                undistorted_matched_uvs = undistorted_matched_uvs.at[c, valid_mask, :].set(ud_chunk)

        points3d = triangulate_points_from_projections(points2d=undistorted_matched_uvs, P_mats=self.all_P, weights=None)

        # Check for valid triangulation points (not nans)
        valid_triangulation_mask = ~jnp.any(jnp.isnan(points3d), axis=1)

        # Reproject all 3D points back to all cameras at once
        all_reprojected_pts, projection_validity = project_to_multiple_cameras(
            object_points=points3d,
            rvec=self.rvecs_w2c,
            tvec=self.tvecs_w2c,
            camera_matrix=self.all_K,
            dist_coeffs=self.all_D
        )

        # Calculate reprojection errors
        # Create a mask of which 2D detections were originally present
        original_visibility_mask = ~jnp.isnan(matched_uvs[:, :, 0])

        # A point is only valid for error calculation if it was originally detected AND re-projects in front of the camera
        combined_visibility_mask = original_visibility_mask.astype(jnp.float32) * projection_validity

        # Calculate per-camera distances (which will contain nans)
        diffs = all_reprojected_pts - matched_uvs
        # Use the mask to zero-out invalid diffs before taking the norm
        valid_diffs = jnp.where(combined_visibility_mask[..., None], diffs, 0.0)
        distances = jnp.linalg.norm(valid_diffs, axis=-1)

        # Calculate mean error for each of the M points
        sum_of_errors = jnp.sum(distances, axis=0)              # sum over camera axis
        num_views = jnp.sum(combined_visibility_mask, axis=0)   # sum over camera axis

        # Add a mask to prevent points with 0 valid views from passing
        has_views_mask = num_views > 0
        reprojection_errors = sum_of_errors / jnp.maximum(num_views, 1)

        # Check against reprojection threshold
        repro_ok_mask = reprojection_errors < self.config['repro_thresh']

        # Combine all masks to get the final list of valid hypotheses
        final_valid_mask = valid_triangulation_mask & repro_ok_mask & has_views_mask

        # Apply the final mask to get the outputs
        valid_indices = np.array(jnp.where(final_valid_mask)[0])
        valid_groups = [groups[i] for i in valid_indices]
        confidences = [len(g) for g in valid_groups]
        errors = list(np.array(reprojection_errors)[valid_indices])

        return points3d[valid_indices], valid_groups, confidences, errors

    def _filter_hypotheses(self,
            points3d:       np.ndarray,
            confidences:    List[int],
            errors:         List[float],
            groups:         List[List[Tuple[int, int]]]
    ) -> Tuple[np.ndarray, List, List]:
        """
        Second pass: Filters and resolves 3D point candidates using a multi-stage,
        evidence-based process (MWIS -> Safe geometric merging)
        """
        num_points = points3d.shape[0]
        if num_points == 0:
            return np.empty((0, 3)), [], []

        cfg = self.config

        # Calculate scores as floats first to maintain precision
        float_scores = np.array([(c * cfg['view_count_weight']) - (e * cfg['repro_error_weight'])
                                 for c, e in zip(confidences, errors)])
        groups_as_sets = [set(g) for g in groups]

        # The max_weight_clique algorithm requires non-negative integer weights

        # Shift scores to be non-negative
        min_score = np.min(float_scores) if float_scores.size > 0 else 0
        if min_score < 0:
            scores_non_negative = float_scores - min_score
        else:
            scores_non_negative = float_scores

        # Scale to integers to preserve precision from reprojection errors
        scaling_factor = 10000
        integer_scores = (scores_non_negative * scaling_factor).astype(int)

        # The Maximum Weight Independent Set of a graph is equivalent to the Maximum Weight Clique of its complement
        conflict_graph = nx.Graph()
        # (we use the original float_scores for everything *except* the MWC algorithm)
        for i in range(num_points):
            # The weight attribute for MWC must be an integer
            conflict_graph.add_node(i, weight=int(integer_scores[i]))

        for i in range(num_points):
            for j in range(i + 1, num_points):
                if not groups_as_sets[i].isdisjoint(groups_as_sets[j]):
                    conflict_graph.add_edge(i, j)

        # Create the complement of the conflict graph
        # In this new graph, an edge means two hypotheses *are compatible*
        complement_graph = nx.complement(conflict_graph)

        # We must copy the node attributes because nx.complement doesnt...
        node_weights = {i: int(integer_scores[i]) for i in range(num_points)}
        nx.set_node_attributes(complement_graph, node_weights, name='weight')

        winner_indices, _ = nx.algorithms.clique.max_weight_clique(complement_graph, weight='weight')

        if not winner_indices:
            return np.empty((0, 3)), [], []

        # we use the original float scores for the subsequent merging step
        winner_points_3d = points3d[np.array(winner_indices)]
        winner_scores = float_scores[winner_indices]
        winner_groups_original = [groups[i] for i in winner_indices]

        if winner_points_3d.shape[0] == 0:
            return np.empty((0, 3)), [], []

        clustering = DBSCAN(eps=cfg['cluster_radius'], min_samples=1).fit(winner_points_3d)
        labels = clustering.labels_

        final_points, final_groups, final_scores = [], [], []
        processed_local_indices = set()

        for i in range(len(winner_indices)):
            if i in processed_local_indices:
                continue

            current_label = labels[i]
            local_indices_in_cluster = np.where(labels == current_label)[0]

            should_merge = False
            if cfg['filter_method'] == 'average' and len(local_indices_in_cluster) > 1:
                
                # Merge if points are geometrically close AND are competing hypotheses for the same thing
                original_indices = [winner_indices[k] for k in local_indices_in_cluster]
                subgraph = conflict_graph.subgraph(original_indices)
                
                if nx.is_connected(subgraph):
                    sets_to_compare = [groups_as_sets[k] for k in original_indices]
                    avg_jaccard = self._calculate_average_jaccard(sets_to_compare)
                    
                    if avg_jaccard > cfg['jaccard_threshold_for_merge']:
                        should_merge = True

            if should_merge:
                cluster_pts = winner_points_3d[local_indices_in_cluster]
                cluster_scores = winner_scores[local_indices_in_cluster]
                weights = self._softmax_weights(cluster_scores, cfg['softmax_temperature'])
                averaged_point = np.sum(cluster_pts * weights[:, np.newaxis], axis=0)

                best_in_cluster_idx = local_indices_in_cluster[np.argmax(cluster_scores)]
                final_points.append(averaged_point)
                final_groups.append(winner_groups_original[best_in_cluster_idx])
                final_scores.append(np.sum(cluster_scores * weights))
                
            else:
                for local_idx in local_indices_in_cluster:
                    final_points.append(winner_points_3d[local_idx])
                    final_groups.append(winner_groups_original[local_idx])
                    final_scores.append(winner_scores[local_idx])

            processed_local_indices.update(local_indices_in_cluster)

        # Step 3: Optional - Aggressive disjoint point merging
        if cfg['enable_disjoint_merge'] and len(final_points) > 1:
            final_points, final_groups, final_scores = self._proximity_merging(
                np.array(final_points), final_groups, np.array(final_scores), cfg['disjoint_merge_radius']
            )

        return np.array(final_points), final_groups, list(final_scores)

    # --------------------------------------------------------------------------

    def _prepare_data(self,
            df_frame:       pd.DataFrame,
            keypoint_names: List[str]
    ) -> Dict[str, List[jnp.ndarray]]:
        """ Extracts and formats 2D detections from a DataFrame for a single frame """
        # TODO: The structure of the df might change anyway so this ultimately won't be needed

        detections_by_keypoint = {}
        for kp_name in keypoint_names:
            dets_per_cam_list = []
            for cam_name in self.camera_names:
                try:
                    df_cam = df_frame.loc[cam_name]
                    # Ensure the keypoint exists as a column level
                    if kp_name in df_cam.columns.get_level_values(0):
                        valid_detections = df_cam[kp_name].dropna()
                        dets_per_cam_list.append(jnp.array(valid_detections[['x', 'y']].values, dtype=jnp.float32))
                    else:
                        dets_per_cam_list.append(jnp.empty((0, 2), dtype=jnp.float32))
                except (KeyError, AttributeError):
                    dets_per_cam_list.append(jnp.empty((0, 2), dtype=jnp.float32))
            detections_by_keypoint[kp_name] = dets_per_cam_list
        return detections_by_keypoint

    def _compute_cost_matrix(self,
            dets_i: jnp.ndarray,
            dets_j: jnp.ndarray,
            i:      int,
            j:      int
        ) -> jnp.ndarray:
        """
        Computes a cost matrix using epipolar segments, constrained by the Volume of Trust
        """

        Ni, Nj = dets_i.shape[0], dets_j.shape[0]
        if Ni == 0 or Nj == 0:
            return jnp.empty((Ni, Nj), dtype=jnp.float32)

        # Get camera parameters
        K_i, D_i, E_i = self.all_K[i], self.all_D[i], self.all_E[i]
        K_j, D_j, E_j = self.all_K[j], self.all_D[j], self.all_E[j]

        # We need world-to-camera rvec/tvec for project_points
        rvec_w2c_j, tvec_w2c_j = extmat_to_rtvecs(E_j)

        # Undistort points in the target camera (j)
        # We will compare distances in this *undistorted* space
        udets_j = undistort_points(dets_j, K_j, D_j)

        # Get the 3D rays for each point in the source camera (i)
        E_c2w_i = jnp.linalg.inv(E_i)
        cam_center_i = E_c2w_i[:3, 3]

        # back_projection handles undistortion internally and gives us a point on the ray
        p_3d_on_ray = back_projection(dets_i, 1.0, K_i, E_c2w_i, dist_coeffs=D_i)
        ray_dirs = p_3d_on_ray - cam_center_i
        ray_dirs /= jnp.linalg.norm(ray_dirs, axis=-1, keepdims=True)

        # Find where these rays intersect the volume of interest (AABB)
        p_near_3d, p_far_3d, has_intersection = bundle_intersection_AABB(cam_center_i, ray_dirs, self.aabb_min, self.aabb_max)

        # Project the 3D segments into the target camera's (j) image plane
        segments_3d = jnp.vstack([p_near_3d, p_far_3d])  # (2 * Ni, 3)

        # Project *without* applying distortion since we are comparing to udets_j
        segments_2d, _ = project_points(
            object_points=segments_3d,
            rvec=rvec_w2c_j,
            tvec=tvec_w2c_j,
            camera_matrix=K_j,
            dist_coeffs=jnp.zeros_like(D_j),  # zero distortion coeffs, important!
            distortion_model='none'
        )

        a_pts = segments_2d[:Ni]  # near points (Ni, 2)
        b_pts = segments_2d[Ni:]  # far points (Ni, 2)

        # Calculate the distance from each undistorted point in j to each projected segment
        p = udets_j[None, :, :]
        a = a_pts[:, None, :]
        b = b_pts[:, None, :]

        ab = b - a
        ap = p - a

        t = jnp.einsum('ijk,ijk->ij', ap, ab) / (jnp.einsum('ijk,ijk->ij', ab, ab) + 1e-6)
        t_clamped = jnp.clip(t, 0.0, 1.0)

        closest_points = a + t_clamped[..., None] * ab
        dists = jnp.linalg.norm(p - closest_points, axis=-1)

        # Apply thresholds to get final cost matrix
        final_costs = jnp.where(has_intersection[:, None], dists, 1e6)
        final_costs = jnp.where(final_costs > self.config['T_epi'], 1e6, final_costs)

        return final_costs

    def _group_points(self,
            dets_per_cam:       list,
            _cost_mat_getter:   Callable[[int, int], jnp.ndarray]
        ) -> list:
        """ Groups 2D detections using a graph-based approach with maximal cliques """

        if sum(d.shape[0] for d in dets_per_cam) < self.config['min_views']:
            return []

        num_dets_per_cam = [d.shape[0] for d in dets_per_cam]
        offsets = np.concatenate(([0], np.cumsum(num_dets_per_cam)[:-1]))
        total_dets = sum(num_dets_per_cam)

        source_indices, target_indices = [], []
        for i in range(self.num_cams):
            for j in range(i + 1, self.num_cams):
                cost_mat = _cost_mat_getter(i, j)
                if cost_mat.size == 0: continue
                match_rows, match_cols = np.where(np.array(cost_mat) < self.config['T_epi'])
                source_indices.extend((offsets[i] + match_rows).tolist())
                target_indices.extend((offsets[j] + match_cols).tolist())

        if not source_indices:
            return []

        adj_matrix = csr_matrix((np.ones(len(source_indices)), (source_indices, target_indices)),
                                shape=(total_dets, total_dets))
        n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)

        all_final_groups = []
        processed_groups = set()

        for i in range(n_components):
            component_indices = np.where(labels == i)[0]
            if len(component_indices) < self.config['min_views']: continue

            def unflatten(idx):
                cam_idx = np.searchsorted(offsets, idx, side='right') - 1
                return int(cam_idx), int(idx - offsets[cam_idx])

            subgraph_adj = adj_matrix[component_indices, :][:, component_indices]
            component_graph = nx.from_scipy_sparse_array(subgraph_adj)

            cliques = find_cliques(component_graph)

            for clique_local_indices in cliques:
                clique_global_indices = [component_indices[k] for k in clique_local_indices]
                if len(clique_global_indices) < self.config['min_views']: continue

                dets_by_cam_in_clique = {}
                for node_idx in clique_global_indices:
                    cam_idx, det_idx = unflatten(node_idx)
                    if cam_idx not in dets_by_cam_in_clique:
                        dets_by_cam_in_clique[cam_idx] = []
                    dets_by_cam_in_clique[cam_idx].append((cam_idx, det_idx))

                if len(dets_by_cam_in_clique) < self.config['min_views']: continue

                for group_tuple in product(*dets_by_cam_in_clique.values()):
                    group = sorted(list(group_tuple))
                    if len(group) >= self.config['min_views']:
                        frozen_group = frozenset(group)
                        if frozen_group not in processed_groups:
                            all_final_groups.append(group)
                            processed_groups.add(frozen_group)

        return all_final_groups

    @staticmethod
    def _proximity_merging(points, groups, scores, radius):
        """ Merges (aggressively) geometrically close points, keeping the best one """

        if points.shape[0] < 2:
            return points, groups, scores

        clustering = DBSCAN(eps=radius, min_samples=1).fit(points)
        labels = clustering.labels_
        final_points, final_groups, final_scores = [], [], []

        for label in np.unique(labels):
            indices = np.where(labels == label)[0]
            best_local_idx = np.argmax(scores[indices])
            best_global_idx = indices[best_local_idx]
            final_points.append(points[best_global_idx])
            final_groups.append(groups[best_global_idx])
            final_scores.append(scores[best_global_idx])

        return np.array(final_points), final_groups, np.array(final_scores)

    @staticmethod
    def _softmax_weights(scores: np.ndarray, temperature: float) -> np.ndarray:

        if temperature == 0:
            weights = np.zeros_like(scores)
            weights[np.argmax(scores)] = 1.0
            return weights

        scores_temp = scores / temperature
        e_scores = np.exp(scores_temp - np.max(scores_temp))

        return e_scores / (e_scores.sum() + 1e-9)

    @staticmethod
    def _calculate_average_jaccard(sets: List[set]) -> float:
        if len(sets) < 2:
            return 0.0

        jaccard_sum, pair_count = 0.0, 0

        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                intersection = len(sets[i].intersection(sets[j]))
                union = len(sets[i].union(sets[j]))
                jaccard_sum += intersection / union if union > 0 else 0
                pair_count += 1

        return jaccard_sum / pair_count if pair_count > 0 else 0


if __name__ == '__main__':

    # mini debug script to reconstruct 1 frame

    folder = Path().home() / 'Desktop' / '3d_ant_data'
    prefix = '240905-1616'
    session = 22
    # DEBUG_FRAME = 926

    df = fileio.load_session(folder / prefix / 'inputs' / 'tracking', session=session)
    df = df.reorder_levels(['camera', 'track', 'frame']).sort_index()

    cal_data = fileio.read_parameters(folder / prefix / 'calibration')
    keypoints, bones = fileio.load_skeleton_SLEAP(folder / prefix / 'inputs' / 'tracking', indices=False)

    volume_bounds = {'x': (-10.5, 13.0), 'y': (-21.0, 11.0), 'z': (180.0, 201.0)}

    reconstructor_config = {
        'repro_thresh': 10.0,
        'cluster_radius': 2.0,
        'view_count_weight': 10.0,
        'repro_error_weight': 1.0
    }

    reconstructor = Reconstructor(
        camera_parameters=cal_data,
        volume_bounds=volume_bounds,
        config=reconstructor_config
    )

    # # Run on a specific frame
    # df_frame = df.loc[pd.IndexSlice[:, :, DEBUG_FRAME], :]
    #
    # reconstructed_3d = reconstructor.reconstruct_frame(
    #     df_frame=df_frame,
    #     keypoint_names=keypoints
    # )
    #
    # print("\nFinal Reconstruction Results")
    # total_points = sum(points.shape[0] for points, confs in reconstructed_3d.values())
    # for name, (points, confs) in reconstructed_3d.items():
    #     if points.shape[0] > 0:
    #         print(f"  {name}: {points.shape[0]} points reconstructed")

    all_reconstructed_points = []
    all_frames = df.index.get_level_values('frame').unique()

    with alive_bar(title='Reconstruction...', total=len(all_frames), length=20, force_tty=True) as bar:
        for frame_idx in all_frames:
            try:
                df_frame = df.loc[pd.IndexSlice[:, :, frame_idx], :]
            except KeyError:
                # No data for this frame
                continue

            # Reconstruct all points for the frame
            reconstructed_3d = reconstructor.reconstruct_frame(
                df_frame=df_frame,
                keypoint_names=keypoints
            )

            frame_data = {
                "frame_idx": frame_idx,
                "points": reconstructed_3d
            }
            all_reconstructed_points.append(frame_data)
            bar()

    pickle.dump(all_reconstructed_points, open('reconstructed_points.pkl', 'wb'))