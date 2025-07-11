import logging
from functools import partial
from dataclasses import dataclass
from pathlib import Path
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Optional, Dict, Tuple, List
import numpy as np
import networkx as nx
from networkx.algorithms.clique import find_cliques
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN
import polars as pl
from mokap.utils import fileio
from mokap.utils.geometry.fitting import bundle_intersection_AABB
from mokap.utils.geometry.projective import (undistort_points, back_projection, triangulate_points_from_projections,
                                             project_points, project_to_multiple_cameras, undistort_multiple)
from mokap.utils.geometry.transforms import (extrinsics_matrix, projection_matrix, invert_rtvecs,
                                             extmat_to_rtvecs, invert_extrinsics_matrix)


logger = logging.getLogger(__name__)


@dataclass
class SoupPoint:
    """ A single reconstructed 3D point """
    frame_idx: int          # frame index
    idx: int                # keypoint's unique index within the frame
    keypoint_type: str
    position: ArrayLike     # x, y, z
    confidence: float       # aggregated confidence from reconstruction


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
            'detection_confidence_weight': 5.0,
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

        # Empty arrays to avoid instanciating thousands of new ones
        self._empty_score_np = np.array((0,), dtype=np.float32)
        self._empty_score_jax = jnp.array((0,), dtype=jnp.float32)
        self._empty_point2d_jax = jnp.empty((0, 2), dtype=jnp.float32)
        self._empty_point3d_np = np.empty((0, 3), dtype=np.float32)
        self._empty_point3d_jax = jnp.empty((0, 3), dtype=jnp.float32)

    def reconstruct_frame(self,
            df_frame:           pl.DataFrame,
            keypoint_names:     List[str]
        ) -> List[SoupPoint]:
        """
        Reconstructs all keypoints for a single frame

        Args:
            df_frame: A polars dataframe containing 2D detections for a single frame
            keypoint_names: A list of keypoint names to reconstruct

        Returns:
            A list of ReconstructedPoint objects, for all successfully
            reconstructed 3D points in the frame
        """

        if df_frame.is_empty():
            return []

        # all rows in df_frame are for the same frame, so we can extract it once
        frame_index = df_frame.select(pl.col("frame")).item(0, 0)

        reconstructed_points = []
        detections_by_keypoint = self._prepare_data(df_frame, keypoint_names)

        for kp_name, dets_per_cam in detections_by_keypoint.items():
            point_id_counter = 0

            points_per_cam = [d[0] for d in dets_per_cam]
            confs_per_cam = [d[1] for d in dets_per_cam]

            logging.debug(f"Reconstructing '{kp_name}' for frame {frame_index}...")
            if sum(d.shape[0] for d in points_per_cam) < self.config['min_views']:
                logging.debug("  -> Not enough detections to reconstruct")
                continue

            final_pts, final_confs = self._reconstruct_keypoint(points_per_cam, confs_per_cam)

            logging.debug(f"  -> Found {final_pts.shape[0]} instances of '{kp_name}'")

            # Convert the array-based results into SoupPoint objects
            for i in range(final_pts.shape[0]):
                point = SoupPoint(
                    frame_idx=frame_index,
                    idx=point_id_counter,
                    keypoint_type=kp_name,
                    position=final_pts[i],
                    confidence=float(final_confs[i])
                )
                reconstructed_points.append(point)
                point_id_counter += 1

        return reconstructed_points

    # --------------------------------------------------------------------------
    # Core reconstruction pipeline
    # --------------------------------------------------------------------------

    def _reconstruct_keypoint(self,
            points_per_cam: List[jnp.ndarray],
            confs_per_cam:  List[jnp.ndarray]
        ) -> Tuple[np.ndarray, np.ndarray]:
        """ Orchestrates the full reconstruction pipeline for a single keypoint """

        # Step 1: Generate all plausible 3D point hypotheses
        all_pts_jax, all_groups, view_counts, summed_confs, all_errors = self._generate_hypotheses(
            points_per_cam, confs_per_cam
        )

        if all_pts_jax.shape[0] == 0:
            return self._empty_point3d_np, self._empty_score_np

        # Step 2: Filter hypotheses to resolve conflicts and merge redundancies
        final_pts, _, final_confs = self._filter_hypotheses(
            all_pts_jax, view_counts, summed_confs, all_errors, all_groups
        )

        return final_pts, final_confs

    def _generate_hypotheses(self,
            points_per_cam:     List[jnp.ndarray],
            confs_per_cam:      List[jnp.ndarray],
        ) -> Tuple[jnp.ndarray, List[List[Tuple[int, int]]], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        First pass of reconstruction: generates all plausible 3D points (hypotheses)
        from the 2D detections without resolving conflicts
        """

        groups = self._group_points(points_per_cam)

        M = len(groups)
        if M == 0:
            return self._empty_point3d_jax, [], self._empty_score_jax, self._empty_score_jax, self._empty_score_jax

        # Create flat index arrays for all detections across all groups
        group_indices = np.array([m for m, g in enumerate(groups) for _ in g])
        cam_indices_src = np.array([cam_idx for g in groups for cam_idx, _ in g])
        det_indices_src = np.array([det_idx for g in groups for _, det_idx in g])

        # Gather the data in a flat format
        # TODO: a better structure here would be padded numpy arrays, but this requires knowing the maximum number
        #  of instances of any keypoint in the full video.
        flat_points_list = [points_per_cam[c][d] for c, d in zip(cam_indices_src, det_indices_src)]
        flat_confs_list = [confs_per_cam[c][d] for c, d in zip(cam_indices_src, det_indices_src)]
        flat_points = jnp.array(flat_points_list)
        flat_confs = jnp.array(flat_confs_list)

        dest_indices = (cam_indices_src, group_indices)
        matched_uvs = jnp.full((self.num_cams, M, 2), jnp.nan, dtype=jnp.float32).at[dest_indices].set(flat_points)
        triangulation_weights = jnp.zeros((self.num_cams, M), dtype=jnp.float32).at[dest_indices].set(flat_confs)

        undistorted_matched_uvs = undistort_multiple(
            matched_uvs,
            self.all_K,
            self.all_D
        )

        points3d = triangulate_points_from_projections(
            points2d=undistorted_matched_uvs,
            P_mats=self.all_P,
            weights=triangulation_weights
        )

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
        sum_of_errors = jnp.sum(distances, axis=0)  # sum over camera axis
        num_views = jnp.sum(combined_visibility_mask, axis=0)  # sum over camera axis

        # Add a mask to prevent points with 0 valid views from passing
        has_views_mask = num_views > 0
        reproj_errors = sum_of_errors / jnp.maximum(num_views, 1)

        # Check against reprojection threshold
        repro_ok_mask = reproj_errors < self.config['repro_thresh']

        # Calculate view counts and summed confs for all hypotheses
        view_counts = jnp.sum(original_visibility_mask, axis=0)
        summed_confs = jnp.sum(jnp.where(original_visibility_mask, triangulation_weights, 0), axis=0)

        # Combine all masks to get the final list of valid hypotheses
        final_valid_mask = valid_triangulation_mask & repro_ok_mask & has_views_mask

        # Apply the final mask to get the outputs
        valid_indices = jnp.where(final_valid_mask)[0]
        if valid_indices.shape[0] == 0:
            return self._empty_point3d_jax, [], self._empty_score_jax, self._empty_score_jax, self._empty_score_jax

        # Groups remain a list of lists because they are ragged
        valid_groups = [groups[i] for i in valid_indices.tolist()]

        # TODO: Ideally, jitting this whole thing and always returning padded arrays would be much faster...

        return (points3d[valid_indices], valid_groups, view_counts[valid_indices],
                summed_confs[valid_indices], reproj_errors[valid_indices])

    def _filter_hypotheses(self,
            points3d:       jnp.ndarray,
            view_counts:    jnp.ndarray,
            summed_confs:   jnp.ndarray,
            errors:         jnp.ndarray,
            groups:         List[List[Tuple[int, int]]]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Second pass: Filters and resolves 3D point candidates using a multi-stage,
        evidence-based process (MWIS -> Safe geometric merging)
        """

        num_points = points3d.shape[0]
        if num_points == 0:
            return self._empty_point3d_np, self._empty_score_np, self._empty_score_np

        # Calculate scores
        float_scores = (
                (view_counts * self.config['view_count_weight'])    # reward for view count
                + (summed_confs * self.config['detection_confidence_weight'])   # reward for 2D confidence
                - (errors * self.config['repro_error_weight'])   # penalty for error
        )

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

        winner_indices_list, _ = nx.algorithms.clique.max_weight_clique(complement_graph, weight='weight')

        if not winner_indices_list:
            return self._empty_point3d_np, self._empty_score_np, self._empty_score_np

        winner_indices = np.array(winner_indices_list)

        # we know which points we need so we can convert back to numpy for sklearn
        winner_points_3d = np.asarray(points3d[winner_indices])
        winner_scores = float_scores[winner_indices]
        winner_groups_original = [groups[i] for i in winner_indices]

        if winner_points_3d.shape[0] == 0:
            return self._empty_point3d_np, self._empty_score_np, self._empty_score_np

        clustering = DBSCAN(eps=self.config['cluster_radius'], min_samples=1).fit(winner_points_3d)
        labels = clustering.labels_

        final_points, final_groups, final_scores = [], [], []
        processed_local_indices = set()

        for i in range(len(winner_indices)):
            if i in processed_local_indices:
                continue

            current_label = labels[i]
            local_indices_in_cluster = np.where(labels == current_label)[0]

            should_merge = False
            if self.config['filter_method'] == 'average' and len(local_indices_in_cluster) > 1:

                # Merge if points are geometrically close AND are competing hypotheses for the same thing
                original_indices = [winner_indices[k] for k in local_indices_in_cluster]
                subgraph = conflict_graph.subgraph(original_indices)

                if nx.is_connected(subgraph):
                    sets_to_compare = [groups_as_sets[k] for k in original_indices]
                    avg_jaccard = self._calculate_average_jaccard(sets_to_compare)

                    if avg_jaccard > self.config['jaccard_threshold_for_merge']:
                        should_merge = True

            if should_merge:
                cluster_pts = winner_points_3d[local_indices_in_cluster]
                cluster_scores = np.asarray(winner_scores[local_indices_in_cluster])
                weights = self._softmax_weights(cluster_scores, self.config['softmax_temperature'])
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

        final_points = np.asarray(final_points, dtype=np.float32)
        final_groups = np.asarray(final_groups, dtype=object)
        final_scores = np.asarray(final_scores, dtype=np.float32)

        # Step 3: Optional - Aggressive disjoint point merging
        if self.config['enable_disjoint_merge'] and len(final_points) > 1:
            final_points, final_groups, final_scores = self._proximity_merging(
                final_points, final_groups, final_scores, self.config['disjoint_merge_radius']
            )

        return final_points, final_groups, final_scores

    # --------------------------------------------------------------------------

    def _prepare_data(self,
            df_frame:           pl.DataFrame,
            keypoint_names:     List[str]
        ) -> Dict[str, List[Tuple[jnp.ndarray, jnp.ndarray]]]:
        """ Extracts and formats 2D detections from a flat Polars DataFrame for a single frame """

        detections_by_keypoint = {}

        # input df_frame is already filtered for a single frame, but we can group by keypoint
        # to process all data for a given keypoint at once
        grouped_by_kp = df_frame.group_by('keypoint')

        kp_dfs = {kp_name[0]: group_df for kp_name, group_df in grouped_by_kp}  # group_by returns keys as tuples!!!
        for kp_name in keypoint_names:
            dets_per_cam_list = []

            # check if this keypoint had any detections in this frame
            if kp_name not in kp_dfs:
                # if not, fill with empty arrays for all cameras
                detections_by_keypoint[kp_name] = [(self._empty_point2d_jax, self._empty_score_jax)] * self.num_cams
                continue

            # if the keypoint exists get its data
            df_kp = kp_dfs[kp_name]
            cam_data = (
                df_kp.group_by("camera")
                .agg(
                    pl.col("x"),
                    pl.col("y"),
                    pl.col("score"),
                )
                .to_dict(as_series=False)
            )

            # create a mapping for fast lookup
            cam_data_map = {cam: i for i, cam in enumerate(cam_data["camera"])}

            for cam_name in self.camera_names:
                if cam_name in cam_data_map:
                    idx = cam_data_map[cam_name]

                    points_np = np.column_stack([cam_data['x'][idx], cam_data['y'][idx]])
                    confs_np = np.array(cam_data['score'][idx])
                    points_jax = jnp.array(points_np, dtype=jnp.float32)
                    confs_jax = jnp.array(confs_np, dtype=jnp.float32)

                    dets_per_cam_list.append((points_jax, confs_jax))
                else:
                    # this camera had no detections for this keypoint in this frame
                    dets_per_cam_list.append((self._empty_point2d_jax, self._empty_score_jax))

            detections_by_keypoint[kp_name] = dets_per_cam_list

        return detections_by_keypoint

    @partial(jax.jit, static_argnums=(0, 3, 4))
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

        # Get camera parameters
        K_i, D_i, E_i = self.all_K[i], self.all_D[i], self.all_E[i]
        K_j, D_j, E_j = self.all_K[j], self.all_D[j], self.all_E[j]

        # We need world-to-camera rvec/tvec for project_points
        rvec_w2c_j, tvec_w2c_j = extmat_to_rtvecs(E_j)

        # Undistort points in the target camera (j)
        # We will compare distances in this *undistorted* space
        udets_j = undistort_points(dets_j, K_j, D_j)

        # Get the 3D rays for each point in the source camera (i)
        E_c2w_i = invert_extrinsics_matrix(E_i)
        cam_center_i = E_c2w_i[:3, 3]

        # back_projection handles undistortion internally and gives us a point on the ray
        p_3d_on_ray = back_projection(dets_i, 1.0, K_i, E_c2w_i, dist_coeffs=D_i)
        ray_dirs = p_3d_on_ray - cam_center_i
        ray_dirs /= jnp.linalg.norm(ray_dirs, axis=-1, keepdims=True)

        # Find where these rays intersect the volume of interest (AABB)
        p_near_3d, p_far_3d, has_intersection = bundle_intersection_AABB(cam_center_i, ray_dirs, self.aabb_min,
                                                                         self.aabb_max)

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

    def _group_points(self, dets_per_cam: List[jnp.ndarray]) -> List:
        """ Groups 2D detections using a graph-based approach with maximal cliques """

        if sum(d.shape[0] for d in dets_per_cam) < self.config['min_views']:
            return []

        nb_dets_per_cam = [d.shape[0] for d in dets_per_cam]
        offsets = np.concatenate(([0], np.cumsum(nb_dets_per_cam)[:-1]))
        total_dets = sum(nb_dets_per_cam)

        source_indices, target_indices = [], []
        for i in range(self.num_cams):
            for j in range(i + 1, self.num_cams):

                if nb_dets_per_cam[i] == 0 or nb_dets_per_cam[j] == 0:
                    continue

                cost_mat = self._compute_cost_matrix(dets_per_cam[i], dets_per_cam[j], i, j)
                if cost_mat.size == 0:
                    continue

                match_rows, match_cols = np.where(np.asarray(cost_mat) < self.config['T_epi'])
                source_indices.extend((offsets[i] + match_rows).tolist())
                target_indices.extend((offsets[j] + match_cols).tolist())

        if not source_indices:
            return []

        adj_matrix = csr_matrix((np.ones(len(source_indices)), (source_indices, target_indices)),
                                shape=(total_dets, total_dets))
        n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)

        all_final_groups = []
        processed_groups = set()

        def unflatten(idx):
            cam_idx = np.searchsorted(offsets, idx, side='right') - 1
            return int(cam_idx), int(idx - offsets[cam_idx])

        for i in range(n_components):
            component_indices = np.where(labels == i)[0]
            if len(component_indices) < self.config['min_views']: continue

            subgraph_adj = adj_matrix[component_indices, :][:, component_indices]
            component_graph = nx.from_scipy_sparse_array(subgraph_adj)
            cliques = find_cliques(component_graph)

            for clique_local_indices in cliques:
                if len(clique_local_indices) < self.config['min_views']: continue

                # Build a small "conflict graph" for this clique
                # Nodes are the original (cam_idx, det_idx) tuples
                # An edge connects two detections if they are from the same camera
                clique_nodes = [unflatten(component_indices[k]) for k in clique_local_indices]

                # Check that we have enough distinct cameras in this clique
                if len(set(cam_idx for cam_idx, det_idx in clique_nodes)) < self.config['min_views']:
                    continue

                conflict_graph = nx.Graph()
                for n1_idx, node1 in enumerate(clique_nodes):
                    conflict_graph.add_node(node1)

                    for n2_idx in range(n1_idx + 1, len(clique_nodes)):
                        node2 = clique_nodes[n2_idx]
                        # If camera index is the same, they are in conflict
                        if node1[0] == node2[0]:
                            conflict_graph.add_edge(node1, node2)

                # Find all maximal independent sets of the conflict graph
                # An independent set has no edges, meaning it contains at most one detection per camera.
                complement_g = nx.complement(conflict_graph)

                for group in find_cliques(complement_g):

                    if len(group) >= self.config['min_views']:
                        # the clique in the complement graph is a valid group
                        sorted_group = sorted(group)
                        frozen_group = frozenset(sorted_group)

                        if frozen_group not in processed_groups:
                            all_final_groups.append(sorted_group)
                            processed_groups.add(frozen_group)

        return all_final_groups

    @staticmethod
    def _proximity_merging(
            points: ArrayLike,
            groups: ArrayLike,
            scores: ArrayLike,
            radius: float
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        return np.asarray(final_points), np.asarray(final_groups), np.asarray(final_scores)

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
    from collections import defaultdict

    # mini debug script to reconstruct 1 frame

    folder = Path().home() / 'Desktop' / '3d_ant_data'
    prefix = '240905-1616'
    session = 22

    df = fileio.load_session(folder / prefix / 'inputs' / 'tracking', session=session, use_polars=True)
    grouped_by_frame = df.group_by('frame', maintain_order=True)
    nb_frames = df.select(pl.col('frame').n_unique()).item()

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

    # Run on the specific debug frame
    DEBUG_FRAME = 926
    df_frame = df.filter(pl.col('frame') == DEBUG_FRAME)

    points_list = reconstructor.reconstruct_frame(
        df_frame=df_frame,
        keypoint_names=keypoints
    )

    print(f"Total points reconstructed in frame {DEBUG_FRAME}: {len(points_list)}\n")
    if points_list:

        points_by_type = defaultdict(list)
        for pt in points_list:
            points_by_type[pt.keypoint_type].append(pt)

        for kp_type, points in points_by_type.items():
            print(f"Keypoint Type: '{kp_type}' ({len(points)} instances)")
            for point in points:
                pos_str = f"[{point.position[0]:.2f}, {point.position[1]:.2f}, {point.position[2]:.2f}]"
                print(
                    f"  - ID: {point.idx:<4} | "
                    f"Frame: {point.frame_idx:<5} | "
                    f"Position: {pos_str:<25} | "
                    f"Confidence: {point.confidence:.4f}"
                )