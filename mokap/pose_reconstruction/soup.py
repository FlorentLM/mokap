"""
3D Point soup reconstruction

Pipeline:
  1. Undistort all 2D detections
  2. For each keypoint type (batched across all frames):
     - Find all epipolar-valid pairs using fundamental matrices
     - Triangulate all pairs in one batch
     - Filter by reprojection error
     - Merge nearby points by re-triangulating with all contributing views
  3. Collect orphan rays for unused detections
"""
import logging
from itertools import combinations
from typing import List, Dict, Tuple
import numpy as np
from scipy.spatial import cKDTree

from lucida import CameraRig
from lucida.geometry.backend import xp, set_at, xp_float
from lucida.geometry import (undistort_points, px_to_ray, transform_vectors, project_full,
                             px_to_norm, triangulate_linear, epipolar_line_distance)

from mokap.pose_reconstruction.datatypes import PointSoup
from mokap.pose_reconstruction.skeleton import SkeletonTopology

logger = logging.getLogger(__name__)


class Reconstructor:
    """
    3D reconstruction via pairwise triangulation and spatial merging.

    For each keypoint type (batched across all frames):
      1. Find epipolar-valid camera pairs
      2. Triangulate each pair
      3. Filter by reprojection error
      4. Merge nearby points by re-triangulating with all contributing views
    """

    def __init__(self,
                 rig: CameraRig,
                 keypoint_names: List[str],
                 min_views: int = 2,
                 epipolar_threshold: float = 10.0,
                 reprojection_threshold: float = 5.0,
                 merge_radius: float = 0.5,
                 ):

        self.rig = rig
        self.nb_cams = len(rig)
        self.cam_pairs = list(combinations(range(self.nb_cams), 2))
        self.keypoint_names = keypoint_names

        # Config
        self.min_views = min_views
        self.epi_thresh = epipolar_threshold
        self.reproj_thresh = reprojection_threshold
        self.merge_radius = merge_radius

        # Cache rig arrays
        self.Ks = rig.K.copy()
        self.Ts = rig.T.copy()
        self.Ds = rig.D.copy()
        self.Tc2w = rig.T_c2w.copy()
        self.F = rig.F.copy()
        self.dist_model = str(rig.distortion_model)

        # Pre-allocate empty arrays to avoid thousands of alloc calls
        self._empty_triangulate_result = (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
            np.empty((0,), dtype=np.uint64),
            np.empty((0,), dtype=np.int32)
        )

    def reconstruct(self, inputs: Dict[str, np.ndarray]) -> PointSoup:
        """Processes a frame (or a chunk of)."""

        coords = xp.asarray(inputs['coords'])
        cam_ids = inputs['cam_ids']
        frame_ids = inputs['frame_indices']
        kp_ids = inputs['kp_type_ids']
        scores = inputs['scores']

        # Undistort all points once
        undist = undistort_points(
            coords,
            self.Ks[cam_ids],
            self.Ds[cam_ids],
            self.dist_model
        )

        used = np.zeros(len(coords), dtype=bool)

        out_pts, out_conf, out_err = [], [], []
        out_kp, out_frame, out_mask = [], [], []

        # Process per keypoint type (frames are batched together)
        for kp_type in np.unique(kp_ids):

            kp_mask = kp_ids == kp_type
            idx = np.where(kp_mask)[0]

            if len(idx) < self.min_views:
                continue

            local_undist = undist[idx]
            local_raw = coords[idx]
            local_cams = cam_ids[idx]
            local_scores = scores[idx]
            local_frames = frame_ids[idx]

            # Triangulate all valid pairs across all frames
            pts, errs, det_indices, cam_masks, pair_frames = self._triangulate_epipolar_pairs(
                local_undist, local_raw, local_cams, local_scores, local_frames, idx
            )
            nb_points = len(pts)

            if nb_points == 0:
                continue

            # Merge nearby points with re-triangulation (respecting frame boundaries)
            pts_merged, errs, det_indices, cam_masks, pair_frames = self._merge_nearby(
                pts, errs, det_indices, cam_masks, pair_frames,
                undist, coords, cam_ids, scores
            )

            for i in range(len(pts_merged)):
                out_pts.append(pts_merged[i])
                out_err.append(errs[i])

                n_views = bin(cam_masks[i]).count('1')
                out_conf.append(n_views * 10.0 - errs[i])

                out_kp.append(kp_type)
                out_frame.append(pair_frames[i])
                out_mask.append(cam_masks[i])

                for det_idx in det_indices[i]:
                    used[det_idx] = True

        # Orphan rays for unused detections
        orphan_rays_mask = ~used & ~np.isnan(inputs['coords'][:, 0])
        orphan_rays = self._get_rays(inputs, undist, orphan_rays_mask)

        return PointSoup(
            positions=np.array(out_pts, dtype=np.float32).reshape(-1, 3),
            confidences=np.array(out_conf, dtype=np.float32),
            reprojection_errors=np.array(out_err, dtype=np.float32),
            keypoint_indices=np.array(out_kp, dtype=np.int16),
            frame_indices=np.array(out_frame, dtype=np.int32),
            camera_masks=np.array(out_mask, dtype=np.uint64),
            **orphan_rays,
            camera_names=self.rig.names,
            keypoint_names=self.keypoint_names
        )

    def _triangulate_reproj_wrapper(self, pixel_coords, weights):
        """Just a tiny internal wrapper for px->norm->3d->reproj because this is used twice."""

        obs_px_normalised = px_to_norm(pixel_coords, self.Ks[:, None])

        pts3d = triangulate_linear(obs_px_normalised, self.Ts, weights=weights)

        reproj = project_full(
            pts3d[None],
            self.Ts[:, None],
            self.Ks[:, None],
            self.Ds[:, None],
            self.dist_model
        )

        return pts3d, reproj

    def _triangulate_epipolar_pairs(
            self,
            undist: xp.ndarray,
            raw: xp.ndarray,
            cam_ids: np.ndarray,
            scores: np.ndarray,
            frame_ids: np.ndarray,
            global_idx: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[List[int]], np.ndarray, np.ndarray]:
        """Triangulate all epipolar-valid pairs (across all frames at once)."""

        all_ii, all_jj = [], []
        all_ci, all_cj = [], []

        for ci, cj in self.cam_pairs:
            idx_i = np.where(cam_ids == ci)[0]
            idx_j = np.where(cam_ids == cj)[0]

            if len(idx_i) == 0 or len(idx_j) == 0:
                continue

            ii, jj = np.meshgrid(idx_i, idx_j, indexing='ij')
            ii, jj = ii.ravel(), jj.ravel()

            # Same frame only
            same_frame = frame_ids[ii] == frame_ids[jj]
            ii, jj = ii[same_frame], jj[same_frame]

            if len(ii) == 0:
                continue

            all_ii.append(ii)
            all_jj.append(jj)
            all_ci.append(np.full(len(ii), ci, dtype=np.int32))
            all_cj.append(np.full(len(ii), cj, dtype=np.int32))

        if not all_ii:
            return self._empty_triangulate_result

        ii = np.concatenate(all_ii)
        jj = np.concatenate(all_jj)
        ci = np.concatenate(all_ci)
        cj = np.concatenate(all_cj)

        pair_frames = frame_ids[ii]

        # Batched epipolar check
        dists = epipolar_line_distance(
            undist[ii],
            undist[jj],
            self.F[ci, cj]
        )
        valid_epi = dists < self.epi_thresh

        ii = ii[valid_epi]
        jj = jj[valid_epi]
        pair_frames = pair_frames[valid_epi]

        nb_pairs = len(ii)
        if nb_pairs == 0:
            return self._empty_triangulate_result

        # Build observation tensor
        pair_indices = np.arange(nb_pairs)

        obs = xp.full((self.nb_cams, nb_pairs, 2), xp.nan, dtype=xp_float)
        obs = set_at(obs, (cam_ids[ii], pair_indices), undist[ii])
        obs = set_at(obs, (cam_ids[jj], pair_indices), undist[jj])

        weights = xp.zeros((self.nb_cams, nb_pairs), dtype=xp_float)
        weights = set_at(weights, (cam_ids[ii], pair_indices), xp.asarray(scores[ii]))
        weights = set_at(weights, (cam_ids[jj], pair_indices), xp.asarray(scores[jj]))

        # Triangulate and reproject
        pts3d, reproj = self._triangulate_reproj_wrapper(obs, weights)

        raw_tensor = xp.full((self.nb_cams, nb_pairs, 2), xp.nan, dtype=xp_float)
        raw_tensor = set_at(raw_tensor, (cam_ids[ii], pair_indices), raw[ii])
        raw_tensor = set_at(raw_tensor, (cam_ids[jj], pair_indices), raw[jj])

        diff = raw_tensor - reproj
        sq_err = xp.sum(xp.square(diff), axis=-1)
        valid_views = ~xp.isnan(raw_tensor[..., 0])

        rmse = xp.sqrt(xp.sum(xp.where(valid_views, sq_err, 0.0), axis=0) / 2.0)

        rmse_np = np.asarray(rmse)
        pts3d_np = np.asarray(pts3d)

        valid_mask = (rmse_np < self.reproj_thresh) & ~np.any(np.isnan(pts3d_np), axis=1)
        valid_idx = np.where(valid_mask)[0]

        pts_out = pts3d_np[valid_idx]
        err_out = rmse_np[valid_idx]
        frames_out = pair_frames[valid_idx]

        det_indices = [
            [int(global_idx[ii[k]]), int(global_idx[jj[k]])]
            for k in valid_idx
        ]

        cam_masks = np.array([
            (1 << int(cam_ids[ii[k]])) | (1 << int(cam_ids[jj[k]]))
            for k in valid_idx
        ], dtype=np.uint64)

        return pts_out, err_out, det_indices, cam_masks, frames_out

    def _merge_nearby(
            self,
            pts: np.ndarray,
            errors: np.ndarray,
            det_indices: List[List[int]],
            cam_masks: np.ndarray,
            frame_ids: np.ndarray,
            undist: xp.ndarray,
            raw: xp.ndarray,
            cam_ids: np.ndarray,
            scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[List[int]], np.ndarray, np.ndarray]:
        """Fast merge using KDTree + union-find, and batched re-triangulation."""

        nb_pts = len(pts)
        if nb_pts <= 1:
            return pts, errors, det_indices, cam_masks, frame_ids

        # Cool trick: add massive offset for frames to Z to prevent cross-frame merging
        frame_shift = self.merge_radius * 1000  # 1000 doesn't matter, just needs to be big
        pts_with_frame = np.column_stack([
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            frame_ids.astype(np.float32) * frame_shift
        ])

        # Find neighbours within merge radius (but in 4D space)
        tree = cKDTree(pts_with_frame)
        pairs = tree.query_pairs(r=self.merge_radius)

        # Union-find to cluster
        parent = np.arange(nb_pts)

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i, j in pairs:
            union(i, j)

        # Group by cluster
        clusters = {}
        for i in range(nb_pts):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)

        cluster_list = list(clusters.values())
        n_clusters = len(cluster_list)

        # Take all detections per cluster
        cluster_dets = []
        cluster_frames = []

        for cluster in cluster_list:
            all_dets = set()

            for idx in cluster:
                all_dets.update(det_indices[idx])

            cluster_dets.append(list(all_dets))
            cluster_frames.append(frame_ids[cluster[0]])

        # Build observation tensor
        obs = xp.full((self.nb_cams, n_clusters, 2), xp.nan, dtype=xp.float32)
        weights = xp.zeros((self.nb_cams, n_clusters), dtype=xp.float32)
        merged_masks = np.zeros(n_clusters, dtype=np.uint64)

        for cluster_idx, dets in enumerate(cluster_dets):
            cluster_mask = np.uint64(0)

            for det_idx in dets:
                c = int(cam_ids[det_idx])
                obs = set_at(obs, (c, cluster_idx), undist[det_idx])
                weights = set_at(weights, (c, cluster_idx), xp.asarray(scores[det_idx]))
                cluster_mask |= np.uint64(1 << c)

            merged_masks[cluster_idx] = cluster_mask

        # Re-triangulate all
        pts3d, reproj = self._triangulate_reproj_wrapper(obs, weights)

        raw_tensor = xp.full((self.nb_cams, n_clusters, 2), xp.nan, dtype=xp.float32)

        for cluster_idx, dets in enumerate(cluster_dets):
            for det_idx in dets:
                c = int(cam_ids[det_idx])
                raw_tensor = set_at(raw_tensor, (c, cluster_idx), raw[det_idx])

        diff = raw_tensor - reproj
        sq_err = xp.sum(xp.square(diff), axis=-1)
        valid_views = ~xp.isnan(raw_tensor[..., 0])
        n_views = xp.sum(valid_views, axis=0)

        rmse = xp.sqrt(xp.sum(xp.where(valid_views, sq_err, 0.0), axis=0) / xp.maximum(n_views, 1))

        pts3d_np = np.asarray(pts3d)
        rmse_np = np.asarray(rmse)
        cluster_frames_np = np.array(cluster_frames, dtype=np.int32)

        return pts3d_np, rmse_np, cluster_dets, merged_masks, cluster_frames_np

    def _get_rays(
            self,
            inputs: Dict[str, np.ndarray],
            undist: xp.ndarray,
            orphan_mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Make 3D rays for unused detections."""

        if not np.any(orphan_mask):
            return {}

        pts = undist[orphan_mask]
        cams = inputs['cam_ids'][orphan_mask]

        n_orphans = len(pts)
        all_dirs = np.zeros((n_orphans, 3), dtype=np.float32)
        all_origins = np.zeros((n_orphans, 3), dtype=np.float32)

        for c in range(self.nb_cams):
            c_mask = cams == c
            if not np.any(c_mask):
                continue

            dirs_cam = px_to_ray(pts[c_mask], self.Ks[c])
            T_c2w = self.Tc2w[c]
            dirs_world = transform_vectors(dirs_cam, T_c2w)

            all_dirs[c_mask] = np.asarray(dirs_world)
            all_origins[c_mask] = np.asarray(T_c2w[:3, 3])

        return {
            'ray_origins': all_origins,
            'ray_directions': all_dirs,
            'ray_confidences': inputs['scores'][orphan_mask],
            'ray_keypoint_indices': inputs['kp_type_ids'][orphan_mask],
            'ray_frame_indices': inputs['frame_indices'][orphan_mask]
        }


if __name__ == "__main__":
    import time
    import polars as pl
    from pathlib import Path
    from mokap.utils import fileio
    from mokap.pose_reconstruction.utils import prepare_reconstruction_input

    BASE_DIR = Path.home() / 'Desktop' / '3d_ant_data'
    PREFIX = '240905-1616'
    SESSION = 22
    CHUNK_SIZE = 500

    calib_dir = BASE_DIR / PREFIX / 'inputs' / 'calibration'
    input_dir = BASE_DIR / PREFIX / 'inputs' / 'tracking'
    output_dir = BASE_DIR / PREFIX / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    rig_file = calib_dir / 'camera_rig.toml'
    soup_file = output_dir / f"soup_session{SESSION}.pkl"

    rig = CameraRig.load(rig_file)
    df = fileio.load_session(input_dir, session=SESSION)
    skeleton = SkeletonTopology.from_sleap(input_dir)

    reconstructor = Reconstructor(
        rig=rig,
        keypoint_names=skeleton.keypoints,  # TODO: not really needed (just for the soup metadata)
        min_views=2,
        epipolar_threshold=10.0,
        reprojection_threshold=5.0,
        merge_radius=0.05,
    )

    all_frames = np.sort(df['frame'].unique().to_numpy())
    batches = []
    total_pts = 0
    total_rays = 0
    t0 = time.time()

    print(f"Processing {len(all_frames)} frames...")

    for i in range(0, len(all_frames), CHUNK_SIZE):
        chunk = all_frames[i: i + CHUNK_SIZE]

        df_chunk = df.filter(pl.col("frame").is_in(chunk))

        # TODO: `prepare_reconstruction_input` will be removed once the I/O formats are unified with CATAR
        inputs = prepare_reconstruction_input(
            df_chunk, rig.names, skeleton.keypoints
        )

        soup = reconstructor.reconstruct(inputs)

        if soup.nb_points > 0 or len(soup.ray_origins) > 0:
            batches.append(soup)
            total_pts += soup.nb_points
            total_rays += len(soup.ray_origins)

        elapsed = time.time() - t0

        frames_done = min(i + CHUNK_SIZE, len(all_frames))
        fps = frames_done / elapsed if elapsed > 0 else 0

        print(f"  Chunk {i // CHUNK_SIZE}: {soup.nb_points} pts, {len(soup.ray_origins)} rays "
              f"({frames_done}/{len(all_frames)} frames, {fps:.1f} fps)")

    if batches:
        final_soup = PointSoup.concatenate(batches)
        final_soup.to_file(soup_file)

        total_time = time.time() - t0

        print(f"\nDone. Saved {total_pts} points and {total_rays} rays to {soup_file}")
        print(f"Total time: {total_time:.2f}s ({len(all_frames) / total_time:.2f} fps)")

    else:
        print("No points reconstructed.")