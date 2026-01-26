"""
3D Point soup reconstruction

Pipeline:
  1. Undistort all 2D detections
  2. For each keypoint type (batched across all frames):
      - Find cliques of mutually epipolar-consistent detections
      - Triangulate each clique using all contributing views
      - Greedily accept non-conflicting points (prefer more views)
  3. Collect orphan rays for unused detections
"""
import logging
from collections import defaultdict, Counter
from typing import List, Dict
import networkx as nx
import numpy as np

from lucida import CameraRig
from lucida.geometry.backend import xp, set_at, xp_float
from lucida.geometry import (undistort_points, px_to_ray, transform_vectors, project_full,
                             px_to_norm, triangulate_linear, epipolar_line_distance)

from mokap.pose_reconstruction.datatypes import PointSoup
from mokap.pose_reconstruction.skeleton import SkeletonTopology

logger = logging.getLogger(__name__)


class Reconstructor:
    """
    3D reconstruction via clique detection and triangulation.

    For each keypoint type (batched across all frames):
      1. Find cliques of mutually epipolar-consistent detections
      2. Triangulate each clique using all contributing views
      3. Greedily accept non-conflicting points (prefer more views)
    """

    def __init__(self,
                 rig: CameraRig,
                 keypoint_names: List[str],
                 min_views: int = 2,
                 epipolar_threshold: float = 10.0,
                 reprojection_threshold: float = 5.0,
                 ):

        self.rig = rig
        self.nb_cams = len(rig)
        self.keypoint_names = keypoint_names

        # Config
        self.min_views = min_views
        self.epi_thresh = epipolar_threshold
        self.reproj_thresh = reprojection_threshold

        # Cache rig arrays
        self.Ks = rig.K.copy()
        self.Ts = rig.T.copy()
        self.Ds = rig.D.copy()
        self.Tc2w = rig.T_c2w.copy()
        self.F = rig.F.copy()
        self.dist_model = str(rig.distortion_model)

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

    def reconstruct(self, inputs: Dict[str, np.ndarray]) -> PointSoup:
        """
        Main entry point. Processes a batch of frames.

        Args:
            inputs: Dict with keys:
                - 'coords': (N, 2) pixel coordinates
                - 'cam_ids': (N,) camera indices
                - 'frame_indices': (N,) frame indices
                - 'kp_type_ids': (N,) keypoint type indices
                - 'scores': (N,) detection confidences

        Returns:
            PointSoup with reconstructed 3D points and orphan rays
        """
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

        # Track which detections get used (for orphan ray generation)
        used = np.zeros(len(coords), dtype=bool)

        out_pts, out_conf, out_err = [], [], []
        out_kp, out_frame, out_mask = [], [], []

        # Process per keypoint type
        for kp_type in np.unique(kp_ids):
            kp_mask = kp_ids == kp_type
            idx = np.where(kp_mask)[0]

            if len(idx) < self.min_views:
                continue

            # Find cliques of mutually consistent detections
            cliques = self._find_detection_cliques(
                undist=undist[idx],
                cam_ids=cam_ids[idx],
                frame_ids=frame_ids[idx],
                local_to_global=idx  # Map local indices back to global
            )

            if not cliques:
                continue

            # Triangulate cliques and resolve conflicts
            results = self._triangulate_cliques(
                cliques=cliques,
                undist=undist,
                raw=coords,
                cam_ids=cam_ids,
                scores=scores,
                frame_ids=frame_ids
            )

            # Collect results
            for res in results:
                out_pts.append(res['position'])
                out_err.append(res['error'])
                out_conf.append(res['n_views'] * 10.0 - res['error'])
                out_kp.append(kp_type)
                out_frame.append(res['frame'])
                out_mask.append(res['cam_mask'])

                for det_idx in res['detections']:
                    used[det_idx] = True

        # Orphan rays for unused detections
        orphan_mask = ~used & ~np.isnan(inputs['coords'][:, 0])
        orphan_rays = self._get_rays(inputs, undist, orphan_mask)

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

    def _find_detection_cliques(
        self,
        undist: xp.ndarray,
        cam_ids: np.ndarray,
        frame_ids: np.ndarray,
        local_to_global: np.ndarray
    ) -> List[List[int]]:
        """
        Find maximal cliques of mutually epipolar-consistent detections.
        Batched epipolar computation for performance.
        """

        # Group by frame
        frame_to_local = defaultdict(list)
        for local_idx, f in enumerate(frame_ids):
            frame_to_local[f].append(local_idx)

        all_cliques = []

        for frame, local_indices in frame_to_local.items():
            n = len(local_indices)
            if n < self.min_views:
                continue

            local_indices = np.array(local_indices, dtype=np.int32)
            local_cams = cam_ids[local_indices]

            # Build all pairs (a, b) where a < b and different cameras
            pairs_a, pairs_b = [], []
            for a in range(n):
                for b in range(a + 1, n):
                    if local_cams[a] != local_cams[b]:
                        pairs_a.append(a)
                        pairs_b.append(b)

            if not pairs_a:
                continue

            pairs_a = np.array(pairs_a, dtype=np.int32)
            pairs_b = np.array(pairs_b, dtype=np.int32)

            # Epipolar distance (batched)
            li_a = local_indices[pairs_a]
            li_b = local_indices[pairs_b]
            ci = local_cams[pairs_a]
            cj = local_cams[pairs_b]

            dists = epipolar_line_distance(
                undist[li_a],
                undist[li_b],
                self.F[ci, cj]
            )
            dists = np.asarray(dists)

            # Build adjacency from valid pairs
            valid_pairs = dists < self.epi_thresh

            # Construct graph edges
            edges = list(zip(pairs_a[valid_pairs], pairs_b[valid_pairs]))
            if not edges:
                # No valid edges, but maybe single-camera situation
                # (shouldn't happen given min_views check, but this to be safe)
                continue

            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from(edges)

            # Find all maximal cliques
            for clique in nx.find_cliques(G):
                if len(clique) >= self.min_views:
                    global_clique = [int(local_to_global[local_indices[c]]) for c in clique]
                    all_cliques.append(global_clique)

        return all_cliques

    def _triangulate_cliques(
        self,
        cliques: List[List[int]],
        undist: xp.ndarray,
        raw: xp.ndarray,
        cam_ids: np.ndarray,
        scores: np.ndarray,
        frame_ids: np.ndarray
    ) -> List[Dict]:
        """
        Triangulate each clique, greedily accepting non-conflicting points.

        Cliques are sorted by (n_views, total_score) descending.
        Once a detection is used, all cliques containing it are skipped.

        Args:
            cliques: List of cliques (each is list of global detection indices)
            undist: All undistorted coordinates
            raw: All raw coordinates (for reprojection error)
            cam_ids: All camera IDs
            scores: All detection scores
            frame_ids: All frame IDs

        Returns:
            List of result dicts with keys: position, error, n_views, detections, cam_mask, frame
        """

        # Score and sort cliques: prefer more views, then higher total confidence
        cliques_scored = []
        for clique in cliques:
            n_views = len(clique)
            total_score = sum(scores[i] for i in clique)
            cliques_scored.append((n_views, total_score, clique))

        cliques_scored.sort(reverse=True, key=lambda x: (x[0], x[1]))

        used_detections = set()
        results = []

        for n_views, total_score, clique in cliques_scored:

            # Skip if any detection already used
            if any(d in used_detections for d in clique):
                continue

            # Build observation tensor for this clique
            obs = xp.full((self.nb_cams, 1, 2), xp.nan, dtype=xp_float)
            weights = xp.zeros((self.nb_cams, 1), dtype=xp_float)

            for det_idx in clique:
                c = int(cam_ids[det_idx])
                obs = set_at(obs, (c, 0), undist[det_idx])
                weights = set_at(weights, (c, 0), xp.asarray(scores[det_idx]))

            # Triangulate
            pt3d, reproj = self._triangulate_reproj_wrapper(obs, weights)

            # Check for degenerate triangulation
            pt3d_np = np.asarray(pt3d[0])
            if np.any(np.isnan(pt3d_np)):
                continue

            # Compute reprojection error against raw (distorted) coordinates
            raw_obs = xp.full((self.nb_cams, 1, 2), xp.nan, dtype=xp_float)
            for det_idx in clique:
                c = int(cam_ids[det_idx])
                raw_obs = set_at(raw_obs, (c, 0), raw[det_idx])

            diff = raw_obs - reproj
            sq_err = xp.sum(xp.square(diff), axis=-1)  # (n_cams, 1)
            valid_mask = ~xp.isnan(raw_obs[..., 0])
            n_valid = int(xp.sum(valid_mask))

            total_sq_err = float(xp.sum(xp.where(valid_mask, sq_err, 0.0)))
            rmse = np.sqrt(total_sq_err / max(n_valid, 1))

            # Filter by reprojection error
            if rmse >= self.reproj_thresh:
                continue

            # Accept point
            used_detections.update(clique)

            cam_mask = np.uint64(0)
            for det_idx in clique:
                cam_mask |= np.uint64(1 << int(cam_ids[det_idx]))

            results.append({
                'position': pt3d_np,
                'error': float(rmse),
                'n_views': n_views,
                'detections': clique,
                'cam_mask': cam_mask,
                'frame': int(frame_ids[clique[0]])
            })

        # print(f"  Points by view count: {dict(Counter(r['n_views'] for r in results))}")

        return results

    def _get_rays(
            self,
            inputs: Dict[str, np.ndarray],
            undist: xp.ndarray,
            orphan_mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Generate 3D rays for unused detections."""

        if not np.any(orphan_mask):
            return {}

        pts = undist[orphan_mask]
        cams = inputs['cam_ids'][orphan_mask]

        n_orphans = int(np.sum(orphan_mask))
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

    calib_dir = BASE_DIR / PREFIX / 'calibration'
    input_dir = BASE_DIR / PREFIX / 'inputs' / 'tracking'
    output_dir = BASE_DIR / PREFIX / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    rig_file = calib_dir / 'camera_rig.toml'
    soup_file = output_dir / f"soup2_session{SESSION}.pkl"

    rig = CameraRig.load(rig_file)
    df = fileio.load_session(input_dir, session=SESSION)
    skeleton = SkeletonTopology.from_sleap(input_dir)

    reconstructor = Reconstructor(
        rig=rig,
        keypoint_names=skeleton.keypoints,  # TODO: not really needed (just for the soup metadata)
        min_views=2,
        epipolar_threshold=10.0,
        reprojection_threshold=5.0
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