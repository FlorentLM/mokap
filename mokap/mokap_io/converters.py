from typing import TYPE_CHECKING, Dict, Sequence
import numpy as np
import polars as pl

from .schemas import empty_dataframe

if TYPE_CHECKING:
    from mokap.pose_reconstruction.datatypes import PointSoup, Tracklet


def dataframe_to_soup(
    dataframe: pl.DataFrame,
    keypoints_order: Sequence[str],
    cameras_order: Sequence[str],
) -> 'PointSoup':
    """
    Convert a Points3D DataFrame to a PointSoup runtime object.

    Args:
        dataframe: DataFrame with Points3D schema
        keypoints_order: Ordered keypoint names (for index mapping)
        cameras_order: Ordered camera names (for mask decoding)

    Returns:
        PointSoup instance
    """
    from mokap.pose_reconstruction.datatypes import PointSoup

    # Split by status
    points_df = dataframe.filter(pl.col("status") == "reconstructed")
    rays_df = dataframe.filter(pl.col("status") == "ray")

    kp_to_idx = {name: i for i, name in enumerate(keypoints_order)}

    # Reconstructed points
    if len(points_df) > 0:
        positions = points_df.select(["x", "y", "z"]).to_numpy().astype(np.float32)
        confidences = points_df["confidence"].to_numpy().astype(np.float32)
        frame_indices = points_df["frame"].to_numpy().astype(np.int32)

        kp_indices = np.array(
            [kp_to_idx.get(kp, -1) for kp in points_df["keypoint"].to_list()],
            dtype=np.int16
        )

        if "reprojection_error" in points_df.columns:
            reproj_errors = points_df["reprojection_error"].to_numpy().astype(np.float32)
        else:
            reproj_errors = np.zeros(len(points_df), dtype=np.float32)

        if "camera_mask" in points_df.columns:
            camera_masks = points_df["camera_mask"].to_numpy().astype(np.uint64)
        else:
            camera_masks = np.zeros(len(points_df), dtype=np.uint64)
    else:
        positions = np.empty((0, 3), dtype=np.float32)
        confidences = np.empty(0, dtype=np.float32)
        frame_indices = np.empty(0, dtype=np.int32)
        kp_indices = np.empty(0, dtype=np.int16)
        reproj_errors = np.empty(0, dtype=np.float32)
        camera_masks = np.empty(0, dtype=np.uint64)

    # Rays
    if len(rays_df) > 0 and "ray_dx" in rays_df.columns:
        ray_origins = rays_df.select(["x", "y", "z"]).to_numpy().astype(np.float32)
        ray_directions = rays_df.select(["ray_dx", "ray_dy", "ray_dz"]).to_numpy().astype(np.float32)
        ray_confidences = rays_df["confidence"].to_numpy().astype(np.float32)
        ray_frame_indices = rays_df["frame"].to_numpy().astype(np.int32)
        ray_kp_indices = np.array(
            [kp_to_idx.get(kp, -1) for kp in rays_df["keypoint"].to_list()],
            dtype=np.int16
        )
    else:
        ray_origins = np.empty((0, 3), dtype=np.float32)
        ray_directions = np.empty((0, 3), dtype=np.float32)
        ray_confidences = np.empty(0, dtype=np.float32)
        ray_frame_indices = np.empty(0, dtype=np.int32)
        ray_kp_indices = np.empty(0, dtype=np.int16)

    return PointSoup(
        positions=positions,
        confidences=confidences,
        reprojection_errors=reproj_errors,
        keypoint_indices=kp_indices,
        frame_indices=frame_indices,
        camera_masks=camera_masks,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        ray_confidences=ray_confidences,
        ray_keypoint_indices=ray_kp_indices,
        ray_frame_indices=ray_frame_indices,
        keypoint_names=list(keypoints_order),
        camera_names=list(cameras_order),
        sort=True,
    )


def soup_to_dataframe(soup: 'PointSoup') -> pl.DataFrame:
    """
    Convert a PointSoup runtime object to a Points3D DataFrame.

    Args:
        soup: PointSoup instance

    Returns:
        DataFrame with Points3D schema
    """
    rows = []

    # Reconstructed points
    for i in range(soup.nb_points):
        kp_idx = soup.keypoint_indices[i]
        kp_name = (
            soup.keypoint_names[kp_idx]
            if 0 <= kp_idx < len(soup.keypoint_names)
            else f"kp_{kp_idx}"
        )

        rows.append({
            "frame": int(soup.frame_indices[i]),
            "keypoint": kp_name,
            "x": float(soup.positions[i, 0]),
            "y": float(soup.positions[i, 1]),
            "z": float(soup.positions[i, 2]),
            "confidence": float(soup.confidences[i]),
            "reprojection_error": (
                float(soup.reprojection_errors[i])
                if len(soup.reprojection_errors) > i else None
            ),
            "camera_mask": (
                int(soup.camera_masks[i])
                if len(soup.camera_masks) > i else 0
            ),
            "status": "reconstructed",
            "instance_id": None,
            "ray_dx": np.nan,
            "ray_dy": np.nan,
            "ray_dz": np.nan,
        })

    # Rays
    for i in range(soup.nb_rays):
        kp_idx = soup.ray_keypoint_indices[i]
        kp_name = (
            soup.keypoint_names[kp_idx]
            if 0 <= kp_idx < len(soup.keypoint_names)
            else f"kp_{kp_idx}"
        )

        rows.append({
            "frame": int(soup.ray_frame_indices[i]),
            "keypoint": kp_name,
            "x": float(soup.ray_origins[i, 0]),
            "y": float(soup.ray_origins[i, 1]),
            "z": float(soup.ray_origins[i, 2]),
            "confidence": float(soup.ray_confidences[i]),
            "reprojection_error": np.nan,
            "camera_mask": 0,
            "status": "ray",
            "instance_id": None,
            "ray_dx": float(soup.ray_directions[i, 0]),
            "ray_dy": float(soup.ray_directions[i, 1]),
            "ray_dz": float(soup.ray_directions[i, 2]),
        })

    if not rows:
        return empty_dataframe('Points3D')

    df = pl.from_dicts(rows)

    # Cast to correct types
    df = df.with_columns([
        pl.col("frame").cast(pl.Int32),
        pl.col("x").cast(pl.Float32),
        pl.col("y").cast(pl.Float32),
        pl.col("z").cast(pl.Float32),
        pl.col("confidence").cast(pl.Float32),
        pl.col("reprojection_error").cast(pl.Float32),
        pl.col("camera_mask").cast(pl.UInt64),
        pl.col("ray_dx").cast(pl.Float32),
        pl.col("ray_dy").cast(pl.Float32),
        pl.col("ray_dz").cast(pl.Float32),
    ])

    return df.sort(["frame", "keypoint", "status"])


def tracklets_to_dataframe(
    tracks: Sequence['Tracklet'],
    frame_idx: int,
) -> pl.DataFrame:
    """
    Convert a sequence of Tracklet objects to a Tracks3D DataFrame.

    Args:
        tracks: Sequence of Tracklet instances
        frame_idx: Current frame index (for timestamp)

    Returns:
        DataFrame with Tracks3D schema
    """
    rows = []

    for tracklet in tracks:
        keypoints = tracklet.hypothesis.positions
        scale = tracklet.estimated_scale
        anatomical_score = tracklet.hypothesis.anatomical_score
        health = tracklet.health
        integrity = tracklet.anatomical_integrity

        # Uncertainty (trace of position covariance)
        pos_unc = tracklet.position_uncertainty
        uncertainty = float(np.sum(pos_unc))

        # Velocity
        velocity = tracklet.velocity
        vel_x, vel_y, vel_z = velocity[0], velocity[1], velocity[2]

        # Create a row for each keypoint
        for kp_name, position in keypoints.items():
            if isinstance(position, np.ndarray):
                x, y, z = position.tolist()
            else:
                x, y, z = position

            rows.append({
                "track_id": int(tracklet.track_idx),
                "frame": int(frame_idx),
                "keypoint": kp_name,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "confidence": 1.0,  # TODO: derive from hypothesis nodes
                "scale": float(scale),
                "anatomical_score": float(anatomical_score),
                "health": float(health),
                "integrity": float(integrity),
                "uncertainty": float(uncertainty),
                "velocity_x": float(vel_x),
                "velocity_y": float(vel_y),
                "velocity_z": float(vel_z),
            })

    if not rows:
        return empty_dataframe('Tracks3D')

    df = pl.from_dicts(rows)

    # Cast to correct types
    df = df.with_columns([
        pl.col("track_id").cast(pl.Int32),
        pl.col("frame").cast(pl.Int32),
        pl.col("x").cast(pl.Float32),
        pl.col("y").cast(pl.Float32),
        pl.col("z").cast(pl.Float32),
        pl.col("confidence").cast(pl.Float32),
        pl.col("scale").cast(pl.Float32),
        pl.col("anatomical_score").cast(pl.Float32),
        pl.col("health").cast(pl.Float32),
        pl.col("integrity").cast(pl.Float32),
        pl.col("uncertainty").cast(pl.Float32),
        pl.col("velocity_x").cast(pl.Float32),
        pl.col("velocity_y").cast(pl.Float32),
        pl.col("velocity_z").cast(pl.Float32),
    ])

    return df.sort(["track_id", "frame", "keypoint"])


def tracklets_to_arrays(
    tracks: pl.DataFrame,
    keypoints_order: Sequence[str],
) -> Dict[int, np.ndarray]:
    """
    Convert a Tracks3D DataFrame to numpy arrays for analysis.

    Args:
        tracks: DataFrame with Tracks3D schema
        keypoints_order: Ordered list of keypoints (defines array columns)

    Returns:
        Dict mapping track_id -> array of shape (n_frames, n_keypoints, 3)
        Missing keypoints are filled with NaN.
    """
    result = {}

    track_ids = tracks["track_id"].unique().to_list()
    n_keypoints = len(keypoints_order)
    kp_to_idx = {kp: i for i, kp in enumerate(keypoints_order)}

    for track_id in track_ids:
        track_df = tracks.filter(pl.col("track_id") == track_id)

        frames = track_df["frame"].unique().sort().to_list()
        n_frames = len(frames)
        frame_to_idx = {f: i for i, f in enumerate(frames)}

        positions = np.full((n_frames, n_keypoints, 3), np.nan, dtype=np.float32)

        for row in track_df.iter_rows(named=True):
            f_idx = frame_to_idx[row["frame"]]
            kp_idx = kp_to_idx.get(row["keypoint"])
            if kp_idx is not None:
                positions[f_idx, kp_idx] = [row["x"], row["y"], row["z"]]

        result[track_id] = positions

    return result