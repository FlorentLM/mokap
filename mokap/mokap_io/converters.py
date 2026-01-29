from typing import TYPE_CHECKING, Dict, List, Sequence
import numpy as np
import polars as pl

from .schemas import empty_dataframe

if TYPE_CHECKING:
    from mokap.pose_reconstruction.datatypes import PointSoup, Tracklet


def df_to_soup(
    df: pl.DataFrame,
    keypoint_names: Sequence[str],
    camera_names: Sequence[str],
) -> 'PointSoup':
    """
    Convert a Points3D DataFrame to a PointSoup runtime object.

    Args:
        df: DataFrame with Points3D schema
        keypoint_names: Ordered keypoint names (for index mapping)
        camera_names: Ordered camera names (for mask decoding)

    Returns:
        PointSoup instance
    """
    from mokap.pose_reconstruction.datatypes import PointSoup

    # Split by status
    points_df = df.filter(pl.col("status") == "reconstructed")
    rays_df = df.filter(pl.col("status") == "ray")

    kp_to_idx = {name: i for i, name in enumerate(keypoint_names)}

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
        keypoint_names=list(keypoint_names),
        camera_names=list(camera_names),
        sort=True,
    )


def soup_to_df(soup: 'PointSoup') -> pl.DataFrame:
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
            "ray_dx": None,
            "ray_dy": None,
            "ray_dz": None,
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
            "reprojection_error": None,
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


def tracklets_to_df(
    tracklets: Sequence['Tracklet'],
    frame_idx: int,
) -> pl.DataFrame:
    """
    Convert a sequence of Tracklet objects to a Tracks3D DataFrame.

    This is the preferred method for serializing tracking output.

    Args:
        tracklets: Sequence of Tracklet instances
        frame_idx: Current frame index (for timestamp)

    Returns:
        DataFrame with Tracks3D schema
    """
    rows = []

    for tracklet in tracklets:
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


def tracklet_records_to_df(
    records: Dict[int, List[dict]]
) -> pl.DataFrame:
    """
    Convert legacy tracklet records dict to Tracks3D DataFrame.

    DEPRECATED: Use tracklets_to_df() with Tracklet objects instead.

    Args:
        records: Dict mapping track_id -> list of per-frame record dicts

    Returns:
        DataFrame with Tracks3D schema
    """
    rows = []

    for track_id, frame_records in records.items():
        for record in frame_records:
            frame_idx = record.get("frame_idx", 0)
            keypoints = record.get("keypoints", {})

            scale = record.get("scale", 1.0)
            anatomical_score = record.get("score", 0.0)
            health = record.get("health", 1.0)
            integrity = record.get("anatomical_integrity", anatomical_score)

            pos_unc = record.get("position_uncertainty", [0, 0, 0])
            uncertainty = sum(pos_unc) if isinstance(pos_unc, (list, tuple)) else float(pos_unc)

            velocity = record.get("velocity", [0, 0, 0])
            vel_x = velocity[0] if len(velocity) > 0 else 0.0
            vel_y = velocity[1] if len(velocity) > 1 else 0.0
            vel_z = velocity[2] if len(velocity) > 2 else 0.0

            for kp_name, position in keypoints.items():
                if isinstance(position, np.ndarray):
                    x, y, z = position.tolist()
                else:
                    x, y, z = position

                rows.append({
                    "track_id": int(track_id),
                    "frame": int(frame_idx),
                    "keypoint": kp_name,
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "confidence": 1.0,
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


def tracks_df_to_numpy(
    df: pl.DataFrame,
    keypoint_order: Sequence[str],
) -> Dict[int, np.ndarray]:
    """
    Convert a Tracks3D DataFrame to numpy arrays for analysis.

    Args:
        df: DataFrame with Tracks3D schema
        keypoint_order: Ordered list of keypoints (defines array columns)

    Returns:
        Dict mapping track_id -> array of shape (n_frames, n_keypoints, 3)
        Missing keypoints are filled with NaN.
    """
    result = {}

    track_ids = df["track_id"].unique().to_list()
    n_keypoints = len(keypoint_order)
    kp_to_idx = {kp: i for i, kp in enumerate(keypoint_order)}

    for track_id in track_ids:
        track_df = df.filter(pl.col("track_id") == track_id)

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


##

# TODO: temporary until Reconstructor uses Polars
def prepare_reconstruction_input(
    df: pl.DataFrame,
    cameras: List[str],
    keypoints: List[str],
) -> dict:
    """
    Convert Points2D DataFrame to flat numpy arrays for the Reconstructor.

    TEMPORARY: Will be removed

    Args:
        df: DataFrame with Points2D schema
        cameras: Ordered list of camera names
        keypoints: Ordered list of keypoint names

    Returns:
        Dict with numpy arrays: frame_indices, kp_type_ids, cam_ids, coords, scores
    """
    df = df.sort(["frame", "keypoint", "camera"])

    cam_map = {cam_name: c for c, cam_name in enumerate(cameras)}
    kp_map = {kp_name: k for k, kp_name in enumerate(keypoints)}

    df = df.with_columns(
        pl.col("keypoint").replace(kp_map).cast(pl.Int16).alias("kp_type_id"),
        pl.col("camera").replace(cam_map).cast(pl.Int8).alias("cam_id"),
    ).sort(
        ["frame", "kp_type_id", "cam_id", "score"],
        descending=[False, False, False, True]
    )

    return {
        "frame_indices": df["frame"].to_numpy(),
        "kp_type_ids": df["kp_type_id"].to_numpy(),
        "cam_ids": df["cam_id"].to_numpy(),
        "coords": df.select(["x", "y"]).to_numpy(),
        "scores": df["score"].to_numpy(),
    }