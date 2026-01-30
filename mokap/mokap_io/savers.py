from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, Sequence, Dict, Any
import json
import polars as pl
import pyarrow.parquet as pq

from .schemas import validate_dataframe

if TYPE_CHECKING:
    from mokap.pose_reconstruction.skeleton import Skeleton, SkeletonStats, SkeletonMetadata
    from mokap.pose_reconstruction.datatypes import Tracklet


def save_dataframe(
    dataframe: pl.DataFrame,
    path: Union[str, Path],
    schema_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    validate: bool = True
) -> None:
    """
    Save a DataFrame to parquet or CSV.

    Args:
        dataframe: DataFrame to save
        path: Output path (.parquet or .csv)
        schema_name: Schema to validate against (required if validate=True)
        metadata: Optional dictionary of key-value pairs to embed (Parquet only)
        validate: Whether to validate before saving
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if validate:
        if schema_name is None:
            raise ValueError("Schema name is required when validate=True")
        validate_dataframe(dataframe, schema_name)

    if path.suffix == ".parquet":
        if metadata:
            table = dataframe.to_arrow()
            existing_meta = table.schema.metadata or {}
            combined_meta = {
                **{k: v for k, v in existing_meta.items()},
                **{k.encode('utf-8'): v.encode('utf-8') for k, v in metadata.items()}
            }
            table = table.replace_schema_metadata(combined_meta)
            pq.write_table(table, str(path))
        else:
            dataframe.write_parquet(path)
    elif path.suffix == ".csv":
        if metadata:
            print("[WARN] Metadata ignored when saving to CSV.")
        dataframe.write_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def save_skeleton(
    skeleton: 'Skeleton',
    path: Union[str, Path]
) -> None:
    """
    Save a skeleton definition to a TOML file.

    Args:
        skeleton: Skeleton instance
        path: Output path (.toml)
    """
    from mokap.pose_reconstruction.skeleton import SkeletonMetadata

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    # Get metadata (use defaults if not present)
    if hasattr(skeleton, 'metadata') and skeleton.metadata:
        meta = skeleton.metadata
    else:
        meta = SkeletonMetadata()

    # Metadata section
    lines.append('# Skeleton definition')
    lines.append('')

    lines.append('[metadata]')
    lines.append(f'species = "{meta.species}"')
    lines.append(f'common_name = "{meta.common_name}"')
    lines.append(f'skeleton_type = "{meta.skeleton_type}"')
    if meta.reference_bone:
        rb = meta.reference_bone
        lines.append(f'reference_bone = ["{rb[0]}", "{rb[1]}"]')
    lines.append(f'units = "{meta.units}"')
    if meta.notes:
        lines.append(f'notes = "{meta.notes}"')
    lines.append(f'version = "{meta.version}"')
    if meta.created_date:
        lines.append(f'created_date = "{meta.created_date}"')
    lines.append('')

    # Keypoints section
    lines.append('[keypoints]')
    kp_list = ', '.join(f'"{kp}"' for kp in skeleton.keypoints)
    lines.append(f'names = [{kp_list}]')
    lines.append('')

    # Skeleton tree section
    lines.append('[skeleton]')
    tree = {kp: [] for kp in skeleton.keypoints}
    for bone in skeleton.bones:
        k1, k2 = bone if isinstance(bone, tuple) else (bone.k1, bone.k2)
        if k1 in tree:
            tree[k1].append(k2)

    for kp in skeleton.keypoints:
        children = tree.get(kp, [])
        if children:
            children_str = ', '.join(f'"{c}"' for c in children)
            lines.append(f'{kp} = [{children_str}]')
        else:
            lines.append(f'{kp} = []')
    lines.append('')

    # Symmetry section
    lines.append('[symmetry]')
    if skeleton.symmetry:
        pairs_str = ', '.join(f'["{p[0]}", "{p[1]}"]' for p in skeleton.symmetry)
        lines.append(f'pairs = [{pairs_str}]')
    else:
        lines.append('pairs = []')
    lines.append('')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def save_skeleton_stats(
    stats: 'SkeletonStats',
    path: Union[str, Path],
    merge_existing: bool = True
) -> None:
    """
    Save learned skeleton statistics to JSON.

    Args:
        stats: SkeletonStats instance
        path: Output path (.json)
        merge_existing: If True, merge with existing file (preserves other sections)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data if merging
    existing_data = {}
    if merge_existing and path.exists():
        try:
            existing_data = json.loads(path.read_text())
        except json.JSONDecodeError:
            pass

    # Build current data
    current_data = {'anatomy': {}, 'dynamics': {}}

    # Anatomy section
    if stats.reference_bone:
        current_data['anatomy'] = {
            'reference_bone': stats.reference_bone.to_key(),
            'reference_length_world': stats.reference_length_world,
            'bones': {b.to_key(): s.to_dict() for b, s in stats.bone_stats.items()}
        }

    # Dynamics section
    if stats.keypoint_dynamics:
        current_data['dynamics'] = {
            k: d.to_dict() for k, d in stats.keypoint_dynamics.items()
        }

    final_data = {
        'anatomy': existing_data.get('anatomy', {}),
        'dynamics': existing_data.get('dynamics', {}),
    }

    if stats.reference_bone:
        final_data['anatomy'] = current_data['anatomy']

    if stats.keypoint_dynamics:
        final_data['dynamics'] = current_data['dynamics']

    with open(path, 'w') as f:
        json.dump(final_data, f, indent=2)


def save_tracks(
    tracks: pl.DataFrame,
    path: Union[str, Path],
) -> None:
    """
    Save tracking results to file.

    Args:
        tracks: DataFrame with Tracks3D schema
        path: Output path (.parquet or .csv)
    """
    save_dataframe(tracks, path, schema_name='Tracks3D', validate=True)


def append_tracks(
    tracks: Sequence['Tracklet'],
    frame_idx: int,
    path: Union[str, Path],
) -> None:
    """
    Append tracking results to an existing file (or create if it doesn't exist).

    Args:
        tracks: Sequence of Tracklet instances
        frame_idx: Current frame index
        path: Output path (.parquet)
    """
    from .converters import tracklets_to_dataframe

    path = Path(path)
    new_df = tracklets_to_dataframe(tracks, frame_idx)

    if path.exists():
        existing_df = pl.read_parquet(path)
        combined_df = pl.concat([existing_df, new_df])
        combined_df.write_parquet(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_df.write_parquet(path)