from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Dict, Any
import json
import yaml
import numpy as np
import polars as pl

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from .schemas import validate_dataframe, add_optional_columns

if TYPE_CHECKING:
    from mokap.pose_reconstruction.skeleton import Skeleton, SkeletonStats
    from mokap.pose_reconstruction.datatypes import PointSoup


def load_config(path: Union[str, Path] = 'config.yaml') -> Dict:
    """Load a YAML configuration file."""

    path = Path(path)

    yaml_file = path.with_suffix('.yaml')
    yml_file = path.with_suffix('.yml')

    if not yaml_file.exists() and not yml_file.exists():
        print('[WARN] Config file not found. Defaulting to example config.')
        path = Path('config_example.yaml')

    elif yaml_file.exists():
        path = yaml_file

    else:
        path = yml_file

    return yaml.safe_load(path.open('r'))


def load_dataframe(
    path: Union[str, Path],
    schema_name: Optional[str] = None,
    validate: bool = True
) -> pl.DataFrame:
    """
    Load a DataFrame from parquet or CSV.

    Args:
        path: Path to .parquet or .csv file
        schema_name: Schema to validate against ('Points2D', 'Points3D', 'Tracks3D')
        validate: Whether to validate (raises on missing required columns)
    """
    path = Path(path)

    if path.suffix == ".parquet":
        df = pl.read_parquet(path)
    elif path.suffix == ".csv":
        df = pl.read_csv(path, comment_prefix="#")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if schema_name and validate:
        validate_dataframe(df, schema_name)
        df = add_optional_columns(df, schema_name)

    return df


def load_point_soup(
    path: Union[str, Path],
    keypoint_names: List[str],
    camera_names: List[str],
) -> 'PointSoup':
    """
    Load a PointSoup from parquet/CSV, or pickle (legacy).

    Args:
        path: Path to data file
        keypoint_names: Ordered keypoint names for index mapping
        camera_names: Ordered camera names for mask decoding
    """
    from .converters import dataframe_to_soup

    path = Path(path)

    # Legacy pickle support
    if path.suffix == ".pkl":
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

    df = load_dataframe(path, schema_name='Points3D', validate=True)
    return dataframe_to_soup(df, keypoint_names, camera_names)


def load_skeleton_toml(path: Union[str, Path]) -> 'Skeleton':
    """
    Load a skeleton definition from a TOML file.

    Args:
        path: Path to .toml file
    """
    from mokap.pose_reconstruction.skeleton import Skeleton, SkeletonMetadata

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Skeleton file not found: {path}")

    data = tomllib.load(path.open('rb'))

    # Parse metadata
    meta = data.get('metadata', {})
    ref_bone = meta.get('reference_bone')

    if ref_bone and isinstance(ref_bone, list) and len(ref_bone) == 2:
        ref_bone = tuple(ref_bone)
    else:
        ref_bone = None

    metadata = SkeletonMetadata(
        species=meta.get('species', 'unknown'),
        common_name=meta.get('common_name', ''),
        skeleton_type=meta.get('skeleton_type', 'articulated'),
        reference_bone=ref_bone,
        units=meta.get('units', 'mm'),
        notes=meta.get('notes', ''),
        version=meta.get('version', '1.0'),
        created_date=meta.get('created_date', ''),
    )

    # Parse keypoints
    kp_section = data.get('keypoints', {})
    keypoints = kp_section.get('names', [])
    if not keypoints:
        raise ValueError("Skeleton TOML must have [keypoints] names = [...]")

    # Parse skeleton tree and derive bones
    skeleton_tree = data.get('skeleton', {})
    bones = _bones_from_tree(skeleton_tree, keypoints)

    # Parse symmetry
    sym_section = data.get('symmetry', {})
    symmetry_raw = sym_section.get('pairs', [])
    symmetry = [tuple(pair) for pair in symmetry_raw if len(pair) == 2]

    return Skeleton(
        keypoints=keypoints,
        bones=bones,
        symmetry=symmetry,
        name=metadata.common_name or metadata.species,
        metadata=metadata
    )


def _bones_from_tree(
    skeleton_tree: dict,
    keypoints: List[str]
) -> List[Tuple[str, str]]:
    """Derive bone list from parent->children tree structure."""

    bones = []
    seen = set()

    for parent, children in skeleton_tree.items():
        if parent not in keypoints:
            continue
        for child in children:
            if child not in keypoints:
                continue
            bone = tuple(sorted([parent, child]))
            if bone not in seen:
                bones.append((parent, child))
                seen.add(bone)

    return bones


def load_skeleton_sleap(path: Union[str, Path]) -> 'Skeleton':
    """
    Load skeleton topology from a SLEAP project file.

    Args:
        path: Path to .slp file or directory containing one

    Returns:
        Skeleton instance
    """
    from mokap.pose_reconstruction.skeleton import Skeleton, SkeletonMetadata
    import sleap_io

    path = Path(path)

    if path.is_dir():
        slp_files = list(path.glob('*.slp'))
        if not slp_files:
            raise FileNotFoundError(f"No .slp files found in {path}")
        path = slp_files[0]

    slp = sleap_io.load_file(str(path))

    keypoints = list(slp.skeleton.node_names)
    bones = list(slp.skeleton.edge_names)
    symmetry = list(slp.skeleton.symmetry_names) if slp.skeleton.symmetry_names else []

    name = slp.skeleton.name if hasattr(slp.skeleton, 'name') else 'SLEAP skeleton'

    return Skeleton(
        keypoints=keypoints,
        bones=bones,
        symmetry=symmetry,
        name=name,
        metadata=SkeletonMetadata(
            species='unknown',
            common_name=name,
            skeleton_type='articulated',
        )
    )


def load_skeleton_stats(
    path: Union[str, Path],
    skeleton: 'Skeleton'
) -> 'SkeletonStats':
    """
    Load learned skeleton statistics from JSON.

    Args:
        path: Path to stats.json
        skeleton: Skeleton instance (required for reconstruction)

    Returns:
        SkeletonStats instance
    """
    from mokap.pose_reconstruction.skeleton import (
        SkeletonStats, BoneStats, KeypointDynamics, Bone
    )

    path = Path(path)
    data = json.loads(path.read_text())

    stats = SkeletonStats(skeleton)

    # Load anatomy
    if 'anatomy' in data and data['anatomy']:
        anat = data['anatomy']
        if 'reference_bone' in anat:
            stats.reference_bone = Bone.from_key(anat['reference_bone'])
            stats.reference_length_world = anat.get('reference_length_world', 1.0)

            for k, v in anat.get('bones', {}).items():
                try:
                    bone = Bone.from_key(k)
                    if bone in skeleton:
                        bs = BoneStats.from_dict(v)
                        if bs.length_world is None:
                            bs.length_world = bs.ratio_length * stats.reference_length_world
                        stats.bone_stats[bone] = bs
                except (KeyError, ValueError):
                    pass

    # Load dynamics
    if 'dynamics' in data and data['dynamics']:
        for k, v in data['dynamics'].items():
            if k in skeleton.keypoints:
                stats.keypoint_dynamics[k] = KeypointDynamics.from_dict(v)

    return stats


def load_tracks(
    path: Union[str, Path],
    validate: bool = True
) -> pl.DataFrame:
    """
    Load tracking results from file.

    Args:
        path: Path to .parquet or .csv file
        validate: Whether to validate schema

    Returns:
        DataFrame with Tracks3D schema
    """
    return load_dataframe(path, schema_name='Tracks3D', validate=validate)


def load_detections_sleap(path: Union[str, Path]) -> pl.DataFrame:
    """
    Load 2D detections from a SLEAP project file.

    Args:
        path: Path to .slp file

    Returns:
        DataFrame with Points2D schema
    """
    import sleap_io

    path = Path(path)
    slp_content = sleap_io.load_file(str(path))

    source_files = [Path(v.filename) for v in slp_content.videos]
    camera_names = set(f.stem.split('_')[-2] for f in source_files)
    sessions = set(f.stem.split('_')[-1] for f in source_files)

    dfs = []
    for session in sessions:
        for cam_name in camera_names:
            df = _sleap_to_polars(slp_content, cam_name, session)
            if len(df) > 0:
                dfs.append(df)

    if not dfs:
        from .schemas import empty_dataframe
        return empty_dataframe('Points2D')

    return pl.concat(dfs)


def _sleap_to_polars(slp_content, camera_name: str, session: str) -> pl.DataFrame:
    """Convert SLEAP content for a specific camera/session to Polars DataFrame."""

    keypoint_names = slp_content.skeleton.node_names
    rows = []

    for frame_content in slp_content.labeled_frames:
        source_video = Path(frame_content.video.filename)
        if camera_name not in source_video.stem or str(session) not in source_video.stem:
            continue

        frame_idx = frame_content.frame_idx
        for i, instance in enumerate(frame_content.instances):
            is_manual = instance in frame_content.user_instances
            track_name = instance.track.name if instance.track else f'instance_{i}'

            for kp_idx, node in enumerate(instance.skeleton.nodes):
                point_data = instance.points[kp_idx]

                x, y = np.nan, np.nan
                score = 0.0

                if 'visible' not in point_data.dtype.names or point_data['visible']:
                    if 'xy' in point_data.dtype.names:
                        x, y = point_data['xy']

                    if 'score' in point_data.dtype.names:
                        score = point_data['score']
                    else:
                        score = 1.0 if is_manual else 0.0

                rows.append({
                    "camera": camera_name,
                    "frame": frame_idx,
                    "instance_id": track_name,
                    "keypoint": keypoint_names[kp_idx],
                    "x": float(x),
                    "y": float(y),
                    "score": float(score),
                    "source": "sleap",
                })

    if not rows:
        return pl.DataFrame()

    return pl.from_dicts(rows)


def load_session(
    path: Union[str, Path],
    session: Any = ''
) -> pl.DataFrame:
    """
    Load a tracking session, checking for cached parquet first.

    Searches for SLEAP (.slp) and CSV files matching the session,
    merges them, and caches the result as parquet.

    Args:
        path: Path to session file or parent directory
        session: Session identifier (extracted from filename if not provided)

    Returns:
        DataFrame with Points2D schema
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Can't find {path.stem}!")

    if path.is_file():
        parent_folder = path.parent
        session = path.name.split('.')[0].split('_')[-1]
    else:
        parent_folder = path

    # Check cache
    cache_file = parent_folder / f"session{session}_tracking.parquet"
    if cache_file.exists():
        print(f"Loading cached data from: {cache_file.name}")
        return pl.read_parquet(cache_file)

    # Find matching files
    files = sorted(
        p.resolve()
        for p in parent_folder.glob(f'**/*session{session}*')
        if p.suffix in {'.csv', '.slp'}
    )

    if not files:
        raise FileNotFoundError(
            f"Can't find any tracking files for session '{session}' in {parent_folder}!"
        )

    dfs = []
    loaded_slp, loaded_csv = 0, 0

    for f in files:
        if f.suffix == '.slp' and 'predictions' in f.stem:
            dfs.append(load_detections_sleap(f))
            loaded_slp += 1
        elif f.suffix == '.csv' and 'predictions' in f.stem:
            dfs.append(pl.read_csv(f, separator=','))
            loaded_csv += 1

    if loaded_slp + loaded_csv == 0:
        print("No files loaded...")
    else:
        parts = []
        if loaded_slp > 0:
            parts.append(f'{loaded_slp} SLEAP slp')
        if loaded_csv > 0:
            parts.append(f'{loaded_csv} SLEAP csv')
        print(f"Loaded {' and '.join(parts)} files.")

    merged_df = _merge_detection_dfs(dfs, reset_tracks=True)

    # Cache for next time
    if not merged_df.is_empty():
        print(f"Saving to cache: {cache_file.name}")
        merged_df.write_parquet(cache_file)

    return merged_df


def _merge_detection_dfs(
    dfs: List[pl.DataFrame],
    reset_tracks: bool = True
) -> pl.DataFrame:
    """Merge and finalise a list of detection DataFrames."""

    if not dfs:
        from .schemas import empty_dataframe
        return empty_dataframe('Points2D')

    merged = pl.concat(dfs)

    if reset_tracks:
        print("Creating globally unique track IDs...")
        merged = merged.with_columns(
            pl.concat_str(['camera', 'instance_id']).alias('instance_id')
        )

    # Ensure consistent column order
    final_cols = ['camera', 'frame', 'instance_id', 'keypoint', 'x', 'y', 'score']
    available = [c for c in final_cols if c in merged.columns]

    return merged.select(available).sort(['camera', 'frame', 'instance_id'])