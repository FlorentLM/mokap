from pathlib import Path
from typing import List
import yaml
import numpy as np
import polars as pl


##

def read_config(config_file='config.yaml'):
    config_file = Path(config_file)

    yaml_file = config_file.with_suffix('.yaml')
    yml_file = config_file.with_suffix('.yml')

    if not yaml_file.exists() and not yml_file.exists():
        print('[WARN] Config file not found. Defaulting to example config.')
        config_file = Path('config_example.yaml')
    elif yaml_file.exists() and not yml_file.exists():
        config_file = yaml_file
    elif yml_file.exists() and not yaml_file.exists():
        config_file = yml_file

    with open(config_file, 'r') as f:
        config_content = yaml.safe_load(f)

    return config_content


##


def SLP_to_polars(slp_content, camera_name: str, session: str) -> pl.DataFrame:

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

                # check point visibility
                if not 'visible' in point_data.dtype.names or not point_data['visible']:
                    score = 0.0
                else:
                    # point visible, get coordinates
                    if 'xy' in point_data.dtype.names:
                        x, y = point_data['xy']

                    # score presence and manual annotation status
                    if 'score' in point_data.dtype.names:
                        score = point_data['score']
                    else:
                        # if score is missing, its value depends on whether it's a manual annotation or not
                        score = 1.0 if is_manual else 0.0

                rows.append({
                    "camera": camera_name,
                    "frame": frame_idx,
                    "track_id": track_name,
                    "keypoint": keypoint_names[kp_idx],
                    "x": float(x),
                    "y": float(y),
                    "score": float(score)
                })

    if not rows:
        return pl.DataFrame()

    return pl.from_dicts(rows)


def read_SLEAP(slp_path):
    import sleap_io

    slp_path = Path(slp_path)
    slp_content = sleap_io.load_file(slp_path.as_posix())

    list_of_dfs = []

    source_files = [Path(v.filename) for v in slp_content.videos]
    cameras_names = set(f.stem.split('_')[-2] for f in source_files)
    sessions = set(f.stem.split('_')[-1] for f in source_files)

    for session in sessions:
        for cam_name in cameras_names:
            df = SLP_to_polars(slp_content, cam_name, session)
            if not len(df) == 0:  # in that case the df is empty, we just skip it
                list_of_dfs.append(df)

    return pl.concat(list_of_dfs) if list_of_dfs else pl.DataFrame()


def load_session(path, session=''):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Can't find {path.stem}!")

    if path.is_file():
        parent_folder = path.parent
        session = path.name.split('.')[0].split('_')[-1]
    else:
        parent_folder = path

    # Check if the cached parquet file exists
    parquet_cache_file = parent_folder / f"session{session}_tracking.parquet"
    if parquet_cache_file.exists():
        print(f"Loading cached data from: {parquet_cache_file.name}")
        return pl.read_parquet(parquet_cache_file)

    files_match = sorted(
        p.resolve() for p in parent_folder.glob(f'**/*session{session}*') if p.suffix in {'.csv', '.slp'})
    if not files_match:
        raise FileNotFoundError(f"Can't find any tracking result files for session '{session}' in {parent_folder}!")

    dfs = []
    loaded_slp, loaded_csv = 0, 0

    for f in files_match:

        if f.suffix == '.slp' and 'predictions' in f.stem:
            dfs.append(read_SLEAP(f))
            loaded_slp += 1

        if f.suffix == '.csv' and 'predictions' in f.stem:
            dfs.append(pl.read_csv(f, separator=','))
            loaded_csv += 1

        # TODO - Add loaders for DLC files

    if loaded_slp + loaded_csv == 0:
        print(f'No files loaded...')
    else:
        slp_txt = f'{loaded_slp} SLEAP slp' if loaded_slp > 0 else ''
        csv_txt = f'{loaded_csv} SLEAP csv' if loaded_csv > 0 else ''
        and_txt = ' and ' if (loaded_slp > 0 and loaded_csv > 0) else ''
        print(f'Loaded {slp_txt}{and_txt}{csv_txt} files.')

    merged_df = merge_polars_dfs(dfs, reset_tracks=True)

    # Save the processed data to the Parquet cache for the next time
    if not merged_df.is_empty():
        print(f"Saving processed data to cache for future use: {parquet_cache_file.name}")
        merged_df.write_parquet(parquet_cache_file)

    return merged_df


def merge_polars_dfs(list_of_dfs: List[pl.DataFrame], reset_tracks: bool = True) -> pl.DataFrame:
    """ Merges a list of Polars DataFrames into a single one and finalizes it """

    if not list_of_dfs:
        return pl.DataFrame()

    multiview_df = pl.concat(list_of_dfs)

    if reset_tracks:
        # unique track ID by combining the camera name and the original track name from SLEAP
        print("Creating globally unique track IDs...")
        multiview_df = multiview_df.with_columns(
            pl.concat_str(['camera', 'track_id']).alias('global_track_id')
        )

    final_cols = ['camera', 'frame', 'global_track_id', 'keypoint', 'x', 'y', 'score']
    return multiview_df.select(final_cols).sort(['camera', 'frame', 'global_track_id'])