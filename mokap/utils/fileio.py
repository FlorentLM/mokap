import re
from pathlib import Path
from typing import Union, List, Dict
import cv2
import yaml
import toml
import numpy as np
import pandas as pd
import polars as pl


##

def exists_check(path):
    """
    Checks if a file or folder of the given name exists. If so, create a suffixed version of the name
    in a smart way. Returns the new, safe to use, name.
    """
    i = 2
    while path.exists():
        if bool(re.match('.+_[0-9]+$', path.stem)):
            # ends with a '_X' number so let's check if there is also a non-suffixed siblings
            parts = path.stem.split('_')
            original_name = ('_').join(parts[:-1])
            suffix = int(parts[-1])

            if (path.parent / f"{original_name}{path.suffix}").exists():
                new_name = f"{original_name}_{suffix + 1}{path.suffix}"
            else:
                new_name = f"{path.stem}_{i}{path.suffix}"
        else:
            new_name = f"{path.stem}_{i}{path.suffix}"

        path = path.parent / new_name
        i += 1
    return path


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


def rm_if_empty(path):
    path = Path(path)
    if not path.exists():
        # if it doesn't exist, nothing to do.
        return
    else:

        # if it exists
        if not any(path.iterdir()):
            # ...and already empty, delete it, done.
            path.rmdir()

        # if not empty, recursively check again
        else:
            for f in path.glob('*'):
                if f.is_file():
                    return
                rm_if_empty(f)


def clean_root_folder(path):
    path = Path(path)
    [rm_if_empty(f) for f in path.glob('*') if f.is_dir()]
    print(f"Cleaned {path}")


def natural_sort_key(s):
    _nsre = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]


def read_parameters(filepath, camera_name=None):
    filepath = Path(filepath)
    if filepath.is_dir():
        filepath = filepath / 'parameters.toml'

    if filepath.is_file():
        with open(filepath, 'r') as f:
            data = toml.load(f)
    else:
        raise FileNotFoundError(f"File not found: {filepath}")

    if camera_name is not None:
        if camera_name in data:
            return {k: np.array(v).squeeze() for k, v in data[camera_name].items()}
        else:
            raise Exception(f'No camera named {camera_name} in {filepath}')
    else:
        for cam_name, cam_data in data.items():
            data[cam_name] = {k: np.array(v).squeeze() for k, v in cam_data.items()}
        return data


def load_skeleton_SLEAP(slp_path, symmetry=False, indices=False):
    import sleap_io

    slp_path = Path(slp_path)

    if slp_path.is_dir():
        slp_content = sleap_io.load_file(next(slp_path.glob('*.slp')))
    else:
        slp_content = sleap_io.load_file(slp_path)

    keypoints = slp_content.skeleton.node_names
    bones = slp_content.skeleton.edge_inds if indices else slp_content.skeleton.edge_names
    symmetry_names = slp_content.skeleton.symmetry_names

    # TODO: this is for backward-compatibility, but we prob want to always return symmetry
    if symmetry:
        return keypoints, bones, symmetry_names
    else:
        return keypoints, bones


def SLP_to_pandas(slp_content, camera_name=None, session=None):
    def instance_to_row(instance, is_manual):

        original_track = instance.track.name if instance.track else ''
        instance_score = float(instance.score) if hasattr(instance, 'score') else int(is_manual)
        tracking_score = float(instance.tracking_score) if hasattr(instance, 'tracking_score') else int(is_manual)

        values = []
        for i, node in enumerate(instance.skeleton.nodes):
            # if not instance.points[node].visible:
            if not instance.points[i]['visible']:
                x = np.nan
                y = np.nan
                s = 0.0
            else:
                # x = float(instance.points[node].x) if hasattr(instance.points[node], 'x') else np.nan
                # y = float(instance.points[node].y) if hasattr(instance.points[node], 'y') else np.nan
                # s = float(instance.points[node].score) if hasattr(instance.points[node], 'score') else 1.0
                try:
                    x, y = instance.points[i]['xy']
                    s = instance.points[i]['score']
                except:
                    x = y = np.nan
                    s = 1.0
                x, y, s = float(x), float(y), float(s)

            values.extend([x, y, s])
        return values + [instance_score, tracking_score, original_track]

    keypoints = slp_content.skeleton.node_names
    columns = (['camera.', 'frame.']
               + [f"{k}.{a}" for k in keypoints for a in ['x', 'y', 'score']]
               + ['comments.instance_score', 'comments.tracking_score', 'comments.instance'])

    rows = []
    for frame_content in slp_content.labeled_frames:
        source_video = Path(frame_content.video.filename)
        if camera_name is None or camera_name in source_video.stem:  # if name is not passed, assume we load everything
            if session is None or str(session) in source_video.stem:
                for i, instance in enumerate(frame_content.instances):
                    is_manual = instance in frame_content.user_instances
                    row = instance_to_row(instance, is_manual)
                    if row[-1] == '':
                        row[-1] = f'instance_{i}'
                    if session is not None:
                        row[-1] = f"{session}_{row[-1]}"  # prepend session in the track nb
                    row = [camera_name, frame_content.frame_idx + 1] + row
                    rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.columns = pd.MultiIndex.from_tuples([col.split('.') for col in df.columns])

    return df

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


def SLP_to_csv(slp_path, output_csv_path=None):
    """ Convert a SLEAP prediction .slp file to a .csv file """

    slp_path = Path(slp_path)

    if output_csv_path is None:
        output_csv_path = slp_path.parent / (slp_path.stem + '.csv')
    else:
        output_csv_path = Path(output_csv_path)

    if output_csv_path.exists():
        print(f"\n{output_csv_path} exists, skipping.")
        return

    try:
        df = read_SLEAP(slp_path, to_polars=False)
        df.to_csv(output_csv_path, index=False)
        if output_csv_path.exists():
            print(f"\nCSV file saved to: {output_csv_path}")
        else:
            print(f"\nError writing {output_csv_path}")

    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")
    except Exception as e:
        print(f"\nUnexpected error processing {slp_path}: {e}")


def read_SLEAP(slp_path, to_polars=True):
    import sleap_io

    slp_path = Path(slp_path)
    slp_content = sleap_io.load_file(slp_path.as_posix())

    list_of_dfs = []

    source_files = [Path(v.filename) for v in slp_content.videos]
    cameras_names = set(f.stem.split('_')[-2] for f in source_files)
    sessions = set(f.stem.split('_')[-1] for f in source_files)

    for session in sessions:
        for cam_name in cameras_names:
            if to_polars:
                df = SLP_to_polars(slp_content, cam_name, session)
            else:
                df = SLP_to_pandas(slp_content, cam_name,
                                   session)  # This particular camera / session might not exist, so
            if not len(df) == 0:  # in that case the df is empty, we just skip it
                list_of_dfs.append(df)

    if to_polars:
        return pl.concat(list_of_dfs) if list_of_dfs else pl.DataFrame()
    else:
        return merge_pandas_dfs(list_of_dfs)


def load_session(path, session='', use_polars=True):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Can't find {path.stem}!")

    if path.is_file():
        parent_folder = path.parent
        session = path.name.split('.')[0].split('_')[-1]
    else:
        parent_folder = path

    # Check if the cached parquet file exists
    if use_polars:
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
            dfs.append(read_SLEAP(f, to_polars=use_polars))
            loaded_slp += 1

        if f.suffix == '.csv' and 'predictions' in f.stem:
            if use_polars:
                dfs.append(pl.read_csv(f, separator=','))
            else:
                dfs.append(pd.read_csv(f, sep=','))
            loaded_csv += 1

        # TODO - Add loaders for DLC files

    if loaded_slp + loaded_csv == 0:
        print(f'No files loaded...')
    else:
        slp_txt = f'{loaded_slp} SLEAP slp' if loaded_slp > 0 else ''
        csv_txt = f'{loaded_csv} SLEAP csv' if loaded_csv > 0 else ''
        and_txt = ' and ' if (loaded_slp > 0 and loaded_csv > 0) else ''
        print(f'Loaded {slp_txt}{and_txt}{csv_txt} files.')

    if use_polars:
        merged_df = merge_polars_dfs(dfs, reset_tracks=True)

        # Save the processed data to the Parquet cache for the next time
        if not merged_df.is_empty():
            print(f"Saving processed data to cache for future use: {parquet_cache_file.name}")
            merged_df.write_parquet(parquet_cache_file)

    else:
        merged_df = merge_pandas_dfs(dfs, reset_tracks=True)

    return merged_df


def merge_pandas_dfs(list_of_dfs, reset_tracks=True):
    list_of_dfs = list_of_dfs.copy()

    if reset_tracks:
        last_nb_tracks = 0
        for df in list_of_dfs:
            track_ids = df[('comments', 'instance')].factorize()[0] + last_nb_tracks
            last_nb_tracks += np.unique(track_ids).shape[0]
            df['track'] = track_ids

    multiview_df = pd.concat(list_of_dfs, join='outer')

    if reset_tracks:
        if 'track' in multiview_df.index.names:
            multiview_df = multiview_df.reset_index('track', drop=True)  # Reset tracks: get rid of the old ones

    if set(multiview_df.index.names) == {None}:
        multiview_df = multiview_df.set_index(['camera', 'track', 'frame'])
    else:
        multiview_df = multiview_df.reset_index().set_index(['camera', 'track', 'frame'])

    # Set the cameras level as a categorical
    multiview_df.index = multiview_df.index.set_levels(
        pd.CategoricalIndex(multiview_df.index.levels[0],
                            categories=sorted(multiview_df.index.levels[0]), ordered=True), level=0)

    # And apply the sorted categorical index for the cameras
    multiview_df = multiview_df.sort_index()

    return multiview_df


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


def sort_multiview_df(in_df, cameras_order=None, keypoints_order=None):
    df = in_df.copy()

    if keypoints_order is None:
        keypoints_order = df.xs('score', level=1, axis=1).columns

    second_level_order = ['x', 'y', 'score', 'disp']

    desired_order = [
        (kp, measures)
        for kp in keypoints_order
        for measures in second_level_order
        if (kp, measures) in df.columns
    ]

    other_columns = [
        ('centroid', 'x'),
        ('centroid', 'y'),
        ('centroid', 'disp'),
        ('comments', 'tracking_score'),
        ('comments', 'instance_score'),
        ('comments', 'instance'),
    ]

    desired_order += [col for col in other_columns if col in df.columns]

    # Apply the columns order
    df = df.reindex(columns=pd.MultiIndex.from_tuples(desired_order))

    # Reorder the levels themselves to the preferred one
    df = df.reorder_levels(['camera', 'track', 'frame'])

    if cameras_order is None:
        # If no custom ordering is passsed we sort alphabetically
        cameras_order = sorted(df.index.levels[0])

    # Set the cameras index as a categorical
    df.index = df.index.set_levels(
        pd.CategoricalIndex(df.index.levels[0],
                            categories=cameras_order, ordered=True), level=0)

    # And apply the sorted categorical index for the cameras
    df_ordered = df.sort_index()

    return df_ordered