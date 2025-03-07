import re
from pathlib import Path
import yaml
import toml
import numpy as np
import pandas as pd

##

COMPRESSED = {'codec': 'libx265',
              'params': '-crf 12 -preset veryfast',
              'ftype': 'mp4'}

LOSSLESS_1 = {'codec': 'ffv1',
              'params': '-level 3',
              'ftype': 'avi'}

LOSSLESS_2 = {'codec': 'libx265',
              'params': '-x265-params lossless=1 -preset veryfast',
              'ftype': 'mp4'}

ENCODE_FORMAT = COMPRESSED


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


def read_config(config_file='config.yml'):
    config_file = Path(config_file)

    if not config_file.exists():
        print('[WARN] Config file not found. Defaulting to example config.')
        config_file = Path('config_example.yaml')

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


def toml_formatter(dictionary):
    # Remove trailing commas
    toml_str = toml.dumps(dictionary).replace(',]', ' ]')
    # Add indents (yes this one-liner is atrocious)
    lines = [l.replace('], [', f'],\n{"".ljust(len(l.split("=")[0]) + 4)}[') for l in toml_str.splitlines()]
    toml_str_formatted = '\n'.join(lines)
    return toml_str_formatted


def intrinsics_to_dict(camera_matrix, dist_coeffs, errors=None):
    intrinsics_dict = {'camera_matrix': np.array(camera_matrix).squeeze().tolist(),
                       'dist_coeffs': np.array(dist_coeffs).squeeze().tolist()}
    if errors is not None:
        errors_arr = np.array(errors)
        if not np.all(errors_arr == np.inf):
            intrinsics_dict['errors'] = errors_arr.tolist()
    return intrinsics_dict


def extrinsics_to_dict(rvec, tvec):
    extrinsics_dict = {'rvec': np.array(rvec).squeeze().tolist(),
                       'tvec': np.array(tvec).squeeze().tolist()}
    return extrinsics_dict


def file_writer(mode, filepath, camera_name, *args):
    filepath = Path(filepath)
    if filepath.is_dir():
        filepath = filepath / 'parameters.toml'

    if filepath.is_file():
        newfile = False
        with open(filepath, 'r') as f:
            data = toml.load(f)
    else:
        newfile = True
        data = {}

    if mode.lower() == 'intrinsics':
        new_dict = intrinsics_to_dict(*args)
    elif mode.lower() == 'extrinsics':
        new_dict = extrinsics_to_dict(*args)
    else:
        raise AttributeError('File writing mode must be intrinsics or extrinsics!')

    if camera_name in data:
        update = True
        data[camera_name] = data[camera_name] | new_dict
    else:
        update = False
        data[camera_name] = new_dict

    with open(filepath, 'w') as f:
        f.write(toml_formatter(data))

    if update:
        print(f'{mode.title()} for camera {camera_name} updated in {filepath}')
    else:
        if newfile:
            print(f'{mode.title()} for camera {camera_name} written to {filepath}')
        else:
            print(f'{mode.title()} for camera {camera_name} added to {filepath}')


def write_intrinsics(filepath, camera_name, camera_matrix, dist_coeffs, errors=None):
    file_writer('intrinsics', filepath, camera_name, camera_matrix, dist_coeffs, errors)


def write_extrinsics(filepath, camera_name, rvec, tvec):
    file_writer('extrinsics', filepath, camera_name, rvec, tvec)


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


def load_skeleton_SLEAP(slp_path, indices=False):
    import sleap_io

    slp_path = Path(slp_path)
    slp_content = sleap_io.load_file(slp_path.as_posix())

    if indices:
        return slp_content.skeleton.node_names, slp_content.skeleton.edge_inds
    else:
        return slp_content.skeleton.node_names, slp_content.skeleton.edge_names


def SLP_to_df(slp_content, camera_name=None, session=None):
    def instance_to_row(instance, is_manual):

        original_track = instance.track.name if instance.track else ''
        instance_score = float(instance.score) if hasattr(instance, 'score') else int(is_manual)
        tracking_score = float(instance.tracking_score) if hasattr(instance, 'tracking_score') else int(is_manual)

        values = []
        for node in instance.skeleton.nodes:
            if not instance.points[node].visible:
                x = np.nan
                y = np.nan
                s = 0.0
            else:
                x = float(instance.points[node].x) if hasattr(instance.points[node], 'x') else np.nan
                y = float(instance.points[node].y) if hasattr(instance.points[node], 'y') else np.nan
                s = float(instance.points[node].score) if hasattr(instance.points[node], 'score') else 1.0
            values.extend([x, y, s])
        return values + [instance_score, tracking_score, original_track]

    keypoints = slp_content.skeleton.node_names
    columns = (['camera.', 'frame.']
               + [f"{k}.{a}" for k in keypoints for a in ['x', 'y', 'score']]
               + ['comments.instance_score', 'comments.tracking_score', 'comments.instance'])

    rows = []
    for frame_content in slp_content.labeled_frames:
        source_video = Path(frame_content.video.filename)
        if camera_name in source_video.stem or camera_name is None:  # if name is not passed, assume we load everything
            if session in source_video.stem or session is None:
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
            df = SLP_to_df(slp_content, cam_name, session)  # This particular camera / session might not exist, so
            if not df.empty:  # in that case the df is empty, we just skip it
                list_of_dfs.append(df)
    return merge_multiview_df(list_of_dfs)


def SLEAP_to_csv(slp_path, output_csv_path=None):
    """
        Convert a SLEAP prediction .slp file to a .csv file
    """
    slp_path = Path(slp_path)

    if output_csv_path is None:
        output_csv_path = slp_path.parent / (slp_path.stem + '.csv')
    else:
        output_csv_path = Path(output_csv_path)

    if output_csv_path.exists():
        print(f"\n{output_csv_path} exists, skipping.")
        return

    try:
        df = read_SLEAP(slp_path)
        df.to_csv(output_csv_path, index=False)
        if output_csv_path.exists():
            print(f"\nCSV file saved to: {output_csv_path}")
        else:
            print(f"\nError writing {output_csv_path}")

    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")
    except Exception as e:
        print(f"\nUnexpected error processing {slp_path}: {e}")


def load_session(path, session=''):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Can't find {path.stem}!")

    if path.is_file():
        parent_folder = path.parent
        session = path.name.split('.')[0].split('_')[-1]
    else:
        parent_folder = path

    files_match = sorted(parent_folder.glob(f'*{session}.*'))
    if len(files_match) == 0:
        raise FileNotFoundError(f"Can't find any tracking result files in {parent_folder}!")

    dfs = []
    loaded_slp = 0
    loaded_csv = 0
    for f in files_match:
        parts = f.name.split('.')
        if 'slp' in parts and 'predictions' in parts:
            dfs.append(read_SLEAP(f))
            loaded_slp += 1
        elif 'csv' in parts and 'predictions' in parts:
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

    merged_df = merge_multiview_df(dfs)

    return merged_df


def merge_multiview_df(list_of_dfs, reset_tracks=True):
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

## These functions below are not to be used anymore - will be deleted
#
# def videos_from_frames(input_folder, output_folder=None, delete_source=False, force=False):
#     input_folder = Path(input_folder).resolve()
#
#     if not input_folder.exists():
#         print(f"{input_folder.stem} not found, skipping.")
#         return
#     if (input_folder / 'recording').exists():
#         print(f"Skipping {input_folder.stem} (still recording).")
#         return
#
#     with open(input_folder / 'metadata.json', 'r') as f:
#         metadata = json.load(f)
#
#     cameras = metadata['sessions'][0]['cameras']
#     converted_ok = [False] * len(cameras)
#
#     if output_folder is not None:
#         output_folder = Path(output_folder).resolve()
#         if input_folder.stem not in output_folder.stem:
#             output_folder = output_folder / input_folder.stem
#     else:
#         output_folder = input_folder / 'videos'
#
#     if not output_folder.exists():
#         output_folder.mkdir(parents=True, exist_ok=False)
#
#     for c, cam in enumerate(cameras):
#         cam_folder = input_folder / f"cam{cam['idx']}_{cam['name']}"
#         assert cam_folder.is_dir()
#
#         if (cam_folder / 'converted').exists() and not force:
#             print(f"Skipping {input_folder.stem} (already converted).")
#             return
#
#         files = [f for f in os.scandir(cam_folder) if f.name != 'converted']    # scandir is the fastest method for that
#         nb_frames = len(files)
#
#         fps_raw = metadata['framerate']
#         total_time_sec = sum([session['end'] - session['start'] for session in metadata['sessions']])
#
#         fps_calc = nb_frames / total_time_sec
#
#         stats = f"Framerate:\n" \
#                 f"---------\n" \
#                 f" Theoretical:\n  {fps_raw:.2f} fps\n" \
#                 f" Actual (mean):\n  {fps_calc:.2f} fps\n" \
#                 f" --> Error = {(1 - (fps_calc / fps_raw)) * 100:.2f}%" \
#                 "\n\n" \
#                 f"Duration:\n" \
#                 f"---------\n" \
#                 f" Sessions:\n  {len(metadata['sessions'])}\n" \
#                 f" Total seconds:\n  {total_time_sec:.2f}"
#
#         print(stats)
#
#         savepath = output_folder / f'{input_folder.stem}_{cam_folder.stem}.{ENCODE_FORMAT["ftype"]}'
#
#         with open(output_folder / f'{savepath.stem}_stats.txt', 'w') as st:
#             st.write(stats)
#
#         in_fmt = (cam_folder / files[0].name).suffix
#
#         process = sp.Popen(shlex.split(
#             f'ffmpeg '
#             f'-framerate {fps_calc:.2f} '                                   # framerate
#             f"-pattern_type glob -i '*{in_fmt}' "                           # glob input
#             f'-r {fps_calc:.2f} '                                           # framerate (again)
#             f'-c:v {ENCODE_FORMAT["codec"]} {ENCODE_FORMAT["params"]} '     # video codec
#             f'-pix_fmt gray '                                               # pixel format
#             f'-an '                                                         # no audio
#             f'-fps_mode vfr '                                               # avoid having duplicate frames
#             f'{savepath.as_posix()}'),                                      # output
#         cwd=cam_folder)
#
#         process.wait()
#         process.terminate()
#
#         if savepath.is_file() and savepath.stat().st_size > 0:
#             (cam_folder / 'converted').touch()
#             converted_ok[c] = True
#             print(f'Done creating {savepath.name}.')
#
#             if delete_source:
#                 shutil.rmtree(cam_folder)
#         else:
#             print(f'Error creating {savepath.name}!')
#
#     shutil.copy(input_folder / 'metadata.json', output_folder / 'metadata.json')
#
#     if delete_source and all(converted_ok) and (output_folder / 'metadata.json').is_file():
#         shutil.rmtree(input_folder)
#
#     print(f'Finished creating videos for record {input_folder.stem}.')
#
#
# def reencode_videos(path, name_filter='*', delete_source=False, force=False):
#     path = Path(path).resolve()
#
#     if not path.exists():
#         return
#     if (path / 'recording').exists() and not force:
#         return
#     if (path / 'converted').exists() and not force:
#         return
#
#     print(f"Converting {path.stem}")
#
#     if path.is_file():
#         to_convert = [path]
#         ftype = 'file'
#     else:
#         to_convert = list(path.glob(name_filter))
#         ftype = 'folder contents'
#
#     for f in to_convert:
#
#         output_name = f.stem + f'.{ENCODE_FORMAT["ftype"]}'
#         savepath = f.parent / output_name
#
#         if output_name == f.name:
#             f = f.rename(f.parent / f'{f.stem}_orig{f.suffix}')
#
#         process = sp.Popen(shlex.split(
#             f'ffmpeg -i {f.as_posix()}'
#             f'-c:v {ENCODE_FORMAT["codec"]} {ENCODE_FORMAT["params"]}'
#             f'-pix_fmt gray'
#             f'-an {savepath.as_posix()}'))
#
#         process.wait()
#         process.terminate()
#
#         if savepath.is_file() and savepath.stat().st_size > 0:
#             print(f'Done reencoding {path.name} {ftype}.')
#
#             if delete_source:
#                 f.unlink(missing_ok=True)
#         else:
#             print(f'Error reencoding {path.name} {ftype}!')