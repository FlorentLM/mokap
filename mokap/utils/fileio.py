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


def read_intrinsics(filepath, camera_name=None):

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
            cam_data = data[camera_name]
            cam_data.pop('rvec', None)
            cam_data.pop('tvec', None)
            return {k: np.array(v).squeeze() for k, v in cam_data.items()}
        else:
            raise Exception(f'No camera named {camera_name} in {filepath}')
    else:
        for cam_name, cam_data in data.items():
            cam_data.pop('rvec', None)
            cam_data.pop('tvec', None)
            data[cam_name] = {k: np.array(v).squeeze() for k, v in cam_data.items()}
        return data


def read_extrinsics(filepath, camera_name=None):

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
            cam_data = data[camera_name]
            cam_data.pop('camera_matrix', None)
            cam_data.pop('dist_coeffs', None)
            cam_data.pop('errors', None)
            return {k: np.array(v).squeeze() for k, v in cam_data.items()}
        else:
            raise Exception(f'No camera named {camera_name} in {filepath}')
    else:
        for cam_name, cam_data in data.items():
            cam_data.pop('camera_matrix', None)
            cam_data.pop('dist_coeffs', None)
            cam_data.pop('errors', None)
            data[cam_name] = {k: np.array(v).squeeze() for k, v in cam_data.items()}
        return data


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

slp_path = 'C:/Users/flolm/Desktop/3d_ant_data/240809-1240/inputs/tracking/240809-1240_cam2_banana_session13.predictions.slp'
def read_SLEAP(slp_path):
    import sleap_io

    def instance_to_row(instance):
        track = instance.track.name if instance.track else ''
        score = float(instance.score) if hasattr(instance, 'score') else 0.0
        tracking_score = float(instance.tracking_score) if hasattr(instance, 'tracking_score') else 0.0
        values = np.array([[instance.points[node].x, instance.points[node].y, instance.points[node].score] for node in instance.skeleton.nodes])
        return [track, score, tracking_score] + values.ravel().tolist()

    slp_path = Path(slp_path)
    cam_name = slp_path.stem.split('_')[2]
    slp_content = sleap_io.load_file(slp_path.as_posix())

    keypoints = slp_content.skeleton.node_names

    columns = ['track', 'instance.instance_score', 'instance.tracking_score'] + [f"{k}.{a}" for k in keypoints for a in ['x', 'y', 'score']]
    index = []
    rows = []
    for frame_idx, frame_content in enumerate(slp_content.labeled_frames):
        for instance in frame_content.instances:
            rows.append(instance_to_row(instance))
            index.append(frame_idx)

    df = pd.DataFrame(rows, columns=columns)
    df.insert(1, "frame", index)
    df.insert(0, "camera", cam_name)
    return df


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
    sorted_df = sort_multiview_df(merged_df, cameras_order=None, keypoints_order=None)

    return sorted_df


def merge_multiview_df(list_of_dfs):
    dfs = []

    # TODO - Check if the csv files contain a track ID or not

    last_nb_tracks = 0
    for df in list_of_dfs:
        df['comments.track_sleap'] = df['track']
        df['track'] = df['track'].factorize()[0] + last_nb_tracks + 1
        last_nb_tracks = df['track'].max()
        df.set_index(['frame', 'camera', 'track'], inplace=True)
        dfs.append(df)

    multiview_df = pd.concat(dfs, join='outer')
    multiview_df.index = multiview_df.index.rename({'frame_idx': 'frame'})
    multiview_df.rename(columns={
        'instance.score': 'comments.instance_score',
        'instance.tracking_score': 'comments.tracking_score',
                       }, inplace=True)
    multiview_df.columns = pd.MultiIndex.from_tuples([col.split('.') for col in multiview_df.columns])

    # Tweak the columns order a little
    columns_order = multiview_df.columns
    columns_order = pd.MultiIndex.from_tuples(list(columns_order[-2:]) + list(columns_order[1:-2]) + [columns_order[0]])
    multiview_df = multiview_df.reindex(columns=columns_order)

    return multiview_df


def sort_multiview_df(df, cameras_order=None, keypoints_order=None):
    df = df.reorder_levels(['camera', 'track', 'frame']).sort_index()

    if keypoints_order is None:
        keypoints_order = [col[0] for col in df.columns if col[1] == 'score']
    second_level_order = ['x', 'y', 'score']

    desired_order = [
        (kp, measures)
        for kp in keypoints_order
        for measures in second_level_order
        if (kp, measures) in df.columns
    ]

    ordered_columns = df.reindex(columns=pd.MultiIndex.from_tuples(desired_order))

    if cameras_order is None:
        cameras_order = sorted(ordered_columns.index.levels[0])

    ordered_columns.index = ordered_columns.index.set_levels(
        pd.CategoricalIndex(ordered_columns.index.levels[0],
                            categories=cameras_order,
                            ordered=True), level=0)

    ordered_both = ordered_columns.sort_index(level=['camera', 'track', 'frame'])

    return ordered_both



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
