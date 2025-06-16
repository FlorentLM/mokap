import re
from pathlib import Path
from typing import Union
import cv2
import yaml
import toml
import numpy as np
import pandas as pd


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mokap.utils.datatypes import ChessBoard, CharucoBoard

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

    if slp_path.is_dir():
        slp_content = sleap_io.load_file(next(slp_path.glob('*.slp')))
    else:
        slp_content = sleap_io.load_file(slp_path)

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


def probe_video(video_path: Path | str):
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(video_path.resolve())

    cap = cv2.VideoCapture(video_path.as_posix())
    r, frame = cap.read()
    if not r:
        raise IOError(f"Can't read video {video_path.resolve()}")

    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return frame.shape, nb_frames


def generate_board_svg(board_params:    Union["ChessBoard", "CharucoBoard"],
                       file_path:       Union[Path, str],
                       multi_size:      bool = False,
                       factor:          float = 2.0,
                       dpi:             int = 1200):

    board_rows = board_params.rows
    board_cols = board_params.cols
    sq_l_w = board_params.square_length

    # We check like this to avoid circular imports
    if board_params.type == 'charuco':
        is_charuco = True
    else:
        is_charuco = False

    if is_charuco:
        marker_bits = board_params.markers_size
        marker_length_mm = board_params.marker_length
        markers_dictionary = board_params.aruco_dict

        mk_l_bits = marker_bits + 2
        sq_l_bits = sq_l_w / marker_length_mm * mk_l_bits
        if not int(sq_l_bits) == sq_l_bits:
            raise AssertionError('Error creating board svg :(')         # TODO make sure this never happens
        else:
            sq_l_bits = int(sq_l_bits)

        # Ratios to convert between world units and bits
        w2b = sq_l_bits / sq_l_w
        margin = int((sq_l_bits - mk_l_bits) / 2)

    else:
        # For classic chessboard, no markers so 1 'bit' = 1 entire square
        # 1 bit = 1 square = the square length in world units
        sq_l_bits = 1
        w2b = 1.0 / sq_l_w     # 1 mm = 1 / sq_l_w bits
        margin = 0

    chessboard_arr = (~np.indices((board_rows, board_cols)).sum(axis=0) % 2).astype(bool)

    A4_mm = np.array([210, 297])
    A4_size_bits = A4_mm * w2b

    text_h_bits = 3.0 * w2b
    board_s_bits = np.array([sq_l_bits * board_cols, sq_l_bits * board_rows])

    # If only one size, use the one that comes with the board object and position in the centre of the page
    if is_charuco:
        filename = f'Charuco{board_rows}x{board_cols}_markers{marker_bits}x{marker_bits}-margin{margin}.svg'
    else:
        filename = f'Chessboard{board_rows}x{board_cols}.svg'

    if multi_size:
        filename = 'Multi_' + filename

        bleed = 15.0 * w2b
        spacing = 25.0 * w2b
        text_width = 30.0 * w2b

        # Compute the scales to generate, from 1/4 of the page width, to the theoretical smallest with current dpi
        min_scale = 1 / dpi * 25.4 * w2b   # Theoretical smallest marker size with visible bits
        max_scale = A4_size_bits[1] / 8 / board_s_bits[1]

        nb_scales = int(np.ceil(np.log(max_scale / min_scale) / np.log(factor)) + 1)

        scales = np.array([max_scale / (factor ** i) for i in range(nb_scales)])

        # Position the boards on the page
        positions = [np.array([bleed, bleed])]

        # These are used when one page width is full
        x_ref, y_ref = positions[0]
        scale_ref = scales[0]

        for i in range(1, nb_scales):
            x, y = positions[i-1]
            next_x = max(x + text_width + spacing, x + board_s_bits[0] * scales[i-1] + spacing)
            if max(next_x + text_width, next_x + board_s_bits[0] * scales[i]) < A4_size_bits[0] - bleed:
                next_pos = np.array([next_x, y_ref])
            else:
                next_y = y_ref + board_s_bits[1] * scale_ref + text_h_bits * 4 + spacing
                next_pos = np.array([x_ref, next_y])
                x_ref, y_ref = next_pos
                scale_ref = scales[i]
            positions.append(next_pos)
    else:
        scales = [1.0]
        positions = [A4_size_bits / 2.0 - board_s_bits / 2.0]    # page centre

    # Start svg content
    svg = [
        f'<svg version="1.1" width="100%" height="100%" viewBox="0 0 {A4_size_bits[0]} {A4_size_bits[1]}" xmlns="http://www.w3.org/2000/svg">',
        f' <rect id="background" x="0" y="0" width="{A4_size_bits[0]}" height="{A4_size_bits[1]}" fill="none" stroke="#000000" stroke-width="0.1"/>'
    ]

    for board_pos, board_scale in zip(positions, scales):

        # Container group with charuco board, cutting guides and info text
        svg.append(f'  <g id="container" transform="translate({board_pos[0]}, {board_pos[1]})">')

        # Add cutting guides
        svg.append(f'    <line x1="{-2 * w2b}" y1="{-2 * w2b}" x2="{-2 * w2b}" y2="{-7 * w2b}" stroke="black" stroke-width="{0.1 * w2b}" />')
        svg.append(f'    <line x1="{-7 * w2b}" y1="{-2 * w2b}" x2="{-2 * w2b}" y2="{-2 * w2b}" stroke="black" stroke-width="{0.1 * w2b}" />')
        svg.append(f'    <line x1="{board_s_bits[0] * board_scale + 2 * w2b}" y1="{-2 * w2b}" x2="{board_s_bits[0] * board_scale + 2 * w2b}" y2="{-7 * w2b}" stroke="black" stroke-width="{0.1 * w2b}" />')
        svg.append(f'    <line x1="{board_s_bits[0] * board_scale + 2 * w2b}" y1="{-2 * w2b}" x2="{board_s_bits[0] * board_scale + 7 * w2b}" y2="{-2 * w2b}" stroke="black" stroke-width="{0.1 * w2b}" />')
        svg.append(f'    <line x1="{-2 * w2b}" y1="{board_s_bits[1] * board_scale + 2 * w2b}" x2="{-2 * w2b}" y2="{board_s_bits[1] * board_scale + 7 * w2b}" stroke="black" stroke-width="{0.1 * w2b}" />')
        svg.append(f'    <line x1="{-7 * w2b}" y1="{board_s_bits[1] * board_scale + 2 * w2b}" x2="{-2 * w2b}" y2="{board_s_bits[1] * board_scale + 2 * w2b}" stroke="black" stroke-width="{0.1 * w2b}" />')
        svg.append(f'    <line x1="{board_s_bits[0] * board_scale + 2 * w2b}" y1="{board_s_bits[1] * board_scale + 2 * w2b}" x2="{board_s_bits[0] * board_scale + 2 * w2b}" y2="{board_s_bits[1] * board_scale + 7 * w2b}" stroke="black" stroke-width="{0.1 * w2b}" />')
        svg.append(f'    <line x1="{board_s_bits[0] * board_scale + 2 * w2b}" y1="{board_s_bits[1] * board_scale + 2 * w2b}" x2="{board_s_bits[0] * board_scale + 7 * w2b}" y2="{board_s_bits[1] * board_scale + 2 * w2b}" stroke="black" stroke-width="{0.1 * w2b}" />')

        # subcontainer group with white background
        svg.append(f'    <g id="subcontainer" transform="scale({board_scale})">')
        svg.append(f'      <rect id="background" x="0" y="0" width="{board_s_bits[0]}" height="{board_s_bits[1]}" fill="#ffffff"/>')

        # Chessboard group
        svg.append('      <g id="chessboard">')
        cc, rr = np.where(chessboard_arr)
        for i, rc in enumerate(zip(rr, cc)):
            svg.append(f'        <rect id="{i}" x="{rc[0] * sq_l_bits}" y="{rc[1] * sq_l_bits}" width="{sq_l_bits}" height="{sq_l_bits}" fill="#000000"/>')
        svg.append('      </g>')

        if is_charuco:
            # Aruco markers group
            svg.append('      <g id="aruco_markers">')
            cc, rr = np.where(~chessboard_arr)
            for i, rc in enumerate(zip(rr, cc)):
                marker = markers_dictionary.generateImageMarker(i, mk_l_bits, mk_l_bits).astype(bool)
                py, px = np.where(marker)
                svg.append(f'        <g id="{i}">')
                svg.append(
                    f'          <rect x="{rc[0] * sq_l_bits + margin}" y="{rc[1] * sq_l_bits + margin}" width="{mk_l_bits}" height="{mk_l_bits}" fill="#000000"/>')
                for x, y in zip(px, py):
                    svg.append(f'          <rect x="{rc[0] * sq_l_bits + x + margin}" y="{rc[1] * sq_l_bits + y + margin}" width="1" height="1" fill="#ffffff"/>')
                svg.append('        </g>')

            svg.append('      </g>')
        svg.append('    </g>')

        # Add text with sizes
        bsize_text = f'{board_scale * sq_l_w * board_rows:.1f} x {board_scale * sq_l_w * board_cols:.1f} mm'
        sqsize_text = f'(squares: {board_scale * sq_l_w:.3f} mm)'
        if is_charuco:
            msize_text = f'(markers: {board_scale * marker_length_mm:.3f} mm)'
        svg.append(f'    <text x="0" y="{board_s_bits[1] * board_scale + text_h_bits * 4}" font-family="monospace" font-size="{text_h_bits}" font-weight="bold">{bsize_text}</text>')
        svg.append(f'    <text x="0" y="{board_s_bits[1] * board_scale + text_h_bits * 5}" font-family="monospace" font-size="{text_h_bits}" font-weight="bold">{sqsize_text}</text>')
        if is_charuco:
            svg.append(f'    <text x="0" y="{board_s_bits[1] * board_scale + text_h_bits * 6}" font-family="monospace" font-size="{text_h_bits}" font-weight="bold">{msize_text}</text>')
        svg.append('  </g>')

    svg.append('</svg>')

    file_path.mkdir(parents=True, exist_ok=True)    # TODO - check if user passed the file name in there...
    with open(file_path / filename, 'w') as f:
        f.write('\n'.join(svg))
        print(f'Saved calibration board as {file_path / filename}')
