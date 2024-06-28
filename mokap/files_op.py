import re
import subprocess as sp
from pathlib import Path
import shutil
import shlex
import json
import os
import yaml

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


##
# def read_config(config_file='config.conf'):
#     confparser = configparser.ConfigParser()
#     try:
#         confparser.read(config_file)
#     except FileNotFoundError:
#         print('[WARN] Config file not found. Defaulting to example config.')
#         confparser.read('example_config.conf')
#
#     return confparser

def read_config(config_file='config.yaml'):
    config_file = Path(config_file)

    if not config_file.exists():
        print('[WARN] Config file not found. Defaulting to example config.')
        config_file = Path('example_config.yaml')

    with open(config_file, 'r') as f:
        config_content = yaml.safe_load(f)

    return config_content


def exists_check(path):
    """ Checks if a file or folder of the given name exists. If so, create a suffixed version of the name
    in a smart way. Returns the new, safe to use, name. """
    i = 2
    while path.exists():

        if bool(re.match('.+_[0-9]+$', path.stem)):
            # ends with a '_X' number so let's check if there is also a non-suffixed sibling
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


def clean_folder(path):
    path = Path(path)
    [f.unlink() for f in path.glob('*')]
    print(f"Cleaned {path}")


def natural_sort_key(s):
    _nsre = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def videos_from_frames(input_folder, output_folder=None, delete_source=False, force=False):
    input_folder = Path(input_folder).resolve()

    if not input_folder.exists():
        print(f"{input_folder.stem} not found, skipping.")
        return
    if (input_folder / 'recording').exists():
        print(f"Skipping {input_folder.stem} (still recording).")
        return

    with open(input_folder / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    cameras = metadata['sessions'][0]['cameras']
    converted_ok = [False] * len(cameras)

    if output_folder is not None:
        output_folder = Path(output_folder).resolve()
        if input_folder.stem not in output_folder.stem:
            output_folder = output_folder / input_folder.stem
    else:
        output_folder = input_folder / 'videos'

    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=False)

    for c, cam in enumerate(cameras):
        cam_folder = input_folder / f"cam{cam['idx']}_{cam['name']}"
        assert cam_folder.is_dir()

        if (cam_folder / 'converted').exists() and not force:
            print(f"Skipping {input_folder.stem} (already converted).")
            return

        files = [f for f in os.scandir(cam_folder) if f.name != 'converted']    # scandir is the fastest method for that
        nb_frames = len(files)

        fps_raw = metadata['framerate']
        total_time_sec = sum([session['end'] - session['start'] for session in metadata['sessions']])

        fps_calc = nb_frames / total_time_sec

        stats = f"Framerate:\n" \
                f"---------\n" \
                f" Theoretical:\n  {fps_raw:.2f} fps\n" \
                f" Actual (mean):\n  {fps_calc:.2f} fps\n" \
                f" --> Error = {(1 - (fps_calc / fps_raw)) * 100:.2f}%" \
                "\n\n" \
                f"Duration:\n" \
                f"---------\n" \
                f" Sessions:\n  {len(metadata['sessions'])}\n" \
                f" Total seconds:\n  {total_time_sec:.2f}"

        print(stats)

        savepath = output_folder / f'{input_folder.stem}_{cam_folder.stem}.{ENCODE_FORMAT["ftype"]}'

        with open(output_folder / f'{savepath.stem}_stats.txt', 'w') as st:
            st.write(stats)

        in_fmt = (cam_folder / files[0].name).suffix

        process = sp.Popen(shlex.split(
            f'ffmpeg '
            f'-framerate {fps_calc:.2f} '                                   # framerate
            f"-pattern_type glob -i '*{in_fmt}' "                           # glob input
            f'-r {fps_calc:.2f} '                                           # framerate (again)
            f'-c:v {ENCODE_FORMAT["codec"]} {ENCODE_FORMAT["params"]} '     # video codec
            f'-pix_fmt gray '                                               # pixel format
            f'-an '                                                         # no audio
            f'-fps_mode vfr '                                               # avoid having duplicate frames
            f'{savepath.as_posix()}'),                                      # output
        cwd=cam_folder)

        process.wait()
        process.terminate()

        if savepath.is_file() and savepath.stat().st_size > 0:
            (cam_folder / 'converted').touch()
            converted_ok[c] = True
            print(f'Done creating {savepath.name}.')

            if delete_source:
                shutil.rmtree(cam_folder)
        else:
            print(f'Error creating {savepath.name}!')

    shutil.copy(input_folder / 'metadata.json', output_folder / 'metadata.json')

    if delete_source and all(converted_ok) and (output_folder / 'metadata.json').is_file():
        shutil.rmtree(input_folder)

    print(f'Finished creating videos for record {input_folder.stem}.')


def reencode_videos(path, name_filter='*', delete_source=False, force=False):
    path = Path(path).resolve()

    if not path.exists():
        return
    if (path / 'recording').exists() and not force:
        return
    if (path / 'converted').exists() and not force:
        return

    print(f"Converting {path.stem}")

    if path.is_file():
        to_convert = [path]
        ftype = 'file'
    else:
        to_convert = list(path.glob(name_filter))
        ftype = 'folder contents'

    for f in to_convert:
        # TODO - Add check to prevent converting to same format, or converting from lossy to lossless!

        output_name = f.stem + f'.{ENCODE_FORMAT["ftype"]}'
        savepath = f.parent / output_name

        if output_name == f.name:
            f = f.rename(f.parent / f'{f.stem}_orig{f.suffix}')

        process = sp.Popen(shlex.split(
            f'ffmpeg -i {f.as_posix()}'
            f'-c:v {ENCODE_FORMAT["codec"]} {ENCODE_FORMAT["params"]}'
            f'-pix_fmt gray'
            f'-an {savepath.as_posix()}'))

        process.wait()
        process.terminate()

        if savepath.is_file() and savepath.stat().st_size > 0:
            print(f'Done reencoding {path.name} {ftype}.')

            if delete_source:
                f.unlink(missing_ok=True)
        else:
            print(f'Error reencoding {path.name} {ftype}!')




