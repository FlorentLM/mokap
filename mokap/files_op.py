import re
import subprocess as sp
from pathlib import Path
import shutil
import shlex
import configparser
import json
import os

##

COMPRESSED = {'codec': 'libx265',
              'params': '-crf 1 -preset veryfast',
              'ftype': 'mp4'}

LOSSLESS_1 = {'codec': 'ffv1',
              'params': '-level 3',
              'ftype': 'avi'}

LOSSLESS_2 = {'codec': 'libx265',
              'params': '-x265-params lossless=1 -preset veryfast',
              'ftype': 'mp4'}

DEFAULT_FMT = COMPRESSED


##
def read_config(config_file='config.conf'):
    confparser = configparser.ConfigParser()
    try:
        confparser.read(config_file)
    except FileNotFoundError:
        print('[WARN] Config file not found. Defaulting to example config.')
        confparser.read('example_config.conf')

    return confparser


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

def rm_if_empty_zarr(path):

    path = Path(path)

    if not path.exists():
        # if it doesn't exist, nothing to do.
        return
    else:
        # if it exists
        if not any(path.iterdir()):
            # ...and already empty, delete it, done.
            path.rmdir()

        # if not empty, check if there is more than one file
        else:
            generator = path.glob('*')
            first_child = next(generator)
            try:
                second_child = next(generator)
                # if more than the zarr, stop
                return
            except StopIteration:
                second_child = None
            # if only one thing, and this file is a .zarr, check if it is empty
            if second_child is None and first_child.suffix == '.zarr':
                generator_2 = (first_child / 'frames').glob('*')
                first_child_2 = next(generator_2)  # This should be the .zarray hidden file
                try:
                    second_child_2 = next(generator)
                    # if more than the .zarray, stop
                    return
                except StopIteration:
                    second_child_2 = None
                    if second_child_2 is None and first_child_2.stem == '.zarray':
                        shutil.rmtree(path)

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

def videos_from_frames(dirpath, format=None, force=False):
    dirpath = Path(dirpath).resolve()

    if not dirpath.exists():
        return
    if (dirpath / 'recording').exists() and not force:
        return

    print(f"Converting {dirpath.stem}...")

    with open(dirpath / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    cameras = metadata['sessions'][0]['cameras']
    nb_cams = len(cameras)

    if format is not None:
        if 'compr' in str(format):
            conv_settings = COMPRESSED
        elif '1' in str(format):
            conv_settings = LOSSLESS_1
        elif '2' in str(format):
            conv_settings = LOSSLESS_2
        else:
            conv_settings = DEFAULT_FMT
    else:
        conv_settings = DEFAULT_FMT

    outfolder = dirpath / 'videos'
    if not outfolder.exists():
        outfolder.mkdir(parents=True, exist_ok=False)

    for c, cam in enumerate(cameras):
        cam_folder = dirpath / f"cam{cam['idx']}_{cam['name']}"
        assert cam_folder.is_dir()

        if (cam_folder / 'converted').exists() and not force:
            return

        files = [f for f in os.scandir(cam_folder) if f.name != 'converted']    # scandir is the fastest method for that
        nb_frames = len(files)

        fps_raw = metadata['framerate']
        total_time_sec = sum([session['end'] - session['start'] for session in metadata['sessions']])

        fps_calc = nb_frames / total_time_sec

        stats = f"\nFramerate:\n" \
                f"---------\n" \
                f"Theoretical:\n  {fps_raw:.2f} fps\n" \
                f"Actual (mean):\n  {fps_calc:.2f} fps\n" \
                f"--> Error = {(1 - (fps_calc / fps_raw)) * 100:.2f}%"

        print(stats)

        savepath = outfolder / (f'{dirpath.stem}_{cam_folder.stem}.{conv_settings["ftype"]}')

        with open(outfolder / (f'{savepath.stem}_stats.txt'), 'w') as st:
            st.write(stats)

        in_fmt = (cam_folder / files[0].name).suffix

        process = sp.Popen(shlex.split(
            f'ffmpeg '
            f'-framerate {fps_calc:.2f} '                                   # framerate
            f"-pattern_type glob -i '*{in_fmt}' "                           # glob input
            f'-r {fps_calc:.2f} '                                           # framerate (again)
            f'-c:v {conv_settings["codec"]} {conv_settings["params"]} '     # video codec
            f'-pix_fmt gray '                                               # pixel format
            f'-an '                                                         # no audio
            f'-fps_mode vfr '                                               # avoid having duplicate frames
            f'{savepath.as_posix()}'),                                      # output
        cwd=cam_folder)

        process.wait()
        process.terminate()

        if savepath.exists():
            (cam_folder / 'converted').touch()

            print(f'Done creating {savepath.name}.')
        else:
            print(f'Error creating {savepath.name}!')



def videos_from_zarr(filepath, cams=None, format=None, force=False):
    import zarr

    filepath = Path(filepath).resolve()

    if not filepath.exists():
        return
    if (filepath / 'recording').exists() and not force:
        return
    if (filepath / 'converted').exists() and not force:
        return

    print(f"Converting {filepath.stem}")

    root = zarr.open((filepath / f'{filepath.stem}.zarr').as_posix(), mode="r")
    nb_cams, nb_frames, h, w = root['frames'].shape
    metadata = root.attrs

    if cams is not None:
        if type(cams) is int:
            cams = {cams}
        else:
            cams = set(cams)
    else:
        cams = set(range(nb_cams))

    if format is not None:
        if 'compr' in str(format):
            conv_settings = COMPRESSED
        elif '1' in str(format):
            conv_settings = LOSSLESS_1
        elif '2' in str(format):
            conv_settings = LOSSLESS_2
        else:
            conv_settings = DEFAULT_FMT
    else:
        conv_settings = DEFAULT_FMT

    for c in cams:

        fps_raw = metadata[str(c)]['framerate']

        sessions = root['times'].shape[0] - 1

        delta = 0
        for s in range(sessions):
            start, end = root['times'][s]
            delta += end - start

        total_time_s = delta  # / np.timedelta64(1, 's')

        fps_calc = nb_frames / total_time_s

        stats = f"\nFramerate:\n" \
                f"---------\n" \
                f"Theoretical:\n  {fps_raw:.2f} fps\n" \
                f"Actual (mean):\n  {fps_calc:.2f} fps\n" \
                f"--> Error = {(1 - (fps_calc / fps_raw)) * 100:.2f}%"

        print(stats)

        outfolder = filepath / 'videos' / filepath.stem
        if not outfolder.exists():
            outfolder.mkdir(parents=True, exist_ok=False)

        outname = f'{outfolder.stem}_cam{c}.'
        with open(outfolder / (outname + 'txt'), 'w') as st:
            st.write(stats)

        savepath = outfolder / (outname + conv_settings["ftype"])

        process = sp.Popen(shlex.split(
            f'ffmpeg '
            f'-s {w}x{h} '                  # set frame size
            f'-f rawvideo '                 # force format
            f'-r {fps_calc:.2f} '           # framerate
            f'-i pipe: '                    # input is piped
            f'-c:v {conv_settings["codec"]} {conv_settings["params"]} '     # video codec
            f'-pix_fmt gray '               # pixel format
            f'-an '                         # no audio
            f'{savepath.as_posix()}'),      # output
            stdin=sp.PIPE)

        for i in range(nb_frames):
            process.stdin.write(root['frames'][c, i].tobytes())

        process.stdin.close()
        process.wait()
        process.terminate()

        (filepath / 'converted').touch()

        print(f'Done creating {outname + conv_settings["ftype"]}.')


def convert_videos(path, name_filter='*', output_format=None, delete_source=False, force=False):
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

    if output_format is not None:
        if 'compr' in str(output_format):
            conv_settings = COMPRESSED
        elif '1' in str(output_format):
            conv_settings = LOSSLESS_1
        elif '2' in str(output_format):
            conv_settings = LOSSLESS_2
        else:
            conv_settings = DEFAULT_FMT
    else:
        conv_settings = DEFAULT_FMT

    for f in to_convert:
        # TODO - Add check to prevent converting to same format, or converting from lossy to lossless!

        output_name = f.stem + f'.{conv_settings["ftype"]}'
        savepath = f.parent / output_name

        if output_name == f.name:
            f = f.rename(f.parent / f'{f.stem}_orig{f.suffix}')

        process = sp.Popen(shlex.split(
            f'ffmpeg -i {f.as_posix()}'
            f'-c:v {conv_settings["codec"]} {conv_settings["params"]}'
            f'-pix_fmt gray'
            f'-an {savepath.as_posix()}'))

        process.wait()
        process.terminate()

        if delete_source:
            if savepath.is_file() and savepath.stat().st_size > 0:
                f.unlink(missing_ok=True)

    print(f'Done converting {path.name} {ftype}.')


# def update_names(path, filter='*'):
#
#     path = Path(path)
#
#     if path.is_file():
#         source_names = [path]
#         ftype = 'file'
#     else:
#         source_names = list(path.glob(filter))
#         ftype = 'folder contents'
#
#     any_renamed = False
#     for f in source_names:
#
#         if bool(re.search('vid_cam[0-9+]_sess[0-9+]', f.stem)):
#
#             new_name = f"{f.parent.stem}_{f.stem.split('_')[1]}{f.suffix}"
#             savepath = exists_check(f.parent / new_name)
#
#             f.rename(savepath)
#             any_renamed = True
#
#     if any_renamed:
#         print(f'Done renaming {path.name} {ftype}.')
#     else:
#         print(f'Nothing needed to be renamed.')
