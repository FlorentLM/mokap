import re
import subprocess as sp
import zarr
from pathlib import Path
from datetime import datetime
import numpy as np
import shutil
import shlex

##

videos_folder = Path('/mnt/data/Videos')
data_folder = Path('/mnt/data/RawFrames')

# videos_folder = Path('F:\\Videos')
# data_folder = Path('F:\\RawFrames')


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

def exists_check(path):
    """ Checks if a file or folder of the given name exists. If so, created a suffixed version of the name
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


def mk_folder(name=''):

    if name == '':
        name = datetime.now().strftime('%y%m%d-%H%M')

    new_path = exists_check(data_folder / name)

    new_path.mkdir(parents=True, exist_ok=False)
    return new_path


def rm_if_empty(path):
    path = Path(path)

    if not path.exists():
        return
    else:
        # if empty, delete folder, done.
        if not any(path.iterdir()):
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
                first_child_2 = next(generator_2)   # This should be the .zarray hidden file
                try:
                    second_child_2 = next(generator)
                    # if more than the .zarray, stop
                    return
                except StopIteration:
                    second_child_2 = None
                    if second_child_2 is None and first_child_2.stem == '.zarray':
                        shutil.rmtree(path)


def clean_folder(path):
    path = Path(path)
    [f.unlink() for f in path.glob('*')]
    print(f"Cleaned {path}")


def natural_sort_key(s):
    _nsre = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def videos_from_zarr(filepath, cams=None, format=None, force=False):

    filepath = Path(filepath)

    if (filepath / 'recording').exists() and not force:
        return

    if (filepath / 'converted').exists() and not force:
        return

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
        print(np.timedelta64(1, 's'))
        total_time_s = delta / np.timedelta64(1, 's')

        fps_calc = nb_frames/total_time_s

        stats = f"\nFramerate:\n"\
              f"---------\n"\
              f"Theoretical:\n  {fps_raw:.2f} fps\n"\
              f"Actual (mean):\n  {fps_calc:.2f} fps\n"\
              f"--> Error = {(1-(fps_calc/fps_raw))*100:.2f}%"

        print(stats)

        outfolder = videos_folder / filepath.stem
        if not outfolder.exists():
            outfolder.mkdir(parents=True, exist_ok=False)

        outname = f'{outfolder.stem}_cam{c}.'
        with open(outfolder / (outname + 'txt'), 'w') as st:
            st.write(stats)

        savepath = outfolder / (outname + conv_settings["ftype"])

        process = sp.Popen(shlex.split(
            f'ffmpeg -y -s {w}x{h} -pix_fmt gray -f rawvideo -r {fps_calc:.2f} -i pipe: -c:v {conv_settings["codec"]} {conv_settings["params"]} -pix_fmt gray -an {savepath.as_posix()}'),
            stdin=sp.PIPE)

        for i in range(nb_frames):
            process.stdin.write(root['frames'][c, i].tobytes())

        process.stdin.close()
        process.wait()
        process.terminate()

        (filepath / 'converted').touch()

        print(f'Done creating {outname + conv_settings["ftype"]}.')


def convert_videos(path, filter='*', output_format=None, delete_source=False):

    path = Path(path)

    if path.is_file():
        to_convert = [path]
        ftype = 'file'
    else:
        to_convert = list(path.glob(filter))
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

        process = sp.Popen(shlex.split(f'ffmpeg -i {f.as_posix()} -c:v {conv_settings["codec"]} {conv_settings["params"]} -pix_fmt gray -an {savepath.as_posix()}'))

        process.wait()
        process.terminate()

        if delete_source:
            if savepath.is_file() and savepath.stat().st_size > 0:
                f.unlink(missing_ok=True)

    print(f'Done converting {path.name} {ftype}.')

#
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