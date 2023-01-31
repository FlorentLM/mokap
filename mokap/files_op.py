import re
import subprocess as sp
import shlex
import zarr
from pathlib import Path
from datetime import datetime
import numpy as np
import shutil


videos_folder = Path('../videos')
data_folder = Path('../data')


def mk_folder(name=''):

    if name == '':
        name = datetime.now().strftime('%y%m%d-%H%M')

    new_path = data_folder / name

    i = 1
    while new_path.exists():
        name = new_path.stem.split('_')[0]
        incremented = name + f"_{i}"
        new_path = new_path.parent / incremented
        i += 1
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


def videos_from_zarr(filepath, cams=None):

    filepath = Path(filepath)

    if (filepath / 'converted').exists():
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

    for c in cams:

        fps_raw = metadata[str(c)]['framerate']

        sessions = root['times'].shape[0] - 1

        delta = 0
        for s in range(sessions):
            start, end = root['times'][s]
            delta += end - start

        total_time_s = delta / np.timedelta64(1, 's')

        fps_calc = nb_frames/total_time_s

        print(f"\nFramerate:\n"
              f"---------\n"
              f"Theoretical:\n  {fps_raw:.2f} fps\n"
              f"Actual (mean):\n  {fps_calc:.2f} fps\n"
              f"--> Error = {(1-(fps_calc/fps_raw))*100:.2f}%")

        outname = f"vid_cam{c}_sess{s}.avi"

        outfolder = videos_folder / filepath.stem
        if not outfolder.exists():
            outfolder.mkdir(parents=True, exist_ok=False)

        savepath = outfolder / outname

        process = sp.Popen(shlex.split(
            f'ffmpeg -y -s {w}x{h} -pixel_format gray -f rawvideo -r {fps_calc:.2f} -i pipe: -vcodec ffv1 -level 3 -pix_fmt gray {savepath.as_posix()}'),
            stdin=sp.PIPE)

        for i in range(nb_frames):
            process.stdin.write(root['frames'][c, i].tobytes())

        process.stdin.close()
        process.wait()
        process.terminate()

        (filepath / 'converted').touch()

        print('Done.')
