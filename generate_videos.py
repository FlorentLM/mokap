#!/usr/bin/env python

from mokap.files_op import *

##

dirpath = Path('/mnt/data/MokapRecordings')

if __name__ == '__main__':
    for folder in dirpath.iterdir():
        print(f'Generating video for {folder}...')
        videos_from_frames(folder)