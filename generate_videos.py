#!/usr/bin/env python

from mokap.files_op import *

##

input_folder = Path('/mnt/data/MokapRecordings')
output_folder = Path('/mnt/storage/MokapVideos')

if __name__ == '__main__':
    for folder in input_folder.iterdir():
        videos_from_frames(folder, output_folder=output_folder, delete_source=False)
