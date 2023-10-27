#!/usr/bin/env python

from mokap.files_op import *

if __name__ == '__main__':
    for folder in data_folder.iterdir():
        print(folder)
        filepath = folder
        videos_from_zarr(folder)

##
