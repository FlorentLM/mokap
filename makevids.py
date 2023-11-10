#!/usr/bin/env python

from mokap.files_op import *

##

# if __name__ == '__main__':
#     for folder in data_folder.iterdir():
#         # print(folder)
#         videos_from_zarr(folder)

##

# if __name__ == '__main__':
#     for folder in data_folder.iterdir():
#         # print(folder)
#         convert_videos(folder, filter='avi')

##

if __name__ == '__main__':
    for folder in data_folder.iterdir():
        print(folder)
        update_names(folder)

