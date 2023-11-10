#!/usr/bin/env python

from mokap.files_op import *

## Use this to reconvert videos into the default format - BE CAREFUL

# if __name__ == '__main__':
#     for folder in videos_folder.iterdir():
#         # print(folder)
#         convert_videos(folder, filter='avi')

## Use this to rename videos into the new naming convention - BE CAREFUL

if __name__ == '__main__':
    for folder in videos_folder.iterdir():
        print(folder)
        update_names(folder)

