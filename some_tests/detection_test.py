#!/usr/bin/env python
import time
from mokap.core import MotionDetector, Manager
from mokap.hardware import enable_usb
from datetime import datetime

## Movement detection

threshold = 5       # Movement detection threshold
preview = True      # Whether to display the movement detection preview
lag = 20            # How long the recording should be trigggered for after a movement is detected

md1 = MotionDetector(cam_id=0, thresh=threshold, lag=lag, preview=preview)
md2 = MotionDetector(cam_id=2, thresh=threshold, lag=lag, preview=preview)
md3 = MotionDetector(cam_id=4, thresh=threshold, lag=lag, preview=preview)

md1.start()
md2.start()
md3.start()

## Recording
#
# enable_usb('4-2')
#
# tomato = '40182207'     # tomato
# leaf = '40166127'       # leaf
# smurf = '40182542'      # smurf
#
# fps = 100
# bin = 2
# exp = 10000
#
# ##
#
# mgr_1 = Manager()
# mgr_1.exposure = exp
# mgr_1.framerate = fps
# mgr_1.binning = bin
# mgr_1.connect(tomato)
#
# mgr_2 = Manager()
# mgr_2.exposure = exp
# mgr_2.framerate = fps
# mgr_2.binning = bin
# mgr_2.connect(leaf)
#
# mgr_3 = Manager()
# mgr_3.exposure = exp
# mgr_3.framerate = fps
# mgr_3.binning = bin
# mgr_3.connect(smurf)
#
# time.sleep(10)
#
# is_active_1 = False
# is_active_2 = False
# is_active_3 = False
#
# print('\n---\n')
#
# while True:
#
#     now = datetime.now().strftime("%y%m%d-%H%M%S")
#
#     if md1.moves and not is_active_1:
#
#         mgr_1.savepath = f'tomato_{now}'
#         mgr_1.on()
#         mgr_1.record()
#         print(f'* Started recording [tomato] : {now}')
#         is_active_1 = True
#
#     elif not md1.moves and is_active_1:
#         mgr_1.pause()
#         mgr_1.off()
#         print(f'* Stopped recording [tomato] : {now}')
#         is_active_1 = False
#
#     else:
#         pass
#
#     ##
#
#     if md2.moves and not is_active_2:
#         mgr_2.savepath = f'leaf_{now}'
#         mgr_2.on()
#         mgr_2.record()
#         print(f'* Started recording [leaf] : {now}')
#         is_active_2 = True
#
#     elif not md2.moves and is_active_2:
#         mgr_2.pause()
#         mgr_2.off()
#         print(f'* Stopped recording [leaf] : {now}')
#         is_active_2 = False
#
#     else:
#         pass
#
#     ##
#
#     if md3.moves and not is_active_3:
#         mgr_3.savepath = f'smurf_{now}'
#         mgr_3.on()
#         mgr_3.record()
#         print(f'* Started recording [smurf] : {now}')
#         is_active_3 = True
#
#     elif not md3.moves and is_active_3:
#         mgr_3.pause()
#         mgr_3.off()
#         print(f'* Stopped recording [smurf] : {now}')
#         is_active_3 = False
#
#     else:
#         pass
#
#     time.sleep(0.1)