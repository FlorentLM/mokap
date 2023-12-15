from mokap.core import MotionDetector


threshold = 1
preview = True
lag = 2

md1 = MotionDetector(cam_id=0, thresh=threshold, lag=lag, preview=preview)
md2 = MotionDetector(cam_id=2, thresh=threshold, lag=lag, preview=preview)
md3 = MotionDetector(cam_id=4, thresh=threshold, lag=lag, preview=preview)

md1.start()
md2.start()
md3.start()

##

from mokap.core import Manager
from mokap.hardware import enable_usb
from datetime import datetime

enable_usb('4-2')

tomato = '40182207'     # tomato
leaf = '40166127'       # leaf
smurf = '40182542'      # smurf


mgr_1 = Manager()
mgr_1.exposure = 10000
mgr_1.framerate = 100
mgr_1.binning = 2
mgr_1.connect(tomato)
mgr_1.savepath = f'tomato_{datetime.now().strftime("%y%m%d-%H%M")}'

mgr_2 = Manager()
mgr_2.exposure = 10000
mgr_2.framerate = 100
mgr_2.binning = 2
mgr_2.connect(leaf)
mgr_2.savepath = f'leaf_{datetime.now().strftime("%y%m%d-%H%M")}'

mgr_3 = Manager()
mgr_3.exposure = 10000
mgr_3.framerate = 100
mgr_3.binning = 2
mgr_3.connect(smurf)
mgr_3.savepath = f'smurf_{datetime.now().strftime("%y%m%d-%H%M")}'
