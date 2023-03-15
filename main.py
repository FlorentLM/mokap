#!/usr/bin/env python

from mokap.interface import GUI
from mokap import hardware as hw

hw.enable_usb()

fps = 180

mgr = hw.Manager(triggered=False)
self = mgr

mgr.set_exposure(5555)     # in µs (max = 4318)
mgr.set_framerate(fps)      # in fps (max = 220)
mgr.set_scale(2)            # in 1/x

mgr.cameras['top'].exposure = 3000

# duration = 50
# mgr.on()
# import time
# mgr.record()
# time.sleep(duration)
# mgr.pause()
# mgr.off()

if __name__ == '__main__':
    GUI(mgr)

