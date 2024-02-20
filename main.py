#!/usr/bin/env python
from mokap.interface import GUI
from mokap.core import ManagerOld
from mokap.hardware import enable_usb, disable_usb

# enable_usb('4-2')
# disable_usb('4-2')

mgr = ManagerOld()

# Set exposure for all cameras in µs
# (maximum exposure time for maximum framerate is 4318 µs)
# mgr.exposure = 5555
# mgr.exposure = 10000
mgr.exposure = 4300

# Enable binning
mgr.binning = 1

mgr.connect()

# Set framerate in images per second
mgr.framerate = 220

if __name__ == '__main__':
    # if mgr.nb_cameras == 0:
    #     exit()
    mgr.on()
    mgr.record()
