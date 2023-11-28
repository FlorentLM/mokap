#!/usr/bin/env python

from mokap.interface import GUI
from mokap.core import Manager
from mokap.hardware import enable_usb

enable_usb('4-2')

mgr = Manager()

# Set exposure for all cameras in µs
# (maximum exposure time for maximum framerate is 4318 µs)
# mgr.exposure = 5555
mgr.exposure = 10000
# mgr.exposure = 4300

# Set framerate in images per second
mgr.framerate = 80

# Enable binning
mgr.binning = 2

mgr.connect()

if __name__ == '__main__':
    if mgr.nb_cameras == 0:
        exit()
    GUI(mgr)

