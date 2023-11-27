#!/usr/bin/env python

from mokap.interface import GUI
from mokap import hardware as hw

hw.enable_usb('4-2')

mgr = hw.Manager()

# Set exposure for all cameras in µs
# (maximum exposure time for maximum framerate is 4318 µs)
# mgr.exposure = 5555
# mgr.exposure = 10000
mgr.exposure = 4318
# mgr.cameras['top'].exposure = 3000

# Set framerate in images per second
mgr.framerate = 100

# Enable binning
mgr.binning = 2

mgr.connect()

if __name__ == '__main__':
    if mgr.nb_cameras == 0:
        exit()
    GUI(mgr)
