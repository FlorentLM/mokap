#!/usr/bin/env python
from mokap.interface import GUI
from mokap.core import Manager

mgr = Manager(config='./config.conf')
mgr.connect()

# Set exposure for all cameras (in µs)
# (maximum exposure time for maximum framerate is 4318 µs)
mgr.exposure = 8300

# Enable binning
mgr.binning = 1
mgr.binning_mode = 'avg'

mgr.gain = 2

# Set framerate in images per second
mgr.framerate = 120

if __name__ == '__main__':
    if mgr.nb_cameras == 0:
        exit()
    GUI(mgr)

