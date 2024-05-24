#!/usr/bin/env python
from mokap.interface import GUI
from mokap.core import Manager

mgr = Manager(config='./config.conf')
mgr.connect()

# Set exposure for all cameras (in µs)
# (maximum exposure time for maximum framerate is 4318 µs)
# mgr.exposure = 15000
mgr.exposure = 4500

# Enable binning
mgr.binning = 2
mgr.binning_mode = 'avg'

# Set framerate in images per second
# mgr.framerate = 60git pu
mgr.framerate = 200

mgr.gamma = 0.35
# mgr.blacks = 1.5
mgr.blacks = 0.5
# mgr.gain = 8.5
mgr.gain = 13.0

if __name__ == '__main__':
    if mgr.nb_cameras == 0:
        exit()
    GUI(mgr)
