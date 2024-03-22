#!/usr/bin/env python
from mokap.interface import GUI
from mokap.core import Manager

mgr = Manager(config='./config.conf')
mgr.connect()

# Set exposure for all cameras (in µs)
# (maximum exposure time for maximum framerate is 4318 µs)
mgr.exposure = 15000

# Enable binning
mgr.binning = 1
mgr.binning_mode = 'avg'

# Set framerate in images per second
mgr.framerate = 60

mgr.gamma = 0.35
mgr.blacks = 1.5
mgr.gain = 8.5

if __name__ == '__main__':
    if mgr.nb_cameras == 0:
        exit()
    GUI(mgr)
