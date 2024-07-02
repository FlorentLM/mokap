#!/usr/bin/env python
from mokap.interface import GUI
from mokap.core import Manager

mgr = Manager(config='./config.yaml', triggered=True, silent=False)

# Set exposure for all cameras (in µs)
mgr.exposure = 4500

# Enable binning
mgr.binning = 1
mgr.binning_mode = 'avg'

# Set framerate in images per second
mgr.framerate = 200

mgr.gamma = 1.0
mgr.blacks = 1.0
mgr.gain = 0.0

if __name__ == '__main__':
    if mgr.nb_cameras == 0:
        exit()
    GUI(mgr)
    exit()