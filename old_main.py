#!/usr/bin/env python
from mokap.interface_tk import MainWindow
from mokap.core import MultiCam

mgr = MultiCam(config='./config.yaml', triggered=True, silent=False)

# Set exposure for all cameras (in µs)
mgr.exposure = 4800

# Enable binning
mgr.binning = 1
mgr.binning_mode = 'avg'

# Set framerate in images per second for all cameras at once
mgr.framerate = 100

mgr.gamma = 1.0
mgr.blacks = 1.0
mgr.gain = 0.0

if __name__ == '__main__':
    if mgr.nb_cameras == 0:
        exit()
    MainWindow(mgr)
    exit()