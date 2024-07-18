#!/usr/bin/env python
from mokap.core import Manager
from mokap.interface_qt import *

mgr = Manager(config='./config.yaml', triggered=False, silent=False)

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
    app = QApplication(sys.argv)

    if mgr.nb_cameras == 0:
        exit()
    main_window = MainWindow(mgr)
    main_window.show()

    sys.exit(app.exec())