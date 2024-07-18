#!/usr/bin/env python
from mokap.core import MultiCam
from mokap.interface_qt import *

mc = MultiCam(config='./config.yaml', triggered=False, silent=False)

# Set exposure for all cameras (in µs)
mc.exposure = 4800

# Enable binning
mc.binning = 1
mc.binning_mode = 'avg'

# Set framerate in images per second for all cameras at once
mc.framerate = 100

mc.gamma = 1.0
mc.blacks = 1.0
mc.gain = 0.0

if __name__ == '__main__':
    app = QApplication(sys.argv)

    if mc.nb_cameras == 0:
        exit()
    main_window = MainWindow(mc)
    main_window.show()

    sys.exit(app.exec())