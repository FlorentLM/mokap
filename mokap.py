#!/usr/bin/env python
import sys
from mokap.core import MultiCam
from mokap.gui import QApplication, MainWindow

mc = MultiCam(config='./config.yaml', triggered=True, silent=False)

# Example:
# Set some default parameters for all cameras at once

mc.exposure = 10000
mc.framerate = 70
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