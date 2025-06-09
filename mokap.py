#!/usr/bin/env python
import sys
from mokap.core.manager import MultiCam
from mokap.gui import QApplication, MainControls, QMessageBox
from mokap.utils import fileio

def main():
    """ Main entry point for the Mokap GUI """
    try:
        config = fileio.read_config('./config.yaml')
    except FileNotFoundError:
        QMessageBox.critical(None, "Error", "Configuration file 'config.yaml' not found. Please create one.")
        sys.exit(1)

    app = QApplication(sys.argv)

    mc = MultiCam(config=config)

    if mc.nb_cameras == 0:
        msg = ("No cameras were found or connected.\n\n"
               "Please check:\n"
               "  - Camera connections and power.\n"
               "  - Vendor SDK installation (Basler Pylon, FLIR Spinnaker...).\n"
               "  - 'sources' configuration in config.yaml")

        QMessageBox.warning(None, "No Cameras Found", msg)

    main_window = MainControls(mc)
    main_window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()