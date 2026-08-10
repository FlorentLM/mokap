#!/usr/bin/env python
import sys
from mokap.gui import QApplication, MainControls, QMessageBox
from mokap.core.controller import CameraController
from mokap.mokap_io import load_config


def main():
    """ Main entry point for the Mokap GUI """

    try:
        config = load_config('./config.yaml')
    except FileNotFoundError:
        QMessageBox.critical(None, "Error", "Configuration file 'config.yaml' not found. Please create one.")
        sys.exit(1)

    app = QApplication(sys.argv)

    cc = CameraController(config=config)

    if cc.nb_cameras == 0:
        msg = ("No cameras were found or connected.\n\n"
               "Please check:\n"
               "  - Camera connections and power.\n"
               "  - Vendor SDK installation (Basler Pylon, FLIR Spinnaker...).\n"
               "  - 'sources' configuration in config.yaml")

        QMessageBox.warning(None, "No Cameras Found", msg)

    main_window = MainControls(cc)
    main_window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()