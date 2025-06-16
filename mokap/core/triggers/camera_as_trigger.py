import logging
from typing import Optional, Dict

from mokap.core.triggers.interface import AbstractTrigger
from mokap.core.cameras.interface import AbstractCamera
from mokap.core.cameras.genicam import GenICamCamera

logger = logging.getLogger(__name__)


class CameraTrigger(AbstractTrigger):
    """
    Uses a GenICam-compliant camera as a hardware trigger source for other cameras

    This trigger configures a designated 'primary' camera to output a signal on one of its GPIO lines.
    This signal can then be used to trigger other cameras
    """

    def __init__(self, primary_camera: AbstractCamera, config: Optional[Dict] = None):
        super().__init__(config=config)
        
        if not isinstance(primary_camera, GenICamCamera):
            raise TypeError("The primary camera for PrimaryCameraTrigger must be a GenICamCamera subclass.")

        self.primary_camera = primary_camera
        self._output_line = self._config.get('output_line', 'Line2')

        logger.debug(
            f"Primary Camera trigger '{self.primary_camera.name}', using {self._output_line} as output.")
        
        self._connect()

    def _connect(self) -> None:
        """ The connection is already managed by the MultiCam class """
        
        if self.primary_camera and self.primary_camera.is_connected:
            self._connected = True
            
            logger.info('Trigger connected successfully.')
            
        else:
            self._connected = False
            logger.error("PrimaryCameraTrigger could not initialize: Primary camera is not connected.")

    def start(self, frequency: float, duty_cycle_percent: int = 50):
        """ Configures the primary camera to start outputting the trigger signal """

        if not self.connected:
            logger.error("Cannot start trigger: not connected to the primary camera.")
            return

        try:
            logger.info(f"Configuring primary camera '{self.primary_camera.name}' to output trigger on {self._output_line}.")

            # The primary camera cannot be hardware-triggered itself
            self.primary_camera.hardware_triggered = False

            # Set the master camera's framerate to the desired trigger frequency
            self.primary_camera.framerate = frequency
            logger.debug(f"Set primary camera '{self.primary_camera.name}' framerate to {self.primary_camera.framerate} Hz.")

            # set the line output
            self.primary_camera._set_feature_value('LineSelector', self._output_line)
            self.primary_camera._set_feature_value('LineMode', 'Output')

            # 'ExposureActive' means the line will be high during exposure
            self.primary_camera._set_feature_value('LineSource', 'ExposureActive')

            logger.info(f"Trigger started at {frequency} Hz.")

        except AttributeError as e:
            logger.error(f"Failed to configure primary camera as trigger. It might not support the required GenICam features: {e}")
            self.stop()  # attempt to clean up
            self._connected = False

    def stop(self):
        """ Configures the primary camera to stop outputting the trigger signal """

        if not self.primary_camera or not self.primary_camera.is_connected:
            return

        try:
            logger.debug(f"Stopping trigger output from primary camera '{self.primary_camera.name}'.")

            self.primary_camera._set_feature_value('LineSelector', self._output_line)

            # set the line mode back to a safe default like input
            self.primary_camera._set_feature_value('LineMode', 'Input')

        except AttributeError as e:
            # this might fail if the camera was disconnected abruptly, which is fine
            logger.warning(f"Could not cleanly reset primary camera trigger line. It might have been disconnected. Error: {e}")

        logger.info('Trigger stopped.')

    def disconnect(self):
        """ Resets the internal state. Does not disconnect the camera """ # TODO: should it?

        if self.connected:
            self.stop()

        self._connected = False

        logger.info('Trigger disconnected.')