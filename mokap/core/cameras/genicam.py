import abc
import logging
from typing import Any, Dict, Optional, Tuple
from mokap.core.cameras.interface import AbstractCamera

logger = logging.getLogger(__name__)


class GenICamCamera(AbstractCamera, abc.ABC):
    """
    An abstract base class for any GenICam-compliant cameras

    This class provides a concrete implementation for most camera properties
    (exposure, gain, etc.) by mapping them to standard GenICam feature names

    Subclasses implement the vendor-specific methods for connecting,
    grabbing, and the low-level feature accessors (_get_feature_value, etc)
    """

    def __init__(self, unique_id: str):
        super().__init__(unique_id)

        # Initialize all cached properties
        self._pixel_format: Optional[str] = None
        self._binning: Optional[int] = None
        self._binning_mode: Optional[str] = None
        self._hardware_triggered: Optional[bool] = None
        self._trigger_line: Any = None
        self._framerate: Optional[float] = None
        self._exposure: Optional[float] = None
        self._gain: Optional[float] = None
        self._blacks: Optional[float] = None
        self._gamma: Optional[float] = None
        self._roi: Optional[Tuple[int, int, int, int]] = None

    def _apply_configuration(self, config: Optional[Dict[str, Any]] = None):
        """ Applies a set of initial parameters to the camera """
        if not self.is_connected:
            raise RuntimeError("Camera is not connected.")

        # Subclasses can override this or provide their own defaults
        settings = {
            'exposure': 5000,
            'gain': 1.0,
            'framerate': 60.0,
            'pixel_format': 'Mono8',
            'hardware_trigger': True,
            'trigger_line': 'Line4',
            'binning': 1,
            'binning_mode': 'Sum',
            'gamma': 1.0,
            'blacks': 1.0,
            'roi': None
        }
        if config:
            settings.update(config)

        # Allow subclasses to run pre-config hooks if needed
        self._pre_apply_configuration(settings)

        # Cache the state before setting hardware
        self._pixel_format = settings['pixel_format']
        self._binning = settings['binning']
        self._binning_mode = settings['binning_mode']
        self._hardware_triggered = settings['hardware_trigger']
        self._trigger_line = settings['trigger_line']
        self._framerate = settings['framerate']
        self._exposure = settings['exposure']
        self._gain = settings['gain']
        self._blacks = settings['blacks']
        self._gamma = settings['gamma']

        # Apply to the hardware by calling the setters
        self.pixel_format = self._pixel_format
        self.binning = self._binning
        self.binning_mode = self._binning_mode
        self.hardware_triggered = self._hardware_triggered
        self.framerate = self._framerate
        self.exposure = self._exposure
        self.gain = self._gain
        self.blacks = self._blacks
        self.gamma = self._gamma

        # Set ROI last, it depends on other stuff
        if settings['roi']:
            self.roi = settings['roi']
        else:
            self._set_feature_value('OffsetX', 0)
            self._set_feature_value('OffsetY', 0)
            self._set_feature_value('Width', self._get_feature_max_value('Width'))
            self._set_feature_value('Height', self._get_feature_max_value('Height'))

        self._roi = (
            self._get_feature_value('OffsetX'), self._get_feature_value('OffsetY'),
            self._get_feature_value('Width'), self._get_feature_value('Height')
        )

        # Allow subclasses to run post-configuration hooks if needed
        self._post_apply_configuration(settings)

    # Hooks for subclasses
    def _pre_apply_configuration(self, settings: Dict[str, Any]):
        """ A hook for subclasses to run code before the main configuration is applied """
        self._set_feature_value('AcquisitionMode', 'Continuous')
        self._set_feature_value('ExposureAuto', 'Off')
        self._set_feature_value('GainAuto', 'Off')

    def _post_apply_configuration(self, settings: Dict[str, Any]):
        """ A hook for subclasses to run code after the main configuration is applied """
        pass

    @abc.abstractmethod
    def _get_feature_value(self, name: str) -> Any:
        """ Vendor-specific implementation to get a feature's value """
        pass

    @abc.abstractmethod
    def _set_feature_value(self, name: str, value: Any) -> Any:
        """ Vendor-specific implementation to set a feature's value. Should return the actual value set """
        pass

    @abc.abstractmethod
    def _get_feature_max_value(self, name: str) -> Any:
        """ Vendor-specific implementation to get a feature's maximum possible value """
        pass

    @property
    def blacks(self) -> str:
        return self._blacks

    @blacks.setter
    def blacks(self, value: float):
        self._blacks = self._set_feature_value('BlackLevel', value)

    @property
    def gain(self) -> str:
        return self._gain

    @gain.setter
    def gain(self, value: float):
        self._gain = self._set_feature_value('Gain', value)

    @property
    def gamma(self) -> str:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float):
        try:  # some cameras apparently require GammaEnable
            self._set_feature_value('GammaEnable', True)
        except AttributeError:
            pass
        self._gamma = self._set_feature_value('Gamma', value)

    @property
    def exposure(self) -> float:
        return self._exposure

    @exposure.setter
    def exposure(self, value: float):
        self._exposure = self._set_feature_value('ExposureTime', value)
        try:
            max_fps = self._get_feature_max_value('AcquisitionFrameRate')
            if self._framerate > max_fps:
                self.framerate = max_fps
        except AttributeError:
            pass  # Not all cameras link exposure and framerate this way

    @property
    def framerate(self) -> float:
        return self._framerate

    @framerate.setter
    def framerate(self, value: float):
        if self.hardware_triggered:
            self._framerate = value
            return
        try:
            self._set_feature_value('AcquisitionFrameRateEnable', True)
            actual_value = self._set_feature_value('AcquisitionFrameRate', value)
            self._framerate = actual_value
        except AttributeError:
            logger.warning(f"Camera {self.unique_id}: Could not set framerate explicitly.")
            self._framerate = self._get_feature_value('AcquisitionFrameRate')

    @property
    def pixel_format(self) -> str:
        return self._pixel_format

    @pixel_format.setter
    def pixel_format(self, value: str):
        was_grabbing = self.is_grabbing
        if was_grabbing:
            self.stop_grabbing()
        try:
            self._pixel_format = self._set_feature_value('PixelFormat', value)
        finally:
            if was_grabbing:
                self.start_grabbing()

    @property
    def hardware_triggered(self) -> bool:
        return self._hardware_triggered

    @hardware_triggered.setter
    def hardware_triggered(self, enabled: bool):
        if enabled:
            # apparently some SDKs use integers, others use strings so we do a bit of voodoo here
            trigger_source = f"Line{''.join([char for char in str(self._trigger_line) if char.isdigit()])}"

            self._set_feature_value('TriggerSelector', 'FrameStart')
            self._set_feature_value('TriggerMode', 'On')
            self._set_feature_value('TriggerSource', trigger_source)
            try:
                self._set_feature_value('AcquisitionFrameRateEnable', False)
            except AttributeError:
                pass
        else:
            self._set_feature_value('TriggerMode', 'Off')
            try:
                self._set_feature_value('AcquisitionFrameRateEnable', True)
            except AttributeError:
                pass
        self._hardware_triggered = enabled
        self.framerate = self._framerate

    @property
    def binning(self) -> int:
        return self._binning

    @binning.setter
    def binning(self, value: int):
        was_grabbing = self.is_grabbing
        if was_grabbing:
            self.stop_grabbing()

        h_val = self._set_feature_value('BinningHorizontal', value)
        v_val = self._set_feature_value('BinningVertical', value)

        if h_val != v_val:
            logger.warning(f"Binning mismatch! H={h_val} V={v_val}")

        self._binning = h_val

        self._set_feature_value('OffsetX', 0)
        self._set_feature_value('OffsetY', 0)
        self._set_feature_value('Width', self._get_feature_max_value('Width'))
        self._set_feature_value('Height', self._get_feature_max_value('Height'))

        self._roi = (0, 0, self._get_feature_value('Width'), self._get_feature_value('Height'))

        if was_grabbing:
            self.start_grabbing()

    @property
    def binning_mode(self) -> str:
        return self._binning_mode

    @binning_mode.setter
    def binning_mode(self, value: str):
        mode = 'Average' if value.lower() in ['a', 'avg', 'average'] else 'Sum'
        h_mode = self._set_feature_value('BinningHorizontalMode', mode)
        self._set_feature_value('BinningVerticalMode', mode)
        self._binning_mode = h_mode

    @property
    def temperature(self) -> Optional[float]:
        try:
            temp = self._get_feature_value('DeviceTemperature')
            return temp if 0 < temp < 420 else None
        except (AttributeError, Exception):
            return None

    @property
    def temperature_state(self) -> Optional[str]:
        try:
            return self._get_feature_value('TemperatureState')
        except (AttributeError, Exception):
            return None

    @property
    def sensor_shape(self) -> Tuple[int, int]:
        try:  # try GenICam standard names
            return self._get_feature_max_value('WidthMax'), self._get_feature_max_value('HeightMax')
        except AttributeError:  # fallback for some SDKs (Spinnaker)
            return self._get_feature_value('SensorWidth'), self._get_feature_value('SensorHeight')

    @property
    def roi(self) -> Tuple[int, int, int, int]:
        return self._roi

    @roi.setter
    def roi(self, value: Tuple[int, int, int, int]):
        was_grabbing = self.is_grabbing

        if was_grabbing:
            self.stop_grabbing()

        off_x, off_y, width, height = value

        actual_off_x = self._set_feature_value('OffsetX', off_x)
        actual_off_y = self._set_feature_value('OffsetY', off_y)
        actual_width = self._set_feature_value('Width', width)
        actual_height = self._set_feature_value('Height', height)

        self._roi = (actual_off_x, actual_off_y, actual_width, actual_height)

        if was_grabbing:
            self.start_grabbing()