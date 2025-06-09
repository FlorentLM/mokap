from pypylon import pylon
from pypylon import genicam as geni
import numpy as np
from typing import Any, Dict, Optional, Tuple

from mokap.core.cameras.interface import AbstractCamera


class BaslerCamera(AbstractCamera):
    """
    Concrete implementation of AbstractCamera for Basler Pylon devices
    """

    def __init__(self, pylon_device_info):
        # The pylon_device_info is the object we get from the factory
        self._device_info = pylon_device_info
        self._ptr: Optional[pylon.InstantCamera] = None
        super().__init__(unique_id=pylon_device_info.GetSerialNumber())

    def connect(self, config: Optional[Dict[str, Any]] = None) -> None:
        if self.is_connected:
            print(f"Camera {self.unique_id} is already connected.")
            return

        try:
            self._ptr = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(self._device_info))
            self._ptr.Open()
            self._is_connected = True

            self._apply_configuration(config)

            print(f"Connected to Basler camera: {self.unique_id}")

        except geni.GenericException as e:
            self._is_connected = False
            raise RuntimeError(f"Failed to connect to Basler camera {self.unique_id}: {e}") from e

    def _apply_configuration(self, config: Optional[Dict[str, Any]] = None):
        """ Applies a set of initial parameters to the camera """

        if not self.is_connected:
            raise RuntimeError("Camera is not connected.")

        # Default configuration
        settings = {
            'exposure': 5000,
            'gain': 1.0,
            'framerate': 60.0,
            'pixel_format': 'Mono8',
            'trigger': True,
            'trigger_line': 4,
            'binning': 1,
            'binning_mode': 'sum',
            'gamma': 1.0,
            'blacks': 1.0,
            'roi': None     # will be set automatically if not provided
        }
        if config:
            settings.update(config)

        # --- Caching the state ---
        # Initialize internal state variables before setting hardware
        self._pixel_format = settings['pixel_format']
        self._binning = settings['binning']
        self._binning_mode = settings['binning_mode']
        self._hardware_triggered = settings['trigger']
        self._trigger_line = settings['trigger_line']
        self._framerate = settings['framerate']
        self._exposure = settings['exposure']
        self._gain = settings['gain']
        self._blacks = settings['blacks']
        self._gamma = settings['gamma']

        # Set GenICam features
        self._set_feature('UserSetSelector', 'Default')
        self._ptr.UserSetLoad.Execute()

        self._set_feature('AcquisitionMode', 'Continuous')
        self._set_feature('ExposureMode', 'Timed')

        # Now, call the property setters to apply the initial cached values to hardware
        self.pixel_format = self._pixel_format
        self.binning = self._binning            # must be set before roi
        self.binning_mode = self._binning_mode  # must be set before roi
        self.hardware_triggered = self._hardware_triggered  # must be set before framerate
        self.framerate = self._framerate
        self.exposure = self._exposure
        self.gain = self._gain
        self.blacks = self._blacks
        self.gamma = self._gamma

        # Set ROI last, as it depends on other settings
        if settings['roi']:
            self.roi = settings['roi']
        else:
            # If no ROI is specified, default to full frame
            self._set_feature('OffsetX', 0)
            self._set_feature('OffsetY', 0)
            self._set_feature('Width', self._get_feature_max('Width'))
            self._set_feature('Height', self._get_feature_max('Height'))

        # After setting, cache the actual ROI
        self._roi = (
            self._get_feature('OffsetX'),
            self._get_feature('OffsetY'),
            self._get_feature('Width'),
            self._get_feature('Height')
        )

    def disconnect(self) -> None:
        if self.is_grabbing:
            self.stop_grabbing()

        if self._ptr and self._ptr.IsOpen():
            self._ptr.Close()

        self._ptr = None
        self._is_connected = False
        print(f"Disconnected from Basler camera: {self.unique_id}")

    def start_grabbing(self) -> None:
        if self.is_connected and not self.is_grabbing:
            self._ptr.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self._is_grabbing = True

    def stop_grabbing(self) -> None:
        if self.is_grabbing:
            self._ptr.StopGrabbing()
            self._is_grabbing = False

    def grab_frame(self, timeout_ms: int = 2000) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.is_grabbing:
            # For single grabs, we start/stop internally
            self._ptr.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        try:
            grab_result = self._ptr.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)

            if grab_result:
                metadata = {
                    'frame_number': grab_result.ImageNumber,
                    'timestamp': grab_result.TimeStamp
                }
                return grab_result.Array, metadata  # TODO: Does Array create a copy or not??
            else:
                raise IOError(f"Grab failed: {grab_result.GetErrorCode()} {grab_result.GetErrorDescription()}")
        finally:
            # Make sure we release the buffer
            if 'grab_result' in locals() and grab_result:
                grab_result.Release()
            if not self.is_grabbing:
                self._ptr.StopGrabbing()

    # --- PROPERTY IMPLEMENTATIONS ---

    @property
    def blacks(self) -> str:
        return self._blacks

    @blacks.setter
    def blacks(self, value: float):
        # The setter updates hardware, then the cache
        actual_value = self._set_feature('BlackLevel', value)
        self._blacks = actual_value

    @property
    def gain(self) -> str:
        return self._gain

    @gain.setter
    def gain(self, value: float):
        # The setter updates hardware, then the cache
        actual_value = self._set_feature('Gain', value)
        self._gain = actual_value

    @property
    def gamma(self) -> str:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float):
        # The setter updates hardware, then the cache
        actual_value = self._set_feature('Gamma', value)
        self._gamma = actual_value

    @property
    def exposure(self) -> float:
        return self._exposure

    @exposure.setter
    def exposure(self, value: float):
        # Set the hardware feature
        actual_value = self._set_feature('ExposureTime', value)     # value might be clamped
        # Update internal cache with the value that was actually set
        self._exposure = actual_value

        # Handle the side effect on framerate
        max_fps = self._get_feature_max('AcquisitionFrameRate')  # query hardware, necessary
        if self._framerate > max_fps:
            # If the current framerate target is now invalid, update it
            # and call the framerate setter
            self.framerate = max_fps

    @property
    def framerate(self) -> float:
        # if self.is_grabbing:
            # try:
            #     # This is the one getter that SHOULD query hardware for real-time feedback
            #     # TODO: Time the impact of this
            #     return self._get_feature('ResultingFrameRate')
            # except (AttributeError, geni.LogicalErrorException):
            #     pass  # Fall through to return the cached target

        # actually no we only want to return the cached version
        return self._framerate

    @framerate.setter
    def framerate(self, value: float):
        # If externally triggered, the camera's internal clock is off. We just cache the target value.
        if self.hardware_triggered:
            self._framerate = value
            return

        # In freerun mode, check against hardware limits
        max_fps = self._get_feature_max('AcquisitionFrameRate')  # query hardware, necessary
        actual_value = min(value, max_fps)

        # Set the hardware
        self._set_feature('AcquisitionFrameRateEnable', True)
        self._set_feature('AcquisitionFrameRate', actual_value)

        # Update the cache
        self._framerate = actual_value

    @property
    def pixel_format(self) -> str:
        return self._pixel_format

    @pixel_format.setter
    def pixel_format(self, value: str):
        # The setter updates hardware, then the cache
        actual_value = self._set_feature('PixelFormat', value)
        self._pixel_format = actual_value

    @property
    def hardware_triggered(self) -> bool:
        return self._hardware_triggered

    @hardware_triggered.setter
    def hardware_triggered(self, enabled: bool):

        if enabled:
            self._set_feature('TriggerSelector', 'FrameStart')
            self._set_feature('TriggerMode', 'On')
            self._set_feature('TriggerSource', f"Line{int(self._trigger_line)}")
            self._set_feature('AcquisitionFrameRateEnable', False)
        else:
            self._set_feature('TriggerMode', 'Off')
            self._set_feature('AcquisitionFrameRateEnable', True)

        # Update the cache
        self._hardware_triggered = enabled

        # Changing trigger mode can affect framerate, so we re-apply the target
        self.framerate = self._framerate

    @property
    def binning(self) -> int:
        return self._binning

    @binning.setter
    def binning(self, value: int):

        # It's safer to stop grabbing before changing binning / ROI
        was_grabbing = self.is_grabbing
        if was_grabbing: self.stop_grabbing()

        # Attempt to set both horizontal and vertical binning
        actual_h = self._set_feature('BinningHorizontal', value)
        actual_v = self._set_feature('BinningVertical', value)

        # Verify that both values were set identically
        if actual_h != actual_v:
            # trust the horizontal value as the primary one, but warn
            print(f"[WARN] Camera {self.name}: Binning mismatch! Horizontal={actual_h}, Vertical={actual_v}. Using {actual_h}.")

        # Cache the confirmed value
        self._binning = actual_h

        # After changing binning, the maximum ROI size and valid offsets change
        # It is safest to reset the ROI to the new maximum to avoid an invalid state
        self._set_feature('OffsetX', 0)
        self._set_feature('OffsetY', 0)
        self._set_feature('Width', self._get_feature_max('Width'))
        self._set_feature('Height', self._get_feature_max('Height'))

        # Update the cached ROI value to reflect the reset
        self._roi = (0, 0, self._get_feature('Width'), self._get_feature('Height'))

        if was_grabbing: self.start_grabbing()

    @property
    def binning_mode(self) -> str:
        return self._binning_mode

    @binning_mode.setter
    def binning_mode(self, value: str):
        if value.lower() in ['s', 'sum']:
            mode = 'Sum'
        elif value.lower() in ['a', 'avg', 'average']:
            mode = 'Average'
        else:
            # Default to sum because it's safer
            mode = 'Sum'

        actual_h_mode = self._set_feature('BinningHorizontalMode', mode)
        self._set_feature('BinningVerticalMode', mode)

        # Update cache
        self._binning_mode = actual_h_mode

    @property
    def temperature(self) -> Optional[float]:
        # Temperature is a read-only sensor value, so it should always query hardware
        try:
            # Check if the feature exists and is readable
            if geni.IsReadable(self._ptr.GetNodeMap().GetNode('DeviceTemperature')):
                temp = self._get_feature('DeviceTemperature')
                # Some cameras return 0 if the sensor is not yet warmed up, or 421 if no sensor
                # (yes a same model can come with or without the temperature sensor, thanks Basler...)
                return temp if (0 < temp < 421) else None

        except (geni.AccessException, geni.LogicalErrorException, AttributeError):
            # This handles cases where the feature doesn't exist or isn't implemented.
            return None

    @property
    def sensor_shape(self) -> Tuple[int, int]:
        """ Returns the maximum sensor resolution (width, height), independent of binning or ROI """

        # this is static for the hardware, so it's ok to query directly
        return self._get_feature_max('WidthMax'), self._get_feature_max('HeightMax')

    @property
    def roi(self) -> Tuple[int, int, int, int]:
        """ Returns the cached ROI (OffsetX, OffsetY, Width, Height) """
        return self._roi

    @roi.setter
    def roi(self, value: Tuple[int, int, int, int]):

        # It's safer to stop grabbing before changing critical ROI settings
        was_grabbing = self.is_grabbing
        if was_grabbing: self.stop_grabbing()

        off_x, off_y, width, height = value

        # Set hardware in the correct order: offset first, then size
        actual_off_x = self._set_feature('OffsetX', off_x)
        actual_off_y = self._set_feature('OffsetY', off_y)
        actual_width = self._set_feature('Width', width)
        actual_height = self._set_feature('Height', height)

        # Update the cache with the values that were actually set
        self._roi = (actual_off_x, actual_off_y, actual_width, actual_height)

        if was_grabbing: self.start_grabbing()

    # --- Generic feature getters / setters ---

    def _get_feature(self, name: str) -> Any:
        try:
            node = self._ptr.GetNodeMap().GetNode(name)
            if not geni.IsReadable(node):
                raise AttributeError(f"Feature '{name}' is not readable.")
            return node.GetValue()
        except geni.GenericException as e:
            raise AttributeError(f"Failed to get feature '{name}': {e}") from e

    def _set_feature(self, name: str, value: Any):
        try:
            node = self._ptr.GetNodeMap().GetNode(name)
            if not geni.IsWritable(node):
                raise AttributeError(f"Feature '{name}' is not writable.")

            # check the node's type using isinstance()
            if isinstance(node, geni.IEnumeration):
                value = str(value)
                node.FromString(value)
                return value

            elif isinstance(node, (geni.IFloat, geni.IInteger)):
                # This handles both float and integer types
                min_val, max_val = node.GetMin(), node.GetMax()

                # Ensure value is of the correct numeric type before clamping
                value = int(value) if isinstance(node, geni.IInteger) else float(value)

                clamped_value = max(min_val, min(max_val, value))
                if clamped_value != value:
                    print(f"[Warning] Clamped {name} from {value} to {clamped_value}")

                node.SetValue(clamped_value)
                return clamped_value

            elif isinstance(node, geni.IBoolean):
                node.SetValue(bool(value))
                return value

            else:
                # Fallback for other types
                node.SetValue(value)
                return value

        except geni.GenericException as e:
            raise AttributeError(f"Failed to set feature '{name}' to '{value}': {e}") from e

    def _get_feature_max(self, name: str) -> Any:
        try:
            node = self._ptr.GetNodeMap().GetNode(name)
            return node.GetMax()
        except geni.GenericException as e:
            raise AttributeError(f"Failed to get max for feature '{name}': {e}") from e