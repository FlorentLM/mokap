import abc
import logging
from typing import Any, Dict, Optional, Tuple, Sequence, List
from mokap.core.cameras.interface import AbstractCamera
from mokap.utils import pretty_microseconds

logger = logging.getLogger(__name__)


class GenICamCamera(AbstractCamera, abc.ABC):
    """
    An abstract base class for any GenICam-compliant cameras.

    This class provides a concrete implementation for most camera properties
    (exposure, gain, etc.) by mapping them to standard GenICam feature names.

    Subclasses implement the vendor-specific methods for connecting,
    grabbing, and the low-level feature accessors (_get_feature_value, etc).
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
        self._black_level: Optional[float] = None
        self._gamma: Optional[float] = None
        self._roi: Optional[Tuple[int, int, int, int]] = None

        self._framerate_range_cache: Optional[Tuple[float, float]] = None

    def _try_set_feature(self, feature_names: Any, value: Any, required: bool = False) -> Any:
        """Attempt to set a feature. If required=False, warn instead of raising."""
        if isinstance(feature_names, str):
            feature_names = [feature_names]

        for name in feature_names:
            try:
                return self._set_feature_value(name, value)
            except AttributeError:
                continue

        # if it gets here none of the names worked
        if required:
            raise AttributeError(f'Could not set any of {feature_names} to {value}. Feature missing or locked.')
        else:
            logger.debug(f'Skipping setting {feature_names[0]} to {value} (feature not supported by {self.name}).')
            return None

    def _apply_configuration(self, config: Optional[Dict[str, Any]] = None):
        """Applies a set of initial parameters to the camera."""
        if not self.is_connected:
            raise RuntimeError("Camera is not connected.")

        # Subclasses can override this or provide their own defaults
        settings = {
            'exposure': 5000,
            'gain': 1.0,
            'framerate': 60.0,
            'pixel_format': 'Mono8',
            'hardware_trigger': True,
            'trigger_line': 4,
            'binning': 1,
            'binning_mode': 'Sum',
            'gamma': 1.0,
            'black_level': 1.0,
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
        self._black_level = settings['black_level']
        self._gamma = settings['gamma']

        # Apply to the hardware by calling the setters
        self.pixel_format = self._pixel_format
        self.binning = self._binning
        self.binning_mode = self._binning_mode
        self.hardware_triggered = self._hardware_triggered
        self.framerate = self._framerate
        self.exposure = self._exposure
        self.gain = self._gain
        self.black_level = self._black_level
        self.gamma = self._gamma

        # Set ROI last, it depends on other stuff
        if settings['roi']:
            self.roi = settings['roi']
        else:
            self._try_set_feature('OffsetX', 0)
            self._try_set_feature('OffsetY', 0)
            self._try_set_feature('Width', self._get_feature_max_value('Width'))
            self._try_set_feature('Height', self._get_feature_max_value('Height'))

        self._roi = (
            self._get_feature_value('OffsetX'), self._get_feature_value('OffsetY'),
            self._get_feature_value('Width'), self._get_feature_value('Height')
        )

        # Allow subclasses to run post-configuration hooks if needed
        self._post_apply_configuration(settings)

    # ────── Hooks ──────

    def _pre_apply_configuration(self, settings: Dict[str, Any]):
        """A hook for subclasses to run code before the main configuration is applied."""
        self._try_set_feature('AcquisitionMode', 'Continuous')
        self._try_set_feature('ExposureAuto', 'Off')
        self._try_set_feature('GainAuto', 'Off')

    def _post_apply_configuration(self, settings: Dict[str, Any]):
        """A hook for subclasses to run code after the main configuration is applied."""
        pass

    def _get_feature_range(self, name: str) -> Tuple[float, float]:
        """Helper to get the min/max of a float-based feature."""
        try:
            min_val = float(self._get_feature_min_value(name))
            max_val = float(self._get_feature_max_value(name))
            return min_val, max_val

        except AttributeError as e:
            logger.error(f"Could not retrieve range for '{name}': {e}")
            return 0.0, 0.0

    # ────── GenICam abstract contract (overridden by vendor implementations) ──────

    @abc.abstractmethod
    def _get_node_map(self) -> Any:
        """Vendor-specific implementation to available nodes."""
        pass

    @abc.abstractmethod
    def _get_feature_value(self, name: str) -> Any:
        """Vendor-specific implementation to get a feature's value."""
        pass

    @abc.abstractmethod
    def _set_feature_value(self, name: str, value: Any) -> Any:
        """Vendor-specific implementation to set a feature's value. Should return the actual value set."""
        pass

    @abc.abstractmethod
    def _get_feature_min_value(self, name: str) -> Any:
        """Vendor-specific implementation to get a feature's maximum possible value."""
        pass

    @abc.abstractmethod
    def _get_feature_max_value(self, name: str) -> Any:
        """Vendor-specific implementation to get a feature's maximum possible value."""
        pass

    @abc.abstractmethod
    def _get_feature_entries(self, name: str) -> List[str]:
        """Vendor-specific implementation to get all enum entries for a feature."""
        pass

    # ────── Core camera control properties ──────

    @property
    def exposure(self) -> float:
        return self._exposure

    @exposure.setter
    def exposure(self, value: float):
        if self.hardware_triggered and self._framerate > 0:
            # Max exposure must leave time for readout between triggers
            # Frame period in µs minus some margin for readout
            frame_period_us = 1e6 / self._framerate
            # Readout overhead depends on camera, but 5% of frame period should be fine
            max_exposure_for_trigger = frame_period_us * 0.95

            hw_min, hw_max = self.exposure_range
            effective_max = min(hw_max, max_exposure_for_trigger)

            if value > effective_max:
                logger.warning(
                    f"[{self.name}] Clamping exposure from {pretty_microseconds(value)}"
                    f" to {pretty_microseconds(effective_max)} (trigger at {self._framerate:.1f} Hz)"
                )
                value = effective_max

        self._exposure = self._try_set_feature('ExposureTime', value)

        if not self.hardware_triggered:
            # In software mode, re-apply framerate in case it needs clamping
            self.framerate = self._framerate

    @property
    def exposure_range(self) -> Tuple[float, float]:
        return self._get_feature_range('ExposureTime')

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, value: float):
        self._gain = self._try_set_feature('Gain', value)

    @property
    def gain_range(self) -> Tuple[float, float]:
        return self._get_feature_range('Gain')

    @property
    def black_level(self) -> float:
        return self._black_level

    @black_level.setter
    def black_level(self, value: float):
        self._black_level = self._try_set_feature('BlackLevel', value)

    @property
    def black_level_range(self) -> Tuple[float, float]:
        return self._get_feature_range('BlackLevel')

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float):
        self._try_set_feature('GammaEnable', True)  # some cameras apparently require this first
        self._gamma = self._try_set_feature('Gamma', value)

    @property
    def gamma_range(self) -> Tuple[float, float]:
        return self._get_feature_range('Gamma')


    @property
    def binning(self) -> int:
        return self._binning

    @binning.setter
    def binning(self, value: int):
        self._framerate_range_cache = None  # binning affects readout time, invalidate the cache

        was_grabbing = self.is_grabbing
        if was_grabbing:
            self.stop_grabbing()

        h_val = self._try_set_feature('BinningHorizontal', value)
        v_val = self._try_set_feature('BinningVertical', value)

        if h_val is None or v_val is None:
            # Fallback for cameras that only expose a single combined Binning feature
            val = self._try_set_feature('Binning', value)
            if val is None:
                logger.warning(f'Camera {self.name} does not support Binning.')
                # ROI update not needed if binning didn't change, so return
                if was_grabbing:
                    self.start_grabbing()
                return
            h_val, v_val = val, val

        if h_val != v_val:
            logger.warning(f"Binning mismatch! H={h_val} V={v_val}")

        self._binning = h_val

        self._try_set_feature('OffsetX', 0)
        self._try_set_feature('OffsetY', 0)
        self._try_set_feature('Width', self._get_feature_max_value('Width'))
        self._try_set_feature('Height', self._get_feature_max_value('Height'))

        self._roi = (0, 0, self._get_feature_value('Width'), self._get_feature_value('Height'))

        if was_grabbing:
            self.start_grabbing()

    @property
    def binning_mode(self) -> str:
        return self._binning_mode

    @binning_mode.setter
    def binning_mode(self, value: str):
        mode = 'Average' if value.lower() in ['a', 'avg', 'average'] else 'Sum'

        h_mode = self._try_set_feature('BinningHorizontalMode', mode)
        if h_mode is not None:
            self._try_set_feature('BinningVerticalMode', mode)
            self._binning_mode = h_mode
        else:
            # Fallback for cameras that only expose a single combined BinningMode feature
            fallback_mode = self._try_set_feature('BinningMode', mode)
            if fallback_mode is not None:
                self._binning_mode = fallback_mode
            else:
                logger.debug(f'Camera {self.name} does not support setting Binning Mode.')

    @property
    def available_binning_modes(self) -> List[str]:
        return self._get_feature_entries('BinningHorizontalMode')

    @property
    def framerate(self) -> float:
        return self._framerate

    @framerate.setter
    def framerate(self, value: float):

        # Always cache the desired value
        # (in hardware trigger mode, this acts as the 'expected' frequency for exposure calculations)
        self._framerate = value

        if not self.hardware_triggered:
            # enable framerate control
            self._try_set_feature('AcquisitionFrameRateEnable', True)

            # set the target framerate. The setter will clamp it
            actual_value_set = self._try_set_feature('AcquisitionFrameRate', value)

            if actual_value_set is not None:
                # update the cached value with what was actually set
                self._framerate = actual_value_set
                logger.debug(f"[{self.name}] Set framerate to {self._framerate} fps.")
            else:
                # Fallback for cameras that might not support explicit framerate control
                logger.warning(f"Camera {self.name} does not support explicit framerate control.")

                # Try to read the current framerate as a best-effort guess
                try:
                    self._framerate = float(self._get_feature_value('ResultingFrameRate'))
                except AttributeError:
                    logger.warning(f"[{self.name}] Could not read ResultingFrameRate.")
                    self._framerate = 0.0
        else:
            logger.warning(
                f"[{self.name}] Hardware trigger is on. Skipping setting framerate directly."
            )

    @property
    def framerate_range(self) -> Tuple[float, float]:

        if self._framerate_range_cache is not None:
            return self._framerate_range_cache

        try:
            min_fps = self._get_feature_min_value('AcquisitionFrameRate')
            # Disabling manual control allows to query the current maximum *possible* framerate
            self._try_set_feature('AcquisitionFrameRateEnable', False)
            max_fps = self._get_feature_value('ResultingFrameRate')

            # reactivate if needed
            if not self.hardware_triggered:
                self._try_set_feature('AcquisitionFrameRateEnable', True)

            self._framerate_range_cache = (float(min_fps), float(max_fps))
            return self._framerate_range_cache

        except AttributeError:
            # fallback if the feature isn't available (not cached, so retry in case it was transient)
            logger.warning(f"Could not determine settable framerate range for {self.unique_id}.")
            return 0.5, 500.0

    # ────── Image format and ROI properties ──────

    @property
    def roi(self) -> Tuple[int, int, int, int]:
        return self._roi

    @roi.setter
    def roi(self, value: Sequence[int]):
        self._framerate_range_cache = None  # ROI affects readout time, invalidate cache

        was_grabbing = self.is_grabbing

        try:
            if len(value) == 4:
                # Standard ROI: (offset_x, offset_y, width, height)
                off_x, off_y, width, height = value

                # Disable auto-centering if it exists, to ensure manual offsets are applied correctly
                # (no-op if the features don't exist)
                self._try_set_feature('CenterX', False)
                self._try_set_feature('CenterY', False)

                # Set size first, then offset
                self._try_set_feature('Width', width)
                self._try_set_feature('Height', height)
                self._try_set_feature('OffsetX', off_x)
                self._try_set_feature('OffsetY', off_y)

            elif len(value) == 2:
                # Centered ROI: (width, height)
                width, height = value

                # Set size first, and get the actual values that were set (they might be clamped)
                actual_width = self._try_set_feature('Width', width)
                actual_height = self._try_set_feature('Height', height)

                # Try camera's built-in centering feature
                center_x = self._try_set_feature('CenterX', True)
                center_y = self._try_set_feature('CenterY', True)

                if center_x is not None and center_y is not None:
                    logger.debug(f"Used camera's built-in centering for ROI ({width}x{height}) on {self.name}.")
                else:
                    # Camera doesn't support CenterX/Y (or Width/Height itself), calculate the offsets
                    logger.debug(f"Camera {self.name} lacks CenterX/Y support. Calculating centered ROI.")

                    # Get max dimensions for offset calculation (respects current binning, etc)
                    max_w = self._get_feature_max_value('Width')
                    max_h = self._get_feature_max_value('Height')

                    # Calculate centered offsets based on actual width/height (fallback to the
                    # requested size if even Width/Height itself wasn't settable)
                    off_x = (max_w - (actual_width if actual_width is not None else width)) // 2
                    off_y = (max_h - (actual_height if actual_height is not None else height)) // 2

                    # The SDK should handle rounding to the nearest valid increment
                    self._try_set_feature('OffsetX', off_x)
                    self._try_set_feature('OffsetY', off_y)
            else:
                raise ValueError(f"ROI must be a sequence of 2 or 4 elements, but got {len(value)}.")

            # After setting, read back the actual values and cache them
            self._roi = (
                self._get_feature_value('OffsetX'), self._get_feature_value('OffsetY'),
                self._get_feature_value('Width'), self._get_feature_value('Height')
            )
            logger.debug(f"ROI for {self.name} set to {self._roi}")

        finally:
            if was_grabbing:
                self.start_grabbing()

    @property
    def pixel_format(self) -> str:
        return self._pixel_format

    @pixel_format.setter
    def pixel_format(self, value: str):
        self._framerate_range_cache = None  # bit depth affects readout time, invalidate cache

        was_grabbing = self.is_grabbing
        if was_grabbing:
            self.stop_grabbing()
        try:
            self._pixel_format = self._try_set_feature('PixelFormat', value)
        finally:
            if was_grabbing:
                self.start_grabbing()

    @property
    def available_pixel_formats(self) -> List[str]:
        return self._get_feature_entries('PixelFormat')

    # ────── Triggering and synchronization ──────

    @property
    def hardware_triggered(self) -> bool:
        return self._hardware_triggered

    @hardware_triggered.setter
    def hardware_triggered(self, enabled: bool):
        self._framerate_range_cache = None  # trigger mode affects the achievable range, invalidate cache

        if enabled:
            # apparently some SDKs use integers, others use strings so we do a bit of voodoo here
            trigger_source = f"Line{''.join([char for char in str(self._trigger_line) if char.isdigit()])}"

            self._try_set_feature('TriggerSelector', 'FrameStart')
            self._try_set_feature('TriggerMode', 'On')
            self._try_set_feature('TriggerSource', trigger_source)
            self._try_set_feature('AcquisitionFrameRateEnable', False)
        else:
            self._try_set_feature('TriggerMode', 'Off')
            self._try_set_feature('AcquisitionFrameRateEnable', True)
        self._hardware_triggered = enabled
        self.framerate = self._framerate

    # ────── Sensor information (ro) ──────

    @property
    def resolution(self) -> Tuple[int, int]:
        try:  # try GenICam standard names
            return self._get_feature_max_value('WidthMax'), self._get_feature_max_value('HeightMax')
        except AttributeError:  # fallback for some SDKs (Spinnaker)
            return self._get_feature_value('SensorWidth'), self._get_feature_value('SensorHeight')

    @property
    def sensor_size(self) -> Optional[Tuple[float, float]]:
        # GenICam doesn't have a universal standard for physical sensor size.
        # Subclasses should override.
        return None

    @property
    def pixel_pitch(self) -> Optional[float]:
        # Similar to sensor_size, PixelSize is not strictly standard in the main namespace.
        return None

    # ────── Other (ro) information properties ──────

    @property
    def model(self) -> str:
        try:
            return self._get_feature_value('DeviceModelName')
        except (AttributeError, Exception):
            return "Unknown GenICam Model"

    @property
    def vendor(self) -> str:
        try:
            return self._get_feature_value('DeviceVendorName')
        except (AttributeError, Exception):
            return "Unknown Vendor"

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