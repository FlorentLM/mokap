import abc
from collections import deque

import numpy as np
from typing import Tuple, Dict, Any, Optional, Sequence, List

CAMERAS_COLOURS = ['#3498db', '#f4d03f', '#27ae60', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#F5A7D4', '#34495e', '#bdc3c7',
               '#2471a3', '#d4ac0d', '#186a3b', '#922b21', '#6c3483', '#d35400', '#117a65', '#e699db', '#1c2833', '#707b7c']


class AbstractCamera(abc.ABC):
    """
    An abstract base class that defines brand-agnostic interface for a camera.
    """

    def __init__(self, unique_id: str):
        self._unique_id = unique_id
        self._name = unique_id  # default name to the serial number
        self._is_connected = False
        self._is_grabbing = False
        self._timestamp_buffer = deque(maxlen=30)  # buffer to calculate actual framerate

    @property
    def unique_id(self) -> str:
        """Unique identifier for the camera, typically the serial number."""
        return self._unique_id

    @property
    def name(self) -> str:
        """user-friendly name for the camera."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Set a user-friendly name for the camera."""
        self._name = value

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_grabbing(self) -> bool:
        return self._is_grabbing

    # ────── Core methods ──────

    @abc.abstractmethod
    def connect(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Connect to the camera and apply an initial configuration."""
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the camera."""
        pass

    @abc.abstractmethod
    def start_grabbing(self) -> None:
        """Start continuous acquisition of frames."""
        pass

    @abc.abstractmethod
    def stop_grabbing(self) -> None:
        """Stop acquisition of frames."""
        pass

    @abc.abstractmethod
    def grab_frame(self, timeout_ms: int = 1000) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Grab a single frame from the camera.

        Returns:
            - The image data
            - A dictionary of metadata (e.g. {'frame_number': int, 'timestamp': int})
        """
        pass

    # ────── Core camera control properties ──────

    @property
    @abc.abstractmethod
    def exposure(self) -> float:
        """Exposure time in microseconds (µs)."""
        pass

    @exposure.setter
    @abc.abstractmethod
    def exposure(self, value: float):
        pass

    @property
    @abc.abstractmethod
    def exposure_range(self) -> Tuple[float, float]:
        """Min and max exposure."""
        pass

    @property
    @abc.abstractmethod
    def gain(self) -> float:
        """Gain in device-specific units (e.g. dB or a raw value)."""
        pass

    @gain.setter
    @abc.abstractmethod
    def gain(self, value: float):
        pass

    @property
    @abc.abstractmethod
    def gain_range(self) -> Tuple[float, float]:
        """Min and max gain."""
        pass

    @property
    @abc.abstractmethod
    def black_level(self) -> float:
        """Black level offset. Sometimes called 'Brightness' or 'Offset'."""
        pass

    @black_level.setter
    @abc.abstractmethod
    def black_level(self, value: float):
        pass

    @property
    @abc.abstractmethod
    def black_level_range(self) -> Tuple[float, float]:
        """Min and max black level."""
        pass

    @property
    @abc.abstractmethod
    def gamma(self) -> float:
        """Gamma correction value."""
        pass

    @gamma.setter
    @abc.abstractmethod
    def gamma(self, value: float):
        pass

    @property
    @abc.abstractmethod
    def gamma_range(self) -> Tuple[float, float]:
        """Min and max gamma."""
        pass

    @property
    @abc.abstractmethod
    def binning(self) -> int:
        """The binning factor (1 for 1x1, 2 for 2x2, etc)."""
        pass

    @binning.setter
    @abc.abstractmethod
    def binning(self, value: int):
        pass

    @property
    @abc.abstractmethod
    def binning_mode(self) -> str:
        """The binning mode, typically 'sum' or 'average'."""
        pass

    @binning_mode.setter
    @abc.abstractmethod
    def binning_mode(self, value: str):
        pass

    @property
    @abc.abstractmethod
    def available_binning_modes(self) -> List[str]:
        """Returns a list of available binning mode strings."""
        pass

    @property
    @abc.abstractmethod
    def framerate(self) -> float:
        """Acquisition framerate in fps."""
        pass

    @framerate.setter
    @abc.abstractmethod
    def framerate(self, value: float):
        pass

    @property
    @abc.abstractmethod
    def framerate_range(self) -> Tuple[float, float]:
        """Min and max framerate."""
        pass

    @property
    def measured_framerate(self) -> float:
        """
        Calculates the actual framerate based on the history of timestamps.
        """
        if len(self._timestamp_buffer) < 2:
            return 0.0

        diffs = np.diff(np.array(self._timestamp_buffer))
        # Filter out crazy values like if a timestamp reset happened or if diff is 0 (duplicate frames)
        valid_diffs = diffs[diffs > 0]

        if len(valid_diffs) == 0:
            return 0.0

        avg_diff_ns = float(np.mean(valid_diffs))
        if avg_diff_ns == 0:
            return 0.0

        return 1e9 / avg_diff_ns

    # ────── Image format and ROI properties ──────

    @property
    @abc.abstractmethod
    def roi(self) -> Tuple[int, int, int, int]:
        """Region of Interest as (X offset, Y offset, width, height)."""
        pass

    @roi.setter
    @abc.abstractmethod
    def roi(self, value: Sequence[int]):
        pass

    @property
    def image_size(self) -> Tuple[int, int]:
        """Image size in pixels (width, height) based on ROI."""
        x, y, w, h = self.roi
        return w, h

    @image_size.setter
    def image_size(self, value: Sequence[int]):
        new_w, new_h = value
        x, y, w, h = self.roi
        self.roi = x, y, new_w, new_h

    @property
    @abc.abstractmethod
    def pixel_format(self) -> str:
        """Pixel format ("Mono8", "BayerRG8", etc)."""
        pass

    @pixel_format.setter
    @abc.abstractmethod
    def pixel_format(self, value: str):
        pass

    @property
    @abc.abstractmethod
    def available_pixel_formats(self) -> List[str]:
        """Returns a list of available pixel format strings."""
        pass

    # ────── Triggering and synchronization ──────

    @property
    @abc.abstractmethod
    def hardware_triggered(self) -> bool:
        """True if hardware trigger is enabled, False otherwise."""
        pass

    @hardware_triggered.setter
    @abc.abstractmethod
    def hardware_triggered(self, enabled: bool):
        pass

    # ────── Sensor information (ro) ──────

    @property
    @abc.abstractmethod
    def resolution(self) -> Tuple[int, int]:
        """The (width, height) of the camera's sensor in pixels (independent of ROI / binning)."""
        pass

    @property
    @abc.abstractmethod
    def sensor_size(self) -> Optional[Tuple[float, float]]:
        """Physical sensor dimensions (width, height) in millimeters."""
        pass

    @property
    @abc.abstractmethod
    def pixel_pitch(self) -> Optional[float]:
        """Pixel pitch in micrometers (μm). Can be derived from sensor_size/sensor_resolution."""
        pass

    # ────── Other (ro) information properties ──────

    @property
    @abc.abstractmethod
    def model(self) -> str:
        """Camera model string (e.g. 'acA1920-40gm')."""
        pass

    @property
    @abc.abstractmethod
    def vendor(self) -> str:
        """Camera vendor/manufacturer (e.g. 'Basler')."""
        pass

    @property
    @abc.abstractmethod
    def temperature(self) -> Optional[float]:
        """Core device temperature in Celsius, if available."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.unique_id}, connected={self.is_connected})"

