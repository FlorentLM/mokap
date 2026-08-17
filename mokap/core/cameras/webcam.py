import logging
import cv2
import numpy as np
import time
from typing import Any, Dict, Optional, Tuple, List, Sequence
from mokap.core.cameras.interface import AbstractCamera

logger = logging.getLogger(__name__)

# Fallback ranges for webcams (since OpenCV cannot query min/max values directly)
_FALLBACK_RANGES = {
    'framerate': (1.0, 120.0),
    'brightness': (0.0, 255.0),
    'contrast': (0.0, 255.0),
    'saturation': (0.0, 255.0),
    'hue': (0.0, 180.0),
    'gain': (0.0, 255.0),
    'exposure': (-15.0, 10.0),  # negative often means log-based in UVC
    'black_level': (0.0, 255.0),
    'gamma': (1.0, 500.0),
    'white_balance': (2000.0, 10000.0),
    'sharpness': (0.0, 255.0),
}


class WebcamCamera(AbstractCamera):
    """
    Concrete implementation of AbstractCamera for generic USB webcams using OpenCV.
    """

    def __init__(self, camera_index: int):
        """
        camera_index: integer index of the camera (for example 0 for /dev/video0 on Unix)
        """
        self._index = camera_index
        self._ptr: Optional[cv2.VideoCapture] = None
        self._frame_counter = 0
        super().__init__(unique_id=f'webcam_{camera_index}')

    def _apply_configuration(self, config: Optional[Dict[str, Any]] = None):
        """Applies a set of initial parameters to the camera."""
        if not self.is_connected:
            raise RuntimeError('Camera is not connected.')

        settings = {
            'framerate': 30.0,
            'pixel_format': 'BGR8',
            'roi': (0, 0, 640, 480),
        }
        if config:
            settings.update(config)

        self._roi = settings['roi']
        self._framerate = settings['framerate']

        # Apply resolution and FPS first
        self.roi = self._roi
        self.framerate = self._framerate

        # Try to disable auto-exposure to allow manual control,
        # but don't crash if it fails (many webcams force auto)
        self._set_cv2_property(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 or 0.75 often maps to Manual on UVC

        # Read back current values to populate cache
        self._exposure = self.exposure
        self._gain = self.gain
        self._gamma = self.gamma
        self._black_level = self.black_level

    def _get_cv2_property(self, prop_id: int) -> float:
        """
        Wrapper to get CV2 property.
        Returns -1.0 if property is unsupported or camera is disconnected.
        """
        if self._ptr and self._ptr.isOpened():
            val = self._ptr.get(prop_id)
            # OpenCV often returns -1 for unsupported features
            return val
        return -1.0

    def _set_cv2_property(self, prop_id: int, value: Any) -> bool:
        if self._ptr and self._ptr.isOpened():
            return self._ptr.set(prop_id, value)
        return False

    def _is_supported(self, prop_id: int) -> bool:
        """Checks if a property is supported by reading it."""
        val = self._get_cv2_property(prop_id)
        return val != -1.0

    # Core methods

    def connect(self, config: Optional[Dict[str, Any]] = None) -> None:
        if self.is_connected:
            logger.warning(f'Camera {self.unique_id} is already connected.')
            return

        try:
            # On Windows, cv2.CAP_DSHOW is often more robust for settings
            # On Linux, default (V4L2) is usually fine.
            self._ptr = cv2.VideoCapture(self._index)
            # self._ptr = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
            # TODO: test more whether CAP_DSHOW is more stable (on Windows probably?)

            if not self._ptr.isOpened():
                raise IOError(f'Cannot open webcam with index {self._index}')

            self._is_connected = True
            self._apply_configuration(config)
            logger.info(f'Connected to Webcam: {self.unique_id}')

        except Exception as e:
            self._is_connected = False
            self._ptr = None
            raise RuntimeError(f'Failed to connect to Webcam {self.unique_id}: {e}') from e

    def disconnect(self) -> None:
        if self._ptr and self._ptr.isOpened():
            self._ptr.release()

        self._ptr = None
        self._is_connected = False
        self._is_grabbing = False
        logger.info(f'Disconnected from Webcam: {self.unique_id}')

    def start_grabbing(self) -> None:
        if self.is_connected:
            self._is_grabbing = True

    def stop_grabbing(self) -> None:
        self._is_grabbing = False

    def grab_frame(self, timeout_ms: int = 2000) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.is_connected or not self._ptr.isOpened():
            raise RuntimeError('Webcam is not connected.')

        ret, frame = self._ptr.read()
        timestamp = time.time_ns()

        if not ret or frame is None:
            raise IOError('Failed to grab frame from webcam.')

        self._frame_counter += 1
        self._timestamp_buffer.append(timestamp)

        metadata = {
            'frame_number': self._frame_counter,
            'timestamp': timestamp
        }
        return frame, metadata

    # Properties

    @property
    def exposure(self) -> float:
        val = self._get_cv2_property(cv2.CAP_PROP_EXPOSURE)
        return val if val != -1.0 else 0.0

    @exposure.setter
    def exposure(self, value: float):
        if self._set_cv2_property(cv2.CAP_PROP_EXPOSURE, value):
            self._exposure = value

    @property
    def exposure_range(self) -> Tuple[float, float]:
        # If camera returns -1, return (0, 0) to disable slider in GUI
        if not self._is_supported(cv2.CAP_PROP_EXPOSURE):
            return 0.0, 0.0
        return _FALLBACK_RANGES['exposure']

    @property
    def gain(self) -> float:
        val = self._get_cv2_property(cv2.CAP_PROP_GAIN)
        return val if val != -1.0 else 0.0

    @gain.setter
    def gain(self, value: float):
        if self._set_cv2_property(cv2.CAP_PROP_GAIN, value):
            self._gain = value

    @property
    def gain_range(self) -> Tuple[float, float]:
        if not self._is_supported(cv2.CAP_PROP_GAIN):
            return 0.0, 0.0
        return _FALLBACK_RANGES['gain']

    @property
    def black_level(self) -> float:
        val = self._get_cv2_property(cv2.CAP_PROP_BRIGHTNESS)
        return val if val != -1.0 else 0.0

    @black_level.setter
    def black_level(self, value: float):
        if self._set_cv2_property(cv2.CAP_PROP_BRIGHTNESS, value):
            self._black_level = value

    @property
    def black_level_range(self) -> Tuple[float, float]:
        if not self._is_supported(cv2.CAP_PROP_BRIGHTNESS):
            return 0.0, 0.0
        return _FALLBACK_RANGES['black_level']

    @property
    def gamma(self) -> float:
        val = self._get_cv2_property(cv2.CAP_PROP_GAMMA)
        return val if val != -1.0 else 0.0

    @gamma.setter
    def gamma(self, value: float):
        if self._set_cv2_property(cv2.CAP_PROP_GAMMA, value):
            self._gamma = value

    @property
    def gamma_range(self) -> Tuple[float, float]:
        if not self._is_supported(cv2.CAP_PROP_GAMMA):
            return 0.0, 0.0
        return _FALLBACK_RANGES['gamma']

    # Industrial Features (Explicitly Unsupported)

    @property
    def binning(self) -> int:
        return 1

    @binning.setter
    def binning(self, value: int):
        if value != 1:
            logger.warning('Webcams do not support hardware binning. Ignoring.')

    @property
    def binning_mode(self) -> str:
        return "N/A"

    @binning_mode.setter
    def binning_mode(self, value: str):
        logger.warning('Webcams do not support hardware binning. Ignoring.')
        pass

    @property
    def available_binning_modes(self) -> List[str]:
        return []

    @property
    def framerate(self) -> float:
        val = self._get_cv2_property(cv2.CAP_PROP_FPS)
        return val if val != -1.0 else 0.0

    @framerate.setter
    def framerate(self, value: float):
        if self._set_cv2_property(cv2.CAP_PROP_FPS, value):
            self._framerate = value

    @property
    def framerate_range(self) -> Tuple[float, float]:
        if not self._is_supported(cv2.CAP_PROP_FPS):
            return 0.0, 0.0
        return _FALLBACK_RANGES['framerate']

    @property
    def roi(self) -> Tuple[int, int, int, int]:
        return self._roi

    @roi.setter
    def roi(self, value: Sequence[int]):
        # Webcams only support changing resolution (width/height), not offsets
        if len(value) == 4:
            _, _, width, height = value
        elif len(value) == 2:
            width, height = value
        else:
            return

        # Attempt to set
        self._set_cv2_property(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._set_cv2_property(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Read back what the hardware actually accepted
        actual_w = int(self._get_cv2_property(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._get_cv2_property(cv2.CAP_PROP_FRAME_HEIGHT))

        # cache the value that was set
        self._roi = (0, 0, actual_w, actual_h)

    @property
    def hardware_triggered(self) -> bool:
        return False

    @hardware_triggered.setter
    def hardware_triggered(self, enabled: bool):
        if enabled:
            logger.warning("Webcams do not support hardware triggering. Ignoring.")

    @property
    def pixel_format(self) -> str:
        return 'BGR8'

    @pixel_format.setter
    def pixel_format(self, value: str):
        logger.warning(f'Webcam pixel format cannot be changed. Ignoring.')

    @property
    def available_pixel_formats(self) -> List[str]:
        return ['BGR8']

    # Read-Only Info

    @property
    def resolution(self) -> Tuple[int, int]:
        # There's no reliable way to get max sensor shape
        # so return the current resolution as a stand-in
        return self.roi[2], self.roi[3]

    @property
    def sensor_size(self) -> Optional[Tuple[float, float]]:
        return None

    @property
    def pixel_pitch(self) -> Optional[float]:
        return None

    # Other (ro) information properties

    @property
    def model(self) -> str:
        return f"Webcam #{self._index}"

    @property
    def vendor(self) -> str:
        return "Generic UVC"

    @property
    def temperature(self) -> Optional[float]:
        return None

    @property
    def temperature_state(self) -> Optional[str]:
        return None
