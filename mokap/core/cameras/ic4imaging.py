import logging
import queue
from typing import Any, Dict, List, Optional, Tuple

import imagingcontrol4 as ic4
import numpy as np
from mokap.core.cameras.genicam import GenICamCamera

logger = logging.getLogger(__name__)


def _IC4_feature_mapping() -> Dict[str, Any]:
    """Mapping from GenICam feature names to IC4 PropId constants."""

    properties = {
        'ExposureTime': 'EXPOSURE_TIME',
        'Gain': 'GAIN',
        'BlackLevel': 'BLACK_LEVEL',
        'Gamma': 'GAMMA',
        'AcquisitionMode': 'ACQUISITION_MODE',
        'ExposureAuto': 'EXPOSURE_AUTO',
        'GainAuto': 'GAIN_AUTO',
        'AcquisitionFrameRate': 'ACQUISITION_FRAME_RATE',
        'ResultingFrameRate': 'RESULTING_FRAME_RATE',
        'Width': 'WIDTH',
        'Height': 'HEIGHT',
        'OffsetX': 'OFFSET_X',
        'OffsetY': 'OFFSET_Y',
        'PixelFormat': 'PIXEL_FORMAT',
        'BinningHorizontal': 'BINNING_HORIZONTAL',
        'BinningVertical': 'BINNING_VERTICAL',
        'BinningHorizontalMode': 'BINNING_HORIZONTAL_MODE',
        'BinningVerticalMode': 'BINNING_VERTICAL_MODE',
        'TriggerSelector': 'TRIGGER_SELECTOR',
        'TriggerMode': 'TRIGGER_MODE',
        'TriggerSource': 'TRIGGER_SOURCE',
    }

    mapping = {}
    for feature_name, prop_name in properties.items():
        try:
            mapping[feature_name] = getattr(ic4.PropId, prop_name)
        except AttributeError:
            logger.debug(f'IC4 PropId.{prop_name} not available in imagingcontrol4')

    return mapping


FEATURE_MAPPING = _IC4_feature_mapping()


class _QueueSinkListener(ic4.QueueSinkListener):
    """Runs on IC4's delivery thread, hands frames to grab_frame()."""

    def __init__(self, buffer_count: int, frame_queue: queue.Queue):
        super().__init__()
        self._buffer_count = buffer_count
        self._queue = frame_queue

    def sink_connected(self, sink, image_type, min_buffers_required) -> bool:
        sink.alloc_and_queue_buffers(max(self._buffer_count, min_buffers_required))
        return True

    def sink_disconnected(self, sink):
        pass

    def frames_queued(self, sink) -> None:
        buffer = sink.pop_output_buffer()
        if buffer is None:
            return
        try:
            image_arr = buffer.numpy_wrap().copy()
            meta = {
                'frame_number': int(buffer.meta_data.device_frame_number),
                'timestamp': int(buffer.meta_data.device_timestamp_ns),
            }
            self._queue.put_nowait((image_arr, meta))
        except queue.Full:
            logger.warning('Internal frame queue overflow. Frame dropped.')
        finally:
            buffer.release()


class IC4ImagingCamera(GenICamCamera):
    """
    Implementation for The Imaging Source IC Imaging Control 4 cameras.
    (only adds IC4-specific connection, grabbing, and feature access)
    """

    def __init__(self, device_info: 'ic4.DeviceInfo'):
        self._device_info = device_info
        self._grabber: Optional[ic4.Grabber] = None
        self._sink: Optional[ic4.QueueSink] = None
        self._sink_buffer_count = 4
        self._frame_queue: Optional[queue.Queue] = None
        self._last_frame_signature: Optional[Tuple[int, int]] = None
        self._warned_features = set()

        super().__init__(unique_id=device_info.serial)

    def connect(self, config: Optional[Dict[str, Any]] = None) -> None:
        if self.is_connected:
            logger.warning(f"Camera {self.unique_id} is already connected.")
            return
        try:
            self._grabber = ic4.Grabber()
            self._grabber.device_open(self._device_info)
            self._is_connected = True
            self._apply_configuration(config)

        except ic4.IC4Exception as e:
            self._is_connected = False
            raise RuntimeError(f'Failed to connect to IC Imaging camera {self.unique_id}: {e}') from e

    def disconnect(self) -> None:
        if self.is_grabbing: self.stop_grabbing()
        if self._grabber:
            try:
                self._grabber.device_close()
            except ic4.IC4Exception as e:
                logger.warning(f'Error closing IC4 device: {e}')
        self._grabber = None
        self._sink = None
        self._is_connected = False

        logger.info(f'Disconnected from IC Imaging camera {self.unique_id}')

    def start_grabbing(self) -> None:

        if self.is_connected and not self.is_grabbing:
            self._frame_queue = queue.Queue(maxsize=10)

            if not self._sink or not self._sink.is_attached:
                self._sink = ic4.QueueSink(
                    _QueueSinkListener(self._sink_buffer_count, self._frame_queue),
                    max_output_buffers=1,
                )

            self._last_frame_signature = None

            try:
                # the sink listener queues buffers during sink_connected so streaming can start immediately
                self._grabber.stream_setup(self._sink, setup_option=ic4.StreamSetupOption.ACQUISITION_START)
                self._is_grabbing = True

            except ic4.IC4Exception as e:
                self._is_grabbing = False
                raise RuntimeError(f"Failed to start grabbing on {self.unique_id}: {e}") from e

    def stop_grabbing(self) -> None:

        if self.is_grabbing and self._grabber:
            try:
                self._grabber.stream_stop()
            except ic4.IC4Exception as e:
                logger.error(f'Error stopping grabbing on {self.unique_id}: {e}')
            finally:
                self._is_grabbing = False

    def grab_frame(self, timeout_ms: int = 2000) -> Tuple[np.ndarray, Dict[str, Any]]:

        if not self._sink or not self._sink.is_attached:
            raise IOError(f"Camera {self.unique_id} sink is not attached; is it grabbing?")

        try:
            while True:
                # frames are pushed onto the queue by _QueueSinkListener, on IC4's own thread
                image_arr, meta = self._frame_queue.get(timeout=timeout_ms / 1000.0)

                frame_signature = (meta['frame_number'], meta['timestamp'])
                if frame_signature == self._last_frame_signature:
                    logger.debug(f"{self.unique_id}: dropping duplicate IC4 buffer {frame_signature}")
                    continue
                self._last_frame_signature = frame_signature

                self._timestamp_buffer.append(meta['timestamp'])
                return image_arr, {'frame_number': meta['frame_number'], 'timestamp': meta['timestamp']}

        except queue.Empty:
            raise TimeoutError(f"Grab timed out after {timeout_ms}ms on {self.unique_id}")

        except Exception as e:
            raise IOError(f"Failed to grab frame from queue: {e}") from e

    # GenICam contract (IC4-specific implementation)

    def _get_node_map(self):
        if not self._grabber or not self.is_connected:
            raise RuntimeError('IC Imaging camera is not initialised.')

        return self._grabber.device_property_map

    def _get_feature_value(self, name: str) -> Any:
        try:
            prop = self._grabber.device_property_map.find(self._get_prop_id(name))
            if prop is None:
                raise AttributeError(f"Feature '{name}' not found")
            return prop.value

        except ic4.IC4Exception as e:
            raise AttributeError(f"Failed to get feature '{name}': {e}") from e

    def _set_feature_value(self, name: str, value: Any) -> Any:
        try:
            prop_id = self._get_prop_id(name)
            self._grabber.device_property_map.set_value(prop_id, value)
            prop = self._grabber.device_property_map.find(prop_id)
            return prop.value if prop else value

        except ic4.IC4Exception as e:
            raise AttributeError(f"Failed to set feature '{name}' to '{value}': {e}") from e

    def _get_feature_min_value(self, name: str) -> Any:
        try:
            prop = self._grabber.device_property_map.find(self._get_prop_id(name))
            if prop is None or not hasattr(prop, 'minimum'):
                raise AttributeError(f"Feature '{name}' has no minimum value")
            return prop.minimum

        except ic4.IC4Exception as e:
            raise AttributeError(f"Failed to get min for feature '{name}': {e}") from e

    def _get_feature_max_value(self, name: str) -> Any:
        try:
            prop = self._grabber.device_property_map.find(self._get_prop_id(name))
            if prop is None or not hasattr(prop, 'maximum'):
                raise AttributeError(f"Feature '{name}' has no maximum value")
            return prop.maximum

        except ic4.IC4Exception as e:
            raise AttributeError(f"Failed to get max for feature '{name}': {e}") from e

    def _get_feature_entries(self, name: str) -> List[str]:
        try:
            prop = self._grabber.device_property_map.find(self._get_prop_id(name))
            if prop is None or not hasattr(prop, 'entries'):
                return []
            return [entry.name for entry in prop.entries]

        except ic4.IC4Exception as e:
            raise AttributeError(f"Failed to get entries for feature '{name}': {e}") from e

    def _get_prop_id(self, name: str) -> Any:
        """Converts a GenICam feature name to an IC4 PropId (mapping -> attribute -> UPPER_SNAKE_CASE -> string)."""

        if name in FEATURE_MAPPING:
            return FEATURE_MAPPING[name]

        try:
            return getattr(ic4.PropId, name)
        except AttributeError:
            pass

        upper_name = ''.join(['_' + c if c.isupper() else c for c in name]).lstrip('_').upper()
        try:
            return getattr(ic4.PropId, upper_name)
        except AttributeError:
            return name

    # IC4 specific overrides
    # (IC4 SDK does not clamp out-of-range values)

    @property
    def framerate(self) -> float:
        if self.hardware_triggered and self._framerate is not None:
            return self._framerate
        try:
            self._framerate = float(self._get_feature_value('AcquisitionFrameRate'))
        except AttributeError:
            pass
        return self._framerate

    @framerate.setter
    def framerate(self, value: float):
        self._framerate = float(value)

        if self.hardware_triggered:
            # cache only (IC4 has no AcquisitionFrameRateEnable)
            return

        try:
            self._try_set_feature('AcquisitionMode', 'Continuous')
            min_fps, max_fps = self.framerate_range
            clamped_value = max(min_fps, min(self._framerate, max_fps))
            self._framerate = self._set_feature_value('AcquisitionFrameRate', clamped_value)

        except AttributeError as e:
            logger.warning(f'Camera {self.name} does not support framerate control: {e}')

    @property
    def framerate_range(self) -> Tuple[float, float]:
        return self._get_feature_range('AcquisitionFrameRate')

    @property
    def exposure(self) -> float:
        try:
            self._exposure = float(self._get_feature_value('ExposureTime'))
        except AttributeError:
            pass
        return self._exposure

    @exposure.setter
    def exposure(self, value: float):
        was_grabbing = self.is_grabbing
        if was_grabbing:
            self.stop_grabbing()
        try:
            min_exp, max_exp = self.exposure_range
            clamped_value = max(min_exp, min(value, max_exp))
            self._exposure = self._set_feature_value('ExposureTime', clamped_value)

        except AttributeError as e:
            logger.warning(f"Camera {self.name} does not support exposure control: {e}")

        finally:
            if was_grabbing:
                self.start_grabbing()

    @property
    def exposure_range(self) -> Tuple[float, float]:
        return self._get_feature_range('ExposureTime')

    @property
    def gain(self) -> float:
        try:
            self._gain = float(self._get_feature_value('Gain'))
        except AttributeError:
            pass
        return self._gain

    @gain.setter
    def gain(self, value: float):
        was_grabbing = self.is_grabbing
        if was_grabbing:
            self.stop_grabbing()
        try:
            min_gain, max_gain = self.gain_range
            clamped_value = max(min_gain, min(value, max_gain))
            self._gain = self._set_feature_value('Gain', clamped_value)

        except AttributeError as e:
            logger.warning(f"Camera {self.name} does not support gain control: {e}")

        finally:
            if was_grabbing:
                self.start_grabbing()

    @property
    def gain_range(self) -> Tuple[float, float]:
        return self._get_feature_range('Gain')

    # Triggering (IC4-specific)

    @property
    def hardware_triggered(self) -> bool:
        return self._hardware_triggered

    @hardware_triggered.setter
    def hardware_triggered(self, enabled: bool):
        self._framerate_range_cache = None

        if enabled:
            try:
                # Selecting trigger target first, since on many TIS sensors this resets TriggerMode to Off
                self._set_feature_value('TriggerSelector', 'FrameStart')
                self._configure_trigger_overlap()

                try:
                    self._set_feature_value('TriggerActivation', 'FallingEdge')
                except AttributeError:
                    self._try_set_feature('TriggerActivation', 'RisingEdge')

                self._set_feature_value('TriggerMode', 'On')
                self._hardware_triggered = True

            except AttributeError as e:
                logger.error(f"{self.unique_id}: failed to configure hardware trigger: {e}")
                self._hardware_triggered = False
                return
        else:
            self._try_set_feature('TriggerMode', 'Off')
            self._hardware_triggered = False

        self.framerate = self._framerate

    def _configure_trigger_overlap(self) -> None:
        """
        Configure overlap/timing registers so exposure N+1 can run while readout for N runs.
        """

        self._try_set_feature('TriggerMode', 'Off')
        self._try_set_feature('TriggerDelay', 0.0)
        self._try_set_feature('TriggerMask', 0.0)
        self._try_set_feature('TriggerDebouncer', 0.0)
        self._try_set_feature('TriggerDenoise', 0.0)
        self._try_set_feature('IMXLowLatencyTriggerMode', False)     # must stay False
        self._set_first_supported_enum('IMXTriggerTiming', ['Fast', 'HighSpeed', 'Overlap'])

        try:
            max_fps = float(self._get_feature_max_value('AcquisitionFrameRate'))
            self._set_feature_value('AcquisitionFrameRate', max_fps if max_fps > 0 else 120.0)
        except AttributeError as e:
            logger.debug(f"{self.unique_id}: could not raise internal frame-clock headroom: {e}")

        self._set_first_supported_enum('TriggerOperation', ['Fast', 'Overlap'])
        self._set_first_supported_enum('TriggerOverlap', ['ReadOut', 'PreviousFrame'])

    def _feature_exists(self, name: str) -> bool:
        try:
            return self._grabber is not None and self._grabber.device_property_map.find(self._get_prop_id(name)) is not None
        except ic4.IC4Exception:
            return False

    def _set_first_supported_enum(self, name: str, candidates: List[str]) -> Optional[Any]:
        """Tries each candidate enum value in order, keeping the first one that actually took."""

        if not self._feature_exists(name):
            return None

        for candidate in candidates:
            try:
                self._set_feature_value(name, candidate)
                readback = self._get_feature_value(name)
            except AttributeError:
                continue

            if str(readback).lower() == str(candidate).lower():
                return readback

        return None
