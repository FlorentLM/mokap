import logging
import time
import queue

import imagingcontrol4 as ic4
import numpy as np
from typing import Any, Dict, Optional, Tuple

from mokap.core.cameras.genicam import GenICamCamera

logger = logging.getLogger(__name__)


def _build_feature_mapping():
    """Build mapping from GenICam names to IC4 PropId constants"""
    mapping = {}
    properties_to_try = {
        'ExposureTime': 'EXPOSURE_TIME',
        'Gain': 'GAIN',
        'BlackLevel': 'BLACK_LEVEL',
        'Gamma': 'GAMMA',
        'AcquisitionMode': 'ACQUISITION_MODE',
        'ExposureAuto': 'EXPOSURE_AUTO',
        'GainAuto': 'GAIN_AUTO',
        # Note: IC4 does not have AcquisitionFrameRateEnable - it uses AcquisitionMode instead
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
    
    for feature_name, prop_name in properties_to_try.items():
        try:
            mapping[feature_name] = getattr(ic4.PropId, prop_name)
        except AttributeError:
            logger.debug(f"IC4 PropId.{prop_name} not available in imagingcontrol4")
    
    return mapping


FEATURE_MAPPING = _build_feature_mapping()

class _QueueSinkListener(ic4.QueueSinkListener):
    def __init__(self, buffer_count: int, frame_queue: queue.Queue):
        super().__init__()
        self._buffer_count = buffer_count
        self._queue = frame_queue

    def sink_connected(self, sink, image_type, min_buffers_required) -> bool:
        sink.alloc_and_queue_buffers(max(self._buffer_count, min_buffers_required))
        return True

    def sink_disconnected(self, sink):
        pass

    def frames_queued(self, sink):
        # Safely extract the buffer on the IC4 thread
        buffer = sink.pop_output_buffer()
        if buffer is not None:
            try:

                # Use getattr to bypass ctypes isolation issues
                last_arrival = getattr(self, '_last_arrival_ms', 0.0)
                
                current_ms = time.monotonic_ns() / 1_000_000.0
                delta_ms = current_ms - last_arrival if last_arrival else 0.0
                
                # Update the attribute safely
                self._last_arrival_ms = current_ms

                # 2. Extract camera internal hardware timestamp
                hw_timestamp_ns = int(buffer.meta_data.device_timestamp_ns)
                hw_timestamp_ms = hw_timestamp_ns / 1_000_000.0
                frame_num = int(buffer.meta_data.device_frame_number)

                # 3. Print the log statement
                logger.info(
                    f"[LISTENER] Frame #{frame_num} received | "
                    f"System Time: {current_ms:.2f} ms | "
                    f"Delta: {delta_ms:.2f} ms | "
                    f"HW Timestamp: {hw_timestamp_ms:.2f} ms"
                )

                # Wrap or copy the array on the callback thread
                image_arr = buffer.numpy_wrap().copy()
                meta = {
                    'frame_number': int(buffer.meta_data.device_frame_number),
                    'timestamp': int(buffer.meta_data.device_timestamp_ns),
                }
                # Hand it off to the consumer thread without blocking
                self._queue.put_nowait((image_arr, meta))
            except queue.Full:
                logger.warning("Internal frame queue overflow. Frame dropped.")
            finally:
                buffer.release()


class IC4ImagingCamera(GenICamCamera):
    """
    Implementation for The Imaging Source IC Imaging Control 4 cameras
    (only adds IC4-specific connection, grabbing, and feature access)
    """

    def __init__(self, device_info: ic4.DeviceInfo):
        self._device_info = device_info
        self._grabber: Optional[ic4.Grabber] = None
        self._sink: Optional[ic4.QueueSink] = None
        self._sink_buffer_count = 4
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

            logger.info(f"Connected to IC Imaging camera {self.unique_id}")

        except ic4.IC4Exception as e:
            self._is_connected = False
            raise RuntimeError(f"Failed to connect to IC Imaging camera {self.unique_id}: {e}") from e

    def disconnect(self) -> None:
        if self.is_grabbing: self.stop_grabbing()
        if self._grabber:
            try:
                self._grabber.device_close()
            except ic4.IC4Exception as e:
                logger.warning(f"Error closing IC4 device: {e}")
        self._grabber = None
        self._sink = None
        self._is_connected = False

        logger.info(f"Disconnected from IC Imaging camera {self.unique_id}")

    def start_grabbing(self) -> None:
        if self.is_connected and not self.is_grabbing:
            self._frame_queue = queue.Queue(maxsize=10) # Prevent memory ballooning
            if not self._sink or not self._sink.is_attached:
                self._sink = ic4.QueueSink(
                    _QueueSinkListener(self._sink_buffer_count,self._frame_queue),
                    max_output_buffers=1,
                )

            self._last_frame_signature = None

            try:
                # The sink listener queues buffers during sink_connected so the stream can start immediately.
                self._grabber.stream_setup(self._sink, setup_option=ic4.StreamSetupOption.ACQUISITION_START)
                
                self._is_grabbing = True
                logger.info(f"Acquisition started on {self.unique_id}")
                
            except ic4.IC4Exception as e:
                self._is_grabbing = False
                raise RuntimeError(f"Failed to start grabbing on {self.unique_id}: {e}") from e

    def stop_grabbing(self) -> None:
        if self.is_grabbing and self._grabber:
            try:
                self._grabber.stream_stop()
                self._is_grabbing = False
            except ic4.IC4Exception as e:
                logger.error(f"Error stopping grabbing on {self.unique_id}: {e}")

    def grab_frame(self, timeout_ms: int = 2000) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self._sink or not self._sink.is_attached:
            return None, None

        try:
            # Convert milliseconds to seconds for Python's queue timeout implementation
            timeout_sec = timeout_ms / 1000.0
            while True:
                #image = self._sink.pop_output_buffer()
                # Retrieve the frame array and metadata pushed by the background Listener thread
                image_arr, meta = self._frame_queue.get(timeout=timeout_sec)

                frame_number = meta['frame_number']
                frame_timestamp = meta['timestamp']
                frame_signature = (frame_number, frame_timestamp)

                if frame_signature == self._last_frame_signature:
                    logger.debug(
                        f"{self.unique_id}: dropping duplicate IC4 buffer frame_number={frame_number} "
                        f"timestamp={frame_timestamp}"
                    )
                    continue

                self._last_frame_signature = frame_signature

                #image_arr = image.numpy_wrap().copy()
                host_arrival_ns = time.monotonic_ns()

                frame_meta = {
                    'frame_number': frame_number,
                    'timestamp': frame_timestamp,
                    'host_arrival_monotonic_ns': host_arrival_ns,
                    'frame_signature': frame_signature,
                }

                return image_arr, frame_meta
                # finally:
                #     image.release()

        except queue.Empty:
            # allow manager loop to continue cleanly
            return None, None
        except Exception as e:
            raise IOError(f"Failed to grab frame from queue: {e}")
        
    # --- GenICamCamera abstract contract ---

    def _get_feature_value(self, name: str) -> Any:
        try:
            prop_id = self._get_prop_id(name)
            prop = self._grabber.device_property_map.find(prop_id)

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
            if name not in self._warned_features:
                logger.debug(f"Feature '{name}' not available or not writable: {e}")
                self._warned_features.add(name)
            return value

    def _get_feature_entries(self, name: str) -> list[str]:
        try:
            prop_id = self._get_prop_id(name)
            prop = self._grabber.device_property_map.find(prop_id)

            if prop is None or not hasattr(prop, 'entries'):
                return []

            return [entry.name for entry in prop.entries]

        except ic4.IC4Exception as e:
            raise AttributeError(f"Failed to get entries for feature '{name}': {e}") from e

    def _get_feature_min_value(self, name: str) -> Any:
        try:
            prop_id = self._get_prop_id(name)
            prop = self._grabber.device_property_map.find(prop_id)
            
            if prop is None:
                logger.debug(f"Property '{name}' not found for min value query")
                return 0
            
            if not hasattr(prop, 'minimum'):
                logger.debug(f"Property '{name}' has no 'minimum' attribute")
                return 0

            return prop.minimum

        except ic4.IC4Exception as e:
            logger.debug(f"IC4 exception getting min for feature '{name}': {e}")
            return 0

    def _get_feature_max_value(self, name: str) -> Any:
        try:
            prop_id = self._get_prop_id(name)
            prop = self._grabber.device_property_map.find(prop_id)
            
            if prop is None:
                logger.debug(f"Property '{name}' not found for max value query")
                return 1000
            
            if not hasattr(prop, 'maximum'):
                logger.debug(f"Property '{name}' has no 'maximum' attribute")
                return 1000

            return prop.maximum

        except ic4.IC4Exception as e:
            logger.debug(f"IC4 exception getting max for feature '{name}': {e}")
            return 1000

    def _get_prop_id(self, name: str) -> Any:
        """Convert GenICam feature name to IC4 PropId (tries: mapping → attribute → UPPER_SNAKE_CASE → string)"""
        if name in FEATURE_MAPPING:
            return FEATURE_MAPPING[name]

        try:
            return getattr(ic4.PropId, name)
        except AttributeError:
            pass

        try:
            upper_name = ''.join(['_' + c if c.isupper() else c for c in name]).lstrip('_').upper()
            return getattr(ic4.PropId, upper_name)
        except AttributeError:
            pass

        return name

    # --- IC4-specific property overrides ---

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
        try:
            self._framerate = float(value)

            if self.hardware_triggered:
                logger.debug(
                    f"{self.unique_id}: hardware trigger is active; caching framerate {self._framerate} "
                    f"without programming AcquisitionFrameRate"
                )
                return

            try:
                self._set_feature_value('AcquisitionMode', 'Continuous')
            except AttributeError:
                pass

            min_fps, max_fps = self.framerate_range
            clamped_value = max(min_fps, min(self._framerate, max_fps))

            actual_value = self._set_feature_value('AcquisitionFrameRate', clamped_value)
            self._framerate = actual_value

        except AttributeError as e:
            logger.warning(f"Camera {self.name} does not support framerate control: {e}")
            self._framerate = 0.0

    @property
    def framerate_range(self) -> Tuple[float, float]:
        try:
            min_fps = float(self._get_feature_min_value('AcquisitionFrameRate'))
            max_fps = float(self._get_feature_max_value('AcquisitionFrameRate'))
            return min_fps, max_fps

        except (AttributeError, ValueError, TypeError):
            logger.warning(f"Could not determine framerate range for {self.unique_id}")
            return 0.5, 500.0

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
            min_exp = float(self._get_feature_min_value('ExposureTime'))
            max_exp = float(self._get_feature_max_value('ExposureTime'))
            clamped_value = max(min_exp, min(value, max_exp))

            actual_value = self._set_feature_value('ExposureTime', clamped_value)
            self._exposure = actual_value

        except AttributeError as e:
            logger.warning(f"Camera {self.name} does not support exposure control: {e}")
            self._exposure = 5000.0

        finally:
            if was_grabbing:
                self.start_grabbing()

    @property
    def exposure_range(self) -> Tuple[float, float]:
        try:
            min_exp = float(self._get_feature_min_value('ExposureTime'))
            max_exp = float(self._get_feature_max_value('ExposureTime'))
            return min_exp, max_exp

        except (AttributeError, ValueError, TypeError):
            logger.warning(f"Could not determine exposure range for {self.unique_id}")
            return 1.0, 1000000.0

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
            min_gain = float(self._get_feature_min_value('Gain'))
            max_gain = float(self._get_feature_max_value('Gain'))
            clamped_value = max(min_gain, min(value, max_gain))

            actual_value = self._set_feature_value('Gain', clamped_value)
            self._gain = actual_value

        except AttributeError as e:
            logger.warning(f"Camera {self.name} does not support gain control: {e}")
            self._gain = 1.0

        finally:
            if was_grabbing:
                self.start_grabbing()

    @property
    def gain_range(self) -> Tuple[float, float]:
        try:
            min_gain = float(self._get_feature_min_value('Gain'))
            max_gain = float(self._get_feature_max_value('Gain'))
            return min_gain, max_gain

        except (AttributeError, ValueError, TypeError):
            logger.warning(f"Could not determine gain range for {self.unique_id}")
            return 0.0, 32.0

    @property
    def hardware_triggered(self) -> bool:
        return self._hardware_triggered

    @hardware_triggered.setter
    def hardware_triggered(self, enabled: bool):
        """
        Configure IC4's trigger configuration.
        Ensures proper register hierarchy: Selector -> Modifiers -> Mode Activation.
        """
        logger.debug(
            f"Configuring hardware_triggered={enabled} for {self.unique_id} "
            f"(current={self._hardware_triggered})"
        )

        if enabled:
            try:
                # 1. ALWAYS select the target state first. 
                # On many TIS sensors, changing this instantly resets TriggerMode to Off.
                logger.debug(f"{self.unique_id}: setting TriggerSelector to FrameStart")
                self._set_feature_value('TriggerSelector', 'FrameStart')
                
                # 2. Configure timing, masks, and Overlap/Fast operation modes while state is flexible
                self._configure_trigger_features()

                # 3. Choose the physical signal edge 
                actual_activation = None
                try:
                    logger.debug(f"{self.unique_id}: setting TriggerActivation to FallingEdge")
                    self._set_feature_value('TriggerActivation', 'FallingEdge')
                    actual_activation = self._get_feature_value('TriggerActivation')
                except Exception as e1:
                    logger.debug(f"{self.unique_id}: FallingEdge unavailable ({e1}); trying RisingEdge")
                    try:
                        logger.debug(f"{self.unique_id}: setting TriggerActivation to RisingEdge")
                        self._set_feature_value('TriggerActivation', 'RisingEdge')
                        actual_activation = self._get_feature_value('TriggerActivation')
                    except Exception as e2:
                        logger.warning(f"{self.unique_id}: TriggerActivation modification failed: {e2}")

                if actual_activation:
                    logger.debug(f"{self.unique_id}: TriggerActivation finalized as {actual_activation}")

                # 4. ARM THE STATE MACHINE LAST.
                # Now the overlap values configured in step 2 will actively latch.
                logger.debug(f"{self.unique_id}: enabling TriggerMode")
                self._set_feature_value('TriggerMode', 'On')
                
                # Verify registration success
                trigger_mode_value = self._get_feature_value('TriggerMode')
                logger.debug(f"{self.unique_id}: TriggerMode verified active as: {trigger_mode_value}")

                self._hardware_triggered = True
                logger.info(f"Hardware trigger enabled on {self.unique_id}")
            
                            # [DIAGNOSTIC TEST BLOCK]
                logger.info(f"=== {self.unique_id} HARDWARE TRIGGER SPEED CAPABILITY PROFILE ===")
                for feature in ['ResultingFrameRate', 'AcquisitionFrameRate', 'AcquisitionFrameRateLimit', 'DeviceLinkThroughputLimit', 'ACQUISITION_BURST_FRAME_COUNT','ACQUISITION_BURST_INTERVAL','ACQUISITION_FRAME_RATE','ACQUISITION_MODE']:
                    try:
                        val = self._get_feature_value(feature)
                        logger.info(f"   > {feature}: {val}")
                        available_entries = self._get_feature_entries(feature)
                        logger.info(f"   > {feature} is currently set to: '{val}' (Available choices: {available_entries})")
                    except Exception:
                        logger.info(f"   > {feature}: Not Supported/Readable")
                # =========================================================================

            except AttributeError as e:
                logger.error(f"{self.unique_id}: Critical error setting up hardware trigger framework: {e}")
                self._hardware_triggered = False
                return

        else:
            # Disarming loop
            try:
                logger.debug(f"{self.unique_id}: disabling TriggerMode")
                self._set_feature_value('TriggerMode', 'Off')
                trigger_mode_value = self._get_feature_value('TriggerMode')
                logger.debug(f"{self.unique_id}: TriggerMode readback after disable is {trigger_mode_value}")
                logger.info(f"Disabled trigger mode on {self.unique_id}")
            except AttributeError:
                logger.debug(f"{self.unique_id}: TriggerMode disable bypassed or unsupported")
            
            self._hardware_triggered = False



    def _feature_exists(self, name: str) -> bool:
        try:
            prop_id = self._get_prop_id(name)
            return self._grabber is not None and self._grabber.device_property_map.find(prop_id) is not None
        except ic4.IC4Exception:
            return False

    def _set_feature_if_available(self, name: str, value: Any) -> Optional[Any]:
        if not self._feature_exists(name):
            return None

        try:
            return self._set_feature_value(name, value)
        except AttributeError:
            return None

    def _set_first_supported_enum(self, name: str, candidates: list[str]) -> Optional[Any]:
        if not self._feature_exists(name):
            return None

        for candidate in candidates:
            actual_value = self._set_feature_value(name, candidate)
            try:
                readback = self._get_feature_value(name)
            except AttributeError:
                readback = actual_value

            if str(readback).lower() == str(candidate).lower():
                return readback

        return None

    def _configure_trigger_features(self) -> None:
            """
            Configure internal camera delays and overlap registers.
            Must run BEFORE TriggerMode is explicitly set to 'On'.
            """

            try:
                self._set_feature_value('TriggerMode', 'Off')
                logger.debug(f"{self.unique_id}: TriggerMode temporarily forced Off for configuration")
            except AttributeError as e:
                logger.warning(f"{self.unique_id}: Could not force TriggerMode Off during configuration: {e}")
                
            logger.debug(f"{self.unique_id}: Tuning trigger-dependent filters and overlap buffers")

            # Strip line noise and line bouncing delays that could miss fast 60Hz pulses
            self._set_feature_if_available('TriggerDelay', 0.0)
            self._set_feature_if_available('TriggerMask', 0.0)
            self._set_feature_if_available('TriggerDebouncer', 0.0)
            self._set_feature_if_available('TriggerDenoise', 0.0)
            self._set_feature_if_available('IMXLowLatencyTriggerMode', False)             # Keep Low Latency FALSE because it explicitly blocks Trigger Overlap

            # Unlock the IMX Sensor's fast timing registers.
            # This clears the internal hardware block that causes 'Access Denied' on TriggerOverlap.
            if self._feature_exists('IMXTriggerTiming'):
                timing_set = self._set_first_supported_enum('IMXTriggerTiming', ['Fast', 'HighSpeed', 'Overlap'])
                logger.debug(f"{self.unique_id}: IMXTriggerTiming configured to: {timing_set}")

            # --- DYNAMIC SPEED GOVERNOR FIX ---
            # Query the camera's actual maximum allowed frame rate and apply it.
            # This opens the internal readiness gate completely.
            try:
                max_fps = float(self._get_feature_max_value('AcquisitionFrameRate'))
                if max_fps > 0:
                    self._set_feature_value('AcquisitionFrameRate', max_fps)
                    logger.debug(f"{self.unique_id}: Internal frame clock window headroom maximized to {max_fps} FPS")
                else:
                    # Fallback to safe overhead if max query fails or returns 0
                    self._set_feature_value('AcquisitionFrameRate', 120.0)
            except Exception as e:
                logger.warning(f"{self.unique_id}: Failed to maximize internal clock window: {e}")

            # Force the sensor into full frame-rate pipeline overlap modes
            # This allows exposure sequence N+1 to run concurrently with readout sequence N.
            operation_set = self._set_first_supported_enum('TriggerOperation', ['Fast', 'Overlap'])
            overlap_set = self._set_first_supported_enum('TriggerOverlap', ['ReadOut', 'PreviousFrame'])
            
            logger.debug(f"{self.unique_id}: TriggerOperation set to {operation_set}, TriggerOverlap set to {overlap_set}")

            # --- DIAGNOSTIC READBACK ---
            logger.info(f"=== {self.unique_id} TRIGGER OVERLAP CONFIGURATION CHECK ===")
            for feat in ['TriggerOperation', 'TriggerOverlap']:
                try:
                    current_val = self._get_feature_value(feat)
                    available_entries = self._get_feature_entries(feat)
                    logger.info(f"   > {feat} is currently set to: '{current_val}' (Available choices: {available_entries})")
                except Exception as e:
                    logger.info(f"   > {feat} read error: {e}")
