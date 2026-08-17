import logging
from pypylon import pylon
from pypylon import genicam as geni
import numpy as np
from typing import Any, Dict, Optional, Tuple, List
from mokap.core.cameras.genicam import GenICamCamera

logger = logging.getLogger(__name__)


class BaslerCamera(GenICamCamera):
    """
    Concrete implementation for Basler cameras.
    Inherits all GenICam logic from the GenICamCamera parent classs.
    (only adds Basler-specific connection, grabbing, and feature access)
    """

    def __init__(self, pylon_device_info):
        self._device_info = pylon_device_info
        self._ptr: Optional[pylon.InstantCamera] = None
        super().__init__(unique_id=pylon_device_info.GetSerialNumber())

    # Hooks

    def _pre_apply_configuration(self, settings: Dict[str, Any]):
        """Basler-specific hook."""

        super()._pre_apply_configuration(settings)  # call parent class's hook

        self._try_set_feature('UserSetSelector', 'Default')
        self._ptr.UserSetLoad.Execute()

        try:
            # Basler's default is 10, se set 50 (gives ~0.5s buffer at 100 fps)
            self._ptr.MaxNumBuffer.Value = 50
        except Exception:
            pass

        self._try_set_feature('ExposureMode', 'Timed')
        self._try_set_feature('TriggerDelay', 0.0)
        self._try_set_feature('LineDebouncerTime', 5.0)     # small debounce so trigger pulses aren't dropped as line noise

    # GenICam abstract contract (Basler-specific implementation)

    def _get_node_map(self):

        if not self._ptr or not self.is_connected:
            raise RuntimeError("Basler camera is not initialized.")

        return self._ptr.GetNodeMap()

    def _get_feature_value(self, name: str) -> Any:
        try:
            node = self._ptr.GetNodeMap().GetNode(name)
            if not geni.IsReadable(node):
                raise AttributeError(f"Feature '{name}' not readable.")
            return node.GetValue()

        except geni.GenericException as e:
            raise AttributeError(f"Failed to get feature '{name}': {e}") from e

    def _set_feature_value(self, name: str, value: Any) -> Any:
        try:
            node = self._ptr.GetNodeMap().GetNode(name)
            if not geni.IsWritable(node):
                raise AttributeError(f"Feature '{name}' not writable.")

            if isinstance(node, geni.IEnumeration):
                node.FromString(str(value))
                return value

            elif isinstance(node, (geni.IFloat, geni.IInteger)):

                min_val, max_val = node.GetMin(), node.GetMax()

                value = type(min_val)(value)  # ensure correct numeric type

                clamped_value = max(min_val, min(max_val, value))

                if clamped_value != value:
                    logger.warning(f"Clamped {name} from {value} to {clamped_value}")

                node.SetValue(clamped_value)
                return clamped_value

            elif isinstance(node, geni.IBoolean):
                node.SetValue(bool(value))
                return bool(value)

            else:
                node.SetValue(value)
                return value

        except geni.GenericException as e:
            raise AttributeError(f"Failed to set feature '{name}' to '{value}': {e}") from e

    def _get_feature_min_value(self, name: str) -> Any:
        try:
            return self._ptr.GetNodeMap().GetNode(name).GetMin()

        except geni.GenericException as e:
            raise AttributeError(f"Failed to get min for feature '{name}': {e}") from e

    def _get_feature_max_value(self, name: str) -> Any:
        try:
            return self._ptr.GetNodeMap().GetNode(name).GetMax()

        except geni.GenericException as e:
            raise AttributeError(f"Failed to get max for feature '{name}': {e}") from e

    def _get_feature_entries(self, name: str) -> List[str]:
        try:
            node = self._ptr.GetNodeMap().GetNode(name)

            if not isinstance(node, geni.IEnumeration):
                raise TypeError(f"Feature '{name}' is not an enumeration.")

            return [entry.GetSymbolic() for entry in node.GetEntries()]

        except geni.GenericException as e:
            raise AttributeError(f"Failed to get entries for feature '{name}': {e}") from e

    # Core methods

    def connect(self, config: Optional[Dict[str, Any]] = None) -> None:
        if self.is_connected:
            logger.warning(f"Camera {self.unique_id} is already connected.")
            return
        try:
            self._ptr = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(self._device_info))
            self._ptr.Open()
            self._is_connected = True
            self._apply_configuration(config)

        except geni.GenericException as e:
            self._is_connected = False
            raise RuntimeError(f"Failed to connect to Basler camera {self.unique_id}: {e}") from e

    def disconnect(self) -> None:
        if self.is_grabbing: self.stop_grabbing()
        if self._ptr and self._ptr.IsOpen(): self._ptr.Close()
        self._ptr = None
        self._is_connected = False

        logger.info(f"Disconnected from Basler camera {self.unique_id}")

    def start_grabbing(self) -> None:
        if self.is_connected and not self.is_grabbing:
            self._ptr.StartGrabbing(pylon.GrabStrategy_OneByOne)
            self._is_grabbing = True

    def stop_grabbing(self) -> None:
        if self.is_grabbing:
            self._ptr.StopGrabbing()
            self._is_grabbing = False

    def grab_frame(self, timeout_ms: int = 2000) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Ensure we are grabbing
        if not self.is_grabbing:
            self._ptr.StartGrabbing(pylon.GrabStrategy_OneByOne)
            self._is_grabbing = True

        grab_result = None
        try:
            grab_result = self._ptr.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
            if grab_result and grab_result.GrabSucceeded():
                ts = grab_result.TimeStamp
                self._timestamp_buffer.append(ts)

                try:
                    # pylon's Array creates a copy by default, so it is safe
                    frame = grab_result.Array
                except ValueError:
                    # Some pylon builds don't expose Bayer/Mono raw formats through Array...
                    width, height = grab_result.GetWidth(), grab_result.GetHeight()
                    padding_x = grab_result.GetPaddingX() if hasattr(grab_result, 'GetPaddingX') else 0
                    raw = np.frombuffer(grab_result.GetBuffer(), dtype=np.uint8)
                    frame = raw.reshape((height, width + padding_x))[:, :width].copy()

                return frame, {'frame_number': grab_result.ImageNumber, 'timestamp': ts}
            else:
                # if grab failed but did not raise an exception, raise one
                desc = grab_result.GetErrorDescription() if grab_result else "Unknown"
                raise IOError(f"Grab failed: {desc}")

        except geni.GenericException as e:
            # Pylon raises this (among other things) on RetrieveResult timeout
            if not self._ptr.IsGrabbing():
                logger.warning(f"Camera {self.unique_id} stopped grabbing unexpectedly. Restarting engine.")
                self._is_grabbing = False
                # Next call to grab_frame will trigger StartGrabbing() above
            raise TimeoutError(f'Grab timed out or failed: {e}') from e

        except Exception as e:
            # Check if the camera stopped grabbing unexpectedly (e.g. buffer cancelled)
            if not self._ptr.IsGrabbing():
                logger.warning(f"Camera {self.unique_id} stopped grabbing unexpectedly. Restarting engine.")
                self._is_grabbing = False
                # Next call to grab_frame will trigger StartGrabbing() above
            raise IOError(f"Grab failed: {e}") from e

        finally:
            if 'grab_result' in locals() and grab_result:
                grab_result.Release()