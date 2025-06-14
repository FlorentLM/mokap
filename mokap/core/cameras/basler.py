from pypylon import pylon
from pypylon import genicam as geni
import numpy as np
from typing import Any, Dict, Optional, Tuple
from mokap.core.cameras.genicam import GenICamCamera


class BaslerCamera(GenICamCamera):
    """
    Concrete implementation for Basler cameras
    Inherits all GenICam logic from the GenICamCamera parent classs
    (only adds Basler-specific connection, grabbing, and feature access)
    """

    def __init__(self, pylon_device_info):
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

    def _pre_apply_configuration(self, settings: Dict[str, Any]):
        """ Basler-specific hook """
        super()._pre_apply_configuration(settings)  # call parent class's hook

        self._set_feature_value('UserSetSelector', 'Default')
        self._ptr.UserSetLoad.Execute()

    def disconnect(self) -> None:
        if self.is_grabbing: self.stop_grabbing()
        if self._ptr and self._ptr.IsOpen(): self._ptr.Close()
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
        # This is specific to pylon's grabbing strategy

        if not self.is_grabbing:
            self._ptr.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        try:
            grab_result = self._ptr.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
            if grab_result:
                # pylon's Array creates a copy by default, so it is safe
                return grab_result.Array, {'frame_number': grab_result.ImageNumber, 'timestamp': grab_result.TimeStamp}
            else:
                raise IOError(f"Grab failed: {grab_result.GetErrorCode()} {grab_result.GetErrorDescription()}")
        finally:
            if 'grab_result' in locals() and grab_result:
                grab_result.Release()
            if not self.is_grabbing:
                self._ptr.StopGrabbing()

    # --- GenICamCamera abstract contract ---

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
                    print(f"[Warning] Clamped {name} from {value} to {clamped_value}")

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

    def _get_feature_max_value(self, name: str) -> Any:
        try:
            return self._ptr.GetNodeMap().GetNode(name).GetMax()

        except geni.GenericException as e:
            raise AttributeError(f"Failed to get max for feature '{name}': {e}") from e