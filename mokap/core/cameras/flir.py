import logging

import PySpin
import numpy as np
import re
from typing import Any, Dict, Optional, Tuple
from mokap.core.cameras.genicam import GenICamCamera
from mokap.utils import pol_to_hsv

logger = logging.getLogger(__name__)


class FLIRCamera(GenICamCamera):
    """
    Concrete implementation for FLIR Spinnaker cameras
    Inherits all GenICam logic from the GenICamCamera parent classs
    (only adds FLIR-specific connection, grabbing, and feature access)
    """

    def __init__(self, pyspin_camera_ptr: PySpin.CameraPtr, pyspin_system: PySpin.SystemPtr):
        """
        pyspin_camera_ptr: pointer object obtained from the PySpin camera list
        pyspin_system: The system instance that created the camera. This is needed for proper cleanup.
        """
        self._cam_ptr: Optional[PySpin.CameraPtr] = pyspin_camera_ptr
        self._system: Optional[PySpin.SystemPtr] = pyspin_system  # store the system instance

        nodemap_tldevice = self._cam_ptr.GetTLDeviceNodeMap()   # has to be before Init()

        node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))

        if PySpin.IsAvailable(node_serial) and PySpin.IsReadable(node_serial):
            unique_id = node_serial.GetValue()
        else:
            raise RuntimeError("Could not retrieve serial number from FLIR camera.")

        super().__init__(unique_id=unique_id)

    def _pre_apply_configuration(self, settings: Dict[str, Any]):
        """ FLIR-specific hook """

        # TODO: check how polarisation is called in other vendors' SDKs

        if 'polarized' in settings.get('pixel_format', '').lower():
            self._polarisation_sensor = True
            self._POL_PATTERN = re.compile(r'polari[sz]ed', re.IGNORECASE)
        else:
            self._polarisation_sensor = False

        super()._pre_apply_configuration(settings)  # call parent class's hook

    def connect(self, config: Optional[Dict[str, Any]] = None) -> None:

        if self.is_connected:
            logger.warning(f"Camera {self.unique_id} is already connected.")
            return
        try:
            self._cam_ptr.Init()
            self._is_connected = True
            self._apply_configuration(config)
            logger.info(f"Connected to FLIR camera {self.unique_id}")

        except PySpin.SpinnakerException as e:
            self._is_connected = False
            raise RuntimeError(f"Failed to connect to FLIR camera {self.unique_id}: {e}") from e

    def disconnect(self) -> None:
        if self.is_grabbing: self.stop_grabbing()

        if self._cam_ptr and self._is_connected: self._cam_ptr.DeInit()

        # PySpin's garbage collection requires deleting the camera pointer object
        # to release the camera itself
        if hasattr(self, '_cam_ptr') and self._cam_ptr is not None:
            del self._cam_ptr
            self._cam_ptr = None

        self._is_connected = False
        logger.info(f"Disconnected from FLIR camera {self.unique_id}")

        # After releasing the camera, we must release the system instance reference
        # that was acquired when this camera was created
        if hasattr(self, '_system') and self._system is not None:
            self._system.ReleaseInstance()
            self._system = None

    def start_grabbing(self) -> None:
        if self.is_connected and not self.is_grabbing and self._cam_ptr:
            self._cam_ptr.BeginAcquisition()
            self._is_grabbing = True

    def stop_grabbing(self) -> None:
        if self.is_grabbing and self._cam_ptr:
            self._cam_ptr.EndAcquisition()
            self._is_grabbing = False

    def grab_frame(self, timeout_ms: int = 2000) -> Tuple[np.ndarray, Dict[str, Any]]:
        # This is specific to PySpin's grabbing strategy

        if not self._cam_ptr:
            raise RuntimeError("Camera is not connected or has been released.")

        if not self.is_grabbing: self._cam_ptr.BeginAcquisition()

        image_result = None
        try:
            image_result = self._cam_ptr.GetNextImage(timeout_ms)

            if image_result.IsIncomplete():
                raise IOError(f"Grab failed: Image incomplete with status {image_result.GetImageStatus()}")
            else:
                frame_meta = {'frame_number': image_result.GetFrameID(), 'timestamp': image_result.GetTimeStamp()}

                if self._polarisation_sensor:

                    # quad_0 = PySpin.ImageUtilityPolarization.ExtractPolarQuadrant(image_result, 0).GetNDArray().copy()
                    # quad_45 = PySpin.ImageUtilityPolarization.ExtractPolarQuadrant(image_result, 1).GetNDArray().copy()
                    # quad_90 = PySpin.ImageUtilityPolarization.ExtractPolarQuadrant(image_result, 2).GetNDArray().copy()
                    # quad_135 = PySpin.ImageUtilityPolarization.ExtractPolarQuadrant(image_result, 3).GetNDArray().copy()
                    #
                    # image_arr = pol_to_hsv(quad_0, quad_45, quad_90, quad_135)
                    # frame_meta['pixel_format'] = 'HSV'

                    quad_0 = PySpin.ImageUtilityPolarization.ExtractPolarQuadrant(image_result, 0)
                    image_arr = quad_0.GetNDArray().copy()
                    frame_meta['pixel_format'] = self._POL_PATTERN.sub('', self._pixel_format)

                else:
                    # IMPORTANT: GetNDArray returns a view. We must copy it!!
                    image_arr = image_result.GetNDArray().copy()

                return image_arr, frame_meta

        finally:
            if image_result: image_result.Release()
            if not self.is_grabbing: self._cam_ptr.EndAcquisition()

    # --- GenICamCamera abstract contract ---

    def _get_nodemap(self):

        if not self._cam_ptr or not self.is_connected:
            raise RuntimeError("FLIR camera is not initialized.")

        return self._cam_ptr.GetNodeMap()

    def _get_feature_value(self, name: str) -> Any:
        try:
            node = self._get_nodemap().GetNode(name)

            if not PySpin.IsAvailable(node) or not PySpin.IsReadable(node):
                raise AttributeError(f"Feature '{name}' not readable.")

            iface = node.GetPrincipalInterfaceType()

            match iface:
                case PySpin.intfIString:
                    return PySpin.CStringPtr(node).GetValue()
                case PySpin.intfIInteger:
                    return PySpin.CIntegerPtr(node).GetValue()
                case PySpin.intfIFloat:
                    return PySpin.CFloatPtr(node).GetValue()
                case PySpin.intfIBoolean:
                    return PySpin.CBooleanPtr(node).GetValue()
                case PySpin.intfIEnumeration:
                    return PySpin.CEnumerationPtr(node).GetCurrentEntry().GetSymbolic()

            raise TypeError(f"Unsupported feature type for '{name}'")

        except PySpin.SpinnakerException as e:
            raise AttributeError(f"Failed to get feature '{name}': {e}") from e

    def _set_feature_value(self, name: str, value: Any) -> Any:
        try:
            node = self._get_nodemap().GetNode(name)

            if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
                raise AttributeError(f"Feature '{name}' not writable.")

            iface = node.GetPrincipalInterfaceType()

            match iface:
                case PySpin.intfIEnumeration:
                    enum_node = PySpin.CEnumerationPtr(node)
                    entry = enum_node.GetEntryByName(str(value))

                    if not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
                        raise ValueError(f"Enum entry '{value}' not found for '{name}'.")

                    enum_node.SetIntValue(entry.GetValue())
                    return str(value)

                case PySpin.intfIFloat:
                    float_node = PySpin.CFloatPtr(node)
                    clamped_value = max(float_node.GetMin(), min(float_node.GetMax(), float(value)))

                    if clamped_value != value:
                        logger.warning(f"Clamped {name} from {value} to {clamped_value}")

                    float_node.SetValue(clamped_value)
                    return clamped_value

                case PySpin.intfIInteger:
                    int_node = PySpin.CIntegerPtr(node)
                    clamped_value = max(int_node.GetMin(), min(int_node.GetMax(), int(value)))

                    if clamped_value != value:
                        logger.info(f"Clamped {name} from {value} to {clamped_value}")

                    int_node.SetValue(clamped_value)
                    return clamped_value

                case PySpin.intfIBoolean:
                    PySpin.CBooleanPtr(node).SetValue(bool(value))
                    return bool(value)

            raise TypeError(f"Unsupported feature type for '{name}'")

        except PySpin.SpinnakerException as e:
            raise AttributeError(f"Failed to set feature '{name}' to '{value}': {e}") from e

    def _get_feature_max_value(self, name: str) -> Any:
        try:
            node = self._get_nodemap().GetNode(name)
            iface = node.GetPrincipalInterfaceType()

            match iface:
                case PySpin.intfIFloat:
                    return PySpin.CFloatPtr(node).GetMax()
                case PySpin.intfIInteger:
                    return PySpin.CIntegerPtr(node).GetMax()

            raise TypeError(f"Cannot get max for feature '{name}' of type {iface}")

        except PySpin.SpinnakerException as e:
            raise AttributeError(f"Failed to get max for feature '{name}': {e}") from e