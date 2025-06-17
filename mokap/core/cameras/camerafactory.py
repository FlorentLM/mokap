import logging
import os
from contextlib import redirect_stderr
from typing import List, Dict, Optional, Union, Any, TYPE_CHECKING
import cv2
from mokap.core.cameras.interface import AbstractCamera
from mokap.core.cameras.webcam import WebcamCamera

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mokap.core.cameras.basler import BaslerCamera
    from mokap.core.cameras.flir import FLIRCamera


def discover_webcams(max_to_check: int = 10):  # increased default just in case
    """ Attempts to find available webcams by trying to open them sequentially """
    found_cams = []
    index = 0

    while len(found_cams) < max_to_check:
        # try to open the camera at the current index
        with open(os.devnull, 'w') as f:
            with redirect_stderr(f):
                cap = cv2.VideoCapture(index)

        if cap.isOpened():
            # if successful, create an instance and release the capture
            logger.debug(f"Found Webcam at index {index}")
            found_cams.append(WebcamCamera(camera_index=index))
            cap.release()
            index += 1
        else:
            # if this fails, we assume there are no more cameras and break
            cap.release()
            break

    return found_cams


class CameraFactory:
    _discovered_devices = []

    @staticmethod
    def discover_cameras() -> List[Dict[str, str]]:
        """
        Scan for all connected cameras from supported vendors.
        Returns a list of dictionaries, each with info about a camera.
        """
        CameraFactory._discovered_devices = []

        # --- Discover Basler Cameras ---
        try:
            from pypylon import pylon as py
            tlf = py.TlFactory.GetInstance()
            pylon_devices = tlf.EnumerateDevices()
            for dev_info in pylon_devices:
                CameraFactory._discovered_devices.append({
                    'vendor': 'Basler',
                    'model': dev_info.GetModelName(),
                    'serial': dev_info.GetSerialNumber(),
                    'native_object': dev_info  # SDK-specific object
                })
        except ImportError:
            logger.debug("Pylon SDK not found. Skipping Basler camera discovery.")
            pass

        except Exception as e:
            logger.error(f"Error during Basler discovery: {e}")

        # --- Discover FLIR Cameras ---
        try:

            import PySpin

            system = PySpin.System.GetInstance()
            cam_list = system.GetCameras()

            for cam in cam_list:
                nodemap_tldevice = cam.GetTLDeviceNodeMap()

                # Get model name
                node_model = PySpin.CStringPtr(nodemap_tldevice.GetNode("DeviceModelName"))
                if not PySpin.IsAvailable(node_model) or not PySpin.IsReadable(node_model):
                    model_name = "Unknown FLIR Model"
                else:
                    model_name = node_model.GetValue()

                # Get Serial number
                node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode("DeviceSerialNumber"))

                if not PySpin.IsAvailable(node_serial) or not PySpin.IsReadable(node_serial):
                    logger.warning("Found a FLIR camera but could not get its serial number. Skipping.")
                    continue  # can't use a camera without a serial number

                else:
                    serial_number = node_serial.GetValue()

                CameraFactory._discovered_devices.append({
                    'vendor': 'FLIR',
                    'model': model_name,
                    'serial': serial_number,
                    'native_object': None       # should not keep a ref to the pointer, otherwise we get device busy
                })

                del cam     #  also we must explicitly delete this to release the reference

                cam_list.Clear()  # This is safe because we are not holding the ref to the pointer
                system.ReleaseInstance()

        except ImportError:
            logger.debug("PySpin SDK not found. Skipping FLIR camera discovery.")
            pass

        # --- Discover Webcams ---
        try:
            # We call the discover_webcams function which returns WebcamCamera instances
            found_webcams = discover_webcams()
            for cam_instance in found_webcams:
                CameraFactory._discovered_devices.append({
                    'vendor': 'Webcam',
                    'model': f'OpenCV Camera Index {cam_instance._index}',
                    'serial': cam_instance.unique_id,
                    'native_object': cam_instance._index  # Store the index needed for creation
                })
                # We don't need the instance itself anymore, just its info
                del cam_instance

        except Exception as e:
            logger.error(f"Error during Webcam discovery: {e}")
            pass

        return CameraFactory._discovered_devices


    @staticmethod
    def get_camera_info(identifier: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Retrieves the discovery information for a camera by index or serial
        NOTE: This does *not* return the native SDK object (for safety)
        """

        if not CameraFactory._discovered_devices:
            CameraFactory.discover_cameras()

        device_info_raw = None
        if isinstance(identifier, int):
            if 0 <= identifier < len(CameraFactory._discovered_devices):
                device_info_raw = CameraFactory._discovered_devices[identifier]

        elif isinstance(identifier, str):
            for dev in CameraFactory._discovered_devices:
                if dev['serial'] == identifier:
                    device_info_raw = dev
                    break

        if device_info_raw:
            # Return a copy of the info dictionary without the native object
            info_copy = device_info_raw.copy()
            info_copy.pop('native_object', None)
            return info_copy

        return None

    @staticmethod
    def get_camera(device_info: Dict[str, Any]) -> Optional[AbstractCamera]:
        """
        Get a camera instance from its discovery information dictionary
        """

        if not device_info:
            return None

        vendor = device_info.get('vendor').lower()

        if vendor == 'basler':
            try:
                from mokap.core.cameras.basler import BaslerCamera

                native_obj = device_info.get('native_object')
                return BaslerCamera(native_obj)

            except ImportError:
                logger.error("Cannot create Basler camera. Is the Pylon SDK installed?")
                return None

        elif vendor == 'flir':
            # For FLIR, we must re-acquire the camera using its serial number
            serial = device_info.get('serial')

            if not serial:
                logger.error("Cannot get FLIR camera without a serial number.")
                return None

            system = None
            try:
                from mokap.core.cameras.flir import FLIRCamera
                import PySpin
                # Get the system instance: this increments the reference count
                system = PySpin.System.GetInstance()

                cam_list = system.GetCameras()
                cam_ptr = cam_list.GetBySerial(serial) # this is the safe way to re-acquire a camera
                cam_list.Clear()    # we can release the list

                if cam_ptr and cam_ptr.IsValid():
                    # If we got a valid camera, return a FLIRCamera instacne
                    # we do NOT release the system instance, the FLIRCamera object needs it :)
                    return FLIRCamera(cam_ptr, system)
                else:
                    # camera was not found (maybe disconnected?)
                    logger.error(f"Could not re-acquire FLIR camera with serial {serial}. Was it disconnected?")

                    if system:
                        system.ReleaseInstance() # clean up the system instance
                    return None

            except ImportError:
                logger.error("Cannot create FLIR camera. Is the PySpin SDK installed?")
                # No need to release system, PySpin wasn't imported
                return None

            except Exception as e:
                logger.error(f"Error during FLIR camera re-acquisition: {e}")
                # clean up if an exception occurred

                if system:
                    system.ReleaseInstance()
                return None

        elif vendor == 'webcam':
            try:
                cam_index = device_info.get('native_object')
                if cam_index is not None:
                    # We already have the WebcamCamera class imported
                    return WebcamCamera(camera_index=cam_index)
                else:
                    logger.error("Webcam device info is missing the camera index.")
                    return None

            except Exception as e:
                logger.error(f"Error creating Webcam instance: {e}")
                return None

        else:
            logger.error(f"Error: Vendor '{vendor}' is not supported (yet).")
            return None