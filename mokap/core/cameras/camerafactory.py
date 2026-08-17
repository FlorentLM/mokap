import logging
import platform
from typing import List, Dict, Optional, Union, Any, TYPE_CHECKING
import cv2
from mokap.core.cameras.interface import AbstractCamera
from mokap.core.cameras.webcam import WebcamCamera

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mokap.core.cameras.basler import BaslerCamera
    from mokap.core.cameras.flir import FLIRCamera
    from mokap.core.cameras.ic4imaging import IC4ImagingCamera


def discover_webcams(max_to_check: int = 10):
    """Attempts to find available webcams by trying to open them sequentially."""

    try:
        # TODO: This does not work
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        pass  # older OpenCV version

    found_cams = []
    index = 0

    while len(found_cams) < max_to_check:
        if platform.system() == 'Windows':
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)

        if cap.isOpened():
            # if successful, create an instance and release the capture
            logger.debug(f"Found Webcam at index {index}")
            found_cams.append(WebcamCamera(camera_index=index))
            cap.release()
            index += 1
        else:
            cap.release()
            # If missed index 0, check 1 just in case (some laptops map rear cam to 1) otherwise assume no more cameras
            if index == 0:
                index += 1
                continue
            break

    return found_cams


class CameraFactory:
    _discovered_devices = []

    @staticmethod
    def discover_cameras(include_webcams: bool = True) -> List[Dict[str, str]]:
        """
        Scan for all connected cameras from supported vendors.
        Returns a list of dictionaries, each with info about a camera.
        """
        CameraFactory._discovered_devices = []

        # Discover Basler cameras
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

        # Discover FLIR cameras
        system = None
        cam_list = None
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
                    'native_object': None  # should not keep a ref to the pointer, otherwise we get device busy
                })

                del cam  # must explicitly delete this to release the reference before next iteration

        except ImportError:
            logger.debug("PySpin SDK not found. Skipping FLIR camera discovery.")

        except Exception as e:
            logger.error(f"Error during FLIR discovery: {e}")

        finally:
            # list should be clearewd and system instance released even if no camera was found
            # (otherwise the reference is leaked and PySpin complains)
            try:
                if cam_list is not None:
                    cam_list.Clear()
                if system is not None:
                    system.ReleaseInstance()
            except Exception as e:
                logger.debug(f"FLIR discovery cleanup failed: {e}")

        # Discover IC4 cameras
        try:
            import imagingcontrol4 as ic4

            try:
                ic4.Library.init()
            except Exception as e:
                if 'already called' not in str(e).lower():
                    raise

            for device_info in ic4.DeviceEnum.devices():
                CameraFactory._discovered_devices.append({
                    'vendor': 'ICImaging',
                    'model': device_info.model_name,
                    'serial': device_info.serial,
                    'native_object': device_info  # SDK-specific object
                })

        except ImportError:
            logger.debug('IC Imaging Control 4 SDK not found. Skipping IC Imaging camera discovery.')

        except Exception as e:
            logger.error(f'Error during IC Imaging discovery: {e}')

        # Discover webcams
        if include_webcams:
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
        else:
            logger.debug("Skipping webcam discovery (no 'webcam' vendor configured).")

        return CameraFactory._discovered_devices

    @staticmethod
    def get_camera_info(identifier: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Retrieves the discovery information for a camera by index or serial.
        Note: This does *not* return the native SDK object.
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
        Get a camera instance from its discovery information dictionary.
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
                cam_ptr = cam_list.GetBySerial(serial)  # this is the safe way to re-acquire a camera
                cam_list.Clear()  # we can release the list

                if cam_ptr and cam_ptr.IsValid():
                    # If we got a valid camera, return a FLIRCamera instacne
                    # we do NOT release the system instance, the FLIRCamera object needs it :)
                    return FLIRCamera(cam_ptr, system)
                else:
                    # camera was not found (maybe disconnected?)
                    logger.error(f"Could not re-acquire FLIR camera with serial {serial}. Was it disconnected?")

                    if system:
                        system.ReleaseInstance()  # clean up the system instance
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

        elif vendor == 'icimaging':
            try:
                from mokap.core.cameras.ic4imaging import IC4ImagingCamera

                native_obj = device_info.get('native_object')
                if native_obj is None:
                    logger.error("IC Imaging device info is missing the DeviceInfo object.")
                    return None
                return IC4ImagingCamera(native_obj)

            except ImportError:
                logger.error("Cannot create IC Imaging camera. Is the imagingcontrol4 SDK installed?")
                return None

            except Exception as e:
                logger.error(f"Error creating IC Imaging camera instance: {e}")
                return None

        elif vendor == 'webcam':
            try:
                cam_index = device_info.get('native_object')
                if cam_index is not None:
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