from typing import List, Dict, Optional, Union, Any
from mokap.core.cameras.interface import AbstractCamera
from mokap.core.cameras.basler import BaslerCamera



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
            print("Pylon SDK not found. Skipping Basler camera discovery.")
        except Exception as e:
            print(f"Error during Basler discovery: {e}")

        # --- Discover FLIR Cameras ---
        # try:
        #     import PySpin
        #     system = PySpin.System.GetInstance()
        #     cam_list = system.GetCameras()
        #     for cam in cam_list:
        #         CameraFactory._discovered_devices.append({
        #             'vendor': 'FLIR',
        #             'model': cam.GetDeviceModelName(),
        #             'serial': cam.GetDeviceSerialNumber(),
        #             'native_object': cam
        #         })
        #     cam_list.Clear()
        #     system.ReleaseInstance()
        # except ImportError:
        #     print("PySpin SDK not found. Skipping FLIR camera discovery.")
        # except Exception as e:
        #     print(f"Error during FLIR discovery: {e}")

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
        Get a camera instance from its discovery information dictionary.
        """
        if not device_info:
            return None

        vendor = device_info.get('vendor')
        native_obj = device_info.get('native_object')

        if vendor == 'Basler':
            return BaslerCamera(native_obj)
        # elif vendor == 'FLIR':
        #     return FlirCamera(native_obj)
        else:
            print(f"Error: Vendor '{vendor}' is not supported.")
            return None