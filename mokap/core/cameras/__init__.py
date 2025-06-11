from typing import List, Dict, Optional, Union, Any
from mokap.core.cameras.interface import AbstractCamera
from mokap.core.cameras.basler import BaslerCamera
from mokap.core.cameras.flir import FLIRCamera


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
                    print("Warning: Found a FLIR camera but could not get its serial number. Skipping.")
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

            cam_list.Clear()    # This is safe because we are not holding the ref to the pointer

            system.ReleaseInstance()

        except ImportError:
            print("PySpin SDK not found. Skipping FLIR camera discovery.")

        except Exception as e:
            print(f"Error during FLIR discovery: {e}")

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
            native_obj = device_info.get('native_object')
            return BaslerCamera(native_obj)

        elif vendor == 'flir':
            # For FLIR, we must re-acquire the camera using its serial number
            serial = device_info.get('serial')
            if not serial:
                print("Error: Cannot get FLIR camera without a serial number.")
                return None

            system = None
            try:
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
                    print(f"Error: Could not re-acquire FLIR camera with serial {serial}. Was it disconnected?")
                    if system:
                        system.ReleaseInstance() # clean up the system instance
                    return None

            except Exception as e:
                print(f"Error during FLIR camera re-acquisition: {e}")
                # clean up if an exception occurred

                if system:
                    system.ReleaseInstance()
                return None

        else:
            print(f"Error: Vendor '{vendor}' is not supported.")
            return None