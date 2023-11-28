from typing import NoReturn, Any, Union
import time
import numpy as np
import mokap.utils as utils
import os
import subprocess
from dotenv import load_dotenv
import pypylon.pylon as py
from numcodecs import blosc
import configparser

import warnings
from cryptography.utils import CryptographyDeprecationWarning
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=CryptographyDeprecationWarning)
    import paramiko

blosc.use_threads = False

config = configparser.ConfigParser()
config.read('config.conf')


##

def setup_ulimit():
    out = os.popen('ulimit -H -n')
    hard_limit = int(out.read().strip('\n'))

    out = os.popen('ulimit -n')
    current_limit = int(out.read().strip('\n'))

    if current_limit < 2048:
        print(f'[WARN] Current file descriptors limit is too small (n={current_limit}), increasing it to maximum value (n={hard_limit})')
        os.popen(f'ulimit -n {hard_limit}')
    else:
        print(f'[INFO] Current file descriptors limit seems fine (n={current_limit})')


def enable_usb(hub_number):
    out = os.popen(f'uhubctl -l {hub_number} -a 1')
    ret = out.read()


def disable_usb(hub_number):
    out = os.popen(f'uhubctl -l {hub_number} -a 0')
    ret = out.read()


def get_devices(max_cams=None, allow_virtual=None) -> tuple[list[py.DeviceInfo], list[py.DeviceInfo]]:

    if max_cams is None:
        if 'GENERAL' in config.sections():
            max_cams = config['GENERAL'].getint('max_cams', 99)
        else:
            max_cams = 99

    if allow_virtual is None:
        if 'GENERAL' in config.sections():
            allow_virtual = config['GENERAL'].getboolean('allow_virtual', False)
        else:
            allow_virtual = False

    instance = py.TlFactory.GetInstance()

    # List connected devices and get pointers to them
    dev_filter = py.DeviceInfo()

    dev_filter.SetDeviceClass("BaslerUsb")
    real_devices = instance.EnumerateDevices([dev_filter, ])
    nb_real = len(real_devices)

    if allow_virtual and nb_real < max_cams:
        os.environ["PYLON_CAMEMU"] = f"{max_cams - nb_real}"
        dev_filter.SetDeviceClass("BaslerCamEmu")
        virtual_devices = instance.EnumerateDevices([dev_filter, ])

    else:
        os.environ.pop("PYLON_CAMEMU", None)
        virtual_devices = []
        real_devices = real_devices[:max_cams]

    return [r for r in real_devices], [v for v in virtual_devices]


def ping(ip: str) -> NoReturn:
    r = subprocess.Popen(["ping", "-c", "1", ip], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode == 1:
        raise ConnectionError(f'{ip} is unreachable :(\nCheck Wireguard status!')

##

class SSHTrigger:
    """ Class to communicate with the hardware Trigger via SSH """
    def __init__(self):
        self._connected = False

        load_dotenv()

        env_ip = os.getenv('ANTBOOTH_IP')
        env_user = os.getenv('ANTBOOTH_USER')
        env_pass = os.getenv('ANTBOOTH_PASS')

        if None in (env_ip, env_user, env_pass):
            raise EnvironmentError(f'Missing {sum([v is None for v in (env_ip, env_user, env_pass)])} variables.')

        ping(env_ip)

        # Open the connection to the Raspberry Pi
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(env_ip, username=env_user, password=env_pass, look_for_keys=False)

        self._connected = True

        print('[INFO] Trigger connected.')

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self, frequency: float, duration: float, highs_pct=50) -> NoReturn:
        """ Starts the trigger loop on the RPi """
        interval, count = utils.to_ticks(frequency, duration + 1)

        interval_micros = interval * 1e6

        high = int(interval_micros * (highs_pct / 100))  # in microseconds
        low = int(interval_micros * ((100 - highs_pct) / 100))  # in microseconds
        cycles = int(count)

        self.client.exec_command(f'~/CameraArray/trigger.sh {cycles} {high} {low}')
        print(f"\n[INFO] Trigger started for {duration} seconds at {frequency} Hz")

    def stop(self) -> NoReturn:
        self.client.exec_command(f'sudo killall pigpiod')
        time.sleep(0.25)
        self.client.exec_command(f'~/CameraArray/zero.sh')
        time.sleep(0.25)
        self.client.exec_command(f'sudo killall pigpiod')


##

class Camera:

    virtual_cams = 0
    unknown_cams = 0

    def __init__(self,
                 framerate=60,
                 exposure=5000,
                 triggered=True,
                 binning=1):

        self._ptr = None
        self._dptr = None

        self._serial = ''
        self._name = 'unknown'

        self._width = config['GENERAL'].getint('sensor_w')
        self._height = config['GENERAL'].getint('sensor_h')

        self._framerate = framerate
        self._exposure = exposure
        self._triggered = triggered
        self._binning = binning

        self._idx = None

        self._connected = False
        self._is_grabbing = False

    def __repr__(self):
        if self._connected:
            return f"{self.name.title()} camera (S/N {self.serial})"
        else:
            return f"{self.name.title()} camera not connected"

    def connect(self, cam_ptr=None) -> NoReturn:

        if cam_ptr is None:
            real_cams, virtual_cams = get_devices()
            devices = real_cams + virtual_cams
            self._ptr = py.InstantCamera(py.TlFactory.GetInstance().CreateDevice(devices[self._idx]))
        else:
            self._ptr = cam_ptr

        self._dptr = self.ptr.DeviceInfo

        self.ptr.GrabCameraEvents = False
        self.ptr.Open()
        self._serial = self.dptr.GetSerialNumber()
        self._connected = True

        known_cams = list(config.sections())
        known_cams.remove('GENERAL')

        if '0815-0' in self.serial:
            self._name = f"virtual_{Camera.virtual_cams}"
            Camera.virtual_cams += 1
            self._color = '#f3a0f2'
        else:
            try:
                self._idx = [config[k]['serial'] for k in known_cams].index(self._serial)
                self._name = config[known_cams[self._idx]].get('name', 'unknown')
                self._color = '#' + config[known_cams[self._idx]].get('color', '#f0a108').lstrip('#')
            except ValueError:
                self._name = f"unkown_{Camera.unknown_cams}"
                Camera.unknown_cams += 1
                self._idx = - Camera.unknown_cams
                self._color = '#f0a108'

        print(f"Camera [S/N {self.serial}]: id={self._idx}, name={self._name}, col={self._color}")

        self.ptr.UserSetSelector.SetValue("Default")
        self.ptr.UserSetLoad.Execute()
        self.ptr.AcquisitionMode.Value = 'Continuous'
        self.ptr.ExposureMode = 'Timed'

        if 'virtual' not in self.name:

            self.ptr.ExposureTimeMode.SetValue('Standard')
            self.ptr.ExposureAuto = 'Off'
            self.ptr.TriggerDelay.Value = 0.0
            self.ptr.LineDebouncerTime.Value = 5.0
            self.ptr.MaxNumBuffer = 20

            if self.triggered:
                self.ptr.LineSelector = "Line4"
                self.ptr.LineMode = "Input"
                self.ptr.TriggerSelector = "FrameStart"
                self.ptr.TriggerMode = "On"
                self.ptr.TriggerSource = "Line4"
                self.ptr.TriggerActivation.Value = 'FallingEdge'
                self.ptr.AcquisitionFrameRateEnable.SetValue(False)
            else:
                self.ptr.TriggerSelector = "FrameStart"
                self.ptr.TriggerMode = "Off"
                self.ptr.AcquisitionFrameRateEnable.SetValue(True)

        self.binning = self._binning
        self.framerate = self._framerate
        self.exposure = self._exposure

    def disconnect(self) -> NoReturn:
        if self._connected:
            if self._is_grabbing:
                self.stop_grabbing()

            self.ptr.Close()
            self._ptr = None
            self._dptr = None
            self._connected = False
            self._serial = ''
            self._name = 'unknown'
            self._width = config['GENERAL'].getint('sensor_w')
            self._height = config['GENERAL'].getint('sensor_h')

    def start_grabbing(self) -> NoReturn:
        if self._connected:
            self.ptr.StartGrabbing(py.GrabStrategy_LatestImageOnly)
            self._is_grabbing = True
        else:
            print(f"{self.name.title()} camera is not connected.")

    def stop_grabbing(self) -> NoReturn:
        if self._connected:
            self.ptr.StopGrabbing()
            self._is_grabbing = False
        else:
            print(f"{self.name.title()} camera is not connected.")

    @property
    def ptr(self) -> py.InstantCamera:
        return self._ptr

    @property
    def dptr(self) -> py.DeviceInfo:
        return self._dptr

    @property
    def idx(self) -> str:
        return self._idx

    @property
    def serial(self) -> str:
        return self._serial

    @property
    def name(self) -> str:
        return self._name

    @property
    def color(self) -> str:
        return self._color

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def exposure(self) -> int:
        return self._exposure

    @property
    def binning(self) -> int:
        return self._binning

    @binning.setter
    def binning(self, value: int) -> NoReturn:
        assert value in [1, 2, 3, 4]
        if self._connected:
            if 'virtual' not in self.name:
                self.ptr.BinningVertical.SetValue(value)
                self.ptr.BinningHorizontal.SetValue(value)
                self.ptr.BinningVerticalMode.SetValue('Average')
                self.ptr.BinningHorizontalMode.SetValue('Average')

            # Actual frame size
            self._width = config['GENERAL'].getint('sensor_w') // value
            self._height = config['GENERAL'].getint('sensor_h') // value

            # Set ROI to full frame
            self.ptr.OffsetX = 0
            self.ptr.OffsetY = 0
            self.ptr.Width = self._width
            self.ptr.Height = self._height

        # And keep a local value to avoid querying the camera every time we read it
        self._binning = value

    @exposure.setter
    def exposure(self, value: int) -> NoReturn:
        if self._connected:
            if 'virtual' in self.name:
                self.ptr.ExposureTimeAbs = value
                self.ptr.ExposureTimeRaw = value
            else:
                self.ptr.ExposureTime = value
        # And keep a local value to avoid querying the camera every time we read it
        self._exposure = value

    @property
    def framerate(self) -> float:
        return self._framerate

    @framerate.setter
    def framerate(self, value: float) -> NoReturn:
        if self._connected:
            if self.triggered:
                print(f'[WARN] Trying to set framerate on a hardware-triggered camera ([{self._name}])')
                self.ptr.AcquisitionFrameRateEnable.SetValue(False)
                self._framerate = self.ptr.ResultingFrameRate.GetValue()
            else:
                if 'virtual' in self.name:
                    self.ptr.AcquisitionFrameRateAbs = 220
                    f = np.round(self.ptr.ResultingFrameRate.GetValue() - 0.5 * 10 ** (-2),
                                 2)  # custom floor with decimals
                    new_framerate = min(value, f)

                    self.ptr.AcquisitionFrameRateAbs = new_framerate
                else:
                    self.ptr.AcquisitionFrameRateEnable.SetValue(True)
                    self.ptr.AcquisitionFrameRate = 220

                    f = np.round(self.ptr.ResultingFrameRate.GetValue() - 0.5 * 10 ** (-2),
                                 2)  # custom floor with decimals
                    new_framerate = min(value, f)

                    self.ptr.AcquisitionFrameRate = new_framerate

                self._framerate = new_framerate

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def temperature(self) -> float:
        degs = None
        if self._connected:
            t = self.ptr.DeviceTemperature.Value
            if t < 400:
                degs = t
        return degs
