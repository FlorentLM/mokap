import time
import math
import numpy as np
import mokap.utils as utils
import os
from dotenv import load_dotenv
import pypylon.pylon as py
import PySpin
os.environ['SPINNAKER_GENTL64_CTI'] = '/Applications/Spinnaker/lib/spinnaker-gentl/Spinnaker_GenTL.cti'

import platform
import subprocess
import cv2
import warnings
from cryptography.utils import CryptographyDeprecationWarning
from pathlib import Path

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=CryptographyDeprecationWarning)
    import paramiko

##

def setup_ulimit(wanted_value=8192, silent=True):
    """
    Sets up the maximum number of open file descriptors for nofile processes.
    It is required to run multiple (i.e. more than 4) Basler cameras at a time.
    """
    out = os.popen('ulimit')
    ret = out.read().strip('\n')

    if ret == 'unlimited':
        hard_limit = np.inf
    else:
        hard_limit = int(ret)

    out = os.popen('ulimit -n')
    current_limit = int(out.read().strip('\n'))

    if current_limit < wanted_value:
        if not silent:
            print(f'[WARN] Current file descriptors limit is too small (n={current_limit}), '
                  f'increasing it to {wanted_value} (max={hard_limit}).')
        os.popen(f'ulimit -n {wanted_value}')
    else:
        if not silent:
            print(f'[INFO] Current file descriptors limit seems fine (n={current_limit})')


def enable_usb(hub_number):
    if 'Linux' in platform.system():
        out = os.popen(f'uhubctl -l {hub_number} -a 1')
        ret = out.read()


def disable_usb(hub_number):
    if 'Linux' in platform.system():
        out = os.popen(f'uhubctl -l {hub_number} -a 0')
        ret = out.read()


def enumerate_basler_devices(virtual_cams=0) -> list[py.DeviceInfo]:

    instance = py.TlFactory.GetInstance()

    # List connected devices and get pointers to them
    dev_filter = py.DeviceInfo()

    dev_filter.SetDeviceClass("BaslerUsb")
    basler_devices = list(instance.EnumerateDevices([dev_filter, ]))

    if virtual_cams > 0:
        os.environ["PYLON_CAMEMU"] = f"{virtual_cams}"
        dev_filter.SetDeviceClass("BaslerCamEmu")
        basler_devices += list(instance.EnumerateDevices([dev_filter, ]))

    return basler_devices


def enumerate_flir_devices(virtual_cams=0):
    # TODO
    return []


def enumarate_webcam_devices():
    pltfm = platform.system().lower()

    if pltfm == "linux" or pltfm == "linux2":
        result = subprocess.run(["ls", "/dev/"],
                                stdout=subprocess.PIPE,
                                text=True)
        devices = [int(v.replace('video', '')) for v in result.stdout.split() if 'video' in v]
    elif pltfm == "darwin":
        # disgusting code block
        command = ['ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', '""']
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        lines = result.stderr.splitlines()
        s = np.argwhere(['AVFoundation video devices:' in l for l in lines])[0][0]
        e = np.argwhere(['AVFoundation audio devices:' in l for l in lines])[0][0]
        devices = [int(l.split('[')[2][0]) for l in lines[s+1:e] if 'Capture screen' not in l]

    elif pltfm == "win32":
        result = subprocess.run(['pnputil', '/enum-devices', '/class', 'Camera', '/connected'],
                                stdout=subprocess.PIPE,
                                text=True)
        devices_ids = [v.replace('Instance ID:', '').strip() for v in result.stdout.splitlines() if 'Instance ID:' in v]
        devices = list(range(len(devices_ids)))
    else:
        raise OSError('Unsupported OS')

    working_ports = []

    prev_log_level = cv2.setLogLevel(0)

    for dev in devices:

        cap = cv2.VideoCapture(dev, 0, (cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY))

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                working_ports.append(dev)
            cap.release()

    cv2.setLogLevel(prev_log_level)
    return working_ports


def ping(host: str) -> bool:

    if 'Windows' in platform.system():
        pop = subprocess.Popen(["ping", "-w", "1", "-n", "1", host], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        pop = subprocess.Popen(["ping", "-W", "1", "-c", "1", host], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pop.wait()
    if pop.returncode == 1:
        raise ConnectionError(f'{host} is unreachable :(\nCheck Wireguard status!')
    elif pop.returncode == 0:
        return True


##

class SSHTrigger:
    """ Class to communicate with the hardware Trigger via SSH """

    def __init__(self, silent=False):

        self._connected = False
        self._silent = silent

        self.PWM_GPIO_PIN = 18  # Should be true for all Raspberry Pis

        load_dotenv()

        env_ip = os.getenv('TRIGGER_HOST')
        env_user = os.getenv('TRIGGER_USER')
        env_pass = os.getenv('TRIGGER_PASS')

        if None in (env_ip, env_user, env_pass):
            raise EnvironmentError(f'Missing {sum([v is None for v in (env_ip, env_user, env_pass)])} variables.')

        if ping(env_ip):

            # Open the connection to the Raspberry Pi
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.client.connect(env_ip, username=env_user, password=env_pass, look_for_keys=False)

            if self.client:
                self._connected = True
                if not self._silent:
                    print('[INFO] Trigger connected')
        else:
            print('[WARN] Trigger unreachable')

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self, frequency: float, highs_pct=50) -> None:
        """ Starts the trigger loop on the RPi """

        pct = int(np.floor(highs_pct * 1e4))
        frq = int(np.floor(frequency))

        if self.client is not None:
            self.client.exec_command(f'pigs hp {self.PWM_GPIO_PIN} {frq} {pct}')
            if not self._silent:
                print(f"[INFO] Trigger started at {frequency} Hz")

    def stop(self) -> None:
        if self.client:
            self.client.exec_command(f'pigs hp {self.PWM_GPIO_PIN} 0 0 && pigs w {self.PWM_GPIO_PIN} 0')
        time.sleep(0.1)
        if not self._silent:
            print(f"[INFO] Trigger stopped")

    def disconnect(self) -> None:
        if self.client:
            self.client.close()
            self.client = False

    def __del__(self):
        self.disconnect()


##

class BaslerCamera:
    instancied_cams = []

    def __init__(self,
                 name='unnamed',
                 framerate=60,
                 exposure=5000,
                 triggered=True,
                 binning=1,
                 binning_mode='sum'):

        self._ptr = None
        self._dptr = None
        self._is_virtual = False

        self._serial = ''
        self._name = name

        self._width = 0
        self._height = 0

        self._framerate = framerate
        self._exposure = exposure
        self._blacks = 0.0
        self._gain = 1.0
        self._gamma = 1.0
        self._triggered = triggered
        self._binning = binning
        self._binning_mode = binning_mode

        self._idx = -1

        self._connected = False
        self._is_grabbing = False

    def __repr__(self):
        if self._connected:
            v = 'Virtual ' if self._is_virtual else ''
            return f"{v}Camera [S/N {self.serial}] (id={self._idx}, name={self._name})"
        else:
            return f"Camera disconnected"

    def _set_roi(self):
        if not self._is_virtual:
            self._width = self.ptr.WidthMax.GetValue() - (16 // self._binning)
            self._height = self.ptr.HeightMax.GetValue() - (8 // self._binning)

            # Apply the dimensions to the ROI
            self.ptr.Width = self._width
            self.ptr.Height = self._height
            self.ptr.CenterX = True
            self.ptr.CenterY = True

        else:
            # We hardcode these for virtual cameras, because the virtual sensor is otherwise 4096x4096 px...
            self._width = 1440
            self._height = 1080
            self.ptr.Width = self._width
            self.ptr.Height = self._height


    def connect(self, cam_ptr=None) -> None:

        available_idx = len(BaslerCamera.instancied_cams)

        if cam_ptr is None:
            real_cams, virtual_cams = enumerate_basler_devices()
            devices = real_cams + virtual_cams
            if available_idx <= len(devices):
                self._ptr = py.InstantCamera(py.TlFactory.GetInstance().CreateDevice(devices[available_idx]))
            else:
                raise RuntimeError("Not enough cameras detected!")
        else:
            self._ptr = cam_ptr

        self._dptr = self.ptr.DeviceInfo

        self.ptr.GrabCameraEvents = True
        self.ptr.Open()
        self._serial = self.dptr.GetSerialNumber()

        if '0815-0' in self.serial:
            self._is_virtual = True
            self._idx = int(self.serial[-1])
            # assert self._idx == available_idx   # This is probably useless
        else:
            self._is_virtual = False
            self._idx = available_idx

        if self._name in BaslerCamera.instancied_cams:
            self._name += f"_{self._idx}"

        self.ptr.UserSetSelector.SetValue("Default")
        self.ptr.UserSetLoad.Execute()
        self.ptr.AcquisitionMode.Value = 'Continuous'
        self.ptr.ExposureMode = 'Timed'

        self.ptr.DeviceLinkThroughputLimitMode.SetValue('On')
        self.ptr.DeviceLinkThroughputLimit.SetValue(342000000)

        if not self._is_virtual:

            self.ptr.ExposureTimeMode.SetValue('Standard')
            self.ptr.ExposureAuto = 'Off'
            self.ptr.GainAuto = 'Off'
            self.ptr.TriggerDelay.Value = 0.0
            self.ptr.LineDebouncerTime.Value = 5.0
            self.ptr.MaxNumBuffer = 20

            self.ptr.TriggerSelector = "FrameStart"

            if self.triggered:
                self.ptr.LineSelector = "Line4"
                self.ptr.LineMode = "Input"
                self.ptr.TriggerMode = "On"
                self.ptr.TriggerSource = "Line4"
                self.ptr.TriggerActivation.Value = 'RisingEdge'
                self.ptr.AcquisitionFrameRateEnable.SetValue(False)
            else:
                self.ptr.TriggerMode = "Off"
                self.ptr.AcquisitionFrameRateEnable.SetValue(True)

        self._set_roi()

        self.binning = self._binning
        self.binning_mode = self._binning_mode

        self.framerate = self._framerate
        self.exposure = self._exposure
        self.blacks = self._blacks
        self.gain = self._gain
        self.gamma = self._gamma

        BaslerCamera.instancied_cams.append(self._name)
        self._connected = True

    def disconnect(self) -> None:
        if self._connected:
            if self._is_grabbing:
                self.stop_grabbing()

            self.ptr.Close()
            self._ptr = None
            self._dptr = None
            self._connected = False
            self._serial = ''
            self._name = 'unnamed'
            self._width = 0
            self._height = 0

        self._connected = False

    def start_grabbing(self) -> None:
        if self._connected:
            # self.ptr.StartGrabbing(py.GrabStrategy_OneByOne, py.GrabLoop_ProvidedByUser)
            self.ptr.StartGrabbing(py.GrabStrategy_OneByOne, py.GrabLoop_ProvidedByInstantCamera)
            self._is_grabbing = True
        else:
            print(f"{self.name.title()} camera is not connected.")

    def stop_grabbing(self) -> None:
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
    def idx(self) -> int:
        return self._idx

    @property
    def serial(self) -> str:
        return self._serial

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name : str) -> None:

        if new_name == self._name or self._name == f"{new_name}_{self._idx}":
            return
        if new_name not in BaslerCamera.instancied_cams and f"{new_name}_{self._idx}" not in BaslerCamera.instancied_cams:
            BaslerCamera.instancied_cams[BaslerCamera.instancied_cams.index(self._name)] = new_name
            self._name = new_name
        elif f"{self._name}_{self._idx}" not in BaslerCamera.instancied_cams:
            BaslerCamera.instancied_cams[BaslerCamera.instancied_cams.index(self._name)] = f"{self._name}_{self._idx}"
            self._name = f"{self._name}_{self._idx}"
        else:
            existing = BaslerCamera.instancied_cams.index(new_name)
            raise ValueError(f"A camera with the name {new_name} already exists: {existing}")    # TODO - handle this case nicely

    @staticmethod
    def pylon_exception_parser(exception):
        exception_message = exception.args[0]
        if 'must be smaller than or equal ' in exception_message:
            value = math.floor(100 * float(
                exception_message.split('must be smaller than or equal ')[1].split('. : OutOfRangeException')[
                    0])) / 100.0
        elif 'must be greater than or equal ' in exception_message:
            value = math.ceil(100 * float(
                exception_message.split('must be greater than or equal ')[1].split('. : OutOfRangeException')[
                    0])) / 100.0
        else:
            raise ValueError(f'[WARN] Unknown exception: {exception_message}')
        return value

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def exposure(self) -> int:
        return self._exposure

    @property
    def blacks(self) -> float:
        return self._blacks

    @property
    def gain(self) -> float:
        return self._gain

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def binning(self) -> int:
        return self._binning

    @property
    def binning_mode(self) -> str:
        return self._binning_mode

    @binning.setter
    def binning(self, value: int) -> None:
        assert value in [1, 2, 3, 4]
        # And keep a local value to avoid querying the camera every time we read it
        self._binning = value

        if self._connected:
            self.ptr.BinningVertical.SetValue(value)
            self.ptr.BinningHorizontal.SetValue(value)

        self._set_roi()

    @binning_mode.setter
    def binning_mode(self, value: str) -> None:
        if value.lower() in ['s', 'sum', 'add', 'addition', 'summation']:
            value = 'Sum'
        elif value.lower() in ['a', 'm', 'avg', 'average', 'mean']:
            value = 'Average'
        else:
            value = 'Sum'

        if self._connected:
            if not self._is_virtual:
                self.ptr.BinningVerticalMode.SetValue(value)
                self.ptr.BinningHorizontalMode.SetValue(value)

        # And keep a local value to avoid querying the camera every time we read it
        self._binning_mode = value

    @exposure.setter
    def exposure(self, value: float) -> None:
        if self._connected:
            try:
                if not self._is_virtual:
                    self.ptr.ExposureTime = value
                else:
                    self.ptr.ExposureTimeAbs = int(value)
                    self.ptr.ExposureTimeRaw = int(value)
            except py.OutOfRangeException as e:
                value = self.pylon_exception_parser(e)

                if not self._is_virtual:
                    self.ptr.ExposureTime = value
                else:
                    self.ptr.ExposureTimeAbs = int(value)
                    self.ptr.ExposureTimeRaw = int(value)

        # And keep a local value to avoid querying the camera every time we read it
        self._exposure = value

    @blacks.setter
    def blacks(self, value: float) -> None:
        if self._connected:
            try:
                self.ptr.BlackLevel.SetValue(value)
            except py.OutOfRangeException as e:
                value = self.pylon_exception_parser(e)
                self.ptr.BlackLevel.SetValue(value)
        # And keep a local value to avoid querying the camera every time we read it
        self._blacks = value

    @gain.setter
    def gain(self, value: float) -> None:
        if self._connected:
            try:
                self.ptr.Gain.SetValue(value)
            except py.OutOfRangeException as e:
                value = self.pylon_exception_parser(e)
                self.ptr.Gain.SetValue(value)
        # And keep a local value to avoid querying the camera every time we read it
        self._gain = value

    @gamma.setter
    def gamma(self, value: float) -> None:
        if self._connected:
            try:
                self.ptr.Gamma.SetValue(value)
            except py.OutOfRangeException as e:
                value = self.pylon_exception_parser(e)
                self.ptr.Gamma.SetValue(value)

        # And keep a local value to avoid querying the camera every time we read it
        self._gamma = value

    @property
    def framerate(self) -> float:
        return self._framerate

    @framerate.setter
    def framerate(self, value: float) -> None:
        if self._connected:
            if self.triggered:
                # print(f'[WARN] Trying to set framerate on a hardware-triggered camera ([{self._name}])')
                self.ptr.AcquisitionFrameRateEnable.SetValue(False)
                self._framerate = np.floor(value)
            else:
                if not self._is_virtual:
                    self.ptr.AcquisitionFrameRateEnable.SetValue(True)
                    self.ptr.AcquisitionFrameRate.SetValue(220.0)

                    # custom floor with decimals
                    f = math.floor(self.ptr.ResultingFrameRate.GetValue() * 100) / 100.0

                    new_framerate = min(value, f)

                    self.ptr.AcquisitionFrameRateEnable.SetValue(True)
                    self.ptr.AcquisitionFrameRate.SetValue(new_framerate)

                else:
                    self.ptr.AcquisitionFrameRateAbs = 220
                    f = math.floor(self.ptr.AcquisitionFrameRateAbs.GetValue() * 100) / 100.0
                    new_framerate = min(value, f)

                    self.ptr.AcquisitionFrameRateAbs = new_framerate

                self._framerate = new_framerate

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height
