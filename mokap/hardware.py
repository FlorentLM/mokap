import random
import string
from collections.abc import MutableSequence
from weakref import WeakValueDictionary
from threading import Lock, RLock, Event, Barrier
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import RawArray
from typing import NoReturn, Any, Union
from pathlib import Path
import time
from datetime import datetime
import numpy as np
import mokap.utils as utils
import os
import subprocess
from dotenv import load_dotenv
import pypylon.pylon as py
import mokap.files_op as files_op
import paramiko
import scipy
from collections import deque
import zarr
from numcodecs import blosc
from numcodecs import Blosc, Delta
# from PIL import Image, ImageFilter
blosc.use_threads = False


NB_CAMS = 5
ALLOW_VIRTUAL = False


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



def enable_usb():
    out = os.popen('uhubctl -l 4-2 -a 1')
    ret = out.read()


def disable_usb():
    out = os.popen('uhubctl -l 4-2 -a 0')
    ret = out.read()


def get_devices(nb_cams=NB_CAMS, allow_virtual=ALLOW_VIRTUAL) -> list[py.DeviceInfo]:
    instance = py.TlFactory.GetInstance()

    # List connected devices and get pointers to them
    dev_filter = py.DeviceInfo()

    dev_filter.SetDeviceClass("BaslerUsb")
    real_devices = instance.EnumerateDevices([dev_filter, ])
    nb_real = len(real_devices)

    if allow_virtual and nb_real < nb_cams:
        os.environ["PYLON_CAMEMU"] = f"{nb_cams - nb_real}"
        dev_filter.SetDeviceClass("BaslerCamEmu")
        virtual_devices = instance.EnumerateDevices([dev_filter, ])

    else:
        os.environ.pop("PYLON_CAMEMU", None)
        virtual_devices = []
        real_devices = real_devices[:nb_cams]

    all_devices = [r for r in real_devices] + [v for v in virtual_devices]
    return all_devices


def ping(ip: str) -> NoReturn:
    r = subprocess.Popen(["ping", "-c", "1", ip], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode == 1:
        raise ConnectionError(f'{ip} is unreachable :(\nCheck Wireguard status!')


class SSHTrigger:

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


class FrameStore:
    def __init__(self, queues=5):
        self._nb_queues = queues
        self._queues = [deque()] * self._nb_queues
        self._lock = RLock()

    def put(self, item: Any, qidx: int):
        with self._lock:
            self._queues[qidx].append(item)

    def get(self, qidx: int) -> list:
        with self._lock:
            try:
                item = self._queues[qidx].popleft()
            except IndexError:
                item = None
        return [item]

    def get_all(self, qidx: int) -> deque:
        with self._lock:
            items, self._queues[qidx] = self._queues[qidx], deque()
        return items

    def __repr__(self):
        return f"FrameStore({[len(i) for i in self._queues]})"


class CamList(MutableSequence):

    def __init__(self, data=None):
        super(CamList, self).__init__()
        if data is not None:
            self._list = list(data)
        else:
            self._list = list()

        self.POS_TO_IDX = {
            # Clockwise
            'top': 0,
            'north-west': 1,
            'north-east': 2,
            'south-east': 3,
            'south-west': 4
        }

    def __repr__(self):
        return f"<{self.__class__.__name__} {self._list}>"

    def __len__(self):
        return len(self._list)

    def __getitem__(self, ii):
        if isinstance(ii, slice):
            return self.__class__(self._list[ii])
        elif type(ii) is str:
            return self._list[self.POS_TO_IDX[ii]]
        else:
            return self._list[ii]

    def __setitem__(self, ii, val):
        # optional: self._acl_check(val)
        self._list[ii] = val

    def __delitem__(self, ii):
         if type(ii) is str:
            del self._list[self.POS_TO_IDX[ii]]
         else:
            del self._list[ii]

    def __str__(self):
        return str(self._list)

    def insert(self, ii, val):
        # optional: self._acl_check(val)
        self._list.insert(ii, val)

    def add(self, val):
        if 'virtual' in val.pos:
            ii = int(val.pos.split('_')[-1])
        else:
            ii = self.POS_TO_IDX[val.pos]
        self.insert(ii, val)


class Camera:

    # This is to make sure cameras are always positioned and ordered the same
    # TODO: use enums instead?

    SERIAL_TO_POS = {
        # Clockwise
        '40166059': 'top',          # red tag
        '40182207': 'north-west',   # green tag
        '40182542': 'north-east',  # yellow tag
        '40166127': 'south-east',  # blue tag
        '40189363': 'south-west'   # zebra tag
    }

    SENSOR_W = 1440
    SENSOR_H = 1080

    unknown_cams = []

    _instances_dict = WeakValueDictionary()
    _instances_indices = []

    def __init__(self,
                 framerate=200,
                 exposure=3200,
                 triggered=True,
                 scale=1):

        self._ptr = None
        self._dptr = None

        self._serial = ''
        self._pos = 'unknown'

        self._width = 0
        self._height = 0

        self._framerate = framerate
        self._exposure = exposure
        self._triggered = triggered
        self._scale = scale

        self._weak_id = f"{random.choice(string.ascii_letters[:26])}{random.randint(100, 999)}"
        Camera._instances_dict[self._weak_id] = self

        existing_instances = np.array(Camera._instances_indices)
        if any(existing_instances < 0):
            self._idx = np.argmin(existing_instances)
            Camera._instances_indices[self._idx] = self._idx
        else:
            self._idx = len(Camera._instances_indices)
            Camera._instances_indices.append(self._idx)

        self._connected = False
        self._is_grabbing = False

    def __del__(self):
        Camera._instances_indices[self._idx] = -1
        Camera._instances_dict.pop(self._weak_id)

    def __repr__(self):
        if self._connected:
            return f"[{self.wid}] {self.pos.title()} camera (S/N {self.serial})"
        else:
            return f"[{self.wid}] Not connected"

    def connect(self, cam_ptr=None) -> NoReturn:

        if cam_ptr is None:
            devices = get_devices()
            self._ptr = py.InstantCamera(py.TlFactory.GetInstance().CreateDevice(devices[self._idx]))
        else:
            self._ptr = cam_ptr

        self._dptr = self.ptr.DeviceInfo

        self.ptr.GrabCameraEvents = False
        self.ptr.Open()
        self._serial = self.dptr.GetSerialNumber()
        self._connected = True

        if '0815-0' in self.serial:
            self._pos = f"virtual_{self._idx}"
        else:
            self._pos = Camera.SERIAL_TO_POS.get(self.serial, 'unknown')

        self.ptr.UserSetSelector.SetValue("Default")
        self.ptr.UserSetLoad.Execute()
        self.ptr.AcquisitionMode.Value = 'Continuous'
        self.ptr.ExposureMode = 'Timed'

        if 'virtual' not in self.pos:

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
                self.ptr.AcquisitionFrameRateEnable = False
            else:
                self.ptr.TriggerSelector = "FrameStart"
                self.ptr.TriggerMode = "Off"
                self.ptr.AcquisitionFrameRateEnable = True

        self.scale = self._scale
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
            self._pos = 'unknown'
            self._width = 0
            self._height = 0

    def start_grabbing(self) -> NoReturn:
        if self._connected:
            # self.ptr.StartGrabbing(py.GrabStrategy_LatestImageOnly)
            self.ptr.StartGrabbing(py.GrabStrategy_OneByOne)
            self._is_grabbing = True
        else:
            print(f"Camera {self.wid} is not connected.")

    def stop_grabbing(self) -> NoReturn:
        if self._connected:
            self.ptr.StopGrabbing()
            self._is_grabbing = False
        else:
            print(f"Camera {self.wid} is not connected.")

    @property
    def ptr(self) -> py.InstantCamera:
        return self._ptr

    @property
    def dptr(self) -> py.DeviceInfo:
        return self._dptr

    @property
    def wid(self) -> str:
        return self._weak_id

    @property
    def serial(self) -> str:
        return self._serial

    @property
    def pos(self) -> str:
        return self._pos

    @property
    def position(self) -> str:
        return self._pos

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def exposure(self) -> int:
        return self._exposure

    @property
    def scale(self) -> int:
        return self._scale

    @scale.setter
    def scale(self, value: int) -> NoReturn:
        if self._connected:
            if 'virtual' not in self.pos:
                self.ptr.BinningVertical.SetValue(value)
                self.ptr.BinningHorizontal.SetValue(value)
                self.ptr.BinningVerticalMode.SetValue('Average')
                self.ptr.BinningHorizontalMode.SetValue('Average')

            # Actual frame size
            self._width = Camera.SENSOR_W // value
            self._height = Camera.SENSOR_H // value

            # Set ROI to full frame
            self.ptr.OffsetX = 0
            self.ptr.OffsetY = 0
            self.ptr.Width = self._width
            self.ptr.Height = self._height

        # And keep a local value to avoid querying the camera every time we read it
        self._scale = value

    @exposure.setter
    def exposure(self, value: int) -> NoReturn:
        if self._connected:
            if 'virtual' in self.pos:
                self.ptr.ExposureTimeAbs = value
                self.ptr.ExposureTimeRaw = value
            else:
                if 'top' in self.pos:
                    # top camera gets a bit more light, 85% exposure is enough
                    value = int(value * 0.85)

                self.ptr.ExposureTime = value
        # And keep a local value to avoid querying the camera every time we read it
        self._exposure = value

    @property
    def framerate(self) -> int:
        return self._framerate

    @framerate.setter
    def framerate(self, value: int) -> NoReturn:
        if self._connected:
            if 'virtual' in self.pos:
                self.ptr.AcquisitionFrameRateAbs = value
                # And keep a local value to avoid querying the camera every time we read it
                self._framerate = value
            else:
                if not self.triggered:
                    self.ptr.AcquisitionFrameRate = value
                    # And keep a local value to avoid querying the camera every time we read it
                    self._framerate = value
                else:
                    self._framerate = self.ptr.ResultingFramerate.GetValue()
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

##


class Manager:

    pti = {
        'top': 0,
        'north-west': 1,
        'north-east': 2,
        'south-east': 3,
        'south-west': 4
    }

    def __init__(self, triggered=True):

        setup_ulimit()

        self._default_scale = 1
        self._default_exposure = 4318
        self._default_framerate = 100
        self._triggered = triggered

        self._savepath = None
        self._acquisition_name = ''

        if self._triggered:
            self.trigger = SSHTrigger()
            if not self.trigger.connected:
                raise AssertionError('Connection problem with the trigger...')
        else:
            self.trigger = None

        devices = get_devices(nb_cams=NB_CAMS, allow_virtual=ALLOW_VIRTUAL)
        self._nb_cams = len(devices)

        self._executor = None

        self._acquiring = Event()
        self._recording = Event()
        self._finished_saving = [Event()] * self._nb_cams
        self._getting_reference = Event()

        self._file_access = Lock()
        self._zarr_length = 36000
        self._z_frames = None

        self._framestore = FrameStore()

        self._grabbed_frames = RawArray('I', self._nb_cams)
        self._saved_frames = RawArray('I', self._nb_cams)

        self.array = py.InstantCameraArray(self._nb_cams)
        self.cameras = CamList([])

        # Create the cameras and put them in auto-sorting CamList
        for i in range(self._nb_cams):
            dptr, cptr = devices[i], self.array[i]
            cptr.Attach(py.TlFactory.GetInstance().CreateDevice(dptr))
            cam = Camera(framerate=self._default_framerate,
                         exposure=self._default_exposure,
                         triggered=self._triggered,
                         scale=self._default_scale)
            cam.connect(cptr)
            self.cameras.add(cam)
            print(f"[INFO-{i}] Attached {cam}.")

        # Once again for the buffers, this time using the sorted CamList
        self._frames_buffers = []
        for i, cam in enumerate(self.cameras):
            self._frames_buffers.append(bytearray(b'\0' * cam.height * cam.width))

        self._reference_image = None
        self._reference_length = 100
        self._reference_buffer = np.zeros((self._reference_length,
                                           self.cameras['top'].height,
                                           self.cameras['top'].width), dtype='<u1')

    def set_framerate(self, value: int) -> NoReturn:
        self._framerate = value
        for cam in self.cameras:
            cam.framerate = value

    def set_exposure(self, value: int) -> NoReturn:
        self._exposure = value
        for cam in self.cameras:
            cam.exposure = value

    def set_scale(self, value: int) -> NoReturn:
        self._scale = value
        for i, cam in enumerate(self.cameras):
            cam.scale = value
            # And update the bytearray to the new size
            self._frames_buffers[i] = bytearray(b'\0' * cam.height * cam.width)

        self._reference_buffer = np.zeros((self._reference_length,
                                           self.cameras['top'].height,
                                           self.cameras['top'].width), dtype='<u1')

    def disconnect(self) -> NoReturn:
        self.array.Close()
        for cam in self.cameras:
            cam.disconnect()

    def _init_storage(self) -> NoReturn:

        # Sanity check - should always be true
        all_w = [c.width for c in self.cameras]
        all_h = [c.height for c in self.cameras]
        assert np.allclose(*all_w)
        assert np.allclose(*all_h)

        w = all_w[0]
        h = all_h[0]

        filters = [Delta(dtype='<u1')]
        compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)

        root = zarr.open(self._savepath / f'{self._acquisition_name}.zarr', mode='w')
        self._z_frames = root.zeros('frames',
                                    shape=(self._nb_cams, self._zarr_length, h, w),
                                    chunks=(1, 100, None, None),
                                    dtype='<u1',
                                    filters=filters, compressor=compressor)

        self._z_times = root.zeros('times', shape=(1, 2), dtype='M8[ns]')

        for i, cam in enumerate(self.cameras):
            root.attrs[i] = {'scale': cam.scale,
                             'framerate': cam.framerate,
                             'exposure': cam.exposure,
                             'triggered': cam.triggered,
                             'pos': cam.pos}

        self._saved_frames = RawArray('I', self._nb_cams)

    def _trim_storage(self, mode='min') -> NoReturn:
        with self._file_access:
            cut = min if mode == 'min' else max
            frame_limit = cut(self._saved_frames)
            sh = self._z_frames.shape
            new_shape = (sh[0], frame_limit, sh[2], sh[3])
            self._z_frames.resize(new_shape)
            print(f'[INFO] Storage trimmed to {frame_limit}.')

    def _extend_storage(self) -> NoReturn:
        with self._file_access:
            sh = self._z_frames.shape
            self._zarr_length *= 2
            new_shape = (sh[0], self._zarr_length, sh[2], sh[3])
            self._z_frames.resize(new_shape)
            print(f'[INFO] Storage extended to {self._zarr_length}.')

    def _writer(self, cam_idx: int, framestore: FrameStore) -> NoReturn:

        fps = self.cameras[cam_idx].framerate

        min_wait = 1/(fps * 0.1)
        max_wait = 1/(fps * 0.5)

        saving_started = False
        while self._acquiring.is_set():

            time.sleep(random.uniform(min_wait, max_wait))

            data = framestore.get_all(qidx=cam_idx)
            nb = len(data)

            if nb > 0:
                if not saving_started:
                    saving_started = True

                self._z_frames[cam_idx, self._saved_frames[cam_idx]:self._saved_frames[cam_idx] + nb, :, :] = data
                self._saved_frames[cam_idx] += nb

                if self._saved_frames[cam_idx] >= self._zarr_length * 0.9:
                    print('[INFO] Storage 90% full: extending...')
                    self._extend_storage()

                if saving_started and not self._recording.is_set():
                    self._finished_saving[cam_idx].set()

    def _grab_frames(self, cam_idx: int, framestore: FrameStore) -> NoReturn:

        cam = self.cameras[cam_idx]
        cam.start_grabbing()

        grabbed_frames = 0

        self.barrier.wait()

        while self._acquiring.is_set():

            with cam.ptr.RetrieveResult(100, py.TimeoutHandling_Return) as res:
                if res and res.GrabSucceeded():
                    frame_idx = res.ImageNumber
                    # buf = res.GetBuffer()
                    framedata = res.GetArray()
                    res.Release()

                    self._frames_buffers[cam_idx][:] = framedata.data.tobytes()
                    self._grabbed_frames[cam_idx] = frame_idx

                    if self._recording.is_set():

                        if frame_idx > grabbed_frames:
                            framestore.put(framedata, qidx=cam_idx)
                            grabbed_frames += 1

    def acquire_reference(self) -> NoReturn:

        if self._acquiring.is_set():
            i = 0
            start = self._grabbed_frames[self.pti['top']]

            print(f'Loading up reference over {self._reference_length} frames...')
            while i < self._reference_length:
                self._reference_buffer[i] = self.get_current_framearray(self.pti['top'])

                if self._grabbed_frames[self.pti['top']] > start:
                    i += 1
        else:
            raise RuntimeError('Must be acquiring!!')

        ref = scipy.ndimage.gaussian_filter(self._reference_buffer.mean(axis=0), 2).astype('<u1')

        # im = Image.fromarray(self._reference_buffer.mean(axis=0))
        # ref = np.array(im.filter(ImageFilter.GaussianBlur(radius=2)).astype('<u1'))

        self._reference_image = np.round((ref / np.max(ref)) * 255).astype('<u1', copy=True)
        self._reference_buffer.fill(0)
        print(f'Reference ready.')

    def clear_reference(self) -> NoReturn:
        self._reference_image = None

    def _reset_name(self):
        if self._savepath is not None:
            files_op.rm_if_empty(self._savepath)
        self._acquisition_name = ''
        self._savepath = None

    def _cleanup(self) -> NoReturn:
        self._trim_storage()

        self._reset_name()

        self._start_times = []
        self._stop_times = []

        self._saved_frames = RawArray('I', self._nb_cams)

    def record(self) -> NoReturn:
        self._recording.set()

        self._z_times[-1, 0] = np.datetime64(datetime.now())

        print('Recording started...')

    def pause(self) -> NoReturn:
        self._recording.clear()

        self._z_times[-1, 1] = np.datetime64(datetime.now())
        self._z_times.append(np.zeros((1, 2), dtype='M8[ns]'))

        print('Finishing saving...')
        [e.wait() for e in self._finished_saving]

        print('Done.')

    def on(self) -> NoReturn:

        if self._triggered:
            # Start trigger thread on the RPi
            self.trigger.start(self._framerate, 250000)
            time.sleep(0.5)

        if self._savepath is None:
            self.savepath = ''

        self._acquiring.set()

        self._executor = ThreadPoolExecutor(max_workers=40)

        self.barrier = Barrier(self._nb_cams, timeout=5)
        for i, cam in enumerate(self.cameras):
            self._executor.submit(self._grab_frames, i, self._framestore)
            self._executor.submit(self._writer, i, self._framestore)

        print(f'\n[INFO] Grabbing started with {self._nb_cams} cameras...')

    def off(self) -> NoReturn:

        if self._recording.is_set():
            self._recording.clear()

        self._acquiring.clear()

        self.array.StopGrabbing()
        for cam in self.cameras:
            cam.stop_grabbing()

        if self._triggered:
            self.trigger.off()

        self._cleanup()

        self._executor = None

        print(f'[INFO] Grabbing stopped.')

    @property
    def savepath(self) -> Path:
        return self._savepath

    @savepath.setter
    def savepath(self, value='') -> NoReturn:
        # Cleanup if a previous folder was created and not used
        self._reset_name()

        self._savepath = files_op.mk_folder(name=value)
        self._acquisition_name = self._savepath.stem

        self._init_storage()

    @property
    def reference(self) -> Union[None, np.array]:
        return self._reference_image

    @property
    def nb_cameras(self) -> int:
        return self._nb_cams

    @property
    def acquiring(self) -> bool:
        return self._acquiring.is_set()

    @property
    def recording(self) -> bool:
        return self._recording.is_set()

    @property
    def indices_buf(self) -> RawArray:
        return self._grabbed_frames

    @property
    def indices(self) -> np.array:
        return np.frombuffer(self._grabbed_frames, dtype=np.uintc)

    def get_current_framebuffer(self, i=None) -> Union[bytearray, list[bytearray]]:
        if i is None:
            return self._frames_buffers
        else:
            return self._frames_buffers[i]

    def get_current_framearray(self, i: Union[str, int]) -> np.array:
        if type(i) is str:
            i = self.pti[i]
        shape = (self.cameras[i].height, self.cameras[i].width)
        return np.frombuffer(self._frames_buffers[i], dtype=np.uint8).reshape(shape)
