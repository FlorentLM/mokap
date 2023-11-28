import random
from threading import Lock, RLock, Event, Barrier
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import RawArray
from typing import NoReturn, Any, Union
from pathlib import Path
import time
from datetime import datetime
import numpy as np
import pypylon.pylon as py
import mokap.files_op as files_op
from collections import deque
import zarr
from numcodecs import Blosc, Delta
from mokap.hardware import SSHTrigger, Camera, setup_ulimit, get_devices

##

class FrameStore:
    """ Simple custom deque for multi-cameras frame buffers """
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

##

class Manager:

    def __init__(self,
                 framerate=220,
                 exposure=4318,
                 triggered=False,
                 binning=1):

        setup_ulimit()

        self._binning = binning
        self._exposure = exposure
        self._framerate = framerate
        self._triggered = triggered

        self._savepath = None
        self._acquisition_name = ''

        self._executor = None

        self._acquiring = Event()
        self._recording = Event()

        # self._barrier = None

        self._file_access_lock = Lock()
        self._zarr_length = 36000
        self._z_frames = None

        self._framestore = FrameStore()

        self._frames_buffers = []
        self._nb_cams = 0
        self._cameras_list = []
        self._cameras_dict = {}

        self.trigger = None
        self.ICarray = None

        self._grabbed_frames_idx = None
        self._saved_frms_idx = None
        self._finished_saving = []

    @property
    def triggered(self) -> bool:
        return self._triggered

    @triggered.setter
    def triggered(self, value: bool):
        if not self._triggered and value is True:
            external_trigger = SSHTrigger()
            if external_trigger.connected:
                self.trigger = external_trigger
                self._triggered = True
                print('Trigger mode enabled.')
            else:
                print("Connection problem with the trigger. Trigger mode can't be enabled.")
                self.trigger = None
                self._triggered = False
        elif self._triggered and value is False:
            self.trigger = None
            self._triggered = False
            print('Trigger mode disabled.')

        # TODO - Refresh cameras on trigger mode change

    def list_devices(self):

        real_cams, virtual_cams = get_devices()
        devices = real_cams + virtual_cams
        print(f"Found {len(devices)} cameras connected ({len(real_cams)} physical, {len(virtual_cams)} virtual).")

        return devices

    def connect(self):
        devices = self.list_devices()

        nb_cams = len(devices)

        self.ICarray = py.InstantCameraArray(nb_cams)

        # Create the cameras and put them in auto-sorting CamList
        for i in range(nb_cams):
            dptr, cptr = devices[i], self.ICarray[i]
            cptr.Attach(py.TlFactory.GetInstance().CreateDevice(dptr))
            cam = Camera(framerate=self._framerate,
                         exposure=self._exposure,
                         triggered=self._triggered,
                         binning=self._binning)
            cam.connect(cptr)
            self._cameras_list.append(cam)
            self._cameras_dict[cam.name] = cam
            print(f"Attached {cam}.")

        self._cameras_list.sort(key=lambda x: (x.idx))

        # Once again for the buffers, this time using the sorted list
        self._frames_buffers = []
        for i, cam in enumerate(self._cameras_list):
            self._frames_buffers.append(bytearray(b'\0' * cam.height * cam.width))

        self._nb_cams = len(self._cameras_list)

        self._grabbed_frames_idx = RawArray('I', self._nb_cams)
        self._saved_frms_idx = RawArray('I', self._nb_cams)

        self._finished_saving = [Event()] * self._nb_cams

    @property
    def framerate(self) -> int:
        return self._framerate

    @framerate.setter
    def framerate(self, value: int) -> NoReturn:
        self._framerate = value
        for cam in self._cameras_list:
            cam.framerate = value

    @property
    def exposure(self) -> int:
        return self._exposure

    @exposure.setter
    def exposure(self, value: int) -> NoReturn:
        self._exposure = value
        for cam in self._cameras_list:
            cam.exposure = value

    @property
    def binning(self) -> int:
        return self._binning

    @binning.setter
    def binning(self, value: int) -> NoReturn:
        self._binning = value
        for i, cam in enumerate(self._cameras_list):
            cam.binning = value
            # And update the bytearray to the new size
            self._frames_buffers[i] = bytearray(b'\0' * cam.height * cam.width)

    def disconnect(self) -> NoReturn:
        self.ICarray.Close()
        for cam in self._cameras_list:
            cam.disconnect()

        self._cameras_list = []
        self.ICarray = None

    def _init_storage(self) -> NoReturn:

        if self.nb_cameras > 1:
            # Sanity check - should always be true
            all_w = [c.width for c in self._cameras_list]
            all_h = [c.height for c in self._cameras_list]

            assert np.allclose(*all_w)
            assert np.allclose(*all_h)

            w = all_w[0]
            h = all_h[0]
        else:
            w = int(self._cameras_list[0].width)
            h = int(self._cameras_list[0].height)

        filters = [Delta(dtype='<u1')]
        compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)

        root = zarr.open(self._savepath / f'{self._acquisition_name}.zarr', mode='w')
        self._z_frames = root.zeros('frames',
                                    shape=(self._nb_cams, self._zarr_length, h, w),
                                    chunks=(1, 100, None, None),
                                    dtype='<u1',
                                    filters=filters, compressor=compressor)

        self._z_times = root.zeros('times', shape=(1, 2), dtype='M8[ns]')

        for i, cam in enumerate(self._cameras_list):
            root.attrs[i] = {'scale': cam.binning,
                             'framerate': cam.framerate,
                             'exposure': cam.exposure,
                             'triggered': cam.triggered,
                             'name': cam.name}

        self._grabbed_frames_idx = RawArray('I', self._nb_cams)
        self._saved_frms_idx = RawArray('I', self._nb_cams)

    def _trim_storage(self, mode='min') -> NoReturn:

        cut = min if mode == 'min' else max
        frame_limit = cut(self._saved_frms_idx)

        with self._file_access_lock:
            sh = self._z_frames.shape
            new_shape = (sh[0], frame_limit, sh[2], sh[3])
            self._z_frames.resize(new_shape)

        print(f'[INFO] Storage trimmed to {frame_limit}.')

    def _extend_storage(self) -> NoReturn:

        with self._file_access_lock:
            sh = self._z_frames.shape
            new_length = self._zarr_length * 2
            self._zarr_length = new_length
            new_shape = (sh[0], self._zarr_length, sh[2], sh[3])
            self._z_frames.resize(new_shape)
        print(f'[INFO] Storage extended to {new_length}.')

    def _writer(self, cam_idx: int, framestore: FrameStore) -> NoReturn:

        fps = self._cameras_list[cam_idx].framerate

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

                self._z_frames[cam_idx, self._saved_frms_idx[cam_idx]:self._saved_frms_idx[cam_idx] + nb, :, :] = data
                self._saved_frms_idx[cam_idx] += nb

                if self._saved_frms_idx[cam_idx] >= self._zarr_length * 0.9:
                    print('[INFO] Storage 90% full: extending...')
                    self._extend_storage()

                if saving_started and not self._recording.is_set():
                    self._finished_saving[cam_idx].set()

    def _grab_frames(self, cam_idx: int, framestore: FrameStore) -> NoReturn:

        cam = self._cameras_list[cam_idx]
        cam.start_grabbing()

        grabbed_frames = 0

        while self._acquiring.is_set():

            # self._barrier.wait()    # TODO - check if this is only needed when no hardware trigger

            with cam.ptr.RetrieveResult(100, py.TimeoutHandling_Return) as res:
                if res and res.GrabSucceeded():
                    frame_idx = res.ImageNumber
                    framedata = res.GetArray()
                    res.Release()

                    self._frames_buffers[cam_idx][:] = framedata.data.tobytes()
                    self._grabbed_frames_idx[cam_idx] = frame_idx

                    if self._recording.is_set():

                        if frame_idx > grabbed_frames:
                            framestore.put(framedata, qidx=cam_idx)
                            grabbed_frames += 1

    def _reset_name(self):
        if self._savepath is not None:
            files_op.rm_if_empty(self._savepath)
        self._acquisition_name = ''
        self._savepath = None

    def _cleanup(self) -> NoReturn:
        if self._z_frames is not None:
            self._trim_storage()
        self._reset_name()

        self._start_times = []
        self._stop_times = []

        self._saved_frms_idx = RawArray('I', self._nb_cams)

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

        # self._barrier = Barrier(self._nb_cams, timeout=5)
        self._executor = ThreadPoolExecutor(max_workers=10)

        for i, cam in enumerate(self._cameras_list):
            self._executor.submit(self._grab_frames, i, self._framestore)
            self._executor.submit(self._writer, i, self._framestore)

        print(f'\n[INFO] Grabbing started with {self._nb_cams} cameras...')

    def off(self) -> NoReturn:

        if self._recording.is_set():
            self._recording.clear()

        self._acquiring.clear()

        self.ICarray.StopGrabbing()
        for cam in self._cameras_list:
            cam.stop_grabbing()

        if self._triggered:
            self.trigger.off()

        self._cleanup()

        self._executor = None
        # self._barrier = None

        print(f'[INFO] Grabbing stopped.')

    @property
    def savepath(self) -> Path:
        return self._savepath

    @savepath.setter
    def savepath(self, value='') -> NoReturn:
        # Cleanup if a previous folder was created and not used
        self._cleanup()

        self._savepath = files_op.mk_folder(name=value)
        self._acquisition_name = self._savepath.stem

        self._init_storage()

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
        if self._grabbed_frames_idx is None:
            print('Please connect at least 1 camera first.')
        return self._grabbed_frames_idx

    @property
    def indices(self) -> np.array:
        if self._grabbed_frames_idx is None:
            print('Please connect at least 1 camera first.')
        return np.frombuffer(self._grabbed_frames_idx, dtype=np.uintc)

    @property
    def cameras(self) -> list[Camera]:
        return self._cameras_list

    def get_current_framebuffer(self, i=None) -> Union[bytearray, list[bytearray]]:
        if i is None:
            return self._frames_buffers
        else:
            return self._frames_buffers[i]

    def get_current_framearray(self, i: Union[str, int]) -> np.array:
        if type(i) is str:
            c = self._cameras_dict[i]
        else:
            c = self._cameras_list[i]
        return np.frombuffer(self._frames_buffers[i], dtype=np.uint8).reshape(c.height, c.width)
