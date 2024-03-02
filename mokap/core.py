from threading import Event, Lock
import random
from multiprocessing import RawArray
from concurrent.futures import ThreadPoolExecutor
from typing import NoReturn, Union, List
from pathlib import Path
import time
import numpy as np
import pypylon.pylon as py
import mokap.files_op as files_op
from datetime import datetime
from collections import deque
# import zarr
# from numcodecs import Blosc, Delta
from mokap.hardware import SSHTrigger, Camera, setup_ulimit, get_basler_devices, config
from mokap.utils import ensure_list
from scipy import ndimage
# import cv2
from multiprocessing import Process
from PIL import Image
import platform

##

class MotionDetector:

    def __init__(self, cam_id=0, learning_rate=-100, thresh=5, lag=0, preview=False, framerate=30, silent=True):
        self._silent = silent
        self._learning_rate = learning_rate

        self._id = cam_id

        self._fgbg = cv2.createBackgroundSubtractorMOG2()

        self._kernel = np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]],
                                np.uint8)
        self._running = Event()
        self._movement = Event()

        log_path = Path(
            files_op.data_folder / f'detection_cam_{self._id}_{time.strftime("%y%m%d-%H%M%S", time.localtime())}.log')

        self._worker = Process(target=self._worker_func, args=(cam_id, thresh, lag, framerate, preview, log_path))

    def start(self):

        self._running.set()
        self._worker.start()
        time.sleep(0.1)
        if not self._silent:
            print('[INFO] Started movement detection...')

    def stop(self):
        self._running.clear()
        time.sleep(0.1)
        if not self._silent:
            print('[INFO] Stopped movement detection.')

    def _worker_func(self, cam_id, thresh, lag, framerate, preview, log_path):

        log_path.touch()

        cap = cv2.VideoCapture(cam_id)

        if not cap.isOpened():
            print("[ERROR] Camera is not open... Try again?")
            return

        success, first_frame = cap.read()
        if not success:
            print("[ERROR] Camera is not ready... Try again?")
            return

        shape = first_frame.shape

        detection_start = time.time()
        tick = time.time()

        loop_duration = 1.0 / float(framerate)
        log_every_n_seconds = 60

        initialised = False
        last_log_time = time.time()
        values_list = []

        if preview:
            cv2.namedWindow(f'Preview (Cam {cam_id})', cv2.WINDOW_NORMAL)

        while self._running.is_set():
            now = time.time()

            time_since_last_loop = now - tick
            time_since_last_log = now - last_log_time

            ret, frame = cap.read()

            if time_since_last_loop >= loop_duration:
                tick = time.time()

                detection = self.process(frame)
                value = detection.sum() / (shape[0] * shape[1]) * 10
                values_list.append(value)

                if not initialised:
                    self._movement.clear()
                    text = 'Initialising...'

                    if value < thresh and tick - detection_start >= 5:
                        initialised = True
                        text = f'{value:.2f}'
                else:
                    text = f'{value:.2f}'

                    if value >= thresh:
                        self._movement.set()
                        detection_start = time.time()
                    else:
                        if tick - detection_start >= lag:
                            self._movement.clear()

                if preview:
                    if self._movement.is_set():
                        text += ' - [ACTIVE]'

                    detection = cv2.putText(frame, text, (30, 30),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=1,
                                            color=(255, 0, 255),
                                            thickness=3)

                    cv2.imshow(f'Preview (Cam {cam_id})', detection)
                    cv2.waitKey(1)

            if time_since_last_log >= log_every_n_seconds:
                log_value = round(sum(values_list) / len(values_list), 3)
                to_log = f'{time.strftime("%d/%m/%y %H:%M:%S -", time.localtime())} {log_value}\n'

                with open(log_path, "a") as log_file:
                    log_file.write(to_log)

                values_list = []
                last_log_time = time.time()

    def process(self, frame):
        motion_mask = self._fgbg.apply(frame, self._learning_rate)

        se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        detection = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, se1)
        detection = cv2.morphologyEx(detection, cv2.MORPH_OPEN, se2)

        filtered = ndimage.gaussian_filter(detection, 1)
        _, filtered = cv2.threshold(filtered, 50, 255, cv2.THRESH_BINARY)

        return filtered

    @property
    def moves(self):
        return self._movement.is_set()


class FrameHandler(py.ImageEventHandler):

    def __init__(self, event, *args):
        self._is_recording = event
        self.indice = 0
        self.frames = deque()
        self.latest = None
        self.rec = False
        super().__init__(*args)

    def OnImagesSkipped(self, camera, nb_skipped):
        print(f"Skipped {nb_skipped} images.")

    def OnImageGrabbed(self, camera, res):
        if res.GrabSucceeded():
            self.indice = res.ImageNumber
            buf = res.GetBuffer()
            if self._is_recording.is_set():
                self.frames.append(buf)
            self.latest = buf


class Manager:

    def __init__(self,
                 framerate=220,
                 exposure=4318,
                 triggered=False,
                 binning=1,
                 binning_mode='sum',
                 gain=1,
                 gamma=1,
                 silent=True):

        if 'Linux' in platform.system():
            setup_ulimit(silent=silent)

        self._display_framerate = 60
        self._silent: bool = silent

        self._binning: int = binning
        self._binning_mode: str = binning_mode
        self._exposure: int = exposure
        self._framerate: int = framerate
        self._gain: float = gain
        self._gamma: float = gamma
        self._triggered: bool = triggered

        self._savepath: Union[Path, None] = None
        self._acquisition_name: str = ''

        self._executor: Union[ThreadPoolExecutor, None] = None

        self._acquiring: Event = Event()
        self._recording: Event = Event()

        self._frames_handlers_list: List[FrameHandler] = []
        self._lastframe_buffers_list: List[RawArray] = []
        self._grabbed_frames_counter: Union[RawArray, None] = None
        self._displayed_frames_counter: Union[RawArray, None] = None
        self._saved_frames_counter: Union[RawArray, None] = None

        self._nb_cams: int = 0
        self._cameras_list: List[Camera] = []
        self._cameras_dict = {}
        self._attributes = {}

        self.trigger: Union[SSHTrigger, None] = None
        self.ICarray: Union[py.InstantCameraArray, None] = None

        self._finished_saving: List[Event] = []

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
                if not self._silent:
                    print('[INFO] Trigger mode enabled.')
            else:
                print("[ERROR] Connection problem with the trigger. Trigger mode can't be enabled.")
                self.trigger = None
                self._triggered = False
        elif self._triggered and value is False:
            self.trigger = None
            self._triggered = False
            if not self._silent:
                print('[INFO] Trigger mode disabled.')

        # TODO - Refresh cameras on trigger mode change

    def list_devices(self):
        real_cams, virtual_cams = get_basler_devices()
        devices = real_cams + virtual_cams
        if not self._silent:
            print(f"[INFO] Found {len(devices)} camera{'s' if self._nb_cams > 1 else ''} connected "
                  f"({len(real_cams)} physical, {len(virtual_cams)} virtual).")

        return devices

    def connect(self, specific_cams=None):

        if specific_cams is not None:
            specific_cams = ensure_list(specific_cams)
            connected_cams = self.list_devices()
            devices = [d for d in connected_cams if d.GetSerialNumber() in specific_cams]
            ignored = len(connected_cams) - len(devices)
            if not self._silent:
                print(f"[WARN] Ignoring {ignored} camera{'s' if ignored > 1 else ''}.")
        else:
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
            if not self._silent:
                print(f"[INFO] Attached {cam}.")

        self._cameras_list.sort(key=lambda x: x.idx)

        # Once again for the buffers, this time using the sorted list
        self._frames_handlers_list = []
        self._lastframe_buffers_list = []
        for cam in self._cameras_list:
            self._frames_handlers_list.append(FrameHandler(self._recording))
            self._lastframe_buffers_list.append(RawArray('B', cam.height * cam.width))

        self._nb_cams = len(self._cameras_list)

        self._grabbed_frames_counter = RawArray('I', self._nb_cams)
        self._displayed_frames_counter = RawArray('I', self._nb_cams)
        self._saved_frames_counter = RawArray('I', self._nb_cams)
        self._grabbed_frames_counter[:] = [0] * self._nb_cams
        self._displayed_frames_counter[:] = [0] * self._nb_cams
        self._saved_frames_counter[:] = [0] * self._nb_cams

        self._finished_saving = [Event()] * self._nb_cams

    @property
    def framerate(self) -> int:
        return self._framerate

    @framerate.setter
    def framerate(self, value: int) -> None:
        self._framerate = value
        for i, cam in enumerate(self._cameras_list):
            cam.framerate = value

    @property
    def exposure(self) -> int:
        return self._exposure

    @exposure.setter
    def exposure(self, value: int) -> None:
        self._exposure = value
        for i, cam in enumerate(self._cameras_list):
            cam.exposure = value

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        self._gain = value
        for i, cam in enumerate(self._cameras_list):
            cam.gain = value

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        self._gamma = value
        for i, cam in enumerate(self._cameras_list):
            cam.gamma = value

    @property
    def binning(self) -> int:
        return self._binning

    @property
    def binning_mode(self) -> str:
        return self._binning_mode

    @binning.setter
    def binning(self, value: int) -> None:
        self._binning = value
        for i, cam in enumerate(self._cameras_list):
            cam.binning = value
            # And update the buffer to the new size
            self._lastframe_buffers_list[i] = RawArray('B', cam.height * cam.width)

    @binning_mode.setter
    def binning_mode(self, value: str) -> None:
        if value.lower() in ['s', 'sum', 'add', 'addition', 'summation']:
            self._binning_mode = 'sum'
        elif value.lower() in ['a', 'm', 'avg', 'average', 'mean']:
            self._binning_mode = 'avg'
        else:
            self._binning_mode = 'sum'
        for i, cam in enumerate(self._cameras_list):
            cam.binning_mode = value

    def disconnect(self) -> None:
        self.ICarray.Close()
        for cam in self._cameras_list:
            cam.disconnect()

        self._cameras_list = []
        self.ICarray = None
        if not self._silent:
            print(f"[INFO] Disconnected {self._nb_cams} camera{'s' if self._nb_cams > 1 else ''}.")
        self._nb_cams = 0

    def _init_storage(self) -> None:
        pass

    def _writer_frames(self, cam_idx: int) -> NoReturn:

        h = self._cameras_list[cam_idx].height
        w = self._cameras_list[cam_idx].width

        folder = self.savepath / f"cam{cam_idx}"
        folder.mkdir(parents=True, exist_ok=True)

        handler = self._frames_handlers_list[cam_idx]

        saving_started = False

        while self._acquiring.is_set():

            # Swap frames buffers
            data, handler.frames = handler.frames, deque()

            if len(data) > 0:
                if not saving_started:
                    saving_started = True

                for frame in data:
                    img = Image.frombuffer("L", (w, h), frame, 'raw', "L", 0, 1)
                    img.save(folder / f"{str(self._saved_frames_counter[cam_idx]).zfill(9)}.bmp")
                    # img.save(folder / f"{str(self._saved_frames_counter[cam_idx]).zfill(9)}.jpg", quality=100, keep_rgb=True)
                    # img.save(folder / f"{str(self._saved_frames_counter[cam_idx]).zfill(9)}.png", compress_level=1)
                    # img.save(folder / f"{str(self._saved_frames_counter[cam_idx]).zfill(9)}.tga")
                    # img.save(folder / f"{str(self._saved_frames_counter[cam_idx]).zfill(9)}.tga", compression='tga_rle')
                    self._saved_frames_counter[cam_idx] += 1

                if saving_started and not self._recording.is_set():
                    self._finished_saving[cam_idx].set()
            else:
                time.sleep((1 / self.framerate) * 0.9)

    def _update_display_buffers(self, cam_idx: int) -> NoReturn:

        handler = self._frames_handlers_list[cam_idx]

        while self._acquiring.is_set():
            time.sleep((1 / self._display_framerate) * 0.9)
            self._lastframe_buffers_list[cam_idx] = handler.latest
            self._displayed_frames_counter[cam_idx] += 1
            self._grabbed_frames_counter[cam_idx] = handler.indice

    def _grab_frames(self, cam_idx: int) -> NoReturn:

        cam = self._cameras_list[cam_idx]

        cam.ptr.RegisterConfiguration(py.SoftwareTriggerConfiguration(),
                                      py.RegistrationMode_ReplaceAll,
                                      py.Cleanup_Delete)

        cam.ptr.RegisterImageEventHandler(self._frames_handlers_list[cam_idx],
                                          py.RegistrationMode_Append,
                                          py.Cleanup_Delete)
        cam.start_grabbing()

        # while self._acquiring.is_set():
        #     # cam.ptr.RetrieveResult(100, py.TimeoutHandling_Return)
        #     time.sleep(1)
        #     if cam.ptr.WaitForFrameTriggerReady(200, py.TimeoutHandling_ThrowException):
        #         cam.ptr.ExecuteSoftwareTrigger()
        #         cam.ptr.RetrieveResult(0, py.TimeoutHandling_Return)


    def _reset_name(self) -> None:
        if self._savepath is not None:
            files_op.rm_if_empty(self._savepath)
        self._acquisition_name = ''
        self._savepath = None

    def _soft_reset(self) -> None:

        if self._savepath is not None:
            (self._savepath / 'recording').unlink(missing_ok=True)

        self._reset_name()

        self._start_times = []
        self._stop_times = []

        self._grabbed_frames_counter = RawArray('I', self._nb_cams)
        self._displayed_frames_counter = RawArray('I', self._nb_cams)
        self._saved_frames_counter = RawArray('I', self._nb_cams)
        self._grabbed_frames_counter[:] = [0] * self._nb_cams
        self._displayed_frames_counter[:] = [0] * self._nb_cams
        self._saved_frames_counter[:] = [0] * self._nb_cams

        self._executor = None

    def record(self) -> None:

        if not self._recording.is_set():
            self._recording.set()

            (self._savepath / 'recording').touch()

            if not self._silent:
                print('[INFO] Recording started...')

    def pause(self) -> None:
        if self._recording.is_set():
            self._recording.clear()

            if not self._silent:
                print('[INFO] Finishing saving...')
            [e.wait() for e in self._finished_saving]

            if not self._silent:
                print('[INFO] Done saving.')

    def on(self) -> None:

        if not self._acquiring.is_set():
            if self._triggered:
                # Start trigger thread on the RPi
                self.trigger.start(self._framerate, 250000)
                time.sleep(0.5)

            if self._savepath is None:
                self.savepath = ''

            self._acquiring.set()

            self._executor = ThreadPoolExecutor(max_workers=20)

            for i, cam in enumerate(self._cameras_list):
                self._executor.submit(self._grab_frames, i)
                self._executor.submit(self._writer_frames, i)
                self._executor.submit(self._update_display_buffers, i)

            if not self._silent:
                print(f"[INFO] Grabbing started with {self._nb_cams} camera{'s' if self._nb_cams > 1 else ''}...")

    def off(self) -> None:

        if self._acquiring.is_set():
            self.pause()
            self._acquiring.clear()

            self.ICarray.StopGrabbing()
            for cam in self._cameras_list:
                cam.stop_grabbing()

            if self._triggered:
                self.trigger.stop()

            self._soft_reset()
            if not self._silent:
                print(f'[INFO] Grabbing stopped.')

    @property
    def savepath(self) -> Path:
        return self._savepath

    @savepath.setter
    def savepath(self, value='') -> None:

        self.pause()
        self._soft_reset()

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
        if self._grabbed_frames_counter is None:
            print('[ERROR] Please connect at least 1 camera first.')
        return self._grabbed_frames_counter

    @property
    def indices(self) -> np.array:
        if self._grabbed_frames_counter is None:
            print('[ERROR] Please connect at least 1 camera first.')
        return np.frombuffer(self._grabbed_frames_counter, dtype=np.uint32)

    @property
    def cameras(self) -> list[Camera]:
        return self._cameras_list

    @property
    def saved_buf(self) -> RawArray:
        if self._saved_frames_counter is None:
            print('[ERROR] Please connect at least 1 camera first.')
        return self._saved_frames_counter

    @property
    def saved(self) -> np.array:
        if self._saved_frames_counter is None:
            print('[ERROR] Please connect at least 1 camera first.')
        return np.frombuffer(self._saved_frames_counter, dtype=np.uint32)



    def get_current_framebuffer(self, i: int = None) -> Union[bytearray, list[bytearray]]:
        if i is None:
            return self._lastframe_buffers_list
        else:
            return self._lastframe_buffers_list[i]

    def get_current_framearray(self, i: Union[str, int]) -> np.array:
        if type(i) is str:
            c = self._cameras_dict[i]
        else:
            c = self._cameras_list[i]
        return np.frombuffer(self._lastframe_buffers_list[i], dtype=np.uint8).reshape(c.height, c.width)

#
# class ManagerOld:
#
#     def __init__(self,
#                  framerate=220,
#                  exposure=4318,
#                  triggered=False,
#                  binning=1,
#                  gain=1,
#                  silent=True):
#
#         if 'Linux' in platform.system():
#             setup_ulimit(silent=silent)
#
#         self._display_framerate = 60
#         self._silent: bool = silent
#
#         self._binning: int = binning
#         self._exposure: int = exposure
#         self._framerate: int = framerate
#         self._gain: float = gain
#         self._triggered: bool = triggered
#
#         self._savepath: Union[Path, None] = None
#         self._acquisition_name: str = ''
#
#         self._executor: Union[ThreadPoolExecutor, None] = None
#
#         self._file_access_lock = Lock()
#         self._zarr_length = 36000
#         self._z_frames = None
#
#         self._framestores = []
#
#         self._frames_buffers = []
#
#         self._acquiring: Event = Event()
#         self._recording: Event = Event()
#
#         self._nb_cams: int = 0
#         self._cameras_list: List[Camera] = []
#         self._cameras_dict = {}
#         self._attributes = {}
#
#         self.trigger: Union[SSHTrigger, None] = None
#         self.ICarray: Union[py.InstantCameraArray, None] = None
#
#         self._grabbed_frames_idx = None
#         self._saved_frms_idx = None
#         self._finished_saving: List[Event] = []
#
#     @property
#     def triggered(self) -> bool:
#         return self._triggered
#
#     @triggered.setter
#     def triggered(self, value: bool):
#         if not self._triggered and value is True:
#             external_trigger = SSHTrigger()
#             if external_trigger.connected:
#                 self.trigger = external_trigger
#                 self._triggered = True
#                 if not self._silent:
#                     print('[INFO] Trigger mode enabled.')
#             else:
#                 print("[ERROR] Connection problem with the trigger. Trigger mode can't be enabled.")
#                 self.trigger = None
#                 self._triggered = False
#         elif self._triggered and value is False:
#             self.trigger = None
#             self._triggered = False
#             if not self._silent:
#                 print('[INFO] Trigger mode disabled.')
#
#         # TODO - Refresh cameras on trigger mode change
#
#     def list_devices(self):
#
#         real_cams, virtual_cams = get_basler_devices()
#         devices = real_cams + virtual_cams
#         if not self._silent:
#             print(f"[INFO] Found {len(devices)} camera{'s' if self._nb_cams > 1 else ''} connected "
#                   f"({len(real_cams)} physical, {len(virtual_cams)} virtual).")
#
#         return devices
#
#     def connect(self, specific_cams=None):
#
#         if specific_cams is not None:
#             specific_cams = ensure_list(specific_cams)
#             connected_cams = self.list_devices()
#             devices = [d for d in connected_cams if d.GetSerialNumber() in specific_cams]
#             ignored = len(connected_cams) - len(devices)
#             if not self._silent:
#                 print(f"[WARN] Ignoring {ignored} camera{'s' if ignored > 1 else ''}.")
#         else:
#             devices = self.list_devices()
#
#         nb_cams = len(devices)
#         self.ICarray = py.InstantCameraArray(nb_cams)
#
#         # Create the cameras and put them in auto-sorting CamList
#         for i in range(nb_cams):
#             dptr, cptr = devices[i], self.ICarray[i]
#             cptr.Attach(py.TlFactory.GetInstance().CreateDevice(dptr))
#             cam = Camera(framerate=self._framerate,
#                          exposure=self._exposure,
#                          triggered=self._triggered,
#                          binning=self._binning)
#             cam.connect(cptr)
#             self._cameras_list.append(cam)
#             self._cameras_dict[cam.name] = cam
#             if not self._silent:
#                 print(f"[INFO] Attached {cam}.")
#
#         self._cameras_list.sort(key=lambda x: x.idx)
#
#         # Once again for the buffers, this time using the sorted list
#         self._frames_buffers = []
#         self._framestores = []
#         for i, cam in enumerate(self._cameras_list):
#             self._frames_buffers.append(bytearray(b'\0' * cam.height * cam.width))
#             self._framestores.append(deque())
#
#         self._nb_cams = len(self._cameras_list)
#
#         self._grabbed_frames_idx = RawArray('I', self._nb_cams)
#         self._saved_frms_idx = RawArray('I', self._nb_cams)
#
#         self._finished_saving = [Event()] * self._nb_cams
#
#     @property
#     def framerate(self) -> int:
#         return self._framerate
#
#     @framerate.setter
#     def framerate(self, value: int) -> None:
#         self._framerate = value
#         for cam in self._cameras_list:
#             cam.framerate = value
#
#     @property
#     def exposure(self) -> int:
#         return self._exposure
#
#     @exposure.setter
#     def exposure(self, value: int) -> None:
#         self._exposure = value
#         for cam in self._cameras_list:
#             cam.exposure = value
#
#     @property
#     def gain(self) -> float:
#         return self._gain
#
#     @gain.setter
#     def gain(self, value: float) -> None:
#         self._gain = value
#         for i, cam in enumerate(self._cameras_list):
#             cam.gain = value
#
#     @property
#     def binning(self) -> int:
#         return self._binning
#
#     @binning.setter
#     def binning(self, value: int) -> None:
#         self._binning = value
#         for i, cam in enumerate(self._cameras_list):
#             cam.binning = value
#             # And update the bytearray to the new size
#             self._frames_buffers[i] = bytearray(b'\0' * cam.height * cam.width)
#
#     def disconnect(self) -> NoReturn:
#         self.ICarray.Close()
#         for cam in self._cameras_list:
#             cam.disconnect()
#
#         self._cameras_list = []
#         self.ICarray = None
#         if not self._silent:
#             print(f"[INFO] Disconnected {self._nb_cams} camera{'s' if self._nb_cams > 1 else ''}.")
#         self._nb_cams = 0
#
#     def _init_storage(self) -> None:
#
#         if self.nb_cameras > 1:
#             # Sanity check - should always be true
#             all_w = [c.width for c in self._cameras_list]
#             all_h = [c.height for c in self._cameras_list]
#
#             assert np.allclose(*all_w)
#             assert np.allclose(*all_h)
#
#             w = all_w[0]
#             h = all_h[0]
#         else:
#             w = int(self._cameras_list[0].width)
#             h = int(self._cameras_list[0].height)
#
#         filters = [Delta(dtype='<u1')]
#         compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)
#
#         root = zarr.open(self._savepath / f'{self._acquisition_name}.zarr', mode='w')
#         self._z_frames = root.zeros('frames',
#                                     shape=(self._nb_cams, self._zarr_length, h, w),
#                                     chunks=(1, 100, None, None),
#                                     dtype='<u1',
#                                     filters=filters, compressor=compressor)
#
#         self._z_times = root.zeros('times', shape=(1, 2), dtype='M8[ns]')
#
#         for i, cam in enumerate(self._cameras_list):
#             root.attrs[i] = {'scale': cam.binning,
#                              'framerate': cam.framerate,
#                              'exposure': cam.exposure,
#                              'triggered': cam.triggered,
#                              'name': cam.name}
#
#         self._grabbed_frames_idx = RawArray('I', self._nb_cams)
#         self._saved_frms_idx = RawArray('I', self._nb_cams)
#
#     def _trim_storage(self, mode='min') -> None:
#
#         cut = min if mode == 'min' else max
#         frame_limit = cut(self._saved_frms_idx)
#
#         with self._file_access_lock:
#             sh = self._z_frames.shape
#             new_shape = (sh[0], frame_limit, sh[2], sh[3])
#             self._z_frames.resize(new_shape)
#
#         if not self._silent:
#             print(f'[INFO] Storage trimmed to {frame_limit}.')
#
#     def _extend_storage(self) -> None:
#
#         with self._file_access_lock:
#             sh = self._z_frames.shape
#             new_length = self._zarr_length * 2
#             self._zarr_length = new_length
#             new_shape = (sh[0], self._zarr_length, sh[2], sh[3])
#             self._z_frames.resize(new_shape)
#         if not self._silent:
#             print(f'[INFO] Storage extended to {new_length}.')
#
#     def _writer(self, cam_idx: int) -> NoReturn:
#
#         fps = self._cameras_list[cam_idx].framerate
#
#         min_wait = 1/(fps * 0.1)
#         max_wait = 1/(fps * 0.5)
#
#         saving_started = False
#         while self._acquiring.is_set():
#
#             time.sleep(random.uniform(min_wait, max_wait))
#
#             data, self._framestores[cam_idx] = self._framestores[cam_idx], deque()
#
#             nb = len(data)
#
#             if nb > 0:
#                 if not saving_started:
#                     saving_started = True
#
#                 self._z_frames[cam_idx, self._saved_frms_idx[cam_idx]:self._saved_frms_idx[cam_idx] + nb, :, :] = data
#                 self._saved_frms_idx[cam_idx] += nb
#
#                 if self._saved_frms_idx[cam_idx] >= self._zarr_length * 0.9:
#                     if not self._silent:
#                         print('[INFO] Storage 90% full. Extending...')
#                     self._extend_storage()
#
#                 if saving_started and not self._recording.is_set():
#                     self._finished_saving[cam_idx].set()
#
#     def _grab_frames(self, cam_idx: int) -> None:
#
#         cam = self._cameras_list[cam_idx]
#         cam.start_grabbing()
#
#         grabbed_frames = 0
#
#         while self._acquiring.is_set():
#
#             with cam.ptr.RetrieveResult(100, py.TimeoutHandling_Return) as res:
#                 if res and res.GrabSucceeded():
#                     frame_idx = res.ImageNumber
#                     framedata = res.GetArray()
#                     res.Release()
#
#                     self._frames_buffers[cam_idx][:] = framedata.data.tobytes()
#                     self._grabbed_frames_idx[cam_idx] = frame_idx
#
#                     if self._recording.is_set():
#
#                         if frame_idx > grabbed_frames:
#                             self._framestores[cam_idx].append(framedata)
#                             grabbed_frames += 1
#
#     def _reset_name(self):
#         if self._savepath is not None:
#             files_op.rm_if_empty(self._savepath)
#         self._acquisition_name = ''
#         self._savepath = None
#
#     def _soft_reset(self) -> None:
#         if self._z_frames is not None:
#             self._trim_storage()
#
#         if self._savepath is not None:
#             (self._savepath / 'recording').unlink(missing_ok=True)
#
#         self._reset_name()
#
#         self._start_times = []
#         self._stop_times = []
#
#         self._saved_frms_idx = RawArray('I', self._nb_cams)
#         self._executor = None
#
#     def record(self) -> None:
#
#         if not self._recording.is_set():
#             self._recording.set()
#
#             (self._savepath / 'recording').touch()
#             self._z_times[-1, 0] = np.datetime64(datetime.now())
#
#             if not self._silent:
#                 print('[INFO] Recording started...')
#
#     def pause(self) -> None:
#         if self._recording.is_set():
#             self._recording.clear()
#
#             self._z_times[-1, 1] = np.datetime64(datetime.now())
#             self._z_times.append(np.zeros((1, 2), dtype='M8[ns]'))
#
#             if not self._silent:
#                 print('[INFO] Finishing saving...')
#             [e.wait() for e in self._finished_saving]
#
#             if not self._silent:
#                 print('[INFO] Done saving.')
#
#     def on(self) -> None:
#
#         if not self._acquiring.is_set():
#             if self._triggered:
#                 # Start trigger thread on the RPi
#                 self.trigger.start(self._framerate, 250000)
#                 time.sleep(0.5)
#
#             if self._savepath is None:
#                 self.savepath = ''
#
#             self._acquiring.set()
#
#             self._executor = ThreadPoolExecutor(max_workers=20)
#
#             for i, cam in enumerate(self._cameras_list):
#                 self._executor.submit(self._grab_frames, i)
#                 self._executor.submit(self._writer, i)
#
#             if not self._silent:
#                 print(f"[INFO] Grabbing started with {self._nb_cams} camera{'s' if self._nb_cams > 1 else ''}...")
#
#     def off(self) -> None:
#
#         if self._acquiring.is_set():
#             self.pause()
#             self._acquiring.clear()
#
#             self.ICarray.StopGrabbing()
#             for cam in self._cameras_list:
#                 cam.stop_grabbing()
#
#             if self._triggered:
#                 self.trigger.off()
#
#             self._soft_reset()
#             if not self._silent:
#                 print(f'[INFO] Grabbing stopped.')
#
#     @property
#     def savepath(self) -> Path:
#         return self._savepath
#
#     @savepath.setter
#     def savepath(self, value='') -> None:
#
#         self.pause()
#         self._soft_reset()
#
#         self._savepath = files_op.mk_folder(name=value)
#         self._acquisition_name = self._savepath.stem
#
#         self._init_storage()
#
#     @property
#     def nb_cameras(self) -> int:
#         return self._nb_cams
#
#     @property
#     def acquiring(self) -> bool:
#         return self._acquiring.is_set()
#
#     @property
#     def recording(self) -> bool:
#         return self._recording.is_set()
#
#     @property
#     def indices_buf(self) -> RawArray:
#         if self._grabbed_frames_idx is None:
#             print('[ERROR] Please connect at least 1 camera first.')
#         return self._grabbed_frames_idx
#
#     @property
#     def indices(self) -> np.array:
#         if self._grabbed_frames_idx is None:
#             print('[ERROR] Please connect at least 1 camera first.')
#         return np.frombuffer(self._grabbed_frames_idx, dtype=np.uintc)
#
#     @property
#     def saved_buf(self) -> RawArray:
#         if self._saved_frames_counter is None:
#             print('[ERROR] Please connect at least 1 camera first.')
#         return self._saved_frames_counter
#
#     @property
#     def saved(self) -> np.array:
#         if self._saved_frames_counter is None:
#             print('[ERROR] Please connect at least 1 camera first.')
#         return np.frombuffer(self._saved_frames_counter, dtype=np.uintc)
#
#
#     @property
#     def cameras(self) -> list[Camera]:
#         return self._cameras_list
#
#     def get_current_framebuffer(self, i: int = None) -> Union[bytearray, list[bytearray]]:
#         if i is None:
#             return self._frames_buffers
#         else:
#             return self._frames_buffers[i]
#
#     def get_current_framearray(self, i: Union[str, int]) -> np.array:
#         if type(i) is str:
#             c = self._cameras_dict[i]
#         else:
#             c = self._cameras_list[i]
#         return np.frombuffer(self._frames_buffers[i], dtype=np.uint8).reshape(c.height, c.width)