from threading import Event
from multiprocessing import RawArray
from concurrent.futures import ThreadPoolExecutor
from typing import NoReturn, Union, List
from pathlib import Path
import time
from datetime import datetime
import numpy as np
import pypylon.pylon as py
import mokap.files_op as files_op
from collections import deque
from mokap.hardware import SSHTrigger, Camera, setup_ulimit, get_basler_devices
from mokap import utils
from PIL import Image
import platform
import configparser
import random

##

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
                 config='config.conf',
                 framerate=220,
                 exposure=4318,
                 triggered=False,   # TODO - support hardware and software trigger modes
                 silent=True):

        self.confparser = configparser.ConfigParser()
        try:
            self.confparser.read(config)
        except FileNotFoundError:
            print('[WARN] Config file not found. Defaulting to example config')
            self.confparser.read('example_config.conf')

        if 'Linux' in platform.system():
            setup_ulimit(silent=silent)

        self._display_framerate = 60
        self._silent: bool = silent

        self._binning: int = 1
        self._binning_mode: str = 'sum'
        self._exposure: int = exposure
        self._framerate: int = framerate
        self._gain: float = 1.0
        self._gamma: float = 1.0
        self._triggered: bool = triggered

        default_base_folder = Path(__file__).parent / 'recordings'
        if 'GENERAL' in self.confparser.sections():
             self._base_folder = Path(self.confparser['GENERAL'].get('base_folder', default_base_folder))
        else:
            self._base_folder = default_base_folder
        self._base_folder.mkdir(parents=True, exist_ok=True)
        self._session_name: Union[Path, None] = None

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
        self._cameras_colours = {}

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

        default_max_cams = 99
        default_allow_virtual = False
        if 'GENERAL' in self.confparser.sections():
            max_cams = self.confparser['GENERAL'].getint('max_cams', default_max_cams)
            allow_virtual = self.confparser['GENERAL'].getboolean('allow_virtual', default_allow_virtual)
        else:
            max_cams = default_max_cams
            allow_virtual = default_allow_virtual

        real_cams, virtual_cams = get_basler_devices(max_cams=max_cams, allow_virtual=allow_virtual)
        devices = real_cams + virtual_cams
        if not self._silent:
            print(f"[INFO] Found {len(devices)} camera{'s' if self._nb_cams > 1 else ''} connected "
                  f"({len(real_cams)} physical, {len(virtual_cams)} virtual).")

        return devices

    def connect(self, specific_cams=None):

        allow_all = self.confparser['GENERAL'].getboolean('allow_all', True)

        connected_cams = self.list_devices()

        cams_in_config_file = list(self.confparser.sections())
        cams_in_config_file.remove('GENERAL')

        if specific_cams is not None:
            specific_cams = utils.ensure_list(specific_cams)

            if allow_all and not self._silent:
                print(f"[WARN] All cameras allowed in config file, but user-provided list is used instead.")

            devices = [d for d in connected_cams if d.GetSerialNumber() in specific_cams]
        else:
            if not allow_all:
                allowed_serials = [self.confparser[k]['serial'] for k in cams_in_config_file]

                devices = [d for d in connected_cams if d.GetSerialNumber() in allowed_serials]
            else:
                devices = connected_cams

        nb_ignored = len(connected_cams) - len(devices)
        if nb_ignored > 0 and not self._silent:
            print(f"[INFO] Ignoring {nb_ignored} camera{'s' if nb_ignored > 1 else ''}.")

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

            if cam.connected:

                cam_name = 'unnamed'
                cam_col = '#' + utils.hls_to_hex(random.randint(0, 360),    # Sample whole Hue range
                                                 random.randint(45, 60),    # Keep somewhate narrow band luminance
                                                 85)                           # Fixed saturation

                # Grab name and colour from config file if they're in there
                for entry in cams_in_config_file:
                    if cam.serial == self.confparser[entry]['serial']:
                        cam_name = self.confparser[entry].get('name', cam_name)
                        cam_col = f'#{self.confparser[entry].get("color", cam_col).lstrip("#")}'

                cam.name = cam_name
                self._cameras_colours[cam.name] = cam_col
                # keep local reference of cameras as list and dict for easy access
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

        folder = self.session_name / f"cam{cam_idx}"
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
                    # TODO - Add support for multiple formats, which could be defined in the config file
                    # img.save(folder / f"{str(self._saved_frames_counter[cam_idx]).zfill(9)}.jpg", quality=100, keep_rgb=True)
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

    def _reset_name(self) -> None:
        if self._session_name is not None:
            files_op.rm_if_empty(self._session_name)
        self._session_name = None

    def _soft_reset(self) -> None:

        if self._session_name is not None:
            (self._session_name / 'recording').unlink(missing_ok=True)

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

            (self._session_name / 'recording').touch()

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

            if self._session_name is None:
                self.session_name = ''

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
    def session_name(self) -> Path:
        return self._session_name

    @session_name.setter
    def session_name(self, session_name='') -> None:

        self.pause()
        self._soft_reset()

        if session_name == '':
            session_name = datetime.now().strftime('%y%m%d-%H%M')

        self._session_name = files_op.exists_check(self._base_folder / session_name)
        self._session_name.mkdir(parents=True, exist_ok=False)

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
    def colours(self) -> dict:
        return self._cameras_colours

    @property
    def colors(self) -> dict:
        return self.colours

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
