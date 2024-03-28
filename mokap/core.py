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
import random
import json

##

class FrameHandler(py.ImageEventHandler):

    def __init__(self, event, *args):
        self._is_recording = event
        self.indice = 0
        self.frames = deque()
        self.latest = None
        self.rec = False
        super().__init__(*args)

    def OnImageEventHandlerRegistered(self, camera):
        siz = camera.Width.GetValue() * camera.Height.GetValue()
        if self.latest is None:
            self.latest = bytearray(siz)

    def OnImagesSkipped(self, camera, nb_skipped):
        print(f"[WARN] Skipped {nb_skipped} images!")

    def OnImageGrabbed(self, camera, res):
        if res.GrabSucceeded():
            self.indice = 0 + res.ImageNumber
            buf = res.GetBuffer()
            self.latest[:] = buf
            if self._is_recording.is_set():
                self.frames.append(buf)


class Manager:

    def __init__(self,
                 config='config.conf',
                 framerate=220,
                 exposure=4318,
                 triggered=False,   # TODO - support hardware and software trigger modes
                 silent=True):

        self.config_dict = files_op.read_config(config)

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
        self._blacks: float = 0.0
        self._triggered: bool = triggered

        default_base_folder = Path('./')
        if 'GENERAL' in self.config_dict.sections():
             self._base_folder = Path(self.config_dict['GENERAL'].get('base_folder', default_base_folder.as_posix()).strip("'").strip('"')) / 'MokapRecordings'
        else:
            self._base_folder = default_base_folder / 'MokapRecordings'
        if self._base_folder.parent == self._base_folder.name:
            self._base_folder = self._base_folder.parent
        self._base_folder.mkdir(parents=True, exist_ok=True)
        self._session_name: str = ''
        self._saving_ext = self.config_dict['GENERAL'].get('save_format', 'bmp').lower().lstrip('.').strip("'").strip('"')

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

        self._metadata = {'framerate': None,
                          'sessions': []}

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
        if 'GENERAL' in self.config_dict.sections():
            max_cams = self.config_dict['GENERAL'].getint('max_cams', default_max_cams)
            allow_virtual = self.config_dict['GENERAL'].getboolean('allow_virtual', default_allow_virtual)
        else:
            max_cams = default_max_cams
            allow_virtual = default_allow_virtual

        real_cams, virtual_cams = get_basler_devices(max_cams=max_cams, allow_virtual=allow_virtual)
        devices = real_cams + virtual_cams
        if not self._silent:
            print(f"[INFO] Found {len(devices)} camera{'s' if len(devices) > 1 else ''} connected "
                  f"({len(real_cams)} physical, {len(virtual_cams)} virtual).")

        return devices

    def connect(self, specific_cams=None):

        allow_all = self.config_dict['GENERAL'].getboolean('allow_all', True)

        connected_cams = self.list_devices()

        cams_in_config_file = list(self.config_dict.sections())
        cams_in_config_file.remove('GENERAL')

        if specific_cams is not None:
            specific_cams = utils.ensure_list(specific_cams)

            if allow_all and not self._silent:
                print(f"[WARN] All cameras allowed in config file, but user-provided list is used instead.")

            devices = [d for d in connected_cams if d.GetSerialNumber() in specific_cams]
        else:
            if not allow_all:
                allowed_serials = [self.config_dict[k]['serial'] for k in cams_in_config_file]

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
                    if cam.serial == self.config_dict[entry]['serial']:
                        cam_name = self.config_dict[entry].get('name', cam_name)
                        cam_col = f'#{self.config_dict[entry].get("color", cam_col).lstrip("#")}'

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
    def framerate(self) -> Union[float, None]:
        if all([c.framerate == self._cameras_list[0].framerate for c in self._cameras_list]):
            return self._cameras_list[0].framerate
        else:
            return None

    @framerate.setter
    def framerate(self, value: int) -> None:
        for i, cam in enumerate(self._cameras_list):
            cam.framerate = value

    @property
    def exposure(self) -> List[float]:
        return [c.exposure for c in self._cameras_list]

    @exposure.setter
    def exposure(self, value: float) -> None:
        for i, cam in enumerate(self._cameras_list):
            cam.exposure = value

    @property
    def gain(self) -> List[float]:
        return [c.gain for c in self._cameras_list]

    @gain.setter
    def gain(self, value: float) -> None:
        for i, cam in enumerate(self._cameras_list):
            cam.gain = value

    @property
    def blacks(self) -> List[float]:
        return [c.blacks for c in self._cameras_list]

    @blacks.setter
    def blacks(self, value: float) -> None:
        for i, cam in enumerate(self._cameras_list):
            cam.blacks = value

    @property
    def gamma(self) -> List[float]:
        return [c.gamma for c in self._cameras_list]

    @gamma.setter
    def gamma(self, value: float) -> None:
        for i, cam in enumerate(self._cameras_list):
            cam.gamma = value

    @property
    def binning(self) -> List[int]:
        return [c.binning for c in self._cameras_list]

    @property
    def binning_mode(self) -> Union[str, None]:
        if all([c.binning_mode == self._cameras_list[0].binning_mode for c in self._cameras_list]):
            return self._cameras_list[0].binning_mode
        else:
            return None

    @binning.setter
    def binning(self, value: int) -> None:
        for i, cam in enumerate(self._cameras_list):
            cam.binning = value
            self._binning = cam.binning
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

    def _writer_frames(self, cam_idx: int) -> NoReturn:

        h = self._cameras_list[cam_idx].height
        w = self._cameras_list[cam_idx].width

        folder = self.full_path / f"cam{cam_idx}_{self._cameras_list[cam_idx].name}"
        handler = self._frames_handlers_list[cam_idx]
        saving_started = False

        while self._acquiring.is_set():

            if self._recording.is_set():
                if not saving_started:
                    saving_started = True

                # Swap frames buffers
                data, handler.frames = handler.frames, deque()

                if len(data) > 0:
                    for frame in data:
                        img = Image.frombuffer("L", (w, h), frame, 'raw', "L", 0, 1)
                        if self._saving_ext == 'bmp':
                            img.save(folder / f"{str(self._saved_frames_counter[cam_idx]).zfill(9)}.{self._saving_ext}")
                        elif self._saving_ext == 'jpg' or self._saving_ext == 'jpeg':
                            img.save(folder / f"{str(self._saved_frames_counter[cam_idx]).zfill(9)}.{self._saving_ext}",
                                     quality=100, keep_rgb=True)
                        elif self._saving_ext == 'png':
                            img.save(folder / f"{str(self._saved_frames_counter[cam_idx]).zfill(9)}.{self._saving_ext}",
                                     compress_level=1)
                        elif self._saving_ext == 'tif' or self._saving_ext == 'tiff':
                            img.save(folder / f"{str(self._saved_frames_counter[cam_idx]).zfill(9)}.{self._saving_ext}",
                                     quality=100)
                        else:
                            img.save(folder / f"{str(self._saved_frames_counter[cam_idx]).zfill(9)}.bmp")
                        self._saved_frames_counter[cam_idx] += 1
            else:
                if not saving_started:
                    self._recording.wait()
                else:
                    self._finished_saving[cam_idx].set()

    def _update_display_buffers(self, cam_idx: int) -> NoReturn:

        handler = self._frames_handlers_list[cam_idx]
        tick = time.time()
        tock = time.time()

        while self._acquiring.is_set():
            if tock - tick >= (1 / self._display_framerate):
                self._lastframe_buffers_list[cam_idx] = handler.latest
                self._displayed_frames_counter[cam_idx] += 1
                self._grabbed_frames_counter[cam_idx] = handler.indice
                tick = tock
            else:
                time.sleep(1 / self._display_framerate)
            tock = time.time()

    def _grab_frames(self, cam_idx: int) -> NoReturn:

        cam = self._cameras_list[cam_idx]

        cam.ptr.RegisterConfiguration(py.SoftwareTriggerConfiguration(),
                                      py.RegistrationMode_ReplaceAll,
                                      py.Cleanup_Delete)

        cam.ptr.RegisterImageEventHandler(self._frames_handlers_list[cam_idx],
                                          py.RegistrationMode_Append,
                                          py.Cleanup_Delete)
        cam.start_grabbing()

    def record(self) -> None:

        if not self._recording.is_set():

            if self.framerate != self._metadata['framerate']:
                print(
                    f"[WARNING] Framerate is different from previous session{'s' if len(self._metadata['sessions']) > 1 else ''}!! Creating a new record...")

                print(f"old: {self.full_path}")
                self.off()
                self.on()

                print(f"Created {self.full_path}")

            (self.full_path / 'recording').touch(exist_ok=True)

            for c in self.cameras:
                (self.full_path / f'cam{c.idx}_{c.name}').mkdir(parents=True, exist_ok=True)

            session_metadata = {'start': datetime.now().timestamp(),
                                'end': 0.0,
                                'cameras': [{
                                    'idx': c.idx,
                                    'name': c.name,
                                    'width': c.width,
                                    'height': c.height,
                                    'exposure': c.exposure,
                                    'gain': c.gain,
                                    'gamma': c.gamma,
                                    'black_level': c.blacks} for c in self.cameras]}

            self._metadata['sessions'].append(session_metadata)
            with open(self.full_path / 'metadata.json', 'w', encoding='utf-8') as f:
                json.dump(self._metadata, f, ensure_ascii=False, indent=4)

            self._recording.set()

            if not self._silent:
                print('[INFO] Recording started...')

    def pause(self) -> None:
        if self._recording.is_set():

            self._recording.clear()

            if not self._silent:
                print('[INFO] Finishing saving...')
            [e.wait() for e in self._finished_saving]

            self._metadata['sessions'][-1]['end'] = datetime.now().timestamp()
            for i, cam in enumerate(self.cameras):
                self._metadata['sessions'][-1]['cameras'][i]['frames'] = self._saved_frames_counter[i]

            with open(self.full_path / 'metadata.json', 'w', encoding='utf-8') as f:
                json.dump(self._metadata, f, ensure_ascii=False, indent=4)

            (self.full_path / 'recording').unlink(missing_ok=True)

            if not self._silent:
                print('[INFO] Done saving.')

    def on(self) -> None:

        if not self._acquiring.is_set():

            # Just in case
            if self._session_name == '':
                self.session_name = ''

            if self._triggered:
                # Start trigger thread on the RPi
                self.trigger.start(self._framerate, 250000)
                time.sleep(0.5)

            if self._metadata['framerate'] is None:
                self._metadata['framerate'] = self.framerate

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

            files_op.rm_if_empty(self._base_folder / self._session_name)
            self._session_name = ''

            self._metadata['framerate'] = None
            self._metadata['sessions'] = []

            self._grabbed_frames_counter = RawArray('I', self._nb_cams)
            self._displayed_frames_counter = RawArray('I', self._nb_cams)
            self._saved_frames_counter = RawArray('I', self._nb_cams)
            self._grabbed_frames_counter[:] = [0] * self._nb_cams
            self._displayed_frames_counter[:] = [0] * self._nb_cams
            self._saved_frames_counter[:] = [0] * self._nb_cams

            self._executor = None

            if not self._silent:
                print(f'[INFO] Grabbing stopped.')

    @property
    def session_name(self) -> str:
        if self._session_name == '':
            self.session_name = ''
        return self._session_name

    @session_name.setter
    def session_name(self, new_name='') -> None:
        was_on = self._acquiring.is_set()
        if was_on:
            self.off()
            was_on = True

        old_name = self._session_name

        # Cleanup old folder if needed
        if old_name != '' and old_name is not None:
            files_op.rm_if_empty(self._base_folder / old_name)

        if new_name == '' or new_name is None:
            new_name = datetime.now().strftime('%y%m%d-%H%M')

        new_folder = files_op.exists_check(self._base_folder / new_name)
        new_folder.mkdir(parents=True, exist_ok=False)

        self._session_name = new_folder.name

        if was_on:
            self.on()


    @property
    def full_path(self) -> Path:
        return self._base_folder / self.session_name

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
    def saving_ext(self):
        return self._saving_ext.lower().lstrip('.').strip("'").strip('"')

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
