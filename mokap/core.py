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
from mokap.hardware import SSHTrigger, BaslerCamera, setup_ulimit, enumerate_basler_devices, enumerate_flir_devices
from mokap import utils
from PIL import Image
import platform
import random
import json
from imageio_ffmpeg import write_frames


##

class FrameHandler(py.ImageEventHandler):

    def __init__(self, event, *args):
        self._is_recording = event
        self.indice = 0
        self.frames = deque()
        self.latest = deque(maxlen=2)
        super().__init__(*args)

    def OnImagesSkipped(self, camera, nb_skipped):
        print(f"[WARN] Skipped {nb_skipped} images!")

    def OnImageGrabbed(self, camera, res):
        img_nb = res.ImageNumber
        frame = res.GetArray()

        self.indice = img_nb
        if res.GrabSucceeded():
            if self._is_recording.is_set():
                self.frames.append((res.ImageNumber, frame))
            self.latest.append(frame)


class Manager:

    def __init__(self,
                 config='config.yaml',
                 framerate=220,
                 exposure=4318,
                 triggered=False,   # TODO - support hardware and software trigger modes
                 silent=True):

        self.config_dict = files_op.read_config(config)

        if 'Linux' in platform.system():
            setup_ulimit(silent=silent)

        self._display_framerate = 30
        self._display_wait_time_s = 1.0/self._display_framerate
        self._display_wait_time_us = self._display_wait_time_s * 1e6

        self._silent: bool = silent

        self._binning: int = 1
        self._binning_mode: str = 'sum'
        self._exposure: int = exposure
        self._framerate: int = framerate
        self._gain: float = 1.0
        self._gamma: float = 1.0
        self._blacks: float = 0.0

        self.trigger: Union[SSHTrigger, None] = None
        self._triggered = False
        self.triggered = triggered

        self._base_folder = Path(self.config_dict.get('base_path', './')) / 'MokapRecordings'
        if self._base_folder.parent == self._base_folder.name:
            self._base_folder = self._base_folder.parent
        self._base_folder.mkdir(parents=True, exist_ok=True)

        self._session_name: str = ''
        self._saving_ext = self.config_dict.get('save_format', 'bmp').lower()

        self._executor: Union[ThreadPoolExecutor, None] = None

        self._acquiring: Event = Event()
        self._recording: Event = Event()

        self._nb_cams: int = 0
        self._sources_list: List[BaslerCamera] = []
        self._sources_dict = {}
        self._cameras_colours = {}

        self._metadata = {'framerate': None,
                          'sessions': []}

        ##

        self.connect_basler_devices()

        ##

        self._sources_list.sort(key=lambda x: x.idx)

        # Initialise buffers and arrays
        self._frames_handlers_list: List[FrameHandler] = []
        self._lastframe_buffers_list: List[RawArray] = []

        for cam in self._sources_list:
            self._frames_handlers_list.append(FrameHandler(self._recording))
            self._lastframe_buffers_list.append(RawArray('B', cam.height * cam.width))

        self._nb_cams = len(self._sources_list)

        self._finished_saving: List[Event] = [Event()] * self._nb_cams

        # Init frames counters
        self._grabbed_frames_counter = RawArray('I', self._nb_cams)
        self._displayed_frames_counter = RawArray('I', self._nb_cams)
        self._saved_frames_counter = RawArray('I', self._nb_cams)

    @property
    def triggered(self) -> bool:
        return self._triggered

    @triggered.setter
    def triggered(self, new_val: bool):
        if not self._triggered and new_val:
            external_trigger = SSHTrigger(silent=self._silent)
            if external_trigger.connected:
                self.trigger = external_trigger
                self._triggered = True
                if not self._silent:
                    print('[INFO] Trigger mode enabled.')
            else:
                print("[ERROR] Connection problem with the trigger. Trigger mode can't be enabled.")
                self.trigger = None
                self._triggered = False
        elif self._triggered and not new_val:
            self.trigger = None
            self._triggered = False
            if not self._silent:
                print('[INFO] Trigger mode disabled.')

        # TODO - Refresh cameras on trigger mode change

    # def connect(self):

        # nb_flir_virtuals = sum([t & v for t, v in zip([(t == 'flir' or t == 'teledyne') for t in source_types], virtual_sources)])
        # flir_devices = enumerate_flir_devices(virtual_cams=nb_flir_virtuals)

        # if specific_cams is not None:
        #     specific_cams = utils.ensure_list(specific_cams)
        #     devices = [d for d in connected_cams if d.GetSerialNumber() in specific_cams]
        # else:
        #     allowed_serials = [self.config_dict[k]['serial'] for k in sources_in_config_file]
        #     devices = [d for d in connected_cams if d.GetSerialNumber() in allowed_serials]

    def connect_basler_devices(self):

        sources_names = list(self.config_dict['sources'].keys())
        source_types = [self.config_dict['sources'][n].get('type').lower() for n in sources_names]

        virtual_sources = [self.config_dict['sources'][n].get('virtual', False) for n in sources_names]
        nb_basler_virtuals = sum([t & v for t, v in zip([t == 'basler' for t in source_types], virtual_sources)])

        avail_basler_devices = enumerate_basler_devices(virtual_cams=nb_basler_virtuals)
        nb_basler_devices = len(avail_basler_devices)

        hues, saturation, luminance = utils.get_random_colors(nb_basler_devices)

        # Create the cameras and put them in auto-sorting CamList
        for i in range(nb_basler_devices):
            dptr = py.TlFactory.GetInstance().CreateDevice(avail_basler_devices[i])
            cptr = py.InstantCamera(dptr)

            source = BaslerCamera(framerate=self._framerate,
                               exposure=self._exposure,
                               triggered=self._triggered,
                               binning=self._binning)
            source.connect(cptr)

            if source.connected:

                source_name = 'unnamed'
                source_col = utils.hls_to_hex(hues[i], luminance[i], saturation[i])

                # Grab name and colour from config file if they're in there
                for n in sources_names:
                    if source.serial == str(self.config_dict['sources'][n].get('serial', 'virtual')):
                        source.name = n
                        source_col = f"#{self.config_dict['sources'][n].get('color', source_col).lstrip('#')}"

                self._cameras_colours[source.name] = source_col
                # keep local reference of cameras as list and dict for easy access
                self._sources_list.append(source)
                self._sources_dict[source.name] = source

                if not self._silent:
                    print(f"[INFO] Attached Basler camera {source}.")

    @property
    def framerate(self) -> Union[float, None]:
        if all([c.framerate == self._sources_list[0].framerate for c in self._sources_list]):
            return self._sources_list[0].framerate
        else:
            return None

    @framerate.setter
    def framerate(self, value: int) -> None:
        for i, cam in enumerate(self._sources_list):
            cam.framerate = value
        self._framerate = value

    @property
    def exposure(self) -> List[float]:
        return [c.exposure for c in self._sources_list]

    @exposure.setter
    def exposure(self, value: float) -> None:
        for i, cam in enumerate(self._sources_list):
            cam.exposure = value

    @property
    def gain(self) -> List[float]:
        return [c.gain for c in self._sources_list]

    @gain.setter
    def gain(self, value: float) -> None:
        for i, cam in enumerate(self._sources_list):
            cam.gain = value

    @property
    def blacks(self) -> List[float]:
        return [c.blacks for c in self._sources_list]

    @blacks.setter
    def blacks(self, value: float) -> None:
        for i, cam in enumerate(self._sources_list):
            cam.blacks = value

    @property
    def gamma(self) -> List[float]:
        return [c.gamma for c in self._sources_list]

    @gamma.setter
    def gamma(self, value: float) -> None:
        for i, cam in enumerate(self._sources_list):
            cam.gamma = value

    @property
    def binning(self) -> List[int]:
        return [c.binning for c in self._sources_list]

    @property
    def binning_mode(self) -> Union[str, None]:
        if all([c.binning_mode == self._sources_list[0].binning_mode for c in self._sources_list]):
            return self._sources_list[0].binning_mode
        else:
            return None

    @binning.setter
    def binning(self, value: int) -> None:
        for i, cam in enumerate(self._sources_list):
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
        for i, cam in enumerate(self._sources_list):
            cam.binning_mode = value

    def disconnect(self) -> None:

        for cam in self._sources_list:
            cam.disconnect()

        self._sources_list = []

        if not self._silent:
            print(f"[INFO] Disconnected {self._nb_cams} camera{'s' if self._nb_cams > 1 else ''}.")
        self._nb_cams = 0

    def _writer_frames(self, cam_idx: int) -> NoReturn:

        def save_frame():
            frame_nb, frame = handler.frames.popleft()
            img = Image.frombuffer("L", (w, h), frame, 'raw', "L", 0, 1)

            if self._saving_ext == 'bmp':
                img.save(folder / f"{str(frame_nb).zfill(9)}.{self._saving_ext}")
            elif self._saving_ext == 'jpg' or self._saving_ext == 'jpeg':
                img.save(folder / f"{str(frame_nb).zfill(9)}.{self._saving_ext}",
                         quality=100, keep_rgb=True)
            elif self._saving_ext == 'png':
                img.save(folder / f"{str(frame_nb).zfill(9)}.{self._saving_ext}",
                         compress_level=1)
            elif self._saving_ext == 'tif' or self._saving_ext == 'tiff':
                img.save(folder / f"{str(frame_nb).zfill(9)}.{self._saving_ext}", quality=100)
            else:
                img.save(folder / f"{str(frame_nb).zfill(9)}.bmp")
            self._saved_frames_counter[cam_idx] += 1

        h = self._sources_list[cam_idx].height
        w = self._sources_list[cam_idx].width

        folder = self.full_path / f"cam{cam_idx}_{self._sources_list[cam_idx].name}"
        handler = self._frames_handlers_list[cam_idx]

        started_saving = False
        finishing = False
        while self._acquiring.is_set():

            if self._recording.is_set():
                if not started_saving:
                    started_saving = True

                if bool(handler.frames):
                    save_frame()

            else:
                if started_saving:
                    if not finishing:
                        print('[INFO] Finishing saving...')
                        finishing = True
                    if bool(handler.frames):
                        save_frame()
                    else:
                        break
                else:
                    self._recording.wait()
        self._finished_saving[cam_idx].set()

    def _writer_video(self, cam_idx: int) -> NoReturn:

        h = self._sources_list[cam_idx].height
        w = self._sources_list[cam_idx].width

        folder = self.full_path / f"cam{cam_idx}_{self._sources_list[cam_idx].name}"
        file_name = folder / f"cam{cam_idx}.mp4"

        handler = self._frames_handlers_list[cam_idx]

        # additional_params = ["-r:v", str(self.framerate),
        #               "-preset", 'superfast',
        #               "-tune", "fastdecode",
        #               "-crf", str(18),
        #               "-bufsize", "20M",
        #               "-maxrate", "10M",
        #               "-bf:v", "4",]

        additional_params = ["-tune", "zerolatency",
                      # "-x264opts", "opencl",
                      "-profile:v", "high",
                      "-framerate", str(self.framerate),
                      # "-preset", 'fast',
                      "-crf", str(10),]

        writer = write_frames(
            file_name.as_posix(),
            (w, h),
            fps=self.framerate,
            codec='libx264',
            pix_fmt_in='gray',  # "bayer_bggr8", "gray", "rgb24", "bgr0", "yuv420p"
            pix_fmt_out="yuv420p",
            ffmpeg_log_level='warning',  # "warning", "quiet", "info"
            input_params=["-an"],
            output_params=additional_params,
            macro_block_size=8
        )
        writer.send(None)

        started_saving = False
        finishing = False
        while self._acquiring.is_set():

            if self._recording.is_set():
                if not started_saving:
                    started_saving = True

                if bool(handler.frames):
                    writer.send(np.frombuffer(handler.frames.popleft()[1], dtype=np.uint8).reshape(h, w))
                    self._saved_frames_counter[cam_idx] += 1
            else:
                if started_saving:
                    if not finishing:
                        print('[INFO] Finishing saving...')
                        finishing = True
                    if bool(handler.frames):
                        writer.send(np.frombuffer(handler.frames.popleft()[1], dtype=np.uint8).reshape(h, w))
                        self._saved_frames_counter[cam_idx] += 1
                    else:
                        break
                else:
                    self._recording.wait()
        print('[INFO] Closing video writers...')
        writer.close()
        self._finished_saving[cam_idx].set()

    def _update_display_buffers(self, cam_idx: int) -> NoReturn:

        handler = self._frames_handlers_list[cam_idx]
        tic = datetime.now()

        while self._acquiring.is_set():
            toc = datetime.now()
            elapsed = (toc - tic).microseconds
            if elapsed >= self._display_wait_time_us:
                self._lastframe_buffers_list[cam_idx] = handler.latest.popleft()
            else:
                time.sleep(self._display_wait_time_s)

            tic = toc

    def _grab_frames(self, cam_idx: int) -> NoReturn:

        cam = self._sources_list[cam_idx]

        cam.ptr.RegisterImageEventHandler(self._frames_handlers_list[cam_idx],
                                          py.RegistrationMode_ReplaceAll,
                                          py.Cleanup_Delete)
        cam.start_grabbing()

    def record(self) -> None:

        if not self._recording.is_set():
            if self._metadata['framerate'] is None:
                self._metadata['framerate'] = self.framerate
            else:
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
                if len(self._metadata['sessions']) > 1:
                    prev = self._metadata['sessions'][-2]['cameras'][i]['frames']
                else:
                    prev = 0
                self._metadata['sessions'][-1]['cameras'][i]['frames'] = self._saved_frames_counter[i] - prev

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
                self.trigger.start(self._framerate)
                time.sleep(0.5)

            if self._metadata['framerate'] is None:
                self._metadata['framerate'] = self.framerate

            self._acquiring.set()

            self._executor = ThreadPoolExecutor(max_workers=3 * self._nb_cams)

            for i, cam in enumerate(self._sources_list):
                self._executor.submit(self._grab_frames, i)
                self._executor.submit(self._writer_frames, i)
                # self._executor.submit(self._writer_video, i)
                self._executor.submit(self._update_display_buffers, i)

            if not self._silent:
                print(f"[INFO] Grabbing started with {self._nb_cams} camera{'s' if self._nb_cams > 1 else ''}...")

    def off(self) -> None:

        if self._acquiring.is_set():
            self.pause()
            self._acquiring.clear()

            for cam in self._sources_list:
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
    def cameras(self) -> list[BaslerCamera]:
        return self._sources_list

    @property
    def colours(self) -> dict:
        return self._cameras_colours

    @property
    def colors(self) -> dict:
        return self.colours

    @property
    def temperature(self) -> float:
        return np.mean([c.temperature for c in self._sources_list if c.temperature not in [0.0, 421.0]])

    @property
    def temperature_state(self) -> list[str]:
        return [c.temperature_state for c in self._sources_list]

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
            c = self._sources_dict[i]
        else:
            c = self._sources_list[i]
        return self._lastframe_buffers_list[i]
