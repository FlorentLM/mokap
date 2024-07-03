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
import json
import os
import fnmatch
from subprocess import Popen, PIPE, STDOUT
import shlex


##

class FrameHandler(py.ImageEventHandler):
    """
    This object is responsible for handling frames received from one camera.
    It puts all the received frames in an unlimited deque, and puts a copy of the latest frame in a short deque.
    The former is used by the saver thread, and the latter is used by the display thread.
    """
    def __init__(self, event, *args):
        self._is_recording = event      # Refers to the recording event in the Manager class

        self.indice = 0     # Last received frame indice (determined by the camera)

        self.frames = deque()   # Unlimited deque for frames that need saving
        self.latest = deque(maxlen=2)   # Short deque for the display buffer updater thread
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
                 triggered=False,
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

        self._triggered = triggered

        if self._triggered:
            self.trigger = SSHTrigger()
        else:
            self.trigger = None

        self._base_folder = Path(self.config_dict.get('base_path', './')) / 'MokapRecordings'
        if self._base_folder.parent == self._base_folder.name:
            self._base_folder = self._base_folder.parent
        self._base_folder.mkdir(parents=True, exist_ok=True)

        self._session_name: str = ''
        self._saving_ext = self.config_dict.get('save_format', 'bmp').lower()
        saving_qual = float(self.config_dict.get('save_quality').rstrip('%'))

        # new_value = (saving_qual / 100) * (new_max - new_min) + new_min

        match self._saving_ext:
            case 'jpg' | 'tif' | 'tiff:':   # tiff quality is only for tiff_jpeg compression
                self._saving_qual = saving_qual
            case 'png':
                self._saving_qual = int(((saving_qual / 100) * -9) + 9)

        self._executor: Union[ThreadPoolExecutor, None] = None

        self._acquiring: Event = Event()
        self._recording: Event = Event()

        self._nb_cams: int = 0
        self._sources_list: List[BaslerCamera] = []
        self._sources_dict = {}
        self._cameras_colours = {}

        self._metadata = {'sessions': []}

        ##

        self.connect_basler_devices()

        ##

        # Sort the sources according to their idx
        self._sources_list.sort(key=lambda x: x.idx)

        # Initialise buffers and arrays
        self._framehandlers: List[FrameHandler] = []
        self._display_buffers: List[RawArray] = []

        for cam in self._sources_list:
            self._framehandlers.append(FrameHandler(self._recording))
            self._display_buffers.append(RawArray('B', cam.height * cam.width))

        self._nb_cams = len(self._sources_list)

        self._finished_saving: List[Event] = [Event()] * self._nb_cams

        # Init frames counters
        self._cnt_grabbed = RawArray('I', self._nb_cams)
        self._cnt_displayed = RawArray('I', self._nb_cams)
        self._cnt_saved = RawArray('I', self._nb_cams)

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

        # Instantiate Basler InstantCameras and link them to our BaslerCamera class
        for i in range(nb_basler_devices):
            dptr = py.TlFactory.GetInstance().CreateDevice(avail_basler_devices[i])
            cptr = py.InstantCamera(dptr)

            source = BaslerCamera(framerate=self._framerate,
                               exposure=self._exposure,
                               triggered=self._triggered,
                               binning=self._binning)
            source.connect(cptr)

            if source.connected:

                source_col = utils.hls_to_hex(hues[i], luminance[i], saturation[i])

                # Grab name and colour from config file if they're in there
                for n in sources_names:
                    if source.serial == str(self.config_dict['sources'][n].get('serial', 'virtual')):
                        source.name = n
                        source_col = f"#{self.config_dict['sources'][n].get('color', source_col).lstrip('#')}"

                self._cameras_colours[source.name] = source_col

                # Keep references of cameras as list and as dict for easy access
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

            # Need to update the display buffers to the new frame size
            self._display_buffers[i] = RawArray('B', cam.height * cam.width)

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

    def _image_writer_thread(self, cam_idx: int) -> NoReturn:
        """
            This thread writes frames to the disk

            Parameters
            ----------
            cam_idx: the index of the camera this threads belongs to
        """

        def save_frame():
            """
                Saves one frame and updates the saved frames counter
            """
            # TODO - This will be a proper method that does either images or videos once the videowriter is working fine

            frame_nb, frame = handler.frames.popleft()

            filepath = folder / f"{str(frame_nb).zfill(9)}.{self._saving_ext}"

            match self._saving_ext:
                case 'bmp':
                    Image.frombuffer("L", (w, h), frame, 'raw', "L", 0, 1).save(filepath)
                case 'jpg' | 'jpeg':
                    Image.frombuffer("L", (w, h), frame, 'raw', "L", 0, 1).save(filepath, quality=self._saving_qual, subsampling='4:2:0')
                case 'png':
                    Image.frombuffer("L", (w, h), frame, 'raw', "L", 0, 1).save(filepath, compress_level=self._saving_qual, optimize=False)
                case 'tif' | 'tiff':
                    if self._saving_qual == 100:
                        Image.frombuffer("L", (w, h), frame, 'raw', "L", 0, 1).save(filepath, compression=None)
                    else:
                        Image.frombuffer("L", (w, h), frame, 'raw', "L", 0, 1).save(filepath, compression='jpeg', quality=self._saving_qual)
                case _:
                    Image.frombuffer("L", (w, h), frame, 'raw', "L", 0, 1).save(filepath)

            # The following is a RawArray, so the count is not atomic!
            # But it is fine as this is only for a rough estimation
            # (the actual number of written files is counted in a safe way when recording stops)
            self._cnt_saved[cam_idx] += 1

        h = self._sources_list[cam_idx].height
        w = self._sources_list[cam_idx].width

        folder = self.full_path / f"cam{cam_idx}_{self._sources_list[cam_idx].name}"
        handler = self._framehandlers[cam_idx]

        started_saving = False
        while self._acquiring.is_set():

            if self._recording.is_set():
                # Recording is set - do actual work

                # Do this just once at the start of a new recording session
                if not started_saving:
                    self._finished_saving[cam_idx].clear()
                    started_saving = True

                # Main state of this thread: If the queue is not empty, save a new frame
                if handler.frames:
                    save_frame()
            else:
                # Recording is not set - either it hasn't started, or it has but hasn't finished yet
                if started_saving:
                    # Recording has been started, so remaining frames still need to be saved
                    while handler.frames:
                        save_frame()

                    # Finished saving, reverting to wait state
                    self._finished_saving[cam_idx].set()
                    started_saving = False
                else:
                    # Default state of this thread: if cameras are acquiring but we're not recording, just wait
                    self._recording.wait()

    def _video_writer_thread(self, cam_idx: int) -> NoReturn:
        """
            This thread writes videos to the disk

            Parameters
            ----------
            cam_idx: the index of the camera this threads belongs to
        """

        h = self._sources_list[cam_idx].height
        w = self._sources_list[cam_idx].width
        fps = self._sources_list[cam_idx].framerate

        filepath = self.full_path / f"cam{cam_idx}_{self._sources_list[cam_idx].name}.mp4"
        handler = self._framehandlers[cam_idx]

        # macOS commands:
        command = f'ffmpeg -threads 1 -y -s {w}x{h} -f rawvideo -framerate {fps} -pix_fmt gray8 -i pipe:0 -an -c:v hevc_videotoolbox -realtime 1 -q:v 100 -tag:v hvc1 -pix_fmt yuv420p -r:v {fps} {filepath.as_posix()}'
        # command = f'ffmpeg -threads 1 -y -s {w}x{h} -f rawvideo -framerate {fps} -pix_fmt gray8 -i pipe:0 -an -c:v h264_videotoolbox -realtime 1 -q:v 100 -pix_fmt yuv420p -r:v {fps} {filepath.as_posix()}'

        # Windows commands:
        # TODO

        # Linux commands:
        # TODO

        process = Popen(shlex.split(command), stdin=PIPE, stdout=PIPE, stderr=STDOUT)

        def save_frame():
            """
                Saves one frame and updates the saved frames counter
            """
            # TODO - This will be a proper method that does either images or videos once the videowriter is working fine

            frame_nb, frame = handler.frames.popleft()
            process.stdin.write(frame.tobytes())

            # The following is a RawArray, so the count is not atomic!
            # But it is fine as this is only for a rough estimation
            # (the actual number of written files is counted in a safe way when recording stops)
            self._cnt_saved[cam_idx] += 1


        started_saving = False
        while self._acquiring.is_set():

            if self._recording.is_set():
                # Recording is set - do actual work

                # Do this just once at the start of a new recording session
                if not started_saving:
                    self._finished_saving[cam_idx].clear()
                    started_saving = True

                # Main state of this thread: If the queue is not empty, save a new frame
                if handler.frames:
                    save_frame()
            else:
                # Recording is not set - either it hasn't started, or it has but hasn't finished yet
                if started_saving:
                    # Recording has been started, so remaining frames still need to be saved
                    while handler.frames:
                        save_frame()

                    # Finished saving, reverting to wait state

                    process.stdin.close()
                    process.wait()

                    self._finished_saving[cam_idx].set()
                    started_saving = False
                else:
                    # Default state of this thread: if cameras are acquiring but we're not recording, just wait
                    self._recording.wait()

    def _display_updater_thread(self, cam_idx: int) -> NoReturn:
        """
            This thread updates the display buffers at a relatively slow pace (not super accurate timing but who cares),
            and otherwise mostly sleeps

            Parameters
            ----------
            cam_idx: the index of the camera this threads belongs to
        """

        # TODO - Rewrite this in a more elegant way

        handler = self._framehandlers[cam_idx]
        tic = datetime.now()

        while self._acquiring.is_set():
            toc = datetime.now()
            elapsed = (toc - tic).microseconds
            if elapsed >= self._display_wait_time_us:
                self._display_buffers[cam_idx] = handler.latest.popleft()
            else:
                time.sleep(self._display_wait_time_s)

            tic = toc

    def _grabber_thread(self, cam_idx: int) -> NoReturn:
        """
            This thread's loop is under control of the camera thanks to the FrameHandler object
            Parameters
            ----------
            cam_idx: the index of the camera this threads belongs to
        """
        cam = self._sources_list[cam_idx]

        cam.ptr.RegisterImageEventHandler(self._framehandlers[cam_idx],
                                          py.RegistrationMode_ReplaceAll,
                                          py.Cleanup_Delete)
        cam.start_grabbing()

    def record(self) -> None:
        """
            Start recording session
        """
        if not self._recording.is_set():

            (self.full_path / 'recording').touch(exist_ok=True)

            for c in self.cameras:
                (self.full_path / f'cam{c.idx}_{c.name}').mkdir(parents=True, exist_ok=True)

            session_metadata = {'start': datetime.now().timestamp(),
                                'end': 0.0,
                                'duration': 0.0,
                                'framerate_theoretical': self.framerate,
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

            self._recording.set()       # Start Recording event

            if not self._silent:
                print('[INFO] Recording started...')

    def pause(self) -> None:
        """
            Stops the current recording session
        """
        if self._recording.is_set():

            # Update the metadata with end time and number of saved frames
            self._metadata['sessions'][-1]['end'] = datetime.now().timestamp()
            duration = self._metadata['sessions'][-1]['end'] - self._metadata['sessions'][-1]['start']
            self._metadata['sessions'][-1]['duration'] = duration

            self._recording.clear()  # End recording event

            if not self._silent:
                print('[INFO] Finishing saving...')

            # Wait for all writer threads to finish saving the current session
            [e.wait() for e in self._finished_saving]

            for i, cam in enumerate(self.cameras):
                # Read back how many frames were recorded in previous sessions of this acquisition
                previsouly_saved = sum([self._metadata['sessions'][p]['cameras'][i].get('frames', 0) for p in range(len(self._metadata['sessions']))])

                # Wait for all files to finish being written and write the number of frames for this session
                saved_frames = self._safe_files_counter(self.full_path / f'cam{i}_{cam.name}')
                saved_frames_curr_sess = saved_frames - previsouly_saved

                self._metadata['sessions'][-1]['cameras'][i]['frames'] = saved_frames_curr_sess
                self._metadata['sessions'][-1]['cameras'][i]['framerate_actual'] = saved_frames_curr_sess / duration

            with open(self.full_path / 'metadata.json', 'w', encoding='utf-8') as f:
                json.dump(self._metadata, f, ensure_ascii=False, indent=4)

            (self.full_path / 'recording').unlink(missing_ok=True)

            if not self._silent:
                print('[INFO] Done saving.')

    def _safe_files_counter(self, path: Union[Path, str]) -> int:
        """
            This counts the number of files in the given path, in a safe manner:
            it keeps checking if new files are still being written

            Parameters
            ----------
            path : Path or str
                The folder to check

            Returns
            -------
            int
            The number of files
        """

        saved_frames = 0
        saved_frames_n = len(fnmatch.filter(os.listdir(path), f'*.{self._saving_ext}'))
        while saved_frames_n > saved_frames:
            saved_frames = saved_frames_n
            time.sleep(0.1)
            saved_frames_n = len(
                fnmatch.filter(os.listdir(path), f'*.{self._saving_ext}'))
        return saved_frames_n

    def on(self) -> None:
        """
            Start acquisition on all cameras
        """

        if not self._acquiring.is_set():

            # Just in case...
            if self._session_name == '':
                self.session_name = ''

            if self._triggered:
                self.trigger.start(self._framerate)
                time.sleep(0.1)

            self._acquiring.set()   # Start Acquiring event

            # Start 3 threads per camera:
            #   - One that grabs frames continuously from the camera
            #   - One that writes frames continuously to disk
            #   - One that (less frequently) updates local buffers for displaying
            self._executor = ThreadPoolExecutor()

            for i, cam in enumerate(self._sources_list):
                self._executor.submit(self._grabber_thread, i)
                self._executor.submit(self._image_writer_thread, i)
                # self._executor.submit(self._video_writer_thread, i)
                self._executor.submit(self._display_updater_thread, i)

            if not self._silent:
                print(f"[INFO] Grabbing started with {self._nb_cams} camera{'s' if self._nb_cams > 1 else ''}...")

    def off(self) -> None:
        """
            Stop acquisition on all cameras
        """

        if self._acquiring.is_set():

            # If we were recording, gracefully stop it
            self.pause()

            self._acquiring.clear()     # End Acquisition event

            for cam in self._sources_list:
                cam.stop_grabbing()     # Ask the cameras to stop grabbing (they control the thread execution loop)

            if self._triggered:
                self.trigger.stop()

            # Clean up the acquisition folder if nothing was written in it
            files_op.rm_if_empty(self._base_folder / self._session_name)

            # Reset everything for next acquisition
            self._session_name = ''
            self._metadata['sessions'] = []

            self._cnt_grabbed = RawArray('I', self._nb_cams)
            self._cnt_displayed = RawArray('I', self._nb_cams)
            self._cnt_saved = RawArray('I', self._nb_cams)

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

        # TODO = Shouldn't this be called Acquisition instead of session? It's *multiple sessions* per acquisition...

        # If we're currently acquiring, temporarily stop in order to rename the session...
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

        # ...and start again
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
    def indices(self) -> np.array:
        """
            Current frames indices for all acquiring cameras
            NB: These indices may be slightly off, they should not be used for anything critical!!

            Returns
            -------
            np.array with shape (n_cams)
        """
        if self._cnt_grabbed is None:
            print('[ERROR] Please connect at least 1 camera first.')
        # The buffer is non-atomic so the counts might be slightly off - they should not be used for anything critical
        return np.frombuffer(self._cnt_grabbed, dtype=np.uint32)

    @property
    def cameras(self) -> list[BaslerCamera]:
        return self._sources_list

    @property
    def colours(self) -> dict:
        return self._cameras_colours

    colors = colours    # An alias for our US American friends :p

    @property
    def temperature(self) -> float:
        return np.mean([c.temperature for c in self._sources_list if c.temperature is not None])

    @property
    def temperature_states(self) -> list[str]:
        return [c.temperature_state for c in self._sources_list]

    @property
    def saving_ext(self):
        return self._saving_ext.lower().lstrip('.').strip("'").strip('"')

    @property
    def saved(self) -> np.array:
        """
            Number of frames saved for all recording cameras
            NB: These counts may be slightly off, they should not be used for anything critical!!

            Returns
            -------
            np.array with shape (n_cams)
        """
        if self._cnt_saved is None:
            print('[ERROR] Please connect at least 1 camera first.')
        # The buffer is non-atomic so the counts might be slightly off - they should not be used for anything critical
        return np.frombuffer(self._cnt_saved, dtype=np.uint32)

    def get_current_framebuffer(self, i: int = None) -> Union[bytearray, list[bytearray]]:
        """
            Returns the current display frame buffer(s) for one or all cameras.
            NB: These arrays are not atomically readable - they are just for visualisation.

            Returns
            -------
            RawArray, or list of RawArray - each RawArray is a buffer of length (width * height)
        """
        if i is None:
            return self._display_buffers
        else:
            return self._display_buffers[i]
