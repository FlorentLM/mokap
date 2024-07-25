import subprocess
from threading import Thread, Event, get_ident
from multiprocessing import RawArray
from typing import NoReturn, Union, List
from pathlib import Path
import time
from datetime import datetime
import cv2
import numpy as np
import pypylon.pylon as py
import mokap.files_op as files_op
from collections import deque
from mokap.hardware import SSHTrigger, BaslerCamera, setup_ulimit, enumerate_basler_devices, enumerate_flir_devices
from PIL import Image
import platform
import json
import os
import fnmatch
from subprocess import Popen, PIPE, STDOUT
import shlex
import sys
import re


class MultiCam:
    COLOURS = ['#3498db', '#f4d03f', '#27ae60', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#F5A7D4', '#34495e', '#bdc3c7',
               '#2471a3', '#d4ac0d', '#186a3b', '#922b21', '#6c3483', '#d35400', '#117a65', '#e699db', '#1c2833', '#707b7c']
    def __init__(self,
                 config='config.yaml',
                 framerate=220,
                 exposure=4318,
                 triggered=False,
                 silent=True):

        self.config_dict = files_op.read_config(config)

        self._ffmpeg_path = 'ffmpeg'

        if 'Linux' in platform.system():
            setup_ulimit(silent=silent)

        self._display_framerate = 30
        self._display_wait_time_s = 1.0/self._display_framerate
        self._display_wait_time_us = self._display_wait_time_s * 1e6

        self._silent: bool = silent

        self._binning: int = 1
        self._binning_mode: str = 'sum'
        self._exposure: int = exposure
        self._framerate: float = framerate
        self._gain: float = 1.0
        self._gamma: float = 1.0
        self._blacks: float = 0.0

        self._triggered = triggered

        if self._triggered:
            self.trigger = SSHTrigger()
        else:
            self.trigger = None

        self._base_folder = Path(self.config_dict.get('base_path', './')) / 'MokapRecordings'
        if self._base_folder.parent.name == self._base_folder.name:
            self._base_folder = self._base_folder.parent
        if re.match(r'[A-Z]:', self._base_folder.parts[0]) and 'Darwin' in platform.system():
            self._base_folder = Path(self._base_folder.as_posix()[2:].lstrip('/'))
        self._base_folder.mkdir(parents=True, exist_ok=True)

        self._session_name: str = ''
        self._saving_ext = self.config_dict.get('save_format', 'bmp').lower()
        saving_qual = float(self.config_dict.get('save_quality'))

        self._config_encoding_params = self.config_dict.get('encoding_parameters', None)
        self._config_encoding_gpu = self.config_dict.get('gpu', False)

        # new_value = (saving_qual / 100) * (new_max - new_min) + new_min

        match self._saving_ext:
            case 'jpg' | 'tif' | 'tiff:':   # tiff quality is only for tiff_jpeg compression
                self._saving_qual = int(saving_qual)
            case 'png':
                self._saving_qual = int(((saving_qual / 100) * -9) + 9)

        self._estim_file_size = None

        # self._executor: Union[ThreadPoolExecutor, None] = None

        self._acquiring: bool = False
        self._recording: bool = False

        self._nb_cams: int = 0

        self._sources_dict = {}
        self._cameras_colours = {}

        self._metadata = {'sessions': []}

        # Initialise the list of sources
        self._sources_list: List[BaslerCamera] = []
        # and populate it    # TODO - Other brands
        self.connect_basler_devices()

        # Initialise the other lists (buffers and events)
        self._l_display_buffers: List[np.array] = []
        self._l_finished_saving: List[Event] = []
        self._l_all_frames: List[deque] = []
        self._l_latest_frames: List[deque] = []

        # Initialise a list of subprocesses
        self._videowriters: List[Union[bool, subprocess.Popen]] = []

        # Sort the sources according to their idx
        self._sources_list.sort(key=lambda x: x.idx)
        self._nb_cams = len(self._sources_list)

        # and populate the lists
        for i, cam in enumerate(self._sources_list):
            self._l_display_buffers.append(np.zeros(cam.shape, dtype=np.uint8))
            self._l_finished_saving.append(Event())
            self._l_all_frames.append(deque())
            self._l_latest_frames.append(deque(maxlen=1))
            self._videowriters.append(False)

        # Init frames counters
        self._cnt_grabbed = RawArray('I', int(self._nb_cams))
        self._cnt_displayed = RawArray('I', int(self._nb_cams))
        self._cnt_saved = RawArray('I', int(self._nb_cams))

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

        if self.config_dict['sources'] is not None:
            config_sources_names = list(self.config_dict['sources'].keys())
            config_source_types = [self.config_dict['sources'][n].get('type').lower() for n in config_sources_names]

            config_virtual_sources = [self.config_dict['sources'][n].get('virtual', False) for n in config_sources_names]
            config_nb_basler_virtuals = sum([t & v for t, v in zip([t == 'basler' for t in config_source_types], config_virtual_sources)])
        else:
            config_sources_names = []
            config_nb_basler_virtuals = 0

        avail_basler_devices = enumerate_basler_devices(virtual_cams=config_nb_basler_virtuals)
        nb_basler_devices = len(avail_basler_devices)

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

                source_col = MultiCam.COLOURS[i]

                # Grab name and colour from config file if they're in there
                for n in config_sources_names:
                    if source.serial == str(self.config_dict['sources'][n].get('serial', 'virtual')):
                        source.name = n
                        source_col = f"#{self.config_dict['sources'][n].get('color', source_col).lstrip('#')}"

                self._cameras_colours[source.name] = source_col

                # Keep references of cameras as list and as dict for easy access
                self._sources_list.append(source)
                self._sources_dict[source.name] = source

                if not self._silent:
                    print(f"[INFO] Attached {source}")

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._sources_list[i]
        if isinstance(i, slice):
            return self._sources_list[i]
        if isinstance(i, str):
            return self._sources_dict[i]

    def __len__(self):
        return self._nb_cams

    @property
    def framerate(self) -> List[float]:
        return [c.framerate for c in self._sources_list]

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
    def binning_mode(self) -> List[str]:
        return [c.binning_mode for c in self._sources_list]

    @binning.setter
    def binning(self, value: int) -> None:
        for i, cam in enumerate(self._sources_list):
            cam.binning = value
            self._binning = cam.binning

            # Need to update the display buffers to the new frame size
            self._l_display_buffers[i] = np.zeros(cam.shape, dtype=np.uint8)

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
            print(f"[INFO] Disconnected {self._nb_cams} camera{'s' if self._nb_cams > 1 else ''}")
        self._nb_cams = 0

    def _init_videowriter(self, cam_idx: int):
        if self._saving_ext == 'mp4':
            cam = self._sources_list[cam_idx]

            if not self._videowriters[cam_idx]:
                dummy_frame = np.zeros((cam.height, cam.width), dtype=np.uint8)
                filepath = self.full_path / f"cam{cam.idx}_{cam.name}_session{len(self._metadata['sessions'])-1}.mp4"

                # TODO - Get available hardware-accelerated encoders on user's system and choose the best one automatically
                # TODO - Why is QSV not working????
                # TODO - h265 only for now, x264 would be nice too

                if len(cam.shape) == 2:
                    fmt = 'gray8'   # TODO - Check if the camera is using 8 or 10 or 12 bits per pixel
                else:
                    fmt = 'rgb8'    # TODO - Check if the camera is using another filter

                input_params = f'{self._ffmpeg_path} -hide_banner -threads 1 -y -s {cam.width}x{cam.height} -f rawvideo -framerate {cam.framerate} -pix_fmt {fmt} -i pipe:0'

                if self._config_encoding_params is not None:
                    output_params = self._config_encoding_params
                else:
                    if 'Linux' in platform.system():
                        if self._config_encoding_gpu:
                            output_params = f'-an -c:v hevc_nvenc -preset llhp -zerolatency 1 -2pass 0 -rc cbr_ld_hq -pix_fmt yuv420p -r:v {cam.framerate}'
                        else:
                            output_params =  f'-an -c:v libx265 -preset veryfast -tune zerolatency -crf 20 -pix_fmt yuv420p -r:v {cam.framerate}'
                    elif 'Windows' in platform.system():
                        if self._config_encoding_gpu:
                            output_params =  f' -an -c:v hevc_nvenc -preset llhp -zerolatency 1 -2pass 0 -rc cbr_ld_hq -pix_fmt yuv420p -r:v {cam.framerate}'
                        else:
                            output_params =  f'-an -c:v libx265 -preset veryfast -tune zerolatency -crf 20 -pix_fmt yuv420p -r:v {cam.framerate}'
                    elif 'Darwin' in platform.system():
                        if self._config_encoding_gpu:
                            output_params = f'-an -c:v hevc_videotoolbox -realtime 1 -q:v 100 -tag:v hvc1 -pix_fmt yuv420p -r:v {cam.framerate}'
                        else:
                            output_params =  f'-an -c:v libx265 -preset veryfast -tune zerolatency -crf 20 -pix_fmt yuv420p -r:v {cam.framerate}'
                    else:
                        raise SystemExit('[ERROR] Unsupported platform')

                command = f'{input_params.strip()} {output_params.strip()} {filepath.as_posix()}'.replace('  ', ' ')

                # p = Popen(shlex.split(command), stdin=PIPE, close_fds=ON_POSIX)     # Debug mode (stderr/stdout on)
                p = Popen(shlex.split(command), stdin=PIPE, stdout=False, stderr=False)
                p.stdin.write(dummy_frame.tobytes())
                self._videowriters[cam_idx] = p
        else:
            self._videowriters[cam_idx] = False

    def _close_videowriter(self, cam_idx: int):
        if self._saving_ext == 'mp4':
            if self._videowriters[cam_idx]:
                self._videowriters[cam_idx].stdin.flush()
                self._videowriters[cam_idx].stdin.close()
                self._videowriters[cam_idx].wait()
                if self._videowriters[cam_idx].returncode() == 0:
                    print(f'Closed video writer for cam {cam_idx}')

                self._videowriters[cam_idx] = False

    def _writer_thread(self, cam_idx: int) -> NoReturn:
        """
            This thread writes frames to the disk

            Parameters
            ----------
            cam_idx: the index of the camera this threads belongs to
        """

        queue = self._l_all_frames[cam_idx]

        h = self._sources_list[cam_idx].height
        w = self._sources_list[cam_idx].width
        folder = self.full_path / f"cam{cam_idx}_{self._sources_list[cam_idx].name}"

        if 'mp4' not in self._saving_ext:
            folder = self.full_path / f"cam{cam_idx}_{self._sources_list[cam_idx].name}"
            folder.mkdir(parents=True, exist_ok=True)

        def save_frame(frame, number):
            """
                Saves one frame and updates the saved frames counter
            """

            # If video mode
            if 'mp4' in self._saving_ext:
                self._videowriters[cam_idx].stdin.write(frame.tobytes())
                if self._estim_file_size is None:
                    self._estim_file_size = -1  # In case of video files, return -1 so the GUI knows what to do

            else:
                # If image mode
                filepath = folder / f"{str(number).zfill(9)}.{self._saving_ext}"

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
                    case 'debug':
                        print('Dummy save')

                # Do this just once after one file has been written
                if self._estim_file_size is None:
                    try:
                        self._estim_file_size = os.path.getsize(filepath)
                    except:
                        pass

            # The following is a RawArray, so the count is not atomic!
            # But it is fine as this is only for a rough estimation
            # (the actual number of written files is counted in a safe way when recording stops)
            self._cnt_saved[cam_idx] += 1

        ##
        timer = Event()

        started_saving = False
        while self._acquiring:
            if self._recording:
                # Recording is set - do actual work
                self._init_videowriter(cam_idx)     # This does nothing if not in video mode

                # Do this just once at the start of a new recording session
                if not started_saving:
                    self._l_finished_saving[cam_idx].clear()
                    started_saving = True

                # Main state of this thread: If the queue is not empty, save a new frame
                if queue:
                    frame_nb, frame = queue.popleft()
                    save_frame(frame, frame_nb)

                # If we're writing fast enough, this thread should wait a bit
                else:
                    timer.wait(0.01)
            else:
                # Recording is not set - either it hasn't started, or it has but hasn't finished yet
                if started_saving:
                    # Recording has been started, so remaining frames still need to be saved
                    if queue:
                        frame_nb, frame = queue.popleft()
                        save_frame(frame, frame_nb)
                    else:
                        self._close_videowriter(cam_idx)     # This does nothing if not in video mode
                        timer.wait(0.1)
                        self._l_finished_saving[cam_idx].set()
                else:
                    # Default state of this thread: if cameras are acquiring but we're not recording, just wait
                    timer.wait(0.1)
        # print(f'[DEBUG] Stopped writer thread {get_ident()}')

    def _display_updater_thread(self, cam_idx: int) -> NoReturn:
        """
            This thread updates the display buffers at a relatively slow pace (not super accurate timing but who cares),
            and otherwise mostly sleeps

            Parameters
            ----------
            cam_idx: the index of the camera this threads belongs to
        """

        queue = self._l_latest_frames[cam_idx]
        timer = Event()

        while self._acquiring:
            timer.wait(0.05)
            if queue:
                np.copyto(self._l_display_buffers[cam_idx], queue.popleft())
                self._cnt_displayed[cam_idx] += 1

        # print(f'[DEBUG] Stopped display thread {get_ident()}')

    def _grabber_thread(self, cam_idx: int) -> NoReturn:
        """
            This grabs frames from the camera and puts them in two buffers (one for displaying and one for saving)

            Parameters
            ----------
            cam_idx: the index of the camera this threads belongs to
        """
        cam = self._sources_list[cam_idx]
        queue_latest = self._l_latest_frames[cam_idx]
        queue_all = self._l_all_frames[cam_idx]

        cam.start_grabbing()

        while self._acquiring:
            with cam.ptr.RetrieveResult(500, py.TimeoutHandling_Return) as res:
                try:
                    if res.GrabSucceeded():
                        img_nb = res.ImageNumber
                        frame = res.GetArray()
                        if self._recording:
                            queue_all.append((img_nb, frame))
                        queue_latest.append(frame)
                        self._cnt_grabbed[cam_idx] += 1
                except py.RuntimeException:     # This might happen if the camera stops grabbing during this loop
                    pass

        cam.stop_grabbing()
        # print(f'[DEBUG] Stopped grabber thread {get_ident()}')

    def record(self) -> None:
        """
            Start recording session
        """
        if self.acquiring:
            if not self._recording:

                (self.full_path / 'recording').touch(exist_ok=True)

                session_metadata = {'start': datetime.now().timestamp(),
                                    'end': 0.0,
                                    'duration': 0.0,
                                    'hardware_triggered': self.triggered,
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

                self._recording = True

                if not self._silent:
                    if 'mp4' in self._saving_ext:
                        print(f'[INFO] Using {"hardware" if self._config_encoding_gpu else "software"} video encoding')
                    else:
                        print(f'[INFO] Using {self._saving_ext} image encoding')
                    print('[INFO] Recording started...')

    def pause(self) -> None:
        """
            Stops the current recording session
        """
        if self.acquiring:
            if self._recording:

                # Update the metadata with end time and number of saved frames
                self._metadata['sessions'][-1]['end'] = datetime.now().timestamp()
                duration = self._metadata['sessions'][-1]['end'] - self._metadata['sessions'][-1]['start']
                self._metadata['sessions'][-1]['duration'] = duration

                self._recording = False

                if not self._silent:
                    print('[INFO] Finishing saving...')

                # Wait for all writer threads to finish saving the current session
                [e.wait() for e in self._l_finished_saving]

                for i, cam in enumerate(self.cameras):
                    if 'mp4' in self._saving_ext:
                        vid = self.full_path / f"cam{i}_{self._sources_list[i].name}_session{len(self._metadata['sessions']) - 1}.mp4"
                        if vid.is_file():
                            # Using cv2 here is much faster than calling ffprobe...
                            cap = cv2.VideoCapture(vid.as_posix())
                            saved_frames_curr_sess = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            cap.release()
                        else:
                            saved_frames_curr_sess = 0
                    else:
                        # Read back how many frames were recorded in previous sessions of this acquisition
                        previsouly_saved = sum([self._metadata['sessions'][p]['cameras'][i].get('frames', 0) for p in
                                                range(len(self._metadata['sessions']))])

                        # Wait for all files to finish being written and write the number of frames for this session
                        saved_frames = self._safe_files_counter(self.full_path / f'cam{i}_{cam.name}')
                        saved_frames_curr_sess = saved_frames - previsouly_saved

                    self._metadata['sessions'][-1]['cameras'][i]['frames'] = saved_frames_curr_sess
                    self._metadata['sessions'][-1]['cameras'][i]['framerate_theoretical'] = cam.framerate
                    self._metadata['sessions'][-1]['cameras'][i]['framerate_actual'] = saved_frames_curr_sess / duration

                with open(self.full_path / 'metadata.json', 'w', encoding='utf-8') as f:
                    json.dump(self._metadata, f, ensure_ascii=True, indent=4)

                (self.full_path / 'recording').unlink(missing_ok=True)

                if not self._silent:
                    print('[INFO] Done saving')

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

        if not self._acquiring:

            # Just in case...
            if self._session_name == '':
                self.session_name = ''

            if self._triggered:
                self.trigger.start(self._framerate)
                time.sleep(0.1)

            self._acquiring = True

            # Start 3 threads per camera:
            #   - One that grabs frames continuously from the camera
            #   - One that writes frames continuously to disk
            #   - One that (less frequently) updates local buffers for displaying

            self._threads = []
            for i, cam in enumerate(self._sources_list):
                g = Thread(target=self._grabber_thread, args=(i, ), daemon=True)
                g.start()
                self._threads.append(g)
                d = Thread(target=self._display_updater_thread, args=(i,), daemon=True)
                d.start()
                self._threads.append(d)
                w = Thread(target=self._writer_thread, args=(i,), daemon=True)
                w.start()
                self._threads.append(w)

            if not self._silent:
                print(f"[INFO] Grabbing started with {self._nb_cams} camera{'s' if self._nb_cams > 1 else ''}...")

    def off(self) -> None:
        """
            Stop acquisition on all cameras
        """

        if self._acquiring:

            # If we were recording, gracefully stop it
            self.pause()

            self._acquiring = False

            # for cam in self._l_sources_list:
            #     cam.stop_grabbing()             # This should not be necessary here

            if self._triggered:
                self.trigger.stop()

            # Clean up the acquisition folder if nothing was written in it
            files_op.rm_if_empty(self._base_folder / self._session_name)

            # Reset everything for next acquisition
            self._session_name = ''
            self._metadata['sessions'] = []

            self._cnt_grabbed = RawArray('I', int(self._nb_cams))
            self._cnt_displayed = RawArray('I', int(self._nb_cams))
            self._cnt_saved = RawArray('I', int(self._nb_cams))

            if not self._silent:
                print(f'[INFO] Grabbing stopped')

    @property
    def session_name(self) -> str:
        if self._session_name == '':
            self.session_name = ''
        return self._session_name

    @session_name.setter
    def session_name(self, new_name='') -> None:

        # TODO = Shouldn't this be called Acquisition instead of session? It's *multiple sessions* per acquisition...

        # If we're currently acquiring, temporarily stop in order to rename the session...
        was_on = self._acquiring
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
        return self._acquiring

    @property
    def recording(self) -> bool:
        return self._recording

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

    def get_current_framebuffer(self, i: int = None) -> Union[np.array, list[np.array]]:
        """
            Returns the current display frame buffer(s) for one or all cameras.
            NB: These arrays are not atomically readable - they are just for visualisation.

            Returns
            -------
            RawArray, or list of RawArray - each RawArray is a buffer of length (width * height)
        """
        if i is None:
            return self._l_display_buffers
        else:
            return self._l_display_buffers[i]
