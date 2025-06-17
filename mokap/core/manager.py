import json
import logging
import os
import queue
import shutil
import subprocess
import time
from datetime import datetime
from multiprocessing.dummy import current_process
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event, Lock, current_thread
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from PIL import Image

from mokap.core.cameras import CameraFactory, AbstractCamera, CAMERAS_COLOURS
from mokap.core.triggers import AbstractTrigger, CameraTrigger, RaspberryTrigger, ArduinoTrigger, FTDITrigger
from mokap.core.writers import FrameWriter, ImageSequenceWriter, FFmpegWriter

from mokap.utils import fileio, is_locked
from mokap.utils.system import setup_ulimit, safe_replace

logger = logging.getLogger(__name__)


class MultiCam:
    """
    An orchestrator for managing multiple cameras for high-speed, synchronized recording
    It handles camera discovery, threading, recording state, and file output
    """

    def __init__(self,
                 config:         Optional[Dict] = None,
                 session_name:   Optional[str] = None):

        # --- Configuration ---
        self.config = config if config else fileio.read_config('config.yaml')
        self._base_folder = Path(self.config.get('base_path', './MokapRecordings'))
        self._base_folder.mkdir(parents=True, exist_ok=True)

        # --- State Management ---
        self._acquiring = Event()
        self._recording = Event()
        self._threads: List[Thread] = []

        self._session_name = ""
        self.session_name = session_name  # use setter to initialize

        setup_ulimit()   # Linux only, does nothing on other platforms

        # internal value for the framerate used to broadcast to all cameras
        self._framerate = self.config.get('framerate', 60)

        # --- Hardware and Cameras ---
        self.cameras: List[AbstractCamera] = []
        self.camera_colours: Dict[str, str] = {}
        self._camera_setting_overrides: Dict[str, Dict] = {}
        self.connect_cameras()

        # Setup hardware trigger (must be done after connecting cameras, in case one of them *is* the trigger)
        self._trigger_instance: Optional[AbstractTrigger] = None
        self._initialize_trigger()

        ## --- Threading Resources ---
        buffer_size = self.config.get('frame_buffer_size', 200)

        # shared state for the most recent frame protected by a lock
        self._latest_frames: List[Optional[Tuple[np.ndarray, Dict]]] = [None] * self.nb_cameras
        self._latest_frame_locks: List[Lock] = [Lock() for _ in self.cameras]
        self._writer_queues: List[Queue] = [Queue(maxsize=buffer_size) for _ in self.cameras]

        # Keep track of writer instances and their parameters to build final metadata
        self._writers: List[Optional[FrameWriter]] = [None] * self.nb_cameras

        self._session_frame_counts: List[int] = [0] * self.nb_cameras
        self._session_encoding_params: List[Dict] = [{} for _ in self.cameras]
        self._session_actual_fps: List[float] = [0.0] * self.nb_cameras

        self._finished_saving_events: List[Event] = [Event() for _ in self.cameras]
        for event in self._finished_saving_events:
            event.set()

        # --- Metadata ---
        self._metadata = {'sessions': []}

    def _initialize_trigger(self):

        if not self.config.get('hardware_trigger', False):
            self._trigger_instance = None
            return

        trigger_conf = self.config.get('trigger', {})
        trigger_type = trigger_conf.get('type', '')

        if not trigger_type:
            logger.error(
                "Config contains 'hardware_trigger: true', but no trigger 'type' found. Running without hardware trigger.")
            self._trigger_instance = None
            return

        if trigger_type == 'camera':
            primary_cam_name = trigger_conf.get('name')
            
            if not primary_cam_name:
                logger.error("Camera trigger requires 'name' in config. Disabling trigger.")
                self._trigger_instance = None
                return

            primary_camera = next((cam for cam in self.cameras if cam.name == primary_cam_name), None)

            if not primary_camera:
                logger.error(
                    f"Camera '{primary_cam_name}' for trigger not found among connected cameras. Disabling trigger.")
                self._trigger_instance = None
                return

            # if primary_camera.hardware_triggered:
            #     logger.warning(
            #         f"Primary camera '{primary_cam_name}' was configured with 'hardware_trigger: true'.\n"
            #         f"This will be overridden, as it cannot be both a primary and a secondary.")

            self._trigger_instance = CameraTrigger(primary_camera=primary_camera, config=trigger_conf)

        elif trigger_type == 'raspberry':
            self._trigger_instance = RaspberryTrigger(config=trigger_conf)

        elif trigger_type == 'arduino':
            self._trigger_instance = ArduinoTrigger(config=trigger_conf)

        elif trigger_type == 'ftdi':
            self._trigger_instance = FTDITrigger(config=trigger_conf)
            logger.info("FTDI Trigger is not recommended for high-precision applications.")

        else:
            logger.error(f"Trigger 'type' '{trigger_type}' is not valid. Running without hardware trigger.")
            self._trigger_instance = None

        if self._trigger_instance and not self._trigger_instance.connected:
            logger.error("Failed to connect to trigger. Running without hardware trigger.")
            self._trigger_instance = None

    def connect_cameras(self):
        """ Discovers and connects to all available cameras using the camera factory """

        logger.debug("Discovering cameras...")

        # Get a list of all physically present devices from the factory
        # TODO: re-implement virtual cameras support
        all_discovered_devices = CameraFactory.discover_cameras()

        if not all_discovered_devices:
            logger.warning("No cameras found.")
            return

        # Create a lookup from serial number to the device info object
        device_lookup = {dev['serial']: dev for dev in all_discovered_devices}

        claimed_serials = set()

        # Get the camera configurations from the config file
        configured_sources = self.config.get('sources', {})
        if not configured_sources:
            logger.warning("No cameras defined in the 'sources' section of the config file.")
            return

        # Define the list of global keys that can be applied to cameras
        valid_global_settings = [
            'exposure', 'gain', 'gamma', 'pixel_format', 'black_level',
            'binning', 'framerate', 'hardware_trigger', 'roi'
        ]

        # Create a base dictionary of global settings found in the config
        base_cam_config = {
            key: self.config[key] for key in valid_global_settings if key in self.config
        }

        for friendly_name, cam_config in configured_sources.items():
            serial = str(cam_config.get('serial', ''))  # ensure serial is a string
            vendor = cam_config.get('vendor')           # for serial-less lookup

            device_info = None

            if serial:
                # Case 1: Serial is provided, find it directly
                device_info = device_lookup.get(serial)
                if not device_info:
                    logger.warning(f"Skipping '{friendly_name}': camera with serial {serial} not found.")
                    continue
            else:
                # Case 2: No serial provided, find an unclaimed camera of the specified vendor
                if not vendor:
                    logger.warning(f"Skipping '{friendly_name}': must provide a 'vendor' if 'serial' is omitted.")
                    continue

                # Find the first available camera of the right vendor that hasn't been claimed
                for dev in all_discovered_devices:
                    if dev['vendor'].lower() == vendor.lower() and dev['serial'] not in claimed_serials:
                        device_info = dev
                        serial = dev['serial']  # Get the serial for future reference
                        logger.info(f"Assigning unclaimed {vendor} camera (S/N: {serial}) to '{friendly_name}'.")
                        break  # stop after finding one

                if not device_info:
                    logger.warning(f"Skipping '{friendly_name}': No unclaimed '{vendor}' cameras available.")
                    continue

            # Once a device is chosen (either by serial or by vendor), get the camera instance
            cam = CameraFactory.get_camera(device_info)

            if cam:
                try:
                    claimed_serials.add(serial)

                    cam.name = friendly_name

                    # start with a copy of the global settings
                    final_settings = base_cam_config.copy()

                    # identify all camera-specific settings from the config
                    camera_specific_settings = {
                        k: v for k, v in cam_config.items()
                        if k in valid_global_settings
                    }
                    nested_settings = cam_config.get('settings', {})
                    if nested_settings:
                        logger.warning(
                            '[DEPRECATION] Nested "settings" block inside cameras configs will be deprecated.')
                        # Merge nested settings into the specific settings
                        camera_specific_settings.update(nested_settings)

                    # Store this dictionary of overrides for later use in metadata
                    self._camera_setting_overrides[friendly_name] = camera_specific_settings

                    # Apply the final, merged configuration to the camera
                    final_settings.update(camera_specific_settings)
                    cam.connect(config=final_settings)

                    # Add the successfully connected camera to our list
                    self.cameras.append(cam)

                    # Use the user-defined color from the config, or fall back to a default
                    color = cam_config.get('color', CAMERAS_COLOURS[len(self.cameras) % len(CAMERAS_COLOURS)])
                    self.camera_colours[cam.unique_id] = f"#{color}"

                    logger.info(f"Successfully connected to {cam}")

                except Exception as e:
                    logger.error(f"Failed to connect or configure camera '{friendly_name}' (S/N: {serial}): {e}")

    def disconnect_cameras(self):
        """ Cleanly disconnect all cameras """

        for cam in self.cameras:
            if cam.is_connected:
                cam.disconnect()

        logger.info("All cameras disconnected.")

    def start_acquisition(self):
        """ Starts all background threads for grabbing and displaying frames """

        if self._acquiring.is_set():
            logger.info("Acquisition is already running.")
            return

        if self.nb_cameras == 0:
            logger.error("No cameras connected. Cannot start acquisition.")
            return

        self._acquiring.set()
        self._threads = []

        if self.hardware_triggered:
            self._trigger_instance.start(self._framerate)

        for i, cam in enumerate(self.cameras):
            # Start one grabber, one writer, and one display thread per camera
            g = Thread(target=self._grabber_thread, args=(i,))
            w = Thread(target=self._writer_thread, args=(i,))
            self._threads.extend([g, w])
            g.start()
            w.start()

        logger.info(f"Acquisition started with {self.nb_cameras} cameras.")

    def stop_acquisition(self):
        """ Stops all background threads """

        if not self._acquiring.is_set():
            return

        if self._recording.is_set():
            self.pause_recording()  # Gracefully finish recording

        self._acquiring.clear()

        logger.debug("Stopping acquisition threads...")

        for thread in self._threads:
            thread.join(timeout=2.0)

        # stop trigger if enabled
        if self.hardware_triggered:
            self._trigger_instance.stop()

        # Clean up session folder if it's empty
        fileio.rm_if_empty(self.full_path)

        logger.info("Acquisition stopped.")

    def start_recording(self):
        """ Begins a recording session, signaling the writer threads to save frames """

        if not self._acquiring.is_set():
            logger.error("Cannot record, acquisition is not running.")
            return

        if self._recording.is_set():
            logger.warning("Already recording.")
            return

        # Prepare metadata snapshot for this session
        keys_order = ['exposure', 'gain', 'gamma', 'black_level', 'pixel_format',
                      'binning', 'binning_mode', 'roi', 'save_format']
        session_config = {k: self.config[k] for k in keys_order if k in self.config}

        if session_config.get('save_format') !=  'mp4' and self.config.get('save_quality'):
            session_config['save_quality'] = self.config.get('save_quality')

        # Prepare metadata for this new session
        session_metadata = {
            'session_nb': len(self._metadata['sessions']),
            'start_time': datetime.now().timestamp(),
            'end_time': None,
            'duration': 0.0,
            'hardware_triggered': self.hardware_triggered
        }
        if self.hardware_triggered:
            session_metadata['trigger_frequency'] = self.framerate
        session_metadata.update(session_config)
        session_metadata['cameras'] = {}    # this will be populated ad the end of a recording

        self._metadata['sessions'].append(session_metadata)

        # Reset the frame counters for the new session
        self._session_frame_counts = [0] * self.nb_cameras
        self._session_actual_fps = [0.0] * self.nb_cameras

        # signal that writers can start
        self._recording.set()

        logger.info(f"Recording started. Saving to: {self.full_path}")

    def pause_recording(self):
        """ Pauses the current recording session, finalizing files """

        if not self._recording.is_set():
            return

        end_ts = datetime.now().timestamp()

        self._recording.clear()

        logger.info("Finishing writing for current session...")

        # we calculate and store session duration BEFORE asking the threads to stop
        curr_session = self._metadata['sessions'][-1]

        start_ts = curr_session['start_time']
        duration = end_ts - start_ts if end_ts > start_ts else 0.0

        curr_session['end_time'] = end_ts
        curr_session['duration'] = duration

        # Wait for all writer threads to confirm they have finished
        for event in self._finished_saving_events:
            event.wait(timeout=5.0)

        cameras_data = {}
        for i, cam in enumerate(self.cameras):
            frames = self._session_frame_counts[i]
            actual_fps = (frames / duration) if duration > 0 else 0.0

            diff = abs(cam.framerate - actual_fps)

            # Correct the video files framerates if needed
            save_format = (self.config.get('sources', {})
                           .get(cam.name, {})
                           .get('save_format', self.config.get('save_format', 'mp4')))

            if save_format == 'mp4' and frames > 0 and diff > 0.05:
                video_path = self.full_path / f"{self.session_name}_{cam.name}_session{curr_session['session_nb']}.mp4"
                self._correct_video_framerate(video_path, actual_fps)

            # Build the final data block for this camera
            cam_data_block = {
                'serial': cam.unique_id,
                'model': CameraFactory.get_camera_info(cam.unique_id)['model'],
                'frames_recorded': frames,
                'theoretical_framerate': cam.framerate,
                'actual_framerate': round(actual_fps, 3),
                'encoding': self._session_encoding_params[i]
            }

            overrides = self._camera_setting_overrides.get(cam.name, {})
            if overrides:
                cam_data_block['settings_overrides'] = overrides

            cameras_data[cam.name] = cam_data_block

        curr_session['cameras'] = cameras_data
        self.save_metadata()

        logger.info("Recording paused. Files saved.")

    def _grabber_thread(self, cam_idx: int):
        """ Dedicated thread to continuously grab frames from a single camera """

        cam = self.cameras[cam_idx]
        writer_queue = self._writer_queues[cam_idx]
        lock = self._latest_frame_locks[cam_idx]

        cam.start_grabbing()
        while self._acquiring.is_set():
            try:
                frame, frame_data = cam.grab_frame(timeout_ms=2000)
                if frame is not None:
                    # Update the shared buffer for any other thread to read
                    with lock:
                        self._latest_frames[cam_idx] = (frame, frame_data)

                    # Feed the high-priority writer queue.
                    if self._recording.is_set():
                        try:
                            writer_queue.put_nowait((frame, frame_data))
                        except queue.Full:
                            logger.warning(f"Cam {cam.name}: Writer queue is full. Recording frame dropped.")

            except (IOError, RuntimeError) as e:
                if self._acquiring.is_set():
                    logger.error(f"Grabber thread for {cam.name} failed: {e}")
                    time.sleep(1)

        cam.stop_grabbing()

    def _writer_thread(self, cam_idx: int):
        """
        Dedicated thread to write frames from a queue to a file
        This thread manages the lifecycle of a FrameWriter object
        """
        cam = self.cameras[cam_idx]
        w_queue = self._writer_queues[cam_idx]
        self._writers[cam_idx] = None

        while self._acquiring.is_set():
            # State A - Waiting to start a new recording session
            if self._recording.is_set() and self._writers[cam_idx] is None:
                self._finished_saving_events[cam_idx].clear()

                try:
                    first_frame, first_frame_data = w_queue.get(timeout=2.0)

                    curr_session_idx = len(self._metadata['sessions']) -1

                    self._writers[cam_idx] = self._create_writer(cam, curr_session_idx)
                    writer = self._writers[cam_idx]

                    logger.debug(f"Writer for {cam.name} created: {type(writer).__name__}")

                    writer.write(first_frame, first_frame_data)

                except Empty:
                    # If we timeout waiting for the first frame, it means the
                    # grabber isn't feeding the writer queue - so we should log it and try again
                    logger.warning(f"Cam {cam.name}: Timed out waiting for first frame to record. Trying again.")
                    continue  # skip to the next loop iteration to try again

                except Exception as e:
                    logger.error(f"Failed to create writer for {cam.name}: {e}")
                    self._recording.clear()  # this is debatable, but prevents getting stuck in an error loop
                    # TODO: Maybe keep a separate status for each camera?

                    self._finished_saving_events[cam_idx].set()
                    continue

            # State B - Actively writing frames
            writer = self._writers[cam_idx]
            if self._recording.is_set() and writer:
                try:
                    frame, frame_data = w_queue.get(timeout=1.0)
                    writer.write(frame, frame_data)
                except Empty:
                    # this is less critical, just means a momentary lull in frames
                    continue

            # State C - Recording has been paused/stopped
            if not self._recording.is_set() and writer:
                # The recording flag was turned off, so we need to finalize
                # First drain any remaining frames from the queue
                while not w_queue.empty():
                    try:
                        frame, frame_data = w_queue.get_nowait()
                        writer.write(frame, frame_data)
                    except Empty:
                        break

                # Now close the writer and report back its info
                self._session_frame_counts[cam_idx] = writer.frame_count
                self._session_encoding_params[cam_idx] = writer.encoding_params

                writer.close()

                self._writers[cam_idx] = None  # set writer to None to signal it's closed
                self._finished_saving_events[cam_idx].set()

            # State D - Idle, not recording
            if not self._recording.is_set() and self._writers[cam_idx] is None:
                # To prevent this thread from busy-waiting, we can sleep
                time.sleep(0.1)

    def _create_writer(self, cam: AbstractCamera, session_idx: int) -> FrameWriter:
        """ Factory method to instantiate the correct writer based on config """

        # Get the camera-specific config, if it exists
        cam_config = self.config.get('sources', {}).get(cam.name, {})

        # Check for a per-camera override, otherwise use the global setting
        save_format = cam_config.get('save_format', self.config.get('save_format', 'mp4'))
        save_quality = cam_config.get('save_quality', self.config.get('save_quality', 90))

        base_path = self.full_path / f"{self.session_name}_{cam.name}_session{session_idx}"

        # Common parameters for all writers
        writer_params = {
            'pixel_format': cam.pixel_format,
            'width': cam.roi[2],
            'height': cam.roi[3],
            'framerate': cam.framerate,
            'cam_name': cam.name,
        }

        if save_format == 'mp4':
            # For video
            ffmpeg_config = self.config.get('ffmpeg', {})
            return FFmpegWriter(
                filepath=base_path.with_suffix('.mp4'),
                ffmpeg_path=ffmpeg_config.get('path', 'ffmpeg'),
                params=ffmpeg_config.get('params', {}),
                use_gpu=ffmpeg_config.get('gpu', False),
                **writer_params
            )
        else:
            # For image sequences
            return ImageSequenceWriter(
                folder=base_path,
                ext=save_format,
                quality=save_quality,
                **writer_params
            )

    def _correct_video_framerate(self, filepath: Path, actual_fps: float):
        """ Corrects the framerate metadata in a video file without re-encoding """

        if not filepath.exists() or actual_fps <= 0:
            return

        ffmpeg_path = shutil.which(self.config.get('ffmpeg', {}).get('path', 'ffmpeg'))
        if not ffmpeg_path or not os.access(ffmpeg_path, os.X_OK):
            return

        logger.info(f"Correcting framerate for {filepath.name} to {actual_fps:.3f} fps.")
        temp_filepath = filepath.with_suffix('.temp.mp4')

        command = [
            ffmpeg_path,
            '-i', str(filepath.resolve()),
            '-c', 'copy',
            '-r', f'{actual_fps:.3f}',
            str(temp_filepath)
        ]

        try:
            p = subprocess.run(command, check=True, capture_output=True, text=True)
            if p.returncode == 0:
                if not safe_replace(str(temp_filepath), str(filepath)):
                    logger.error(f"Could not rename {temp_filepath}.")

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"FFmpeg failed to correct framerate for {filepath.name}: {e}")

            if isinstance(e, subprocess.CalledProcessError):
                logger.error(f"FFmpeg stderr:\n{e.stderr}")

            if temp_filepath.exists():
                os.remove(temp_filepath)

    def take_snapshot(self) -> bool:
        """ Takes a snapshot from all cameras (tries to sync) """

        if not self.acquiring:
            logger.warning("Cannot take snapshot, acquisition is not running.")
            return False

        self.full_path.mkdir(parents=True, exist_ok=True)

        # Atomically copy the latest frames from the shared state
        current_frames = []
        for i in range(self.nb_cameras):
            with self._latest_frame_locks[i]:
                frame_data = self._latest_frames[i]

            if frame_data is None:
                logger.error(f"Snapshot failed: No frame has been received from camera {self.cameras[i].name} yet.")
                return False

            current_frames.append(frame_data)

        # Reconcile frames
        frames, frames_datas = zip(*current_frames)

        # We only have one frame per camera, so we check if their IDs match
        # This is a much simpler check than the previous buffer search
        first_frame_id = frames_datas[0].get('frame_id')
        is_synchronized = False
        if first_frame_id is not None:
            if all(dat.get('frame_id') == first_frame_id for dat in frames_datas[1:]):
                is_synchronized = True

        # Save
        if is_synchronized:
            logger.info(f"Saving synchronized frame set with ID: {first_frame_id}")
        else:
            logger.warning("Frame set is not synchronized. Saving latest available frames.")

        success = True

        now = datetime.now().strftime('%y%m%d-%H%M%S')
        for i, cam in enumerate(self.cameras):
            try:
                Image.fromarray(frames[i]).save(self.full_path / f"snapshot_{now}_{cam.name}.png")
                # TODO: use cv2 and the config's quality settings

            except Exception as e:
                logger.error(f"Could not save snapshot for camera {cam.name}: {e}")
                success = False

        if success:
            logger.info(f"Snapshot saved in {self.full_path}")

        return success

    def set_all_cameras(self, parameter: str, value: Any):
        """ Broadcasts a setting to all cameras """
        # TODO: we probably want to have setters that will call this, like the framerate one for other properties

        logger.debug(f"Broadcasting '{parameter} = {value}' to all cameras.")

        # The trigger framerate must be set via its own property
        if parameter == 'framerate' and self.hardware_triggered:
            self.framerate = value

        for cam in self.cameras:
            try:
                # use setattr to dynamically set the property (e.g., 'exposure', 'gain')
                setattr(cam, parameter, value)
            except Exception as e:
                logger.error(f"Could not set '{parameter}' on camera {cam.name}: {e}")

    # --- Session and Metadata Management ---

    @property
    def session_name(self) -> str:
        return self._session_name

    @session_name.setter
    def session_name(self, name: Optional[str]):
        if self._acquiring.is_set():
            raise RuntimeError("Cannot change session name while acquisition is running.")

        if not name:
            name = datetime.now().strftime('%y%m%d-%H%M%S')

        new_folder = fileio.exists_check(self._base_folder / name)
        new_folder.mkdir(parents=True, exist_ok=False)
        self._session_name = new_folder.name

    @property
    def full_path(self) -> Path:
        return self._base_folder / self.session_name

    def save_metadata(self):
        """ Saves the current session metadata to a JSON file """
        meta_path = self.full_path / 'metadata.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=4)

    def __enter__(self):
        """ Context manager entry: starts acquisition """
        self.start_acquisition()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Context manager exit: ensures threads and cameras are cleaned up """
        self.stop_acquisition()
        self.disconnect_cameras()

    @property
    def framerate(self) -> Optional[float]:
        return self._framerate

    @framerate.setter
    def framerate(self, value: float):
        """ Sets the framerate for the whole system (incl external trigger) """

        # TODO: Similar setters for other properties

        new_framerate = float(value)

        # for hardware trigger, update internal value and the trigger itself
        self._framerate = new_framerate
        if self._acquiring.is_set() and self._trigger_instance:
            # only update if already running (if not, trigger will pick up new value when it starts)
            self._trigger_instance.start(self._framerate)

        else:
            # for software trigger, broadcast to all cameras and then verify
            self.set_all_cameras('framerate', new_framerate)

            all_cams_synced = True
            for cam in self.cameras:
                if cam.framerate != new_framerate:
                    logger.warning(f"Camera {cam.name} could not be set to {new_framerate} fps. Actual: {cam.framerate} fps.")
                    all_cams_synced = False
                    break

            if all_cams_synced:
                self._framerate = new_framerate
                logger.debug(f"All cameras successfully set to {new_framerate} fps.")
            else:
                self._framerate = None
                logger.warning("Not all cameras could be set to the requested framerate. System framerate is now undefined.")

    @property
    def nb_cameras(self) -> int:
        return len(self.cameras)

    @property
    def acquiring(self) -> bool:
        """ Provides backward compatibility for the old '.acquiring' property """
        return self._acquiring.is_set()

    @property
    def recording(self) -> bool:
        """ Provides backward compatibility for the old '.recording' property """
        return self._recording.is_set()

    @property
    def hardware_triggered(self) -> bool:
        return self._trigger_instance is not None and self._trigger_instance.connected

    @property
    def saved(self) -> int:
        """ Provides a real-time sum of frames saved in the current session """
        return sum(self._session_frame_counts)

    @property
    def colours(self) -> Dict[str, str]:
        """ Alias for camera_colours """
        return self.camera_colours

    def on(self):
        """ Alias for start_acquisition() """
        self.start_acquisition()

    def off(self):
        """ Alias for stop_acquisition() """
        self.stop_acquisition()

    def record(self):
        """ Alias for start_recording() """
        self.start_recording()

    def pause(self):
        """ Alias for pause_recording() """
        self.pause_recording()