import json
import queue
import time
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event, Lock
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from PIL import Image

from mokap.core.cameras.interface import AbstractCamera, CAMERAS_COLOURS
from mokap.core.cameras.camerafactory import CameraFactory
from mokap.core.triggers.interface import AbstractTrigger
from mokap.core.triggers.raspberry import RaspberryTrigger
from mokap.core.triggers.arduino import ArduinoTrigger
from mokap.core.triggers.ftdi import FTDITrigger
from mokap.core.writers import FrameWriter, ImageSequenceWriter, FFmpegWriter
from mokap.utils import fileio
from mokap.utils.system import setup_ulimit

METADATA_KEYS = {
# which keys from the main config are relevant to a recording session
    'framerate', 'gain', 'blacks', 'exposure', 'roi', 'pixel_format',
    'binning', 'save_format', 'save_quality', 'hardware_trigger'
}

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
        self._acquiring = False
        self._recording = False
        self._threads: List[Thread] = []

        self._session_name = ""
        self.session_name = session_name  # use setter to initialize

        self._silent = self.config.get('silent', False)

        setup_ulimit(silent=self._silent)   # Linux only, does nothing on other platforms

        # internal value for the framerate used to broadcast to all cameras
        self._trigger_framerate = self.config.get('framerate', 60)

        # --- Hardware and Cameras ---
        self.cameras: List[AbstractCamera] = []
        self.camera_colours: Dict[str, str] = {}
        self.connect_cameras()

        # setup hardware trigger
        self._trigger_instance: Optional[AbstractTrigger] = None
        if self.config.get('hardware_trigger', False):

            trigger_conf = self.config.get('trigger', {})
            trigger_kind = trigger_conf.get('kind', '')

            if trigger_kind == 'raspberry':
                print(trigger_kind)
                self._trigger_instance = RaspberryTrigger(config=trigger_conf, silent=self._silent)
            elif trigger_kind == 'arduino':
                self._trigger_instance = ArduinoTrigger(config=trigger_conf, silent=self._silent)
            elif trigger_kind == 'ftdi':
                self._trigger_instance = FTDITrigger(config=trigger_conf, silent=self._silent)
                print("[INFO] FTDI Trigger is not recommended for high-precision applications.")
            else:
                print("[ERROR] No valid trigger 'kind' found in config. Running without hardware trigger.")
                self._trigger_instance = None

            if self._trigger_instance and not self._trigger_instance.connected:
                print("[ERROR] Failed to connect to trigger. Running without hardware trigger.")
                self._trigger_instance = None

        ## --- Threading Resources ---
        buffer_size = self.config.get('frame_buffer_size', 200)
        self._session_frame_counts: List[int] = [0] * self.nb_cameras
        self._display_queues: List[Queue] = [Queue(maxsize=2) for _ in self.cameras]

        # The 'tee' queues that will be used to 'snoop' on frames for display
        self._display_tee_queues: List[Queue] = [Queue(maxsize=2) for _ in self.cameras]

        # shared state for the most recent frame protected by a lock
        self._latest_frames: List[Optional[Tuple[np.ndarray, Dict]]] = [None] * self.nb_cameras
        self._latest_frame_locks: List[Lock] = [Lock() for _ in self.cameras]

        self._writer_queues: List[Queue] = [Queue(maxsize=buffer_size) for _ in self.cameras]

        # Keep track of writer instances and their parameters to build final metadata
        self._writers: List[Optional[FrameWriter]] = [None] * self.nb_cameras
        self._session_encoding_params: List[Dict] = [{} for _ in self.cameras]

        self._finished_saving_events: List[Event] = [Event() for _ in self.cameras]
        for event in self._finished_saving_events:
            event.set()

        # --- Metadata ---
        self._metadata = {'sessions': []}

    @property
    def nb_cameras(self) -> int:
        return len(self.cameras)

    def connect_cameras(self):
        """ Discovers and connects to all available cameras using the camera factory """

        if not self._silent:
            print("[INFO] Discovering cameras...")

        # Get a list of all physically present devices from the factory
        # TODO: re-implement virtual cameras support
        all_discovered_devices = CameraFactory.discover_cameras()
        if not all_discovered_devices:
            print("[WARN] No cameras found.")
            return

        # Create a lookup from serial number to the device info object
        device_lookup = {dev['serial']: dev for dev in all_discovered_devices}

        claimed_serials = set()

        # Get the camera configurations from the config file
        configured_sources = self.config.get('sources', {})
        if not configured_sources:
            print("[WARN] No cameras defined in the 'sources' section of the config file.")
            return

        # Define the list of global keys that can be applied to cameras
        valid_global_settings = [
            'exposure', 'gain', 'gamma', 'pixel_format', 'blacks',
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
                    print(f"[WARN] Skipping '{friendly_name}': camera with serial {serial} not found.")
                    continue
            else:
                # Case 2: No serial provided, find an unclaimed camera of the specified vendor
                if not vendor:
                    print(f"[WARN] Skipping '{friendly_name}': must provide a 'vendor' if 'serial' is omitted.")
                    continue

                # Find the first available camera of the right vendor that hasn't been claimed
                for dev in all_discovered_devices:
                    if dev['vendor'].lower() == vendor.lower() and dev['serial'] not in claimed_serials:
                        device_info = dev
                        serial = dev['serial']  # Get the serial for future reference
                        print(f"[INFO] Assigning unclaimed {vendor} camera (S/N: {serial}) to '{friendly_name}'.")
                        break  # stop after finding one

                if not device_info:
                    print(f"[WARN] Skipping '{friendly_name}': No unclaimed '{vendor}' cameras available.")
                    continue

            # Once a device is chosen (either by serial or by vendor), get the camera instance
            cam = CameraFactory.get_camera(device_info)

            if cam:
                try:
                    claimed_serials.add(serial)

                    cam.name = friendly_name

                    # Start with a copy of the global settings
                    final_settings = base_cam_config.copy()

                    # Get camera-specific overrides and merge them in
                    camera_specific_settings = {
                        k: v for k, v in cam_config.items()
                        if k in valid_global_settings
                    }
                    final_settings.update(camera_specific_settings)

                    # We used to have a 'settings' block in the cameras, let's keep this for now but we'll trash it
                    nested_settings = cam_config.get('settings', {})
                    if nested_settings:
                        print('[DEPRECATION WARNING] Nested "settings" block inside cameras configs will be deprecated.')
                        final_settings.update(nested_settings)

                    # Connect and apply the settings
                    cam.connect(config=final_settings)

                    # Add the successfully connected camera to our list
                    self.cameras.append(cam)

                    # Use the user-defined color from the config, or fall back to a default
                    color = cam_config.get('color', CAMERAS_COLOURS[len(self.cameras) % len(CAMERAS_COLOURS)])
                    self.camera_colours[cam.unique_id] = f"#{color}"

                    if not self._silent:
                        print(f"[INFO] Successfully connected to {cam}")

                except Exception as e:
                    print(f"[ERROR] Failed to connect or configure camera '{friendly_name}' (S/N: {serial}): {e}")

    def disconnect_cameras(self):
        """ Cleanly disconnect all cameras """

        for cam in self.cameras:
            if cam.is_connected:
                cam.disconnect()

        if not self._silent:
            print("[INFO] All cameras disconnected.")

    def start_acquisition(self):
        """ Starts all background threads for grabbing and displaying frames """
        if self._acquiring:
            print("[WARN] Acquisition is already running.")
            return

        if self.nb_cameras == 0:
            print("[ERROR] No cameras connected. Cannot start acquisition.")
            return

        self._acquiring = True
        self._threads = []

        if self._trigger_instance and self._trigger_instance.connected:
            self._trigger_instance.start(self._trigger_framerate)

        for i, cam in enumerate(self.cameras):
            # Start one grabber, one writer, and one display thread per camera
            g = Thread(target=self._grabber_thread, args=(i,))
            w = Thread(target=self._writer_thread, args=(i,))
            d = Thread(target=self._display_updater_thread, args=(i,))
            self._threads.extend([g, w, d])
            g.start()
            w.start()
            d.start()

        if not self._silent:
            print(f"[INFO] Acquisition started with {self.nb_cameras} cameras.")

    def stop_acquisition(self):
        """ Stops all background threads """

        if not self._acquiring:
            return

        if self._recording:
            self.pause_recording()  # Gracefully finish recording

        self._acquiring = False
        if not self._silent:
            print("[INFO] Stopping acquisition threads...")

        for thread in self._threads:
            thread.join(timeout=2.0)

        # Stop trigger if enabled
        if self._trigger_instance and self._trigger_instance.connected:
            self._trigger_instance.stop()

        # Clean up session folder if it's empty
        fileio.rm_if_empty(self.full_path)

        if not self._silent:
            print("[INFO] Acquisition stopped.")

    def start_recording(self):
        """ Begins a recording session, signaling the writer threads to save frames """
        if not self._acquiring:
            print("[ERROR] Cannot record, acquisition is not running.")
            return

        if self._recording:
            print("[WARN] Already recording.")
            return

        # Prepare a lean configuration snapshot for this session
        session_config = {
            key: self.config[key] for key in METADATA_KEYS if key in self.config
        }

        # Prepare metadata for this new session
        session_metadata = {
            'start_timestamp': datetime.now().timestamp(),
            'end_timestamp': None,
            'duration_seconds': 0.0,
            'requested_framerate': self._trigger_framerate,
            'total_frames_recorded': 0,
            'session_config': session_config,
            'cameras': {}  # This will be populated in pause_recording
        }

        self._metadata['sessions'].append(session_metadata)

        # Reset the frame counters for the new session
        self._session_frame_counts = [0] * self.nb_cameras

        # signal that writers can start
        self._recording = True

        if not self._silent:
            print(f"[INFO] Recording started. Saving to: {self.full_path}")

    def pause_recording(self):
        """ Pauses the current recording session, finalizing files """

        if not self._recording:
            return

        self._recording = False
        if not self._silent:
            print("[INFO] Finishing writing for current session...")

        # Wait for all writer threads to confirm they have finished
        for event in self._finished_saving_events:
            event.wait(timeout=5.0)

        # Update metadata with final counts, timestamps, and calculated values
        session_idx = len(self._metadata['sessions']) - 1
        current_session = self._metadata['sessions'][session_idx]

        # timestamps and duration
        start_ts = current_session['start_timestamp']
        end_ts = datetime.now().timestamp()
        duration = end_ts - start_ts if end_ts > start_ts else 0.0

        current_session['end_timestamp'] = end_ts
        current_session['duration_seconds'] = round(duration, 4)
        current_session['total_frames_recorded'] = sum(self._session_frame_counts)

        cameras_data = {}
        global_config = current_session.get('session_config', {})

        for i, cam in enumerate(self.cameras):
            frames = self._session_frame_counts[i]
            actual_fps = (frames / duration) if duration > 0 else 0.0

            # Find the actual settings that differ from the global config
            overrides = {}
            for param in METADATA_KEYS:
                if hasattr(cam, param):
                    actual_value = getattr(cam, param)
                    global_value = global_config.get(param)

                    # Store the parameter if it differs from the global setting
                    if actual_value is not None and actual_value != 'N/A' and actual_value != global_value:
                        overrides[param] = actual_value

            # Build the final data block for this camera
            cam_data_block = {
                'serial': cam.unique_id,
                'model': CameraFactory.get_camera_info(cam.unique_id)['model'],
                'frames_recorded': frames,
                'actual_framerate': round(actual_fps, 2),
                'encoding': self._session_encoding_params[i]
            }

            # Only add the overrides key if there were any actual differences
            if overrides:
                cam_data_block['settings_overrides'] = overrides

            cameras_data[cam.name] = cam_data_block
        current_session['cameras'] = cameras_data

        self.save_metadata()

        if not self._silent:
            print("[INFO] Recording paused. Files saved.")

    def _grabber_thread(self, cam_idx: int):
        """ Dedicated thread to continuously grab frames from a single camera """

        cam = self.cameras[cam_idx]
        writer_queue = self._writer_queues[cam_idx]
        display_tee_queue = self._display_tee_queues[cam_idx]

        cam.start_grabbing()
        while self._acquiring:
            try:
                frame, metadata = cam.grab_frame(timeout_ms=2000)
                if frame is not None:

                    # also put the same frame in the display "tee" queue, non-blocking
                    if not display_tee_queue.full():
                        display_tee_queue.put_nowait((frame, metadata))

                    if self._recording:
                        try:
                            writer_queue.put_nowait((frame, metadata))
                        except queue.Full:
                            # This is the expected behavior when the writer can't keep up
                            # it's better to log it and continue grabbing than to block
                            if not self._silent:
                                print(f"[WARN] Cam {cam.name}: Writer queue is full. Recording frame dropped.")

            except (IOError, RuntimeError) as e:
                if self._acquiring:  # avoid spamming errors on shutdown
                    print(f"[ERROR] Grabber thread for {cam.name} failed: {e}")
                    time.sleep(1)  # prevent flooding error loops

        cam.stop_grabbing()

    def _display_updater_thread(self, cam_idx: int):
        """
        This thread's job is to manage the flow of preview frames
        1. It gets a frame from the grabber's tee queue
        2. It updates a shared variable with this "latest" frame for snapshots
        3. It tries to push the frame to the GUI queue for live display
        """
        tee_queue = self._display_tee_queues[cam_idx]
        display_queue = self._display_queues[cam_idx]
        lock = self._latest_frame_locks[cam_idx]

        while self._acquiring:
            try:
                # Block and wait for a new frame from the grabber
                frame, metadata = tee_queue.get(timeout=1.0)

                # Update the shared state for snapshots
                with lock:
                    self._latest_frames[cam_idx] = (frame, metadata)

                #Try to push to the actual GUI queue
                # If the GUI is lagging, we just drop the frame and continue
                if not display_queue.full():
                    display_queue.put_nowait((frame, metadata))

            except queue.Empty:
                # No frame came from the grabber in 1s, just continue
                continue

    def _writer_thread(self, cam_idx: int):
        """
        Dedicated thread to write frames from a queue to a file
        This thread manages the lifecycle of a FrameWriter object
        """
        cam = self.cameras[cam_idx]
        queue = self._writer_queues[cam_idx]
        self._writers[cam_idx] = None

        while self._acquiring:
            # State A - Waiting to start a new recording session
            if self._recording and self._writers[cam_idx] is None:
                self._finished_saving_events[cam_idx].clear()

                try:
                    first_frame, first_metadata = queue.get(timeout=2.0)

                    # Now that we have a frame, create the writer
                    session_idx = len(self._metadata['sessions']) - 1
                    self._writers[cam_idx] = self._create_writer(cam, session_idx)
                    writer = self._writers[cam_idx]

                    if not self._silent:
                        print(f"[INFO] Writer for {cam.name} created: {type(writer).__name__}")

                    writer.write(first_frame, first_metadata)

                except Empty:
                    # If we timeout waiting for the first frame, it means the
                    # grabber isn't feeding the writer queue - so we should log it and try again
                    if not self._silent:
                        print(
                            f"[WARN] Cam {cam.name}: Timed out waiting for first frame to record. Writer not started.")
                    continue  # Skip to the next loop iteration to try again

                except Exception as e:
                    print(f"[ERROR] Failed to create writer for {cam.name}: {e}")
                    self._recording = False  # this is debatable, but prevents getting stuck in an error loop
                    # TODO: Maybe we want to keep a separate status for each camera?
                    self._finished_saving_events[cam_idx].set()
                    continue

            # State B - Actively writing frames
            writer = self._writers[cam_idx]
            if self._recording and writer:
                try:
                    frame, metadata = queue.get(timeout=1.0)
                    writer.write(frame, metadata)
                except Empty:
                    # this is less critical, just means a momentary lull in frames
                    continue

            # State C - Recording has been paused/stopped
            if not self._recording and writer:
                # The recording flag was turned off, so we need to finalize
                # First drain any remaining frames from the queue
                while not queue.empty():
                    try:
                        frame, metadata = queue.get_nowait()
                        writer.write(frame, metadata)
                    except Empty:
                        break

                # Now close the writer and report back its info
                self._session_frame_counts[cam_idx] = writer.frame_count
                self._session_encoding_params[cam_idx] = writer.encoding_params
                writer.close()
                self._writers[cam_idx] = None  # set writer to None to signal it's closed
                self._finished_saving_events[cam_idx].set()

            # State D - Idle, not recording
            if not self._recording and self._writers[cam_idx] is None:
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
            'width': cam.roi[2],
            'height': cam.roi[3],
            'framerate': cam.framerate,
        }

        if save_format == 'mp4':
            # For video
            ffmpeg_config = self.config.get('ffmpeg', {})
            return FFmpegWriter(
                filepath=base_path.with_suffix('.mp4'),
                ffmpeg_path=ffmpeg_config.get('path', 'ffmpeg'),
                params=ffmpeg_config.get('params', {}),
                use_gpu=ffmpeg_config.get('gpu', False),
                pixel_format=cam.pixel_format,
                **writer_params
            )
        else:
            # For image sequences
            return ImageSequenceWriter(
                folder=base_path,
                ext=save_format,
                quality=save_quality,
                pixel_format=cam.pixel_format,
                **writer_params
            )

    def take_snapshot(self) -> bool:
        """ Takes a snapshot from all cameras (tries to sync) """

        if not self.acquiring:
            print("[WARN] Cannot take snapshot, acquisition is not running.")
            return False

        self.full_path.mkdir(parents=True, exist_ok=True)

        # Atomically copy the latest frames from the shared state
        current_frames = []
        for i in range(self.nb_cameras):
            with self._latest_frame_locks[i]:
                frame_data = self._latest_frames[i]

            if frame_data is None:
                print(f"[ERROR] Snapshot failed: No frame has been received from camera {self.cameras[i].name} yet.")
                return False

            current_frames.append(frame_data)

        # Reconcile frames
        frames, metadatas = zip(*current_frames)

        # We only have one frame per camera, so we check if their IDs match
        # This is a much simpler check than the previous buffer search
        first_frame_id = metadatas[0].get('frame_id')
        is_synchronized = False
        if first_frame_id is not None:
            if all(meta.get('frame_id') == first_frame_id for meta in metadatas[1:]):
                is_synchronized = True

        # Save
        if is_synchronized:
            print(f"[INFO] Saving synchronized frame set with ID: {first_frame_id}")
        else:
            print("[WARN] Frame set is not synchronized. Saving latest available frames.")

        success = True

        now = datetime.now().strftime('%y%m%d-%H%M%S')
        for i, cam in enumerate(self.cameras):
            try:
                Image.fromarray(frames[i]).save(self.full_path / f"snapshot_{now}_{cam.name}.png")
            except Exception as e:
                print(f"[ERROR] Could not save snapshot for camera {cam.name}: {e}")
                success = False

        if success:
            print(f"[INFO] Snapshot saved in {self.full_path}")

        return success

    def set_all_cameras(self, parameter: str, value: Any):
        """ Broadcasts a setting to all cameras """
        # TODO: we probably want to have framerate, exposure, etc setters that will call this

        print(f"[INFO] Broadcasting '{parameter} = {value}' to all cameras.")

        # The trigger framerate must be set via its own property
        if parameter == 'framerate' and self.hardware_triggered:
            self.trigger_framerate = value

        for cam in self.cameras:
            try:
                # use setattr to dynamically set the property (e.g., 'exposure', 'gain')
                setattr(cam, parameter, value)
            except Exception as e:
                print(f"[ERROR] Could not set '{parameter}' on camera {cam.name}: {e}")

    # --- Session and Metadata Management ---

    @property
    def session_name(self) -> str:
        return self._session_name

    @session_name.setter
    def session_name(self, name: Optional[str]):
        if self._acquiring:
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
    def trigger_framerate(self) -> float:
        return self._trigger_framerate

    @trigger_framerate.setter
    def trigger_framerate(self, value: float):
        """ Sets the framerate for the external trigger """

        self._trigger_framerate = float(value)

        # If acquisition is running, we can update the trigger on the fly
        if self._acquiring and self._trigger_instance and self._trigger_instance.connected:
            self._trigger_instance.start(self._trigger_framerate)

    @property
    def acquiring(self) -> bool:
        """ Provides backward compatibility for the old '.acquiring' property """
        return self._acquiring

    @property
    def recording(self) -> bool:
        """ Provides backward compatibility for the old '.recording' property """
        return self._recording

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