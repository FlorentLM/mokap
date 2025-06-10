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
from mokap.core.cameras import CameraFactory
from mokap.core.triggers.interface import AbstractTrigger
from mokap.core.triggers.raspberry import RaspberryTrigger
from mokap.core.triggers.arduino import ArduinoTrigger
from mokap.core.writers import FrameWriter, ImageSequenceWriter, FFmpegWriter
from mokap.utils import fileio
from mokap.utils.system import setup_ulimit


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

            if not self._trigger_instance.connected:
                print("[ERROR] Failed to connect to trigger. Disabling triggered mode.")
                self._trigger_instance = None

        # --- Threading Resources ---
        buffer_size = self.config.get('frame_buffer_size', 200)

        self._session_frame_counts: List[int] = [0] * self.nb_cameras

        # Queues for the GUI's blocking consumer thread
        self._display_queues: List[Queue] = [Queue(maxsize=2) for _ in self.cameras]
        # Parallel state for the synchronized snapshots
        self._latest_display_frames: List[Optional[Tuple[np.ndarray, Dict]]] = [None] * self.nb_cameras
        self._display_locks: List[Lock] = [Lock() for _ in self.cameras]

        # Writer queues for saving
        self._writer_queues: List[Queue] = [Queue(maxsize=buffer_size) for _ in self.cameras]
        self._finished_saving_events: List[Event] = [Event() for _ in self.cameras]

        for event in self._finished_saving_events:
            event.set()  # default to 'finished' state

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
            serial = str(cam_config.get('serial'))  # ensure serial is a string

            if not serial:
                print(f"[WARN] Skipping '{friendly_name}': no serial number provided in config.")
                continue

            # Find the corresponding physical device using our lookup map
            device_info = device_lookup.get(serial)

            if not device_info:
                print(f"[WARN] Skipping '{friendly_name}': camera with serial {serial} not found.")
                continue

            cam = CameraFactory.get_camera(device_info)

            if cam:
                try:
                    cam.name = friendly_name

                    # Start with a copy of the global settings
                    final_settings = base_cam_config.copy()

                    # Get camera-specific overrides and merge them in
                    camera_overrides = cam_config.get('settings', {})
                    if camera_overrides:
                        final_settings.update(camera_overrides)

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
            # Start one grabber and one writer thread per camera
            g = Thread(target=self._grabber_thread, args=(i,), daemon=True)
            w = Thread(target=self._writer_thread, args=(i,), daemon=True)
            self._threads.extend([g, w])
            g.start()
            w.start()

        if not self._silent:
            print(f"[INFO] Acquisition started with {self.nb_cameras} cameras.")

    def stop_acquisition(self):
        """ Stops all background threads and disconnects cameras """
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

        # Cleanly disconnect all cameras
        for cam in self.cameras:
            if cam.is_connected:
                cam.disconnect()

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

        # Prepare metadata for this new session
        session_metadata = {
            'start_timestamp': datetime.now().timestamp(),
            'end_timestamp': None,
            'duration': 0.0,
            'config': self.config,  # Store a snapshot of the config
            'cameras': {cam.unique_id: {
                'name': cam.name,
                'model': CameraFactory.get_camera_info(cam.unique_id)['model'],
                'frames_recorded': 0
            } for cam in self.cameras
            }
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

        # Update metadata with final counts
        session_idx = len(self._metadata['sessions']) - 1
        current_session = self._metadata['sessions'][session_idx]
        # Read the counts that the writer threads have reported back
        for i, cam in enumerate(self.cameras):
            count = self._session_frame_counts[i]
            current_session['cameras'][cam.unique_id]['frames_recorded'] = count

        self.save_metadata()

        if not self._silent:
            print("[INFO] Recording paused. Files saved.")

    def _grabber_thread(self, cam_idx: int):
        """ Dedicated thread to continuously grab frames from a single camera """

        cam = self.cameras[cam_idx]
        writer_queue = self._writer_queues[cam_idx]

        display_queue = self._display_queues[cam_idx]
        display_lock = self._display_locks[cam_idx]

        cam.start_grabbing()
        while self._acquiring:
            try:
                frame, metadata = cam.grab_frame(timeout_ms=2000)

                if frame is not None:
                    # --- For Display (GUI) ---
                    if display_queue.full():
                        try:
                            display_queue.get_nowait()  # Discard old frame
                        except queue.Empty:
                            pass
                    # The GUI thread will get this via its blocking .get() call
                    display_queue.put_nowait((frame, metadata))

                    # --- For snapshots ---
                    # Also update the shared variable for non-destructive reads
                    with display_lock:
                        # we store the *same* frame here
                        self._latest_display_frames[cam_idx] = (frame, metadata)

                    # --- For Recording ---
                    if self._recording:
                        try:
                            writer_queue.put((frame, metadata), timeout=1.0)
                        except queue.Full:
                            if not self._silent:
                                print(f"[WARN] Cam {cam.name}: Writer queue full. Frame dropped.")

            except (IOError, RuntimeError) as e:
                if self._acquiring:  # avoid spamming errors on shutdown
                    print(f"[ERROR] Grabber thread for {cam.name} failed: {e}")
                    time.sleep(1)  # prevent flooding error loops

        cam.stop_grabbing()

    # In class MultiCam:
    def _writer_thread(self, cam_idx: int):
        """
        Dedicated thread to write frames from a queue to a file
        This thread manages the lifecycle of a FrameWriter object
        """
        cam = self.cameras[cam_idx]
        queue = self._writer_queues[cam_idx]
        writer: Optional[FrameWriter] = None

        while self._acquiring:
            # State A - Waiting to start a new recording session
            if self._recording and writer is None:
                self._finished_saving_events[cam_idx].clear()

                try:
                    # **THE FIX**: Wait for the first frame BEFORE creating the writer.
                    # This guarantees we have data the moment FFmpeg starts.
                    # Use a longer timeout to be safe, especially at session start.
                    first_frame, first_metadata = queue.get(timeout=2.0)

                    # Now that we have a frame, create the writer.
                    session_idx = len(self._metadata['sessions']) - 1
                    writer = self._create_writer_for_camera(cam, session_idx)

                    if not self._silent:
                        print(f"[INFO] Writer for {cam.name} created: {type(writer).__name__}")

                    # Immediately write the first frame we just retrieved.
                    # This is the "no-dummy-frame" way to prime the pump.
                    writer.write(first_frame, first_metadata)

                except Empty:
                    # If we time out waiting for the first frame, it means the
                    # grabber isn't feeding the writer queue. This is a real problem,
                    # so we should log it and try again.
                    if not self._silent:
                        print(
                            f"[WARN] Cam {cam.name}: Timed out waiting for first frame to record. Writer not started.")
                    continue  # Skip to the next loop iteration to try again.
                except Exception as e:
                    # Catch potential errors from _create_writer_for_camera
                    print(f"[ERROR] Failed to create writer for {cam.name}: {e}")
                    # Ensure we don't get stuck in a loop if creation fails
                    self._recording = False  # This is debatable, but prevents a fast error loop.
                    self._finished_saving_events[cam_idx].set()
                    continue

            #  State B - Actively writing frames
            if self._recording and writer:
                try:
                    # The main writing loop.
                    frame, metadata = queue.get(timeout=0.2)  # Use a slightly longer timeout
                    writer.write(frame, metadata)
                except Empty:
                    # This is now less critical, just means a momentary lull in frames.
                    continue

            # State C - Recording has been paused/stopped
            if not self._recording and writer:
                # The recording flag was turned off, so we need to finalize.
                # First, drain any remaining frames from the queue.
                while not queue.empty():
                    try:
                        frame, metadata = queue.get_nowait()
                        writer.write(frame, metadata)
                    except Empty:
                        break

                # Now close the writer.
                self._session_frame_counts[cam_idx] = writer.frame_count
                writer.close()
                writer = None  # Set writer to None to signal it's closed.
                self._finished_saving_events[cam_idx].set()

            # State D - Idle, not recording
            if not self._recording and writer is None:
                # To prevent this thread from busy-waiting, we can sleep
                time.sleep(0.1)

    def _create_writer_for_camera(self, cam: AbstractCamera, session_idx: int) -> FrameWriter:
        """ Factory method to instantiate the correct writer based on config """
        save_format = self.config.get('save_format', 'mp4')
        base_path = self.full_path / f"{self.session_name}_cam{cam.name}_session{session_idx}"

        # Common parameters for all writers
        writer_params = {
            'width': cam.roi[2],
            'height': cam.roi[3],
            'framerate': cam.framerate,
            'pixel_format': cam.pixel_format
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
                quality=self.config.get('save_quality', 90),
                **writer_params
            )

    def take_snapshot(self, folder: Path, base_name: str) -> bool:
        """
        Grabs the most recent frame from each camera in a synchronized manner
        by reading from a shared state, without interfering with the GUI queues.
        """

        if not self.acquiring:
            print("[WARN] Cannot take snapshot, acquisition is not running.")
            return False

        folder.mkdir(parents=True, exist_ok=True)

        frames_to_save = []

        # Acquire all locks first to get consistent set of frames across cameras
        # This briefly pauses all grabber threads from updating the _latest_display_frames variables
        for lock in self._display_locks:
            lock.acquire()

        for i in range(self.nb_cameras):
            frame_data = self._latest_display_frames[i]
            if frame_data:
                # Important: copy of the numpy array
                frames_to_save.append((frame_data[0].copy(), self.cameras[i].name))
            else:
                frames_to_save.append((None, self.cameras[i].name))

        # of course release the locks
        for lock in self._display_locks:
            lock.release()

        success = True
        for i, cam in enumerate(self.cameras):
            frame_data = frames_to_save[i]
            if frame_data:
                try:
                    # TODO: We prob want to use the Writer class here. Or define a new snapshot writer that never compresses? idk
                    frame, _ = frame_data
                    img = Image.fromarray(frame)
                    filepath = folder / f"{base_name}_{cam.name}.png"
                    img.save(filepath)

                except Exception as e:
                    print(f"[ERROR] Could not save snapshot for camera {cam.name}: {e}")
                    success = False
            else:
                # This will only happen if a camera has not produced a single frame yet
                print(f"[WARN] No frame has ever been received from camera {cam.name} to snapshot.")
                success = False

        if success:
            print(f"[INFO] Snapshot saved in {folder}")
        return success

    def get_latest_frame_for_display(self, cam_idx: int) -> Optional[np.ndarray]:
        """
        Thread-safe method for the GUI to get the latest frame for display
        This is a non-destructive read
        """
        if not (0 <= cam_idx < self.nb_cameras):
            return None

        with self._display_locks[cam_idx]:
            frame_data = self._latest_display_frames[cam_idx]
            if frame_data:
                # Return a copy so the GUI's processing doesn't affect the stored frame
                return frame_data[0].copy()
        return None

    def set_all_cameras(self, parameter: str, value: Any):
        """ Broadcasts a setting to all cameras """
        # TODO: we probably want to have framerate, exposure, etc setters that will call this

        print(f"[INFO] Broadcasting '{parameter} = {value}' to all cameras.")

        # If setting framerate in hardware trigger mode, we also set the trigger speed
        if parameter == 'framerate' and self.config.get('triggered', False):
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

    @property
    def trigger_framerate(self) -> float:
        return self._trigger_framerate

    @trigger_framerate.setter
    def trigger_framerate(self, value: float):
        """ Sets the framerate for the external trigger """

        self._trigger_framerate = float(value)
        # If acquisition is running, we can even update the trigger on the fly
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