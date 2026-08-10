import logging
import os
import shutil
import subprocess
import shlex
import platform
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEBUG = False


class FrameWriter(ABC):
    """
    Abstract base class for writing frames to disk.
    It defines the common interface for all writer types (e.g. video, image sequence).
    """

    def __init__(self, filepath: Union[Path, str], pixel_format: str, width: int, height: int, framerate: float,
                 cam_name: str):
        self.filepath = Path(filepath)
        self.pixel_format = pixel_format
        self.width = width
        self.height = height
        self.framerate = framerate
        self.cam_name = cam_name
        self.frame_count = 0

        # Each writer subclass must define its specific encoding parameters (for metadata logging)
        self._encoding_params: Dict[str, Any] = {}

    @property
    def encoding_params(self) -> Dict[str, Any]:
        """Returns the specific encoding parameters used by this writer instance."""
        return self._encoding_params

    def write(self, frame: np.ndarray, frame_data: Dict[str, Any]):
        """
        Writes a single frame
        Calls the internal format-specific writing method
        Increments the internal counter
        """
        self._write_frame(frame, frame_data)
        self.frame_count += 1  # this counter is only incremented if _write_frame succeeds

    @abstractmethod
    def _write_frame(self, frame: np.ndarray, frame_data: Dict[str, Any]):
        """The specific implementation for writing a frame."""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Finalizes the writing process and closes any open resources."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(path='{self.filepath}', frames={self.frame_count})"


class ImageSequenceWriter(FrameWriter):
    """
    Writes frames as a sequence of individual image files (PNG, BMP, JPG, etc.)
    The filepath provided in the constructor is treated as the base name
    and is used as the folder to store the images
    """

    def __init__(self, folder: Union[Path, str], ext: str, quality: int, **kwargs):
        # The filepath for the base class is the folder itself
        super().__init__(folder, **kwargs)

        self.ext = ext.lstrip('.').lower()
        self.quality = int(quality)
        self._imwrite_params = []

        # Store the specific parameters for metadata logging
        self._encoding_params = {
            'format': 'images',
            'extension': self.ext,
        }

        # Determine if the chosen format supports 16-bit depth
        self._supports_16bit = self.ext in ('png', 'tif', 'tiff')

        # Configure OpenCV save parameters based on extension and quality
        if self.ext in ('jpg', 'jpeg'):
            # For JPEG, the quality parameter is a direct 0-100 scale
            self._encoding_params['quality'] = int(self.quality)
            self._imwrite_params = [cv2.IMWRITE_JPEG_QUALITY, int(self.quality)]

        elif self.ext == 'png':
            # For PNG the parameter is 'compression' (0-9)
            # High quality (100) -> Low compression (1) -> Faster write, larger file.
            # Low quality (0) -> High compression (9) -> Slower write, smaller file.

            compr_level = int(np.round(np.interp(self.quality, [0, 100], [9, 1])))
            self._encoding_params['compression'] = compr_level
            self._imwrite_params = [cv2.IMWRITE_PNG_COMPRESSION, compr_level]
            logger.info(f"ImageSequenceWriter: PNG quality {self.quality} mapped to compression level {compr_level}.")

        elif self.ext in ('tif', 'tiff'):
            # For TIFF, simple choice
            # High quality (>= 95): No compression. Fast, huge files, raw data.
            # Lower quality (< 95): Use lossy JPEG compression inside the TIFF container.

            if self.quality >= 95:
                # value 1 for no compression (ZLIB would be 8)
                self._imwrite_params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
                self._encoding_params['lossless'] = True
                logger.info(f"ImageSequenceWriter: TIFF quality {self.quality} >= 95."
                            f" Using no compression.")

            else:
                # JPEG compression (value 7) and pass the quality setting
                self._imwrite_params = [cv2.IMWRITE_TIFF_COMPRESSION, 7, cv2.IMWRITE_JPEG_QUALITY, int(self.quality)]
                self._encoding_params['lossless'] = False
                self._encoding_params['quality'] = int(self.quality)
                logger.info(f"ImageSequenceWriter: TIFF quality {self.quality} < 95."
                            f" Using JPEG compression inside TIFF.")

        elif self.ext == 'bmp':
            self._encoding_params['lossless'] = True
            logger.info(f"ImageSequenceWriter: BMP. Lossless.")

        # Ensure output directory exists
        self.filepath.mkdir(parents=True, exist_ok=True)

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Converts the input frame to a format savable by cv2.imwrite."""

        # High bit-depth monochrome
        if self.pixel_format in ('Mono10', 'Mono12', 'Mono16'):
            if self._supports_16bit:
                # Preserve bit depth for PNG/TIFF
                # Scale up to full 16-bit range
                if self.pixel_format == 'Mono10':
                    return frame.astype(np.uint16) << 6

                if self.pixel_format == 'Mono12':
                    return frame.astype(np.uint16) << 4

                return frame  # Mono16 is already uint16
            else:
                # Convert to 8-bit for JPG/BMP etc
                # (this is a lossy conversion)
                shift = {'Mono10': 2, 'Mono12': 4, 'Mono16': 8}[self.pixel_format]
                return (frame >> shift).astype(np.uint8)

        # Bayer pattern to BGR Conversion
        bayer_map = {
            'BayerRG8': cv2.COLOR_BAYER_RG2BGR, 'BayerGR8': cv2.COLOR_BAYER_GR2BGR,
            'BayerGB8': cv2.COLOR_BAYER_GB2BGR, 'BayerBG8': cv2.COLOR_BAYER_BG2BGR,
        }
        if self.pixel_format in bayer_map:
            return cv2.cvtColor(frame, bayer_map[self.pixel_format])

        # Standard color format conversions
        if self.pixel_format == 'RGB8':
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.pixel_format == 'RGBA8':
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # 8-bit Mono or already BGR
        # if it's Mono8, BGR8, or unknown, return as is
        return frame

    def _write_frame(self, frame: np.ndarray, frame_data: Dict[str, Any]):

        image_path = self.filepath / f"{str(self.frame_count).zfill(9)}.{self.ext}"

        try:
            img_to_write = self._prepare_frame(frame)

            success = cv2.imwrite(str(image_path.resolve()), img_to_write, self._imwrite_params)
            if not success:
                raise IOError("cv2.imwrite() failed, check file path and permissions.")

        except Exception as e:
            # Catch any potential PIL or OS error during saving
            logger.error(f"Failed to save frame {self.frame_count} to {image_path}: {e}")

            # re-raising an error to prevent frame_count from being incremented in the parent write() method
            raise IOError(f"Disk write failed for frame {self.frame_count}") from e

    def close(self):
        # For image sequences, nothing to finalize
        pass


class FFmpegWriter(FrameWriter):
    """Writes frames to a video file by piping them to an FFmpeg subprocess."""

    _available_encoders = None
    _ffmpeg_version = None
    _encoders_lock = threading.Lock()

    def __init__(self, filepath: Union[Path, str], ffmpeg_path: Union[Path, str], params: Dict,
                 use_gpu: bool, profile: Optional[str] = None, **kwargs):
        super().__init__(filepath, **kwargs)

        self.proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None

        which_ffmpeg = shutil.which(str(ffmpeg_path))
        if not which_ffmpeg:
            raise OSError(f"Can't find FFmpeg. Is it installed?")

        ffmpeg_path = Path(which_ffmpeg)
        if not os.access(ffmpeg_path, os.X_OK):
            raise PermissionError(f"Can't run FFmpeg from `{ffmpeg_path}`. Is it executable?")

        self.ffmpeg_path = ffmpeg_path

        # Keep last FFmpeg stderr lines around for diagnostics when something dies mid-recording
        self._stderr_tail: deque = deque(maxlen=30)

        # Map camera format to FFmpeg input format
        input_format_map = {
            'Mono8': 'gray',
            'BayerRG8': 'bayer_rggb8',
            'BayerGR8': 'bayer_grbg8',
            'BayerGB8': 'bayer_gbrg8',
            'BayerBG8': 'bayer_bggr8',
            'RGB8': 'rgb24',
            'BGR8': 'bgr24',
            'Mono10': 'gray10le',
            'Mono12': 'gray12le',
            'Mono16': 'gray16le',
        }
        input_pixel_fmt = input_format_map.get(self.pixel_format)

        if not input_pixel_fmt:
            raise ValueError(f"Unsupported pixel_format '{self.pixel_format}' for FFmpegWriter.")

        # Determine output format and encoder-specific setup
        high_bitdepth = self.pixel_format in ('Mono10', 'Mono12', 'Mono16')

        # An encoder being listed by ffmpeg doesn't mean it actually works on this machine
        # (missing GPU, outdated driver, etc) so we try candidates in priority order and keep the first one that works
        candidates = self._get_priority_candidates(params, use_gpu, profile)

        success = False
        last_error = 'No candidate encoder profiles available'

        for param_key in candidates:
            command = self._build_ffmpeg_command(
                filepath=filepath,
                param_key=param_key,
                encoder_params_str=params[param_key],
                input_pixel_fmt=input_pixel_fmt,
                high_bitdepth=high_bitdepth,
            )
            logger.debug(f"Attempting FFmpeg profile '{param_key}' for '{self.cam_name}': {' '.join(command)}")

            try:
                proc = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    bufsize=10 ** 8
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Could not spawn FFmpeg with profile '{param_key}': {e}")
                continue

            # Just to give FFmpeg a moment to initialise
            time.sleep(0.25)

            if proc.poll() is None:
                # Still running, looks good
                self.proc = proc

                # Drain stderr from now on: unread pipe fills up and deadlocks FFmpeg,
                # which stops gets killed at close() -> corrupted file
                self._stderr_thread = threading.Thread(target=self._drain_stderr, args=(proc,), daemon=True)
                self._stderr_thread.start()

                self._encoding_params = {
                    'format': 'ffmpeg_video',
                    'encoder_profile': param_key,
                    'command': ' '.join(command)
                }
                logger.info(f"FFmpeg initialized for '{self.cam_name}' using profile '{param_key}'")
                success = True
                break
            else:
                # crashed: remember why and try next candidate
                try:
                    _, err_data = proc.communicate(timeout=2)
                    last_error = err_data.decode('utf-8', errors='replace').strip() or 'Unknown error'
                except Exception:
                    last_error = 'Unknown error'

                logger.warning(f"FFmpeg profile '{param_key}' failed for '{self.cam_name}':"
                               f" {last_error.splitlines()[-1]}")
                continue

        if not success:
            raise RuntimeError(
                f"FFmpegWriter failed to start any viable encoder for '{self.cam_name}'. Last error: {last_error}")

    def _get_priority_candidates(self, params: Dict, use_gpu: bool, profile: Optional[str]) -> List[str]:
        """
        Returns the list of profile keys to try, from most to least preferred:
        explicitly requested profile first, then version- and availability-gated
        hardware profiles (if use_gpu), then the CPU profiles.
        """

        candidates: List[str] = []

        if profile:
            if profile in params:
                candidates.append(profile)
            else:
                logger.warning(f"Requested FFmpeg profile '{profile}' not found in config's 'params' section."
                               f" Falling back to auto-selection.")

        if use_gpu:
            ffmpeg_version = self._get_ffmpeg_version(self.ffmpeg_path)
            available_encoders = self._get_available_encoders(self.ffmpeg_path)
            system = platform.system()

            # (profile_key, encoder_name, minimum_ffmpeg_version as tuple)
            PRIORITY_MAP = {
                'Linux': [
                    ('gpu_nvenc_h264', 'h264_nvenc', (4, 0, 0)),
                    ('gpu_nvenc_h265', 'hevc_nvenc', (4, 0, 0)),
                    ('gpu_vulkan_h264', 'h264_vulkan', (8, 0, 0)),
                    ('gpu_vulkan_h265', 'hevc_vulkan', (8, 0, 0)),
                    ('gpu_arc_av1', 'av1_qsv', (5, 0, 0)),
                    ('gpu_vaapi', 'hevc_vaapi', (4, 0, 0)),
                    ('gpu_arc_hevc', 'hevc_qsv', (5, 0, 0)),
                ],
                'Windows': [
                    ('gpu_nvenc_h264', 'h264_nvenc', (4, 0, 0)),
                    ('gpu_nvenc_h265', 'hevc_nvenc', (4, 0, 0)),
                    ('gpu_vulkan_h264', 'h264_vulkan', (8, 0, 0)),
                    ('gpu_vulkan_h265', 'hevc_vulkan', (8, 0, 0)),
                    ('gpu_arc_av1', 'av1_qsv', (5, 0, 0)),
                    ('gpu_amf', 'hevc_amf', (4, 0, 0)),
                    ('gpu_arc_hevc', 'hevc_qsv', (5, 0, 0)),
                ],
                'Darwin': [
                    ('gpu_videotoolbox', 'hevc_videotoolbox', (4, 0, 0)),
                ]
            }

            for profile_key, encoder_name, min_version in PRIORITY_MAP.get(system, []):
                if ffmpeg_version < min_version:
                    continue
                if profile_key in params and encoder_name in available_encoders and profile_key not in candidates:
                    candidates.append(profile_key)

        # Always add CPU fallbacks at the end ('cpu_x*' are legacy config key names)
        for cpu_key in ('cpu_h264', 'cpu_h265', 'cpu_x264', 'cpu_x265'):
            if cpu_key in params and cpu_key not in candidates:
                candidates.append(cpu_key)

        return candidates

    def _build_ffmpeg_command(self, filepath: Path, param_key: str, encoder_params_str: str,
                              input_pixel_fmt: str, high_bitdepth: bool) -> List[str]:
        """Build the complete FFmpeg command based on encoder type."""

        extra_input_args: List[str] = []
        extra_encoder_args: List[str] = []

        is_cpu = param_key.startswith('cpu')

        # Vulkan encoders need special handling
        if 'vulkan' in param_key:
            # Vulkan requires hardware device initialization
            # The filter chain handles format conversion
            extra_input_args = ['-init_hw_device', 'vulkan=vk', '-filter_hw_device', 'vk']

            # For grayscale input, we need to convert to a format Vulkan can handle
            if input_pixel_fmt == 'gray':
                extra_encoder_args = ['-vf', 'format=nv12,hwupload']
            else:
                extra_encoder_args = ['-vf', 'hwupload']

        elif 'vaapi' in param_key:
            # VAAPI needs a filter chain with hwupload
            vaapi_format = 'p010' if high_bitdepth else 'nv12'
            extra_encoder_args = ['-vf', f'format={vaapi_format},hwupload']

        elif 'videotoolbox' in param_key:
            # Inject the correct profile if not already specified
            if '-profile' not in encoder_params_str:
                extra_encoder_args = ['-profile', 'main10' if high_bitdepth else 'main']

        elif '-pix_fmt' not in encoder_params_str:  # Covers cpu, nvenc, qsv, amf
            # These encoders use the standard -pix_fmt flag at the end
            if high_bitdepth:
                output_pixel_fmt = 'yuv420p10le' if is_cpu else 'p010le'
            else:
                output_pixel_fmt = 'yuv420p' if is_cpu else 'nv12'
            extra_encoder_args = ['-pix_fmt', output_pixel_fmt]

        command = [str(self.ffmpeg_path), '-hide_banner']
        command += extra_input_args
        command += [
            '-thread_queue_size', '1024', '-y',
            '-s', f'{self.width}x{self.height}',
            '-f', 'rawvideo',
            '-framerate', f'{self.framerate:.3f}',
            '-pix_fmt', input_pixel_fmt,
            '-i', 'pipe:0'
        ]
        command += shlex.split(encoder_params_str)
        command += extra_encoder_args
        # Fragmented mp4 so that the file stays playable even if FFmpeg is killed before finalising
        command += ['-movflags', '+frag_keyframe+empty_moov']
        command.append(str(filepath))

        return command

    def _drain_stderr(self, proc: subprocess.Popen):
        """
        Drain stderr to prevent buffer deadlock, keeps a tail of output for diagnostics.
        """
        try:
            for line in proc.stderr:
                decoded = line.decode('utf-8', errors='replace').strip()
                if decoded:
                    self._stderr_tail.append(decoded)
                    if DEBUG:
                        logger.debug(f"FFmpeg [{self.cam_name}]: {decoded}")
        except Exception:
            pass

    @staticmethod
    def _get_ffmpeg_version(ffmpeg_path: Union[Path, str]) -> Tuple[int, int, int]:
        """
        Gets the FFmpeg version as a tuple (major, minor, patch).
        """
        if FFmpegWriter._ffmpeg_version is not None:
            return FFmpegWriter._ffmpeg_version

        with FFmpegWriter._encoders_lock:
            if FFmpegWriter._ffmpeg_version is not None:
                return FFmpegWriter._ffmpeg_version

            try:
                result = subprocess.check_output(
                    [ffmpeg_path, '-version'],
                    stderr=subprocess.STDOUT
                ).decode('utf-8')

                # Parse version from first line, e.g., "ffmpeg version 8.0.1-full_build ..."
                import re
                match = re.search(r'ffmpeg version (\d+)\.(\d+)(?:\.(\d+))?', result)
                if match:
                    major = int(match.group(1))
                    minor = int(match.group(2))
                    patch = int(match.group(3)) if match.group(3) else 0
                    FFmpegWriter._ffmpeg_version = (major, minor, patch)
                    logger.debug(f"Detected FFmpeg version: {major}.{minor}.{patch}")
                else:
                    logger.warning("Could not parse FFmpeg version, assuming 6.0.0+")
                    FFmpegWriter._ffmpeg_version = (6, 0, 0)

                return FFmpegWriter._ffmpeg_version

            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.error(f"Could not query FFmpeg version: {e}")
                FFmpegWriter._ffmpeg_version = (6, 0, 0)
                return FFmpegWriter._ffmpeg_version

    @staticmethod
    def _get_available_encoders(ffmpeg_path: Union[Path, str]) -> set:
        """
        Gets a set of all available encoders from the ffmpeg executable.
        Results are cached in the class to avoid repeated calls to the subprocess from multiple threads.
        """

        if FFmpegWriter._available_encoders is not None:
            return FFmpegWriter._available_encoders

        with FFmpegWriter._encoders_lock:
            # Double-check in case another thread just populated it
            if FFmpegWriter._available_encoders is not None:
                return FFmpegWriter._available_encoders

            logger.debug("Querying FFmpeg for available encoders...")
            try:
                result = subprocess.check_output(
                    [ffmpeg_path, '-hide_banner', '-encoders'],
                    stderr=subprocess.STDOUT
                ).decode('utf-8')

                encoders = set()
                # Parsing the output of ffmpeg -encoders
                # Line format is like: ' V..... h264_nvenc   NVIDIA NVENC H.264 encoder (codec h264)'
                for line in result.splitlines():
                    if "Encoders:" in line:
                        continue  # Skip header

                    parts = line.strip().split()
                    if len(parts) > 1 and parts[0].startswith('V'):  # 'V' means video encoder
                        encoders.add(parts[1])

                FFmpegWriter._available_encoders = encoders
                logger.debug(f"Found encoders: {FFmpegWriter._available_encoders}")
                return FFmpegWriter._available_encoders

            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.error(f"Could not query FFmpeg for encoders: {e}")
                FFmpegWriter._available_encoders = set()  # cache failure
                return FFmpegWriter._available_encoders

    def _write_frame(self, frame: np.ndarray, frame_data: Dict[str, Any]):

        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.write(memoryview(frame))

            except (IOError, BrokenPipeError) as e:

                # This can happen if FFmpeg closes unexpectedly
                logger.error(f"Failed to write to FFmpeg process for '{self.cam_name}': {e}")
                if self._stderr_tail:
                    logger.error("Last FFmpeg output:\n" + '\n'.join(self._stderr_tail))
                self.close()
                raise IOError("FFmpeg process terminated unexpectedly.") from e

    def close(self):
        if not self.proc:
            return

        # Close input pipe so FFmpeg knows stream is done
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            # Process might already be dead, which is fine
            pass

        # Wait for FFmpeg to finish writing the file
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning(f"FFmpeg process for {self.cam_name} timed out during close. Forcing termination.")
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            except Exception:
                pass

        self.proc = None