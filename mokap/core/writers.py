import logging
import os
import shutil
import subprocess
import shlex
import platform
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameWriter(ABC):
    """
    Abstract base class for writing frames to disk. It defines the common
    interface for all writer types (e.g., video, image sequence)
    """

    def __init__(self, filepath: Path, pixel_format: str, width: int, height: int, framerate: float, cam_name: str):
        self.filepath = filepath
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
        """ Returns the specific encoding parameters used by this writer instance """
        return self._encoding_params

    def write(self, frame: np.ndarray, frame_data: Dict[str, Any]):
        """
        Writes a single frame
        Calls the internal format-specific writing method
        Increments the internal counter
        """
        self._write_frame(frame, frame_data)
        self.frame_count += 1   # this counter is only incremented if _write_frame succeeds

    @abstractmethod
    def _write_frame(self, frame: np.ndarray, frame_data: Dict[str, Any]):
        """ The specific implementation for writing a frame """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """ Finalizes the writing process and closes any open resources """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(path='{self.filepath}', frames={self.frame_count})"


class ImageSequenceWriter(FrameWriter):
    """
    Writes frames as a sequence of individual image files (PNG, BMP, JPG, etc.)
    The filepath provided in the constructor is treated as the base name
    and is used as the folder to store the images
    """

    def __init__(self, folder: Path, ext: str, quality: int, **kwargs):
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
            #
            # higher compression level means a smaller file but slower write time
            #
            # We want:
            # High Quality (100) -> Low Compression (1) -> Faster write, larger file
            # Low Quality (0) -> High Compression (9) -> Slower write, smaller file
            #
            compression_level = int(np.round(np.interp(self.quality, [0, 100], [9, 1])))
            self._encoding_params['compression'] = compression_level
            self._imwrite_params = [cv2.IMWRITE_PNG_COMPRESSION, compression_level]
            logger.info(f"ImageSequenceWriter: PNG quality {self.quality} mapped to compression level {compression_level}.")

        elif self.ext in ('tif', 'tiff'):
            # For TIFF we can offer a simple choice
            # - High quality (>= 95): No compression. Fast, huge files, raw data
            # - Lower quality (< 95): Use lossy JPEG compression inside the TIFF container
            if self.quality >= 95:
                # Use value 1 for no compression (ZLIB would be 8)
                self._imwrite_params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
                self._encoding_params['lossless'] = True
                logger.info(f"ImageSequenceWriter: TIFF quality {self.quality} >= 95. Using no compression.")

            else:
                # Use JPEG compression (value 7) and pass the quality setting
                self._imwrite_params = [cv2.IMWRITE_TIFF_COMPRESSION, 7, cv2.IMWRITE_JPEG_QUALITY, int(self.quality)]
                self._encoding_params['lossless'] = False
                self._encoding_params['quality'] = int(self.quality)
                logger.info(f"ImageSequenceWriter: TIFF quality {self.quality} < 95. Using JPEG compression inside TIFF.")

        elif self.ext in ('bmp'):
            self._encoding_params['lossless'] = True
            logger.info(f"ImageSequenceWriter: BMP. Lossless.")

        # Ensure the output directory exists
        self.filepath.mkdir(parents=True, exist_ok=True)

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """ Converts the input frame to a format savable by cv2.imwrite """

        # High Bit-Depth monochrome
        if self.pixel_format in ('Mono10', 'Mono12', 'Mono16'):
            if self._supports_16bit:
                # Preserve bit depth for PNG/TIFF
                # Scale up to full 16-bit range for better visualization
                if self.pixel_format == 'Mono10':
                    return (frame.astype(np.uint16) << 6)

                if self.pixel_format == 'Mono12':
                    return (frame.astype(np.uint16) << 4)

                return frame  # Mono16 is already uint16
            else:
                # Convert to 8-bit for JPG/BMP etc
                # This is a lossy conversion
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
            # Prepare the frame (debayer, scale bit depth, etc)
            img_to_write = self._prepare_frame(frame)

            # Save the image using OpenCV
            success = cv2.imwrite(str(image_path.resolve()), img_to_write, self._imwrite_params)
            if not success:
                raise IOError("cv2.imwrite returned False, check file path and permissions.")

        except Exception as e:
            # Catch any potential PIL or OS error during the save operation
            logger.error(f"Failed to save frame {self.frame_count} to {image_path}: {e}")

            # re-raising an error to prevent the frame_count from being incremented in the parent write() method
            raise IOError(f"Disk write failed for frame {self.frame_count}") from e

    def close(self):
        # For image sequences, there's nothing to finalize
        pass


class FFmpegWriter(FrameWriter):
    """ Writes frames to a video file by piping them to an FFmpeg subprocess """

    _available_encoders = None
    _encoders_lock = threading.Lock()  # we generally create writers from multiple threads at once

    def __init__(self, filepath: Path, ffmpeg_path: str, params: Dict, use_gpu: bool, **kwargs):
        super().__init__(filepath, **kwargs)

        self.proc: Optional[subprocess.Popen] = None

        which_ffmpeg = shutil.which(ffmpeg_path)
        if not which_ffmpeg:
            raise OSError(f"Can't find FFmpeg. Is it installed?")

        path = Path(which_ffmpeg)
        if not os.access(path, os.X_OK):
            raise PermissionError(f"Can't run FFmpeg from `{path.as_posix()}`. Is it executable?.")

        self.ffmpeg_path = path.resolve()
        print(self.ffmpeg_path)

        if use_gpu:
            # Allow user override first
            param_key = params.get('profile')
            if not param_key:
                # If no override, auto-detect
                param_key = self._get_best_profile_key(ffmpeg_path, params)
        else:
            param_key = 'cpu_x265'

        encoder_params_str = params.get(param_key)
        if not encoder_params_str:
            raise ValueError(f"FFmpeg profile '{param_key}' not found in config's 'params' section.")

        # Pixel formats

        # map camera format to FFmpeg input format
        input_format_map = {
            'Mono8': 'gray',
            'BayerRG8': 'bayer_rggr8',
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

        # Determine output format, and if we are doing a high-bit-depth encode
        high_bitdepth = self.pixel_format in ('Mono10', 'Mono12', 'Mono16')
        extra_encoder_args = ""

        if 'vaapi' in param_key:
            # VAAPI needs a filter chain with hwupload
            vaapi_format = 'p010' if high_bitdepth else 'nv12'
            extra_encoder_args = f"-vf format={vaapi_format},hwupload"

        elif 'videotoolbox' in param_key:
            # Inject the correct profile if not already specified by the user in the config
            if "-profile" not in encoder_params_str:
                profile_arg = "-profile main10" if high_bitdepth else "-profile main"
                extra_encoder_args = profile_arg

        else:  # Covers cpu, nvenc, qsv, amf
            # These encoders use the standard -pix_fmt flag at the end
            if high_bitdepth:
                # Hardware encoders prefer p010le, software prefers yuv420p10le
                output_pixel_fmt = 'p010le' if use_gpu else 'yuv420p10le'
            else:
                output_pixel_fmt = 'yuv420p'
            extra_encoder_args  = f"-pix_fmt {output_pixel_fmt}"

        # Build the command
        input_args = (
            f"-y -s {self.width}x{self.height} -f rawvideo "
            f"-framerate {self.framerate:.3f} -pix_fmt {input_pixel_fmt} -i pipe:0"
        )

        # Add the dynamically determined pixel format to the encoder params
        full_encoder_params = f"{encoder_params_str} {extra_encoder_args}"

        command = f"{shlex.quote(str(self.ffmpeg_path))} -hide_banner {input_args} {full_encoder_params} {shlex.quote(str(filepath))}"

        logger.debug(f"FFmpeg command for '{self.cam_name}': {command}")

        # Store metadata
        self._encoding_params = {
            'format': 'ffmpeg_video',
            'encoder_profile': param_key,
            'command': command
        }

        # Start the subprocess, redirecting stdout/stderr to prevent console spam
        self.proc = subprocess.Popen(
            shlex.split(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
            # stdout=subprocess.PIPE,         # for debug
            # stderr=subprocess.PIPE          # for debug
        )

    @staticmethod
    def _get_available_encoders(ffmpeg_path: str) -> set:
        """
        Gets a set of all available encoders from the ffmpeg executable
        Results are cached in the class to avoid repeated calls to the subprocess from multiple threads
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
                # Line format is like: ' V..... h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)'
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

    def _get_best_profile_key(self, ffmpeg_path: str, params: Dict) -> str:
        """
        Automatically determines the best encoder profile to use
        (based on OS, available hardware, and a predefined priority list)
        """

        # Priority is defined as: Best quality/efficiency first
        # We prefer AV1 > HEVC, and Hardware > Software
        PRIORITY_MAP = {
            'Linux': [
                ('gpu_arc_av1', 'av1_qsv'),
                ('gpu_nvenc_h265', 'hevc_nvenc'),
                ('gpu_nvenc_h264', 'h264_nvenc'),
                ('gpu_vaapi', 'hevc_vaapi'),
                ('gpu_arc_hevc', 'hevc_qsv'),
                ('cpu_h265', 'libx265'),
                ('cpu_h264', 'libx264'),
            ],
            'Windows': [
                ('gpu_arc_av1', 'av1_qsv'),
                ('gpu_nvenc_h265', 'hevc_nvenc'),
                ('gpu_nvenc_h264', 'h264_nvenc'),
                ('gpu_amf', 'hevc_amf'),
                ('gpu_arc_hevc', 'hevc_qsv'),
                ('cpu_h265', 'libx265'),
                ('cpu_h264', 'libx264'),
            ],
            'Darwin': [  # macOS
                ('gpu_videotoolbox', 'hevc_videotoolbox'),
                ('cpu_h265', 'libx265'),
                ('cpu_h264', 'libx264'),
            ]
        }

        available_encoders = self._get_available_encoders(ffmpeg_path)
        system = platform.system()

        priority_list = PRIORITY_MAP.get(system, [])
        if not priority_list:
            logger.warning(f"Unsupported OS '{system}' for auto-selection. Falling back to CPU.")
            return 'cpu_x265'  # a safe default

        for profile_key, encoder_name in priority_list:
            if profile_key in params and encoder_name in available_encoders:
                logger.info(f"Auto-selected FFmpeg profile: '{profile_key}' (using '{encoder_name}')")
                return profile_key

        logger.warning("No suitable high-priority encoder found. Check FFmpeg build and drivers.")
        return 'cpu_x264'  # absolute fallback

    def _write_frame(self, frame: np.ndarray, frame_data: Dict[str, Any]):

        if self.proc and self.proc.stdin:
            try:
                # Write the raw bytes of the frame to the stdin of the FFmpeg process
                self.proc.stdin.write(frame.tobytes())

            except (IOError, BrokenPipeError) as e:

                # This can happen if FFmpeg closes unexpectedly
                logger.error(f"Failed to write to FFmpeg process: {e}")

                # We can try to get more info from stderr if it was captured
                self.close()  # Attempt to clean up
                raise IOError("FFmpeg process terminated unexpectedly.") from e

    def close(self):
        if not self.proc:
            return

        # Gracefully close the input pipe and wait for FFmpeg to finish encoding
        if self.proc.stdin:
            self.proc.stdin.flush()
            self.proc.stdin.close()
        self.proc.wait(timeout=10)
        self.proc = None