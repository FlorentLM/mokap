import os
import subprocess
import shlex
import platform
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image


class FrameWriter(ABC):
    """
    Abstract base class for writing frames to disk. It defines the common
    interface for all writer types (e.g., video, image sequence)
    """

    def __init__(self, filepath: Path, width: int, height: int, framerate: float, pixel_format: str):
        self.filepath = filepath
        self.width = width
        self.height = height
        self.framerate = framerate
        self.pixel_format = pixel_format
        self.frame_count = 0

        # Each writer subclass must define its specific encoding parameters (for metadata logging)
        self._encoding_params: Dict[str, Any] = {}

    @property
    def encoding_params(self) -> Dict[str, Any]:
        """ Returns the specific encoding parameters used by this writer instance """
        return self._encoding_params

    def write(self, frame: np.ndarray, metadata: Dict[str, Any]):
        """
        Writes a single frame
        Calls the internal format-specific writing method
        Increments the internal counter
        """
        self._write_frame(frame, metadata)
        self.frame_count += 1   # this counter is only incremented if _write_frame succeeds

    @abstractmethod
    def _write_frame(self, frame: np.ndarray, metadata: Dict[str, Any]):
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
            print(f"[INFO] ImageSequenceWriter: PNG quality {self.quality} mapped to compression level {compression_level}.")

        elif self.ext in ('tif', 'tiff'):
            # For TIFF we can offer a simple choice
            # - High quality (>= 95): No compression. Fast, huge files, raw data
            # - Lower quality (< 95): Use lossy JPEG compression inside the TIFF container
            if self.quality >= 95:
                # Use value 1 for no compression (ZLIB would be 8)
                self._imwrite_params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
                self._encoding_params['lossless'] = True
                print(f"[INFO] ImageSequenceWriter: TIFF quality {self.quality} >= 95. Using no compression.")

            else:
                # Use JPEG compression (value 7) and pass the quality setting
                self._imwrite_params = [cv2.IMWRITE_TIFF_COMPRESSION, 7, cv2.IMWRITE_JPEG_QUALITY, int(self.quality)]
                self._encoding_params['lossless'] = False
                self._encoding_params['quality'] = int(self.quality)
                print(f"[INFO] ImageSequenceWriter: TIFF quality {self.quality} < 95. Using JPEG compression inside TIFF.")

        elif self.ext in ('bmp'):
            self._encoding_params['lossless'] = True
            print(f"[INFO] ImageSequenceWriter: BMP. Lossless.")

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

    def _write_frame(self, frame: np.ndarray, metadata: Dict[str, Any]):

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
            print(f"[ERROR] Failed to save frame {self.frame_count} to {image_path}: {e}")

            # re-raising an error to prevent the frame_count from being incremented in the parent write() method
            raise IOError(f"Disk write failed for frame {self.frame_count}") from e

    def close(self):
        # For image sequences, there's nothing to finalize
        pass


class FFmpegWriter(FrameWriter):
    """ Writes frames to a video file by piping them to an FFmpeg subprocess """

    def __init__(self, filepath: Path, ffmpeg_path: str, params: Dict, use_gpu: bool, **kwargs):
        super().__init__(filepath, **kwargs)

        self.proc: Optional[subprocess.Popen] = None
        self.ffmpeg_path = ffmpeg_path

        # Determine if we use CPU or GPU
        param_key = self._get_platform_param_key(use_gpu)
        encoder_params = params.get(param_key)

        # Store the specific parameters for metadata logging
        self._encoding_params = {
            'format': 'ffmpeg_video',
            'encoder_profile': param_key,
            'encoder_options': encoder_params
        }

        if not encoder_params:
            raise ValueError(f"FFmpeg parameters for '{param_key}' not found in config.")

        format_map = {
            # 8-bit Monochrome and Bayer
            'Mono8': 'gray',
            'BayerRG8': 'bayer_rggr8',
            'BayerGR8': 'bayer_grbg8',
            'BayerGB8': 'bayer_gbrg8',
            'BayerBG8': 'bayer_bggr8',
            # 8-bit Color
            'RGB8': 'rgb24',
            'BGR8': 'bgr24',
            # High Bit-Depth Monochrome
            'Mono10': 'gray10le',
            'Mono12': 'gray12le',
            'Mono16': 'gray16le',
        }
        input_pixel_format = format_map.get(self.pixel_format)
        if not input_pixel_format:
            raise ValueError(f"Unsupported pixel_format '{self.pixel_format}' for FFmpegWriter.")

        # Build the FFmpeg command
        input_args = (
            f"-y -s {self.width}x{self.height} -f rawvideo "
            f"-framerate {self.framerate:.3f} -pix_fmt {input_pixel_format} -i pipe:0"
        )
        command = f"{self.ffmpeg_path} -hide_banner {input_args} {encoder_params} {shlex.quote(str(filepath))}"
        print(f"[DEBUG] FFmpeg command: {command}")

        # Start the subprocess, redirecting stdout/stderr to prevent console spam
        self.proc = subprocess.Popen(
            shlex.split(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
            # stdout=subprocess.PIPE,         # for debug
            # stderr=subprocess.PIPE          # for debug
        )

    def _get_platform_param_key(self, use_gpu: bool) -> str:
        """ Determines the correct key for FFmpeg params based on OS and GPU flag """

        if not use_gpu:
            return 'cpu'

        # TODO: support AMD GPUs

        system = platform.system()
        if system == 'Windows':
            return 'gpu_nvenc'  # Assuming NVIDIA on Windows
        elif system == 'Linux':
            return 'gpu_nvenc'  # Assuming NVIDIA on Linux
        elif system == 'Darwin':
            return 'gpu_videotoolbox'
        else:
            print(f"[WARN] Unsupported OS '{system}' for GPU encoding. Falling back to CPU.")
            return 'cpu'

    def _write_frame(self, frame: np.ndarray, metadata: Dict[str, Any]):

        if self.proc and self.proc.stdin:
            try:
                # Write the raw bytes of the frame to the stdin of the FFmpeg process
                self.proc.stdin.write(frame.tobytes())

            except (IOError, BrokenPipeError) as e:

                # This can happen if FFmpeg closes unexpectedly
                print(f"[ERROR] Failed to write to FFmpeg process: {e}")

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