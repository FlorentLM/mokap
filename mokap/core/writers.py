import subprocess
import shlex
import platform
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

import numpy as np
from PIL import Image


class FrameWriter(ABC):
    """
    Abstract base class for writing frames to disk. It defines the common
    interface for all writer types (e.g., video, image sequence)
    """

    def __init__(self, filepath: Path, width: int, height: int, framerate: float):
        self.filepath = filepath
        self.width = width
        self.height = height
        self.framerate = framerate
        self.frame_count = 0

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
        self.quality = quality

        # Ensure the output directory exists
        self.filepath.mkdir(parents=True, exist_ok=True)

    def _write_frame(self, frame: np.ndarray, metadata: Dict[str, Any]):

        image_path = self.filepath / f"{str(self.frame_count).zfill(9)}.{self.ext}"

        try:
            # Create a PIL Image from the raw buffer without copying data
            # TODO: This assumes an 8-bit grayscale but it should also work for colour cameras!!!
            # TODO: Maybe we want to use cv2.imwrite instead
            img = Image.frombuffer("L", (self.width, self.height), frame, 'raw', "L", 0, 1)

            # Apply format-specific save options
            if self.ext == 'png':
                # PNG compress_level: 0=none, 1=fastest, 9=best
                # We map a 0-100 quality scale to the 9-1 range
                compress_level = int(np.interp(self.quality, [0, 100], [9, 1]))
                img.save(image_path, compress_level=compress_level, optimize=False)

            elif self.ext in ('jpg', 'jpeg'):
                img.save(image_path, quality=self.quality, subsampling='4:2:0')

            elif self.ext in ('tif', 'tiff'):
                if self.quality >= 99:
                    img.save(image_path, compression=None)

                else:
                    # TIFF with JPEG compression
                    img.save(image_path, compression='jpeg', quality=self.quality)

            else:
                # For BMP and other simple formats
                img.save(image_path)

        except Exception as e:
            # Catch any potential PIL or OS error during the save operation
            print(f"[ERROR] Failed to save frame {self.frame_count} to {image_path}: {e}")

            # re-raising an error to prevent the frame_count from being incremented in the parent write() method
            raise IOError(f"Disk write failed for frame {self.frame_count}") from e

    def close(self):
        # For image sequences, there's nothing to finalize
        pass


class FFmpegWriter(FrameWriter):
    """
    Writes frames to a video file by piping them to an FFmpeg subprocess
    """

    def __init__(self, filepath: Path, ffmpeg_path: str, params: Dict, use_gpu: bool, pixel_format: str, **kwargs):
        super().__init__(filepath, **kwargs)

        self.proc: subprocess.Popen = None

        # Determine if we use CPU or GPU
        param_key = self._get_platform_param_key(use_gpu)
        encoder_params = params.get(param_key)

        if not encoder_params:
            raise ValueError(f"FFmpeg parameters for '{param_key}' not found in config.")

        format_map = {
            'Mono8': 'gray8',
            'Mono10': 'gray10le',
            'Mono12': 'gray12le',
            'BayerBG8': 'bayer_bggr8',
            'BayerRG8': 'bayer_rggr8',
        }
        # Use the provided format (fallback to gray8)
        input_pixel_format = format_map.get(pixel_format, 'gray8')

        # Build the FFmpeg command
        # TODO: color formats for video
        input_args = (
            f"-y -s {self.width}x{self.height} -f rawvideo "
            f"-framerate {self.framerate:.3f} -pix_fmt {input_pixel_format} -i pipe:0"
        )

        command = f"{ffmpeg_path} -hide_banner {input_args} {encoder_params} {shlex.quote(str(filepath))}"

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
        if self.proc:
            if self.proc.stdin:
                # Gracefully close the input pipe and wait for FFmpeg to finish encoding
                self.proc.stdin.flush()
                self.proc.stdin.close()

            # Wait for the process to terminate - SUPER IMPORTANT
            self.proc.wait(timeout=10)
        self.proc = None