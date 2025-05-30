from dataclasses import dataclass
from copy import deepcopy
import cv2
import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=5)
from numpy.typing import ArrayLike
from typing import Optional
from pathlib import Path
from mokap.utils import generate_charuco, generate_board_svg
from mokap.utils import fileio


@dataclass
class BoardParams:
    rows: int
    cols: int
    square_length: float    # in real-life units (e.g. mm)
    markers_size: int = 4
    margin: int = 1

    def to_opencv(self) -> cv2.aruco.CharucoBoard:
        return generate_charuco(self.rows, self.cols, self.square_length, self.markers_size, self.margin)

    def to_file(self, file_path: Path | str, multi_size=False, factor=2.0, dpi=1200) -> None:
        generate_board_svg(self.to_opencv(), file_path, multi_size=multi_size, factor=factor, dpi=dpi)

    def object_points(self) -> np.ndarray:
        """
        Returns the theoretical 3D locations (X, Y, Z=0) of the inner chessboard corners
        """
        xs = np.arange(1, self.cols) * self.square_length
        ys = np.arange(1, self.rows) * self.square_length

        xx, yy = np.meshgrid(xs, ys)

        return np.stack((xx, yy, np.zeros_like(xx)), axis=-1).reshape(-1, 3)

    def copy(self):
        return deepcopy(self)

##

@dataclass
class ErrorsPayload:
    errors: Optional[ArrayLike] = None

@dataclass
class OriginCameraPayload:
    camera_name: str

@dataclass
class PosePayload:
    """
    Monocular estimation of the extrinsics (camera pose)
    """
    frame: int
    rvec: np.ndarray
    tvec: np.ndarray

@dataclass
class DetectionPayload:
    """
    Monocular detection of points 2D
    """
    frame: int
    points2D: np.ndarray
    pointsIDs: np.ndarray

@dataclass
class IntrinsicsPayload:
    """
    Monocular intrinsics parameters
    """
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    errors: Optional[ArrayLike] = None

    @classmethod
    def from_file(cls, filepath, camera_name: Optional[str] = None):
        params = fileio.read_parameters(filepath, camera_name)
        return cls(camera_matrix=params['camera_matrix'], dist_coeffs=params['dist_coeffs'], errors=params.get('errors'))

@dataclass
class ExtrinsicsPayload:
    """
    Multiview extrinsics parameters (global arrangement)
    """
    rvec: np.ndarray
    tvec: np.ndarray

    @classmethod
    def from_file(cls, filepath, camera_name: Optional[str] = None):
        params = fileio.read_parameters(filepath, camera_name)
        return cls(rvec=params['rvec'], tvec=params['tvec'])

@dataclass
class CalibrationData:
    """
    Encapsulation of a payload with the camera name
    """
    camera_name: str
    payload: IntrinsicsPayload | ExtrinsicsPayload | DetectionPayload | PosePayload | ErrorsPayload | OriginCameraPayload

    def to_file(self, filepath):
        if isinstance(self.payload, IntrinsicsPayload):
            fileio.write_intrinsics(filepath, self.camera_name, self.payload.camera_matrix, self.payload.dist_coeffs, self.payload.errors)
        elif isinstance(self.payload, ExtrinsicsPayload):
            fileio.write_extrinsics(filepath, self.camera_name, self.payload.rvec, self.payload.tvec)
