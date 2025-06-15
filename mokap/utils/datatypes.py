from dataclasses import dataclass
from copy import deepcopy
import cv2
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Union
from pathlib import Path
from mokap.utils.fileio import generate_board_svg
from mokap.utils import fileio



class ChessBoard:
    def __init__(self,
                 rows: int,
                 cols: int,
                 square_length: float):  # in real-life units (e.g. mm)

        self.rows = rows
        self.cols = cols
        self.square_length = square_length
        self.kind = 'chessboard'

    def object_points(self) -> np.ndarray:
        """
        Returns the theoretical 3D locations (X, Y, Z=0) of the inner chessboard corners
        """
        xs = np.arange(1, self.cols) * self.square_length
        ys = np.arange(1, self.rows) * self.square_length

        xx, yy = np.meshgrid(xs, ys)

        return np.stack((xx, yy, np.zeros_like(xx)), axis=-1).reshape(-1, 3).astype(np.float32)

    def to_opencv(self):
        raise NotImplementedError()

    def to_image(self, imsize_hw):
        w, h = imsize_hw    # yes it's flipped...
        yy = (np.arange(h) * self.rows) // h
        xx = (np.arange(w) * self.cols) // w
        grid_y = yy[:, None]
        grid_x = xx[None, :]
        checker = ((grid_x + grid_y) % 2)
        return np.where(checker == 0, 0, 255).astype(np.uint8)

    def to_file(self, file_path: Union[Path, str], multi_size=False, factor=2.0, dpi=1200):
        generate_board_svg(self, file_path, multi_size=multi_size, factor=factor, dpi=dpi)

    def copy(self):
        return deepcopy(self)


class CharucoBoard(ChessBoard):
    def __init__(self,
                 rows:          int,
                 cols:          int,
                 square_length: float,      # in real-life units (e.g. mm)
                 markers_size:  int = 4,
                 margin:        int = 1,
                 padding:       int = 1,    # Black margin inside the markers (i.e. OpenCV's borderBits)
                 ):
        super().__init__(rows, cols, square_length)

        self.markers_size = markers_size
        self.margin = margin
        self.kind = 'charuco'

        self.all_dict_sizes = [50, 100, 250, 1000]
        self.padding = padding

        mk_l_bits = self.markers_size + padding * 2
        sq_l_bits = mk_l_bits + self.margin * 2

        self.marker_length = mk_l_bits / sq_l_bits * self.square_length

        dict_size = next(s for s in self.all_dict_sizes if s >= self.rows * self.cols)
        self.dict_name = f'DICT_{self.markers_size}X{self.markers_size}_{dict_size}'

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dict_name))

    def to_opencv(self) -> cv2.aruco.CharucoBoard:
        return cv2.aruco.CharucoBoard((self.cols, self.rows),  # number of chessboard squares in x and y directions
                                      self.square_length,  # chessboard square side length (real world units)
                                      self.marker_length,  # marker side length (same unit as square_length)
                                      self.aruco_dict)

    def to_image(self, imsize_hw):
        return self.to_opencv().generateImage(imsize_hw)

##

@dataclass
class ErrorsPayload:
    errors: Optional[ArrayLike] = None

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

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    errors: Optional[ArrayLike] = None

    @classmethod
    def from_file(cls, filepath, camera_name: Optional[str] = None):
        params = fileio.read_parameters(filepath, camera_name)
        return cls(camera_matrix=params['camera_matrix'], dist_coeffs=params['dist_coeffs'], errors=params.get('errors'))

@dataclass
class ExtrinsicsPayload:

    rvec: np.ndarray
    tvec: np.ndarray
    error: Optional[float] = None

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
    payload: IntrinsicsPayload | ExtrinsicsPayload | DetectionPayload | ErrorsPayload

    def to_file(self, filepath):
        if isinstance(self.payload, IntrinsicsPayload):
            fileio.write_intrinsics(filepath, self.camera_name, self.payload.camera_matrix, self.payload.dist_coeffs, self.payload.errors)
        elif isinstance(self.payload, ExtrinsicsPayload):
            fileio.write_extrinsics(filepath, self.camera_name, self.payload.rvec, self.payload.tvec)
