from dataclasses import dataclass, field
from copy import deepcopy
import cv2
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Union, Literal
from pathlib import Path
from mokap.utils.fileio import generate_board_svg
from mokap.utils import fileio

DistortionModel = Literal['none', 'simple', 'standard', 'full', 'rational']


@dataclass
class CalibrateCameraResult:
    """ A container for the results of an intrinsic camera calibration """
    success: bool
    rms_error: float = np.inf
    K_new: Optional[np.ndarray] = None
    D_new: Optional[np.ndarray] = None
    rvecs: Optional[np.ndarray] = None
    tvecs: Optional[np.ndarray] = None
    std_devs_intrinsics: Optional[np.ndarray] = None
    error_message: str = ""
    # field avoids including the large error array in the __repr__
    per_view_errors: Optional[np.ndarray] = field(default=None, repr=False)


class ChessBoard:

    type = 'ChessBoard'

    def __init__(self,
                 rows: int,
                 cols: int,
                 square_length: float):  # in real-life units (e.g. mm)

        if rows < 2 or cols < 2:
            raise ValueError(f"{self.type.title()} must be least 2x2 squares.")

        self.rows = rows
        self.cols = cols
        self.square_length = square_length

        # Cache object points in 3D
        xs = np.arange(1, self.cols) * self.square_length
        ys = np.arange(1, self.rows) * self.square_length
        xx, yy = np.meshgrid(xs, ys)
        self._object_points = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).reshape(-1, 3).astype(np.float32)

        self._N = self._object_points.shape[0]

        # Cache board outer corners in 3D
        self._corners = np.array([
            [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]
        ], dtype=np.float32) * [self.cols, self.rows, 0] * self.square_length

    @property
    def object_points(self) -> np.ndarray:
        """ Returns the theoretical 3D locations (X, Y, Z=0) of the inner points """
        return self._object_points

    @property
    def corner_points(self) -> np.ndarray:
        """ Returns the theoretical 3D locations (X, Y, Z=0) of the outer corners """
        return self._corners

    @property
    def nb_points(self) -> int:
        return self._N

    def to_opencv(self):
        raise NotImplementedError()

    def to_image(self, square_size_px: Optional[int] = None):
        f = square_size_px if square_size_px else 1
        return ((np.indices((self.rows * f, self.cols * f)) // f).sum(axis=0) % 2).astype(np.uint8) * 255

    def to_file(self, file_path: Union[Path, str], multi_size=False, factor=2.0, dpi=1200):
        generate_board_svg(self, file_path, multi_size=multi_size, factor=factor, dpi=dpi)

    def copy(self):
        return deepcopy(self)


class CharucoBoard(ChessBoard):

    type = 'CharucoBoard'

    def __init__(self,
                 rows:          int,
                 cols:          int,
                 square_length: float,      # in real-life units (e.g. mm)
                 markers_size:  int = 4,
                 margin:        int = 1,
                 padding:       int = 1,    # Black margin inside the markers (i.e. OpenCV's borderBits)
                 ):
        super().__init__(rows, cols, square_length)

        self.markers_size = max(1, markers_size)
        self.margin = max(0, margin)
        self.type = 'charuco'

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

    def to_image(self, square_size_px: Optional[int] = None):
        side_pixels = self.markers_size + 2 * self.padding + 2 * self.margin
        w = self.cols * side_pixels + 2 * self.margin
        h = self.rows * side_pixels + 2 * self.margin
        f = square_size_px if square_size_px else 1
        return self.to_opencv().generateImage((w * f, h * f), marginSize=self.margin, borderBits=self.padding)

##

# TODO: merge some of these payloads

@dataclass
class ReprojectionPayload:
    """ Payload for reprojected points for visualization """
    all_points_2d: np.ndarray       # Reprojection of all board points
    detected_ids: np.ndarray        # IDs of currently detected points

@dataclass
class CoveragePayload:
    """ Payload for general state information for UI text fields """
    grid: np.ndarray  # The boolean cumulative grid (H, W)
    coverage_percent: float
    nb_samples: int
    total_points: int

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
    payload: IntrinsicsPayload | ExtrinsicsPayload | DetectionPayload | ErrorsPayload | ReprojectionPayload | CoveragePayload

    def to_file(self, filepath):
        if isinstance(self.payload, IntrinsicsPayload):
            fileio.write_intrinsics(filepath, self.camera_name, self.payload.camera_matrix, self.payload.dist_coeffs, self.payload.errors)
        elif isinstance(self.payload, ExtrinsicsPayload):
            fileio.write_extrinsics(filepath, self.camera_name, self.payload.rvec, self.payload.tvec)
