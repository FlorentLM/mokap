import logging
from typing import Tuple, Union, Optional
import cv2
import numpy as np
from numpy.typing import ArrayLike
from mokap.utils.datatypes import CharucoBoard, ChessBoard

logger = logging.getLogger(__name__)


class ChessboardDetector:
    """ Detects a standard chessboard """

    def __init__(self, board: ChessBoard, downsample_size: int = 480):

        self._board = board

        # Maximum number of board points and distances
        self._nb_points = board.nb_points

        # OpenCV expects this tuple
        self._n_inner_size: Tuple[int, int] = (board.cols - 1, board.rows - 1)

        # chessboard detections always returns either all or no points, so we fix the points_ids once
        self._points2d_ids = np.arange(np.prod(self._n_inner_size), dtype=np.int32)

        # classic chessboard detection is much slower than charuco so we kinda have to downsample
        self._downsample_size = downsample_size

        # Detection flags
        self._detection_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                                 cv2.CALIB_CB_NORMALIZE_IMAGE |
                                 cv2.CALIB_CB_FAST_CHECK |     # quickly dismisses frames with no board in view
                                 cv2.CALIB_CB_FILTER_QUADS)    # pre‐filters candidate quads before full points grouping

        # Criteria for subpixel refinement
        self._win_size: Tuple[int, int] = (11, 11)
        self._zero_zone: Tuple[int, int] = (-1, -1)
        self._subpix_criteria: Tuple[int, int, float] = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            20,  # max iterations
            0.1  # epsilon is the minimum allowed movement (in pixels) of a point from one iteration to the next
        )

    def detect(self,
               frame: ArrayLike,
                refine_points: bool = False,
               **kwargs
               ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Downsample image if needed - otherwise chessboard detection is way too slow
        h_full, w_full = frame.shape[:2]
        max_dim = max(h_full, w_full)
        scale = self._downsample_size / float(max_dim) if max_dim > self._downsample_size else 1.0

        if scale < 1.0:
            new_w = int(w_full * scale)
            new_h = int(h_full * scale)
            frame_small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_small = frame

        found, chessboard_points = cv2.findChessboardCorners(
            frame_small if scale < 1.0 else frame,
            self._n_inner_size,
            flags=self._detection_flags
        )

        if not found:
            return None, None

        chessboard_points = chessboard_points.astype(np.float32) / scale

        if refine_points:
            try:
                chessboard_points = cv2.cornerSubPix(frame,
                                                     chessboard_points,
                                                     winSize=self._win_size,
                                                     zeroZone=self._zero_zone,
                                                     criteria=self._subpix_criteria)
            except cv2.error as e:
                logger.error(e)

        # we want (N, 2)
        points2d_coords = chessboard_points.reshape(-1, 2).astype(np.float32)

        return points2d_coords, self._points2d_ids

    @property
    def board(self) -> ChessBoard:
        return self._board


class CharucoDetector(ChessboardDetector):
    """
    Detects a Charuco board
    """

    def __init__(self, board: CharucoBoard):
        super().__init__(board)

        self._board = board
        self._board_opencv = board.to_opencv()

        # we need to keep references to the OpenCV board object and detector parameters
        self._detector_parameters = cv2.aruco.DetectorParameters()
        self._detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(self._board_opencv.getDictionary(),
                                                 detectorParams=self._detector_parameters)

        # Criteria for subpixel refinement
        self._win_size: Tuple[int, int] = (11, 11)
        self._zero_zone: Tuple[int, int] = (-1, -1)
        self._subpix_criteria: Tuple[int, int, float] = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            20,  # max iterations
            0.1  # epsilon is the minimum allowed movement (in pixels) of a point from one iteration to the next
        )

    def detect(self,
               frame: ArrayLike,
               camera_matrix: Optional[ArrayLike] = None,
               dist_coeffs: Optional[ArrayLike] = None,
               refine_markers: bool = True,
               refine_points: bool = False
               ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detect and refine aruco markers
        markers_coords, marker_ids, rejected = self._detector.detectMarkers(frame)

        if marker_ids is None or len(marker_ids) == 0:
            return None, None

        if refine_markers:
            markers_coords, marker_ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(
                image=frame,
                board=self._board_opencv,
                detectedCorners=markers_coords,     # Input/Output /!\
                detectedIds=marker_ids,             # Input/Output /!\
                rejectedCorners=rejected,           # Input/Output /!\
                parameters=self._detector_parameters,
                # Known bug with refineDetectedMarkers, fixed in OpenCV 4.9: https://github.com/opencv/opencv/pull/24139
                cameraMatrix=camera_matrix if cv2.getVersionMajor() >= 4 and cv2.getVersionMinor() >= 9 else None,
                distCoeffs=dist_coeffs)

        if marker_ids is None or len(marker_ids) == 0:
            return None, None

        # If any marker has been detected, try to detect the chessboard points
        nb_chessboard_points, chessboard_points, chessboard_points_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=markers_coords,
            markerIds=marker_ids,
            image=frame,
            board=self._board_opencv,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            minMarkers=1)

        if chessboard_points is None or nb_chessboard_points <= 1:
            return None, None

        if refine_points and chessboard_points is not None:
            try:
                # Refine the chessboard corners
                chessboard_points = cv2.cornerSubPix(frame,
                                                     chessboard_points,
                                                     winSize=self._win_size,
                                                     zeroZone=self._zero_zone,
                                                     criteria=self._subpix_criteria)
            except cv2.error as e:
                logger.error(e)

        return chessboard_points[:, 0, :], chessboard_points_ids[:, 0]

    @property
    def board(self) -> ChessBoard:
        return self._board