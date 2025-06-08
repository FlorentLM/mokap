import numpy as np
from numpy.typing import ArrayLike
np.set_printoptions(precision=3, suppress=True, threshold=5)
import cv2
import scipy.stats as stats
from scipy.spatial.distance import cdist
import jax
import jax.numpy as jnp
from collections import deque
from typing import Tuple, Optional, Iterable, Union
from mokap.utils.datatypes import ChessBoard, CharucoBoard, DetectionPayload
from mokap.calibration import monocular_2, bundle_adjustment_2
from mokap.utils import geometry_jax, geometry_2, pad_dist_coeffs, outliers_rejection


def _maybe_put(x):
    return jax.device_put(x) if x is not None else None


class ChessboardDetector:
    """
    Detects a standard chessboard
    """

    def __init__(self,
                 board: CharucoBoard,
                 downsample_size: int = 480):

        self._n_cols, self._n_rows = board.cols, board.rows

        if self._n_cols < 2 or self._n_rows < 2:
            raise ValueError("BoardParams must have at least 2x2 squares for a valid chessboard.")

        # Create 3D coordinates for board corners (in board-centric coordinates)
        self._board_points_3d = board.object_points()
        self._board_corners_3d = (
                np.array([[0, 0, 0],
                          [0, 1, 0],
                          [1, 1, 0],
                          [1, 0, 0]], dtype=np.float32) * [self._n_cols, self._n_rows, 0] * board.square_length)

        # Maximum number of board points and distances
        self._total_points: int = len(self._board_points_3d)
        self._total_distances: int = int((self._total_points * (self._total_points - 1)) / 2.0)

        # OpenCV expects this tuple
        self._n_inner_size: Tuple[int, int] = (self._n_cols - 1, self._n_rows - 1)

        # chessboard detections always returns either all or no points, so we fix the points_ids once
        self._points2d_ids = np.arange(np.prod(self._n_inner_size), dtype=np.int32)

        # classic chessboard detection is much slower than charuco so we kinda have to downsample
        self._downsample_size = downsample_size

        # Detection flags
        self._detection_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                                 cv2.CALIB_CB_NORMALIZE_IMAGE |
                                 cv2.CALIB_CB_FAST_CHECK |  # quickly dismisses frames with no board in view
                                 cv2.CALIB_CB_FILTER_QUADS)  # pre‐filters candidate quads before full points grouping

        # Criteria for subpixel refinement
        self._win_size: Tuple[int, int] = (11, 11)
        self._zero_zone: Tuple[int, int] = (-1, -1)
        self._subpix_criteria: Tuple[int, int, float] = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            20,  # max iterations
            0.1  # epsilon is the minimum allowed movement (in pixels) of a point from one iteration to the next
        )

    @property
    def points3d(self) -> ArrayLike:
        """ Returns the (board-centric) coordinates of the chessboard points in 3D """
        return self._board_points_3d

    @property
    def corners3d(self) -> ArrayLike:
        """ Returns the (board-centric) coordinates of the chessboard outer corners in 3D """
        return self._board_corners_3d

    @property
    def nb_points(self) -> int:
        return self._total_points

    @property
    def board_dims(self) -> Tuple[int, int]:
        return self._n_cols, self._n_rows

    def detect(self,
               frame: ArrayLike,
               refine_points: bool = False
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
                print(e)

        # we want (N, 2)
        points2d_coords = chessboard_points.reshape(-1, 2).astype(np.float32)

        return points2d_coords, self._points2d_ids


class CharucoDetector(ChessboardDetector):
    """
    Detects a Charuco board
    """

    def __init__(self, board_params: CharucoBoard):
        super().__init__(board_params)

        # we need to keep references to the OpenCV board object and detector parameters
        self.board = board_params.to_opencv()
        self._detector_parameters = cv2.aruco.DetectorParameters()
        self._detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(self.board.getDictionary(), detectorParams=self._detector_parameters)

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

        if refine_markers:
            markers_coords, marker_ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(
                image=frame,
                board=self.board,
                detectedCorners=markers_coords,  # Input/Output /!\
                detectedIds=marker_ids,  # Input/Output /!\
                rejectedCorners=rejected,  # Input/Output /!\
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
            board=self.board,
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
                print(e)

        return chessboard_points[:, 0, :], chessboard_points_ids[:, 0]


class MonocularCalibrationTool:
    """
    This object is stateful for the intrinsics *only*
    """

    def __init__(self,
                 board_params: Union[ChessBoard, CharucoBoard],
                 imsize_hw: Optional[Iterable[int]] = None,  # OpenCV order (height, width)
                 min_stack: int = 15,
                 max_stack: int = 100,
                 focal_mm: Optional[int] = None,
                 sensor_size: Optional[Union[Tuple[float], str]] = None,
                 verbose: bool = False):

        if type(board_params) is ChessBoard:
            self.dt: ChessboardDetector = ChessboardDetector(board_params)
        else:
            self.dt: CharucoDetector = CharucoDetector(board_params)

        self._verbose: bool = verbose

        # TODO: these 3 alternatives should be selectable from the config file
        self._min_pts: int = 4  # SQPNP method needs at least 3 points but in practice 4 is safer
        # self._min_pts: int = 4    # ITERATIVE method needs at least 4 points
        # self._min_pts: int = 6    # DLT algorithm needs at least 6 points for pose estimation

        # TODO: grid parameters should be configurable from the config file
        self._nb_grid_cells: int = 15
        self._cells_gamma: float = 2.0
        self._min_cells_weight: float = 0.25  # cells at centre get ~ min weight and cells at the edge get ~ 1.0

        self.h, self.w = None, None
        self._sensor_size: Union[ArrayLike, None] = None

        self._points2d = None
        self._points_ids = None

        self._points3d = np.asarray(self.dt.points3d)
        self._points3d_j: jnp.ndarray = jnp.asarray(self._points3d)
        self._cornersd3d_j: jnp.ndarray = jnp.asarray(self.dt.corners3d)

        self._zero_coeffs = jnp.zeros(8, dtype=jnp.float32)

        self._th_camera_matrix_j: Union[jnp.ndarray, None] = None

        self._camera_matrix: Union[np.ndarray, None] = None
        self._dist_coeffs: Union[np.ndarray, None] = None

        self._camera_matrix_j: Union[jnp.ndarray, None] = None
        self._dist_coeffs_j: Union[jnp.ndarray, None] = None

        self._rvec: Union[np.ndarray, None] = None
        self._tvec: Union[np.ndarray, None] = None

        self._rvec_j: Union[jnp.ndarray, None] = None
        self._tvec_j: Union[jnp.ndarray, None] = None

        if imsize_hw is not None:
            self._update_imsize(imsize_hw)
        # otherwise arrays will be initialised on the first detection

        # Process sensor size input
        if isinstance(sensor_size, str):
            self._sensor_size = monocular_2.SENSOR_SIZES.get(f'''{sensor_size.strip('"')}"''', None)
        elif isinstance(sensor_size, (tuple, list, set, np.ndarray)) and len(sensor_size) == 2:
            self._sensor_size = sensor_size

        # compute theoretical camera matrix if possible
        # (this allows to fix the fx/fy ratio and helps the first estimation)
        if None not in (focal_mm, self._sensor_size, self.h, self.w):
            self._th_camera_matrix_j = _maybe_put(monocular_2.estimate_camera_matrix(
                focal_mm,
                self._sensor_size,
                (self.w, self.h))
            )

            self._camera_matrix_j = _maybe_put(self._th_camera_matrix_j.copy())
            self._dist_coeffs_j = _maybe_put(np.zeros(8, dtype=np.float32))

        # Samples stack
        self._min_stack: int = min_stack
        self._max_stack: int = max_stack
        self.stack_points2d: deque = deque(maxlen=self._max_stack)
        self.stack_points_ids: deque = deque(maxlen=self._max_stack)

        # Error metrics
        self._intrinsics_errors: ArrayLike = np.array([np.inf])
        self._pose_error: float = np.inf

        # TODO: visualisation scale should be inferred from image size?
        self.set_visualisation_scale(scale=1)

    def _update_imsize(self, imsize_hw):
        """ Internal method to set or update arrays related to image size """

        self.h, self.w = imsize_hw[:2]

        # needs to be color frame for annotations
        self._frame_buf = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # TODO: Error normalisation factor: we want to use image diagonal to normalise the errors
        self._err_norm = 1

        self._grid_shape = np.array([self._nb_grid_cells, int(np.round((self.w / self.h) * self._nb_grid_cells))],
                                    dtype=np.uint32)
        self._cumul_grid = np.zeros(self._grid_shape, dtype=bool)  # Keeps the total covered area
        self._temp_grid = np.zeros(self._grid_shape, dtype=bool)  # buffer reset at each new sample
        self._green_overlay = np.zeros_like(self._frame_buf, dtype=np.uint8)

        # we want to weight the cells based on distance from the image centre (to avoid oversampling the centre)
        grid_h, grid_w = self._grid_shape
        cell_h = self.h / grid_h
        cell_w = self.w / grid_w

        # cell centers
        xs = (np.arange(grid_w) + 0.5) * cell_w
        ys = (np.arange(grid_h) + 0.5) * cell_h
        grid_x, grid_y = np.meshgrid(xs, ys)

        # distance from centre of the image
        center_x, center_y = self.w / 2, self.h / 2
        distances = np.sqrt((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)  # max dist is from the centre to one of the corners
        norm_dist = distances / max_distance

        # cells near centre (norm_dist ~ 0) get near min weight, and cells at the edge get near 1.0
        self._grid_weights = self._min_cells_weight + (1 - self._min_cells_weight) * (norm_dist ** self._cells_gamma)

    def set_visualisation_scale(self, scale=1):
        """ Sets or updates the scale for the visualisation elements """
        self._bit_shift = 4
        self._vis_scale = scale
        self._shift_factor = 2 ** self._bit_shift
        self._draw_params = {'shift': self._bit_shift, 'lineType': cv2.LINE_AA}
        self._text_params = {'fontFace': cv2.FONT_HERSHEY_DUPLEX,
                             'fontScale': 0.8 * self._vis_scale,
                             'color': (255, 255, 255),
                             'thickness': self._vis_scale,
                             'lineType': cv2.LINE_AA}

    @property
    def detection(self) -> Tuple[ArrayLike, ArrayLike]:
        return self._points2d, self._points_ids

    @property
    def intrinsics(self) -> Tuple[ArrayLike, ArrayLike]:
        return self._camera_matrix, self._dist_coeffs

    @property
    def extrinsics(self) -> Tuple[ArrayLike, ArrayLike]:
        return self._rvec, self._tvec

    @property
    def has_detection(self) -> bool:
        return all(x is not None for x in self.detection) and len(self._points2d) >= self._min_pts

    @property
    def has_intrinsics(self) -> bool:
        return all(x is not None for x in self.intrinsics)

    @property
    def has_extrinsics(self) -> bool:
        return all(x is not None for x in self.extrinsics)

    @property
    def nb_points(self) -> int:
        return self._points2d.shape[0] if self._points2d is not None else 0

    @property
    def nb_samples(self) -> int:
        return len(self.stack_points2d)

    @property
    def coverage(self) -> float:
        return float(np.sum(self._cumul_grid) / self._cumul_grid.size) * 100

    @property
    def pose_error(self):
        return self._pose_error

    @property
    def intrinsics_errors(self) -> ArrayLike:
        return self._intrinsics_errors

    @property
    def focal(self) -> float:
        return float(self._camera_matrix[np.diag_indices(2)].sum() / 2.0) if self.has_intrinsics else 0.0

    @property
    def focal_mm(self) -> float:
        if any(x is None for x in (self._camera_matrix, self._sensor_size, self.h, self.w)):
            return 0.0
        return float(
            (self._camera_matrix[np.diag_indices(2)] * (self._sensor_size / np.array([self.w, self.h]))).sum() / 2.0)

    def set_intrinsics(self, camera_matrix: ArrayLike, dist_coeffs: ArrayLike, errors: Optional[ArrayLike] = None):

        self._camera_matrix = np.asarray(camera_matrix)
        self._camera_matrix_j = _maybe_put(self._camera_matrix)

        self._dist_coeffs = pad_dist_coeffs(dist_coeffs)
        self._dist_coeffs_j = _maybe_put(self._dist_coeffs)

        if errors is not None:
            self._intrinsics_errors = np.asarray(errors)
        else:
            self._intrinsics_errors = np.array([np.inf])

    def clear_intrinsics(self):

        if self._th_camera_matrix_j is not None:
            self._camera_matrix_j = _maybe_put(self._th_camera_matrix_j)
            self._dist_coeffs_j = _maybe_put(self._zero_coeffs)
        else:
            self._camera_matrix_j = None
            self._dist_coeffs_j = None
        self._intrinsics_errors = np.array([np.inf])

    @staticmethod
    def _check_new_errors(errors_new: ArrayLike, errors_prev: ArrayLike, p_val=0.05, confidence_lvl=0.95):

        mean_new, se_new, l_new = np.mean(errors_new), stats.sem(errors_new), len(np.atleast_1d(errors_new))
        mean_prev, se_prev, l_prev = np.mean(errors_prev), stats.sem(errors_prev), len(np.atleast_1d(errors_prev))

        # Cumulative scores and return choice
        scores = np.zeros(2, dtype=np.uint8)
        ret = (True, False)

        # T-test to compare means
        t_stat, p_value = stats.ttest_ind(errors_new, errors_prev, equal_var=False)
        if mean_new < mean_prev:
            scores[0] += 1
        else:
            scores[1] += 1

        if p_value < p_val:  # If the means are significantly different, go with the smallest one and move on
            return ret[np.argmax(scores)]

        # If the means are close to each other, keep the fight going
        if se_new < se_prev:
            scores[0] += 1
        else:
            scores[1] += 1

        ci_new = stats.t.interval(confidence_lvl, l_new - 1, loc=mean_new, scale=se_new)
        ci_prev = stats.t.interval(confidence_lvl, l_prev - 1, loc=mean_prev, scale=se_prev)
        ci_new_spread = ci_new[1] - ci_new[0]
        ci_prev_spread = ci_prev[1] - ci_prev[0]

        overlapping = not (ci_new[1] < ci_prev[0] or ci_prev[1] < ci_new[0])

        if overlapping:
            if ci_new_spread < ci_prev_spread:
                scores[0] += 1
            else:
                scores[1] += 1
        else:
            if ci_new[1] < ci_prev[0]:
                scores[0] += 1
            else:
                scores[1] += 1

        return ret[np.argmax(scores)]

    def register_sample(self):
        if self.has_detection:
            # cv2.calibrateCamera() will complain if the deques contain (N, 2) arrays, it expects (1, N, 2)
            # TODO: maybe we should not squeeze the arrays in the detector lol
            self.stack_points2d.append(self._points2d[np.newaxis, :, :])
            self.stack_points_ids.append(self._points_ids[np.newaxis, :])

    def clear_stacks(self):
        self._cumul_grid.fill(False)
        self._temp_grid.fill(False)
        self.stack_points2d.clear()
        self.stack_points_ids.clear()

    def compute_intrinsics(self, clear_stack=True, fix_aspect_ratio=True, simple_distortion=False,
                           complex_distortion=False):

        if simple_distortion and complex_distortion:
            raise AttributeError("Can't enable simple and complex distortion modes at the same time!")

        # If there is fewer than 5 images (no matter the self._min_stack value), this will NOT be enough
        if len(self.stack_points2d) < 5:
            return  # Abort but keep the stacks

        if self._camera_matrix_j is None and fix_aspect_ratio:
            print('[WARN] [MonocularCalibrationTool] No current camera matrix guess, unfixing aspect ratio.')
            fix_aspect_ratio = False

        calib_flags = 0
        if fix_aspect_ratio:
            calib_flags |= cv2.CALIB_FIX_ASPECT_RATIO  # This locks the ratio of fx and fy
        if simple_distortion or not self.has_intrinsics:  # The first iteration will use the simple model - this helps
            calib_flags |= cv2.CALIB_FIX_K3
        if complex_distortion:
            calib_flags |= cv2.CALIB_RATIONAL_MODEL
        if self.has_intrinsics:
            calib_flags |= cv2.CALIB_USE_INTRINSIC_GUESS  # Important, otherwise it ignores the existing intrinsics

        # calib_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

        # We need to copy to a new array, because OpenCV uses these as Input/Output buffers
        if self.has_intrinsics:
            current_camera_matrix = np.copy(self._camera_matrix)
            current_dist_coeffs = np.copy(self._dist_coeffs)
        else:
            current_camera_matrix = None
            current_dist_coeffs = None

        try:  # compute intrinsics using all the frames in the stacks
            if type(self.dt) is ChessboardDetector:

                object_points_list = [self.dt.points3d.astype(np.float32)] * len(self.stack_points2d)

                (global_intr_error,
                 new_camera_matrix, new_dist_coeffs,
                 stack_rvecs, stack_tvecs,
                 std_intrinsics, std_extrinsics,
                 stack_intr_errors) = cv2.calibrateCameraExtended(

                    objectPoints=object_points_list,
                    imagePoints=self.stack_points2d,
                    imageSize=(self.w, self.h),
                    cameraMatrix=current_camera_matrix,  # Input/Output /!\
                    distCoeffs=current_dist_coeffs,  # Input/Output /!\
                    flags=calib_flags
                )

            else:
                calib_flags |= cv2.CALIB_USE_LU
                # calib_flags |= cv2.CALIB_USE_QR

                (global_intr_error,
                 new_camera_matrix, new_dist_coeffs,
                 stack_rvecs, stack_tvecs,
                 std_intrinsics, std_extrinsics,
                 stack_intr_errors) = cv2.aruco.calibrateCameraCharucoExtended(

                    charucoCorners=self.stack_points2d,
                    charucoIds=self.stack_points_ids,
                    board=self.dt.board,
                    imageSize=(self.w, self.h),
                    cameraMatrix=current_camera_matrix,  # Input/Output /!\
                    distCoeffs=current_dist_coeffs,  # Input/Output /!\
                    flags=calib_flags
                )

            # std_cam_mat, std_dist_coeffs = np.split(std_intrinsics.squeeze(), [4])
            # std_rvecs, std_tvecs =  std_extrinsics.reshape(2, -1, 3)
            # TODO: Use these std values in the plot - or to decide if the new round of calibrateCamera is good or not?

            new_dist_coeffs = new_dist_coeffs.squeeze()
            # stack_rvecs = np.stack(stack_rvecs).squeeze()       # Unused for now
            # stack_tvecs = np.stack(stack_tvecs).squeeze()       # Unused for now
            stack_intr_errors = stack_intr_errors.squeeze() / self._err_norm  # TODO: Normalise errors on image diagonal

            # Note:
            # ------
            #
            # The per-view reprojection error as returned by calibrateCamera() is:
            #   the square root of the sum of the 2 means in x and y of the squared diff
            #       np.sqrt(np.sum(np.mean(sq_diff, axis=0)))
            #
            # This is NOT the same as the per-view reprojection error as returned by solvePnP():
            #   this one is the square root of the mean of the squared diff over both x and y
            #        np.sqrt(np.mean(sq_diff, axis=(0, 1)))
            #
            # ...in other words, the first one is larger by a factor sqrt(2)
            #
            # ----------------------------------------------
            #
            # The global calibration error in calibrateCamera() is:
            #       np.sqrt(np.sum([sq_diff for view in stack])) / np.sum([len(view) for view in stack]))

            # The following should never happen; if any happens, then we trash the stack and abort
            if (new_camera_matrix < 0).any() or (new_camera_matrix[:2, 2] >= np.array([self.w, self.h])).any():
                self.clear_stacks()
                return

            # store intrinsics if it is the very first estimation
            if not self.has_intrinsics or np.inf in self._intrinsics_errors:
                # TODO: if we loaded intrinsics without errors values, then they default to np.inf so are always overwritten here...

                self._camera_matrix = new_camera_matrix
                self._dist_coeffs = new_dist_coeffs
                self._camera_matrix_j = _maybe_put(self._camera_matrix)
                self._dist_coeffs_j = _maybe_put(self._dist_coeffs)

                self._intrinsics_errors = stack_intr_errors

                if self._verbose:
                    print(f"[INFO] [MonocularCalibrationTool] Computed intrinsics")

            # or update them if this stack's errors are better
            elif self._check_new_errors(stack_intr_errors, self._intrinsics_errors):

                self._camera_matrix = new_camera_matrix
                self._dist_coeffs = new_dist_coeffs
                self._camera_matrix_j = _maybe_put(self._camera_matrix)
                self._dist_coeffs_j = _maybe_put(self._dist_coeffs)

                self._intrinsics_errors = stack_intr_errors

                if self._verbose:
                    print(f"[INFO] [MonocularCalibrationTool] Updated intrinsics")

        except cv2.error as e:
            print(f"[WARN] [MonocularCalibrationTool] OpenCV Error in calibrateCamera:\n\n{e}")

        if clear_stack:
            self.clear_stacks()

    def compute_extrinsics(self, refine=True):

        # We need a detection and intrinsics to compute the extrinsics
        if not self.has_detection or not self.has_intrinsics:
            self._rvec, self._tvec = None, None
            self._rvec_j, self._tvec_j = None, None
            self._pose_error = np.inf
            return

        if type(self.dt) is CharucoDetector:
            # TODO: Check collinearity for classic chessboards too?

            # If the points are collinear, extrinsics estimation is garbage, so abort
            if cv2.aruco.testCharucoCornersCollinear(self.dt.board, self._points_ids):
                self._rvec, self._tvec = None, None
                self._rvec_j, self._tvec_j = None, None
                self._pose_error = np.inf
                return

        # pnp_flags = cv2.SOLVEPNP_ITERATIVE

        # SQPNP:
        # - "A Consistently Fast and Globally Optimal Solution to the Perspective-n-Point Problem", 2020,
        #   George Terzakis and Manolis Lourakis, 10.1007/978-3-030-58452-8_28
        pnp_flags = cv2.SOLVEPNP_SQPNP

        try:
            nb_solutions, rvecs, tvecs, solutions_errors = cv2.solvePnPGeneric(
                self._points3d[self._points_ids],
                self._points2d,
                self._camera_matrix,
                self._dist_coeffs,
                flags=pnp_flags
            )

        except cv2.error as e:
            print(f"[WARN] [MonocularCalibrationTool] OpenCV Error in PnP:\n\n{e}")

            self._rvec, self._tvec = None, None
            self._rvec_j, self._tvec_j = None, None
            self._pose_error = np.inf
            return

        # If no solution, or if multiple solutions were found, abort
        # TODO: Classic chessboard will likely often have 2 solutions because of the 180 degrees ambiguity - need to be dealt with
        if nb_solutions != 1:
            self._rvec, self._tvec = None, None
            self._rvec_j, self._tvec_j = None, None
            self._pose_error = np.inf
            return

        # if only one solution, continue
        rvec, tvec = rvecs[0], tvecs[0]
        self._pose_error = float(solutions_errors.squeeze()) / self._err_norm  # TODO: Normalise error on image diagonal

        if refine:
            # Virtual Visual Servoing:
            # - "Visual servo control. I. Basic approaches", 2006,
            #   François Chaumette, Seth Hutchinson, 10.1109/MRA.2006.250573
            # - "Pose Estimation for Augmented Reality: A Hands-On Survey", 2015
            #   Eric Marchand, Hideaki Uchiyama, Fabien Spindler, 10.1109/TVCG.2015.2513408
            rvec, tvec = cv2.solvePnPRefineVVS(
                objectPoints=self.dt.points3d[self._points_ids],
                imagePoints=self._points2d,
                cameraMatrix=self._camera_matrix,
                distCoeffs=self._dist_coeffs,
                rvec=rvec,  # Input/Output /!\
                tvec=tvec,  # Input/Output /!\
                VVSlambda=1.0)

            # TODO: Test whether the Levenberg-Marquardt alternative solvePnPRefineLM() is better or not

        self._rvec, self._tvec = rvec.squeeze(), tvec.squeeze()
        self._rvec_j, self._tvec_j = _maybe_put(self._rvec), _maybe_put(self._tvec)

    def _compute_new_area(self) -> float:

        if not self.has_detection:
            return 0.0

        cells_indices = np.fliplr(
            np.clip((self._points2d // ((self.h, self.w) / self._grid_shape)).astype(np.int32), [0, 0],
                    np.flip(self._grid_shape - 1)))
        self._temp_grid[*cells_indices.T] = True

        # Novel area = cells that are in the current grid but not in the cumulative grid
        novel_cells = self._temp_grid & (~self._cumul_grid)
        novel_weight = self._grid_weights[novel_cells].sum()
        total_weight = self._grid_weights.sum()

        # update cumulative coverage and clear current temporary grid
        self._cumul_grid |= self._temp_grid
        self._temp_grid.fill(False)

        return float(novel_weight / total_weight) * 100

    def detect(self, frame):

        # initialise or update the internal arrays to match frame size if needed
        if None in (self.h, self.w) or np.any([self.h, self.w] != np.asarray(frame.shape)[:2]):
            self._update_imsize(frame.shape)

        if frame.ndim == 2:
            np.copyto(self._frame_buf[:, :, 0], frame)
            np.copyto(self._frame_buf[:, :, 1], frame)
            np.copyto(self._frame_buf[:, :, 2], frame)
        else:
            np.copyto(self._frame_buf, frame)

        # Detect
        if type(self.dt) is ChessboardDetector:
            self._points2d, self._points_ids = self.dt.detect(frame, refine_points=True)
        else:
            self._points2d, self._points_ids = self.dt.detect(frame,
                                                              camera_matrix=self._camera_matrix,
                                                              dist_coeffs=self._dist_coeffs,
                                                              refine_markers=True,
                                                              refine_points=True)

    def auto_register_area_based(self,
                                 area_threshold: float = 0.2,
                                 nb_points_threshold: int = 4
                                 ) -> bool:

        novel_area = self._compute_new_area()
        if novel_area >= area_threshold and self.nb_points >= nb_points_threshold:
            self.register_sample()
            return True
        else:
            return False

    def auto_compute_intrinsics(self,
                                coverage_threshold: float = 80.0,
                                stack_length_threshold: int = 15,
                                simple_focal: bool = False,
                                simple_distortion: bool = False,
                                complex_distortion: bool = False
                                ) -> bool:
        """
        Triggers computation if the percentage of grid cells marked as covered exceeds the threshold and if
        there are enough samples
        """
        if self.coverage >= coverage_threshold and self.nb_samples > stack_length_threshold:
            self.compute_intrinsics(fix_aspect_ratio=simple_focal,
                                    simple_distortion=simple_distortion,
                                    complex_distortion=complex_distortion)
            return True
        else:
            return False

    def visualise(self, errors_mm=False):

        # visualisation does not run at the same speed as detection, so we need to copy the buffer to avoid flickers
        frame_out = self._frame_buf.copy()

        if self.has_detection:
            # if corners have been found show them as red dots
            detected_points_int = (self._points2d * self._shift_factor).astype(np.int32)

            for xy in detected_points_int:
                frame_out = cv2.circle(frame_out, xy, 4 * self._vis_scale, (0, 0, 255), 4 * self._vis_scale,
                                       **self._draw_params)

        if self.has_intrinsics and self.has_extrinsics:
            # Display reprojected points: currently detected corners as yellow dots, the others as white dots
            reproj_points_j = geometry_jax.project_points(
                self._points3d_j,
                self._rvec_j,
                self._tvec_j,
                self._camera_matrix_j,
                self._dist_coeffs_j
            )

            reproj_points = np.array(reproj_points_j)
            reproj_points_int = (reproj_points * self._shift_factor).astype(np.int32)

            for i, xy in enumerate(reproj_points_int):
                if i in self._points_ids:
                    frame_out = cv2.circle(frame_out, xy, 2 * self._vis_scale, (0, 255, 255), 4 * self._vis_scale,
                                           **self._draw_params)
                else:
                    frame_out = cv2.circle(frame_out, xy, 4 * self._vis_scale, (255, 255, 255), 4 * self._vis_scale,
                                           **self._draw_params)

            # Compute errors in mm for each point
            if errors_mm:
                # Get each detected point's distance to the camera, and its error in pixels
                cam_points_dists = cdist([self._tvec], self._points3d[self._points_ids]).squeeze()
                per_point_error = np.nanmean(np.abs(self._points2d - reproj_points[self._points_ids]), axis=-1)

                # Horizontal field of view in pixels, and the pixel-angle (i.e. how many rads per pixels)
                f = self.focal
                if f:
                    pixel_angle = 2 * np.arctan(self.h / (2 * f)) / self.h

                    # Determine the per-point error in millimeters
                    error_arc = pixel_angle * per_point_error
                    error_mm = error_arc * cam_points_dists

                    for i, err in enumerate(error_mm):
                        frame_out = cv2.putText(frame_out, f"{err:.3f}",
                                                self._points2d[i].astype(int) + 6,
                                                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                                fontScale=0.3 * self._vis_scale,
                                                color=(0, 255, 255),
                                                thickness=1 * self._vis_scale,
                                                lineType=cv2.LINE_AA)

        # Draw grid-based coverage overlay
        # get mask of covered cells at image resolution
        mask = cv2.resize(self._cumul_grid.astype(np.uint8),
                          (self.w, self.h), interpolation=cv2.INTER_NEAREST).astype(bool)
        # refresh and apply the new green overlay
        self._green_overlay.fill(0)
        self._green_overlay[mask] = (0, 255, 0)
        frame_out = cv2.addWeighted(self._green_overlay, 0.3, frame_out, 0.7, 0)

        # Undistort image
        if self.has_intrinsics:
            optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self._camera_matrix,
                self._dist_coeffs,
                (self.w, self.h),
                1.0,
                (self.w, self.h)
            )
            frame_out = cv2.undistort(
                frame_out,
                self._camera_matrix,
                self._dist_coeffs,
                None,
                optimal_camera_matrix)

            # Display board perimeter in purple (after undistortion so that the lines are straight)
            if self.has_extrinsics:
                # we also need to undistort the corner points with the undistorted ('optimal') camera matrix
                # Note: dist coeffs are 'included' in this optimal camera matrix so we have to pass None here
                corners2d_j = geometry_jax.project_points(
                    self._cornersd3d_j,
                    camera_matrix=jnp.asarray(optimal_camera_matrix),
                    dist_coeffs=self._zero_coeffs,
                    rvec=self._rvec_j,
                    tvec=self._tvec_j)

                pts_int = np.asarray((corners2d_j * self._shift_factor)).astype(np.int32)

                frame_out = cv2.polylines(frame_out,
                                          [pts_int],
                                          True,
                                          (255, 0, 255),
                                          1 * self._vis_scale,
                                          **self._draw_params)

        # Add information text to the visualisation image
        frame_out = cv2.putText(frame_out,
                                f"Points: {self.nb_points}/{self.dt.nb_points}",
                                (30, 30 * self._vis_scale),
                                **self._text_params)
        frame_out = cv2.putText(frame_out,
                                f"Area: {self.coverage:.2f}% ({len(self.stack_points2d)} snapshots)",
                                (30, 60 * self._vis_scale),
                                **self._text_params)

        txt = f"{self._pose_error:.3f} px" if np.all(self._pose_error != np.inf) else '-'
        frame_out = cv2.putText(frame_out,
                                f"Current reprojection error: {txt}",
                                (30, 90 * self._vis_scale),
                                **self._text_params)

        avg_intr_err = np.nanmean(self._intrinsics_errors)
        txt = f"{avg_intr_err:.3f} px" if np.all(avg_intr_err != np.inf) else '-'
        frame_out = cv2.putText(frame_out,
                                f"Best average reprojection error: {txt}",
                                (30, 120 * self._vis_scale),
                                **self._text_params)

        f_mm = self.focal_mm
        txt = f"{f_mm:.2f} mm" if f_mm else '-'
        frame_out = cv2.putText(frame_out,
                                f"Estimated focal: {txt}",
                                (30, 150 * self._vis_scale),
                                **self._text_params)

        return frame_out


class MultiviewCalibrationTool:
    def __init__(self,
                 nb_cameras: int,
                 images_sizes_wh: ArrayLike,
                 origin_idx: int,
                 init_cam_matrices: ArrayLike,
                 init_dist_coeffs: ArrayLike,
                 object_points: ArrayLike,
                 intrinsics_window: int = 10,
                 min_detections: int = 15,
                 max_detections: int = 100,
                 angular_thresh: float = 15.0,        # in degrees
                 translational_thresh: float = 15.0,  # in object_points' units
                 refine_intrinsics_online: bool = True,
                 debug_print = True):

        # TODO: Typing and optimising this class

        self._debug_print = debug_print

        self.nb_cameras = nb_cameras
        self.origin_idx = origin_idx

        images_sizes_wh = np.asarray(images_sizes_wh)
        assert images_sizes_wh.ndim == 2 and images_sizes_wh.shape[0] == self.nb_cameras
        self.images_sizes_wh = images_sizes_wh[:, :2]

        self._refine_intrinsics_online = refine_intrinsics_online

        self._angular_thresh = angular_thresh
        self._translational_thresh = translational_thresh

        # Known 3D board model points (N, 3)
        self._object_points = np.asarray(object_points, dtype=np.float32)

        # buffers for incoming frames
        self._detection_buffer = [dict() for _ in range(nb_cameras)]
        self._last_frame = np.full(nb_cameras, -1, dtype=int)

        # extrinsics state
        self._has_ext = [False] * nb_cameras
        self._rvecs_cam2world = jnp.zeros((nb_cameras, 3), dtype=jnp.float32)
        self._tvecs_cam2world = jnp.zeros((nb_cameras, 3), dtype=jnp.float32)
        self._estimated = False

        # intrinsics state & buffer per camera
        self._cam_matrices = [np.asarray(init_cam_matrices[c], dtype=np.float32) for c in range(nb_cameras)]
        self._dist_coeffs = [np.asarray(init_dist_coeffs[c], dtype=np.float32) for c in range(nb_cameras)]
        self._intrinsics_buffer = {c: deque(maxlen=intrinsics_window) for c in range(nb_cameras)}
        self._intrinsics_window = intrinsics_window

        # triangulation & BA buffers
        self.ba_samples = deque(maxlen=max_detections)
        self.min_detections = min_detections

        # bs results
        self._refined_intrinsics = None
        self._refined_extrinsics = None
        self._refined_board_poses = None
        self._ba_points2d = None
        self._ba_pointsids = None

    def register(self, cam_idx: int, detection: DetectionPayload):

        if detection.pointsIDs is None or detection.points2D is None:
            return

        if len(detection.pointsIDs) < 4:
            return

        # Get the current best intrinsics for this camera
        K = self._cam_matrices[cam_idx]
        D = self._dist_coeffs[cam_idx]

        # Prepare the 3D and 2D points for solvePnP
        ids = detection.pointsIDs
        obj_pts_subset = self._object_points[ids]
        img_pts_subset = detection.points2D

        # Reestimate the board-to-camera pose and validate it
        try:
            success, rvec, tvec = cv2.solvePnP(
                objectPoints=obj_pts_subset,
                imagePoints=img_pts_subset,
                cameraMatrix=K,
                distCoeffs=D,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            # If PnP fails, return
            if not success:
                return

            rvec = rvec.squeeze()
            tvec = tvec.squeeze()

            # if PnP placed the board behind the camera, return
            if tvec[2] <= 0:
                return

        except cv2.error:
            return

        #--- From here on, we know rvec tvec are sane ---

        f = detection.frame
        self._last_frame[cam_idx] = f

        E_b2c = geometry_jax.extrinsics_matrix(
            jnp.asarray(rvec), jnp.asarray(tvec)
        )
        self._detection_buffer[cam_idx][f] = (E_b2c, detection.points2D, detection.pointsIDs)

        if cam_idx == self.origin_idx and not self._has_ext[cam_idx]:
            self._has_ext[cam_idx] = True
            # The pose is already (0, 0, 0) we can continue

        self._flush_frames()

    def _find_stale_frames(self):
        global_min = int(self._last_frame.min())
        pending = set()
        for buf in self._detection_buffer:
            pending.update(buf.keys())
        stale = [f for f in pending if f < global_min]
        return stale

    def _flush_frames(self):
        for f in self._find_stale_frames():
            cams = [c for c in range(self.nb_cameras) if f in self._detection_buffer[c]]
            if len(cams) < 2:
                for c in cams:
                    self._detection_buffer[c].pop(f, None)
                continue
            entries = [(c, *self._detection_buffer[c].pop(f)) for c in cams]
            self._process_frame(entries)

    def _process_frame(self, entries):
        """
        Uses a 'surgical reset' of extrinsics whenever intrinsics are updated,
        to prevent stale state propagation
        """

        if not any(self._has_ext):
            return

        if self._refine_intrinsics_online:
            # STEP 1- Refine intrinsics and check if a system-wide reset is needed
            # ================================================================

            intrinsics_were_updated = self._refine_intrinsics_per_camera()

            if intrinsics_were_updated:
                if self._debug_print:
                    print("[STATE_RESET] Intrinsics updated. Invalidating all non-origin extrinsics.")
                # We acknowledge that the world geometry is now different, so we force every non-origin camera
                # to be re-seeded from scratch
                for c in range(self.nb_cameras):
                    if c != self.origin_idx:
                        self._has_ext[c] = False

                # Abort the rest of this frame's processing. The system will recover
                # on subsequent frames by re-seeding the cameras one by one
                return

        # If we are here, the system state is stable. Proceed with normal estimation
        # ===========================================================================

        # STEP 2- Recompute board-to-camera poses for this frame
        recomputed_entries = []
        for cam_idx, _, pts2D, ids in entries:
            K = self._cam_matrices[cam_idx]
            D = self._dist_coeffs[cam_idx]
            obj_pts_subset = self._object_points[np.asarray(ids).flatten()]
            img_pts_subset = np.asarray(pts2D)
            if len(ids) < 4: continue

            success, rvec, tvec = cv2.solvePnP(obj_pts_subset, img_pts_subset, K, D)
            if not success or tvec.squeeze()[2] <= 0:
                continue

            rvec, tvec = rvec.flatten(), tvec.flatten()
            E_b2c = geometry_jax.extrinsics_matrix(jnp.asarray(rvec), jnp.asarray(tvec))
            recomputed_entries.append((cam_idx, E_b2c, pts2D, ids))

        if not recomputed_entries:
            return

        known = [e for e in recomputed_entries if self._has_ext[e[0]]]

        # STEP 3- Estimate board-to-world pose
        E_b2w = None
        if known:
            E_votes = []
            for cam_idx, E_b2c, _, _ in known:
                E_c2w_current = geometry_jax.extrinsics_matrix(
                    self._rvecs_cam2world[cam_idx], self._tvecs_cam2world[cam_idx]
                )
                E_votes.append(E_c2w_current @ E_b2c)

            E_stack = jnp.stack(E_votes, axis=0)

            # Convert all pose votes into (quaternion, translation) format
            r_stack, t_stack = geometry_jax.extmat_to_rtvecs(E_stack)
            q_stack = geometry_2.axisangle_to_quaternion_batched(r_stack)
            rt_stack = jnp.concatenate([q_stack, t_stack], axis=1)  # Shape: (num_known, 7)

            # robust filtering and averaging
            q_med, t_med = outliers_rejection.filter_rt_samples(
                rt_stack=rt_stack,
                ang_thresh=np.deg2rad(self._angular_thresh),
                trans_thresh=self._translational_thresh
            )

            E_b2w = geometry_jax.extrinsics_matrix(
                geometry_2.quaternion_to_axisangle(q_med), t_med
            )

            # Update all non-origin camera extrinsics based on the new E_b2w
            # (this new state will be used in the next iteration of this loop)
            if E_b2w is not None:
                for cam_idx, E_b2c, _, _ in recomputed_entries:
                    if cam_idx != self.origin_idx:
                        E_c2b = geometry_jax.invert_extrinsics_matrix(E_b2c)
                        E_c2w = E_b2w @ E_c2b
                        r_c2w, t_c2w = geometry_jax.extmat_to_rtvecs(E_c2w)
                        self._rvecs_cam2world = self._rvecs_cam2world.at[cam_idx].set(r_c2w)
                        self._tvecs_cam2world = self._tvecs_cam2world.at[cam_idx].set(t_c2w)
                    self._has_ext[cam_idx] = True

        # STEP 4- Final buffering (using the stabilized state)
        if E_b2w is not None:
            board_pts_hom = np.hstack([self._object_points, np.ones((self._object_points.shape[0], 1))])
            world_pts = (E_b2w @ board_pts_hom.T).T[:, :3]

            for cam_idx, E_b2c, pts2D, ids in recomputed_entries:
                uv_obs = np.asarray(pts2D, dtype=np.float32)

                if cam_idx == self.origin_idx:
                    r_b2c, t_b2c = geometry_jax.extmat_to_rtvecs(E_b2c)
                    obj_pts_local = self._object_points[ids]
                    proj_pts, _ = cv2.projectPoints(
                        np.asarray(obj_pts_local).reshape(-1, 1, 3), np.asarray(r_b2c), np.asarray(t_b2c),
                        self._cam_matrices[cam_idx], self._dist_coeffs[cam_idx]
                    )
                else:
                    current_r_c2w = self._rvecs_cam2world[cam_idx]
                    current_t_c2w = self._tvecs_cam2world[cam_idx]
                    r_w2c, t_w2c = geometry_jax.invert_extrinsics(current_r_c2w, current_t_c2w)
                    obj_pts_world = world_pts[ids]
                    proj_pts, _ = cv2.projectPoints(
                        np.asarray(obj_pts_world).reshape(-1, 1, 3), np.asarray(r_w2c), np.asarray(t_w2c),
                        self._cam_matrices[cam_idx], self._dist_coeffs[cam_idx]
                    )

                uv_proj = proj_pts.reshape(-1, 2)
                errs = np.linalg.norm(uv_proj - uv_obs, axis=1)
                if self._debug_print:
                    print(f"[REPROJ_ERR] cam={cam_idx}, mean={errs.mean():.2f}px, max={errs.max():.2f}px")

            self._estimated = True

        # Buffer data for final BA
        for cam_idx, _, pts2D, ids in recomputed_entries:
            if len(ids) >= 6:
                board_pts_subset = self._object_points[ids].astype(np.float32)
                img_pts = np.asarray(pts2D, dtype=np.float32)
                self._intrinsics_buffer[cam_idx].append((board_pts_subset, img_pts))

        self.ba_samples.append(recomputed_entries)

    def _refine_intrinsics_per_camera(self):
        """
        Refines intrinsics for any camera with a full buffer of views
        """
        update_happened = False

        for ci in range(self.nb_cameras):
            buf = self._intrinsics_buffer[ci]
            if len(buf) < self._intrinsics_window:
                continue

            if self._debug_print:
                print(f"[REFINE_INTR] Starting for cam={ci} with {len(buf)} views.")

            object_points_views, image_points_views = zip(*buf)
            initial_K = self._cam_matrices[ci].copy()
            initial_D = self._dist_coeffs[ci].copy()

            try:
                ret, K_new, D_new, _, _, _, _, _ = cv2.calibrateCameraExtended(
                    objectPoints=list(object_points_views),
                    imagePoints=list(image_points_views),
                    imageSize=self.images_sizes_wh[ci],
                    cameraMatrix=initial_K,
                    distCoeffs=initial_D,

                    # flags=(cv2.CALIB_USE_INTRINSIC_GUESS |
                    #        cv2.CALIB_FIX_PRINCIPAL_POINT |
                    #        cv2.CALIB_FIX_K3)

                    # flags=(cv2.CALIB_USE_INTRINSIC_GUESS)

                    flags=(cv2.CALIB_USE_INTRINSIC_GUESS |
                           cv2.CALIB_FIX_PRINCIPAL_POINT |
                           cv2.CALIB_FIX_ASPECT_RATIO |  # Very important for stability
                           cv2.CALIB_ZERO_TANGENT_DIST |  # Often improves stability
                           cv2.CALIB_FIX_K3 |  # Solve for k1, k2 only
                           cv2.CALIB_FIX_K4 |
                           cv2.CALIB_FIX_K5 |
                           cv2.CALIB_FIX_K6
                           )
                )

                # Sanity checks to prevent numerical instability from bad optimizations
                is_valid_K = np.all(np.isfinite(K_new))
                is_valid_D = np.all(np.isfinite(D_new))
                focal_lengths_ok = K_new[0, 0] > 0 and K_new[1, 1] > 0
                w, h = self.images_sizes_wh[ci]
                principal_point_ok = (0 < K_new[0, 2] < w) and (0 < K_new[1, 2] < h)
                dist_coeffs_ok = np.all(np.abs(D_new.flatten()) < 100.0)

                if is_valid_K and is_valid_D and focal_lengths_ok and principal_point_ok and dist_coeffs_ok:

                    D_new_squeezed = D_new.squeeze()

                    # Pad the shorter array with zeros to match the length of the longer one
                    if len(initial_D) > len(D_new_squeezed):
                        padded_D_new = np.zeros_like(initial_D)
                        padded_D_new[:len(D_new_squeezed)] = D_new_squeezed
                        D_to_compare = padded_D_new
                    elif len(D_new_squeezed) > len(initial_D):
                        padded_initial_D = np.zeros_like(D_new_squeezed)
                        padded_initial_D[:len(initial_D)] = initial_D
                        initial_D = padded_initial_D
                        D_to_compare = D_new_squeezed
                    else:
                        D_to_compare = D_new_squeezed

                    # Now we can safely compare them
                    k_changed = not np.allclose(initial_K, K_new, atol=1e-2, rtol=1e-2)
                    d_changed = not np.allclose(initial_D, D_to_compare, atol=1e-3, rtol=1e-3)

                    if k_changed or d_changed:
                        if self._debug_print:
                            print(f"[REFINE_INTR] cam={ci} finished. RMS: {ret:.4f}px. Update ACCEPTED.")
                        self._cam_matrices[ci] = K_new
                        self._dist_coeffs[ci] = D_to_compare
                        print(self._cam_matrices)
                        update_happened = True
                    else:
                        if self._debug_print:
                            print(f"[REFINE_INTR] cam={ci} finished. RMS: {ret:.4f}px. No significant change.")

                self._intrinsics_buffer[ci].clear()

            except cv2.error as e:
                if self._debug_print:
                    print(f"[REFINE_INTR] cam={ci} failed with OpenCV error: {e}")
                self._intrinsics_buffer[ci].clear()

        return update_happened

    def refine_all(self,
                   simple_focal: bool = False,
                   simple_distortion: bool = False,
                   complex_distortion: bool = False,
                   shared: bool = False,
                   ) -> bool:
        """
        Performs a global bundle adjustment (BA) over all collected samples

            - Format the 2D detections and visibility masks from all samples
            - Use the current best estimates for camera intrinsics and extrinsics
            - For each sample frame, calculate a robust initial guess for the board's world pose
            by averaging the poses implied by each camera that saw it
            - Call the BA solver
            - Store the final globally optimized results
        """

        if not self._estimated:
            print("[BA] Error: Initial extrinsics have not been estimated yet.")
            return False

        P = self.ba_sample_count
        if P < self.min_detections:
            print(f"[BA] Not enough samples for bundle adjustment. Have {P}, need {self.min_detections}.")
            return False

        if self._debug_print:
            print(f"[BA] Starting Bundle Adjustment with {P} samples.")

        C = self.nb_cameras
        N = self._object_points.shape[0]

        # Prepare 2D Detections and Visibility Mask
        # =========================================================================
        pts2d_buf = np.full((C, P, N, 2), 0.0, dtype=np.float32)
        vis_buf = np.zeros((C, P, N), dtype=bool)

        for p_idx, entries in enumerate(self.ba_samples):
            for cam_idx, _, pts2D, ids in entries:
                pts2d_buf[cam_idx, p_idx, ids, :] = pts2D
                vis_buf[cam_idx, p_idx, ids] = True

        # Prepare initial guess for board poses (one per sample)
        # =========================================================================
        r_board_w_list = []
        t_board_w_list = []

        # Get the current best camera-to-world extrinsics
        E_c2w_all = geometry_jax.extrinsics_matrix(self._rvecs_cam2world, self._tvecs_cam2world)

        for p_idx, entries in enumerate(self.ba_samples):
            E_b2w_votes = []
            for cam_idx, E_b2c, _, _ in entries:
                # For each camera in the sample, calculate its 'vote' for the board's world pose
                E_c2w = E_c2w_all[cam_idx]
                E_b2w_vote = E_c2w @ E_b2c
                E_b2w_votes.append(E_b2w_vote)

            # Robustly average the votes to get the initial guess for this sample's board pose
            E_stack = jnp.stack(E_b2w_votes, axis=0)
            r_stack, t_stack = geometry_jax.extmat_to_rtvecs(E_stack)
            q_stack = geometry_2.axisangle_to_quaternion_batched(r_stack)
            q_med = geometry_2.quaternion_average(q_stack)
            t_med = jnp.median(t_stack, axis=0)

            r_board_w_list.append(geometry_2.quaternion_to_axisangle(q_med))
            t_board_w_list.append(t_med)

        # Convert lists to final np arrays for the BA function
        r_board_w_init = np.asarray(jnp.stack(r_board_w_list))  # (P, 3)
        t_board_w_init = np.asarray(jnp.stack(t_board_w_list))  # (P, 3)

        # Prepare initial guesses for camera parameters
        # =========================================================================
        # Use the latest refined parameters as starting point
        K_init = np.stack(self._cam_matrices)
        D_init = np.stack(self._dist_coeffs)
        cam_r_init = np.asarray(self._rvecs_cam2world)
        cam_t_init = np.asarray(self._tvecs_cam2world)

        # Run the Bundle Adjustment
        # =========================================================================

        success, retvals = bundle_adjustment_2.run_bundle_adjustment(
            K_init,                 # (C, 3, 3)
            D_init,                 # (C, <=8)
            cam_r_init,             # (C, 3)
            cam_t_init,             # (C, 3)
            r_board_w_init,         # (P, 3)
            t_board_w_init,         # (P, 3)
            pts2d_buf,              # (C, P, N, 2)
            vis_buf,                # (C, P, N)
            self._object_points,    # (N, 3)
            self.images_sizes_wh,
            priors_weight=0.0,
            simple_focal=simple_focal,
            simple_distortion=simple_distortion,
            complex_distortion=complex_distortion,
            shared=shared
        )

        # Store Refined Results
        # =========================================================================
        if success:
            print("[BA] Bundle Adjustment successful.")
            self.ba_samples.clear()

            self._refined_intrinsics = (retvals['K_opt'], retvals['D_opt'])
            self._refined_extrinsics = (retvals['cam_r_opt'], retvals['cam_t_opt'])
            self._refined_board_poses = (retvals['board_r_opt'], retvals['board_t_opt'])

            self._ba_points2d = pts2d_buf   # for visualization
            self._ba_pointsids = vis_buf    # for visualization

            self._refined = True
            return True

        else:
            print("[BA] Bundle Adjustment failed.")
            return False

    @property
    def initial_extrinsics(self):
        return np.array(self._rvecs_cam2world), np.array(self._tvecs_cam2world)

    @property
    def refined_intrinsics(self):
        return self._refined_intrinsics

    @property
    def refined_extrinsics(self):
        return self._refined_extrinsics

    @property
    def refined_board_poses(self):
        return self._refined_board_poses

    @property
    def image_points(self):
        return self._ba_points2d, self._ba_pointsids

    @property
    def ba_sample_count(self):
        return len(self.ba_samples)
