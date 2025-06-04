import numpy as np
from numpy.typing import ArrayLike
np.set_printoptions(precision=3, suppress=True, threshold=5)
import cv2
import scipy.stats as stats
from scipy.spatial.distance import cdist
import jax
import jax.numpy as jnp
from collections import deque
from typing import Dict, Set, Tuple, List, Optional, Iterable, Union
from mokap.utils.datatypes import ChessBoard, CharucoBoard, PosePayload, DetectionWithPosePayload
from mokap.calibration import monocular_2, bundle_adjustment_2
from mokap.utils import geometry_jax, geometry_2, pad_dist_coeffs


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

        if marker_ids is None:
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

        mean_new, se_new, l_new = np.mean(errors_new), stats.sem(errors_new), len(np.atleast_1d(errors_prev))
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

        calib_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

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
                 origin_camera_idx: int = 0,
                 min_poses: int = 15,
                 max_poses: int = 100,
                 min_detections: int = 15,
                 max_detections: int = 100):

        self.nb_cameras = nb_cameras
        self.origin_idx = origin_camera_idx

        self._min_poses = min_poses
        self._max_poses = max_poses

        # Track last processed frame
        self._last_frame = np.full(self.nb_cameras, -1, dtype=int)

        # ----------- PHASE 1: pose collection -----------
        self._frame_pose_buffer: Dict[int, Dict[int, jnp.ndarray]] = {}
        self._frames_stale: Set[int] = set()

        # per‐camera deque of samples in world frame for median estimation
        self._cams_world_poses: Dict[int, deque] = {
            ci: deque(maxlen=self._max_poses) for ci in range(self.nb_cameras)
        }
        self._cams_world_medians: Dict[int, Tuple[jnp.ndarray, jnp.ndarray]] = {}

        # Buffers for [cam -> world] extrinsics (set after initial best-guess)
        self._rvecs_cam2world_j = jnp.zeros((self.nb_cameras, 3), dtype=jnp.float32)
        self._tvecs_cam2world_j = jnp.zeros((self.nb_cameras, 3), dtype=jnp.float32)

        self._origin_seeded = False
        self._estimated = False

        # ---------- PHASE 2: detection + BA ----------
        self._min_detections = min_detections
        self._max_detections = max_detections

        # _detection_buffer[c][f] = (E_board2cam, pts2d, pts_ids)
        self._detection_buffer: List[Dict[int, Tuple[jnp.ndarray, np.ndarray, np.ndarray]]] = [
            {} for _ in range(self.nb_cameras)
        ]
        self.ba_samples: deque = deque(maxlen=self._max_detections)

        self._refined = False

        # Final outputs
        self._cam_matrices_refined: Optional[jnp.ndarray] = None
        self._dist_coeffs_refined: Optional[jnp.ndarray] = None
        self._rvecs_refined: Optional[np.ndarray] = None
        self._tvecs_refined: Optional[np.ndarray] = None

    @property
    def has_estimate(self) -> bool:
        return self._estimated

    @property
    def is_refined(self) -> bool:
        return self._refined

    @property
    def initial_extrinsics(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._estimated:
            return None, None
        return np.asarray(self._rvecs_cam2world_j), np.asarray(self._tvecs_cam2world_j)

    @property
    def ba_sample_count(self) -> int:
        return len(self.ba_samples)

    @property
    def refined_intrinsics(self) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        if self._refined:
            return self._cam_matrices_refined, self._dist_coeffs_refined
        return None, None

    @property
    def refined_extrinsics(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._refined:
            return self._rvecs_refined, self._tvecs_refined
        return None, None

    def _find_stale_frames(self, buffers: List[Dict[int, any]]) -> List[int]:
        """
        Return frames in any buffer that are stale:
        we mark frames as 'stale' only when all cameras have reported newer frames
        """
        global_min = int(np.min(self._last_frame))
        pending = set()
        for buf in buffers:
            pending.update(buf.keys())
        return [f for f in pending if f < global_min]

    def _flush_frames(self):
        """
        This marks any frame waiting as ready to be processed (or it discards it if it can't be processed)
        """
        if not self._estimated:
            # Phase 1: if a stale frame has fewer than the required cameras to compute a root‐world chain,
            # it waits until enough data arrives
            stale = self._find_stale_frames([self._frame_pose_buffer])
            self._frames_stale.update(stale)
        else:
            # Phase 2: we immediately discard any stale frame that has fewer than two cameras
            stale = self._find_stale_frames(self._detection_buffer)
            for f in stale:
                cams = [c for c in range(self.nb_cameras) if f in self._detection_buffer[c]]
                if len(cams) >= 2:
                    # build BA sample
                    sample = []
                    for ci in range(self.nb_cameras):
                        if ci in cams:
                            ext_mat, pts2d, pts_ids = self._detection_buffer[ci].pop(f)
                            sample.append((ext_mat, pts2d, pts_ids))
                        else:
                            sample.append((None, None, None))
                    self.ba_samples.append(sample)
                else:
                    # discard stale detection
                    for cam in cams:
                        self._detection_buffer[cam].pop(f, None)

    # -------------------------- PHASE 1 --------------------------

    def register_pose(self, cam_idx: int, pose: PosePayload):

        if self._estimated:
            return

        frame_idx = pose.frame

        # Update last seen frame for cam_idx
        self._last_frame[cam_idx] = frame_idx

        # Insert (rvec, tvec) into the pose-buffer for this frame
        if frame_idx not in self._frame_pose_buffer:
            self._frame_pose_buffer[frame_idx] = {}
        E_b2c = geometry_jax.extrinsics_matrix(jnp.asarray(pose.rvec), jnp.asarray(pose.tvec))
        self._frame_pose_buffer[frame_idx][cam_idx] = E_b2c

        # Only the first time the origin camera appears
        if not self._origin_seeded and cam_idx == self.origin_idx:
            # seed [origin -> world] = identity matrix
            rt_flat = jnp.concatenate([geometry_2.ID_QUAT, geometry_2.ZERO_T], axis=0)  # (7,)
            self._cams_world_poses[self.origin_idx].append(rt_flat)
            self._cams_world_medians[self.origin_idx] = (geometry_2.ID_QUAT, geometry_2.ZERO_T)
            self._origin_seeded = True

            # We can process that frame immediately and return because origin is now known
            self._attempt_process_pending()
            return

        self._flush_frames()
        # Try to process any stale frames whose root camera is now known
        self._attempt_process_pending()

    def _attempt_process_pending(self):
        """
        Repeatedly scan pending phase 1 frames to see if any can now be linked to an existing world camera
        Keep looping until no more frames in _frames_stale qualify
        """

        made_progress = True
        while made_progress:
            made_progress = False

            for f in list(self._frames_stale):
                cams = set(self._frame_pose_buffer[f].keys())

                # If a camera in this frame is already known (in cam_to_world), use it as root
                known = cams.intersection(self._cams_world_medians.keys())
                if known:
                    root_cam = next(iter(known))
                else:
                    # No available root in this frame, skip for now
                    continue

                # We have a root_cam so we can process frame f
                self._process_pose(f, root_cam)
                self._frames_stale.remove(f)
                made_progress = True
                break  # rescan from scratch because cam_to_world has changed

    def _process_pose(self, frame_idx: int, root_cam: int):
        """
        Propagate world extrinsic from the root cam in this frame to all other cams in this frame
        """

        cams = list(self._frame_pose_buffer[frame_idx].keys())  # which cameras saw this frame

        # Gather all M [board -> cam] matrices
        E_board2cam_stack = jnp.stack(
            [self._frame_pose_buffer[frame_idx][cam] for cam in cams],
            axis=0  # (M, 4, 4)
        )

        # [root -> world] already in self._cams_world_medians[root_cam] as a 7 rt vector
        q_med, t_med = self._cams_world_medians[root_cam]
        rvec_med = geometry_2.quaternion_to_axisangle(q_med)
        E_root2world_j = geometry_jax.extrinsics_matrix(rvec_med, t_med)

        # Remap all cams in this frame from [board -> cam] to [cam -> world]
        rvecs_cam2world_j, tvecs_cam2world_j = self._remap_phase1(E_board2cam_stack,
                                                                  E_root2world_j,
                                                                  cams.index(root_cam))
        quaternions_cam2world_j = geometry_2.axisangle_to_quaternion_batched(rvecs_cam2world_j) # (M, 4)

        for i, cam in enumerate(cams):
            rt_flat = jnp.concatenate([quaternions_cam2world_j[i], tvecs_cam2world_j[i]], axis=0)  # (7,)
            self._cams_world_poses[cam].append(rt_flat)

            # Now recompute that camera's median
            arr_rt = jnp.stack(list(self._cams_world_poses[cam]), axis=0)  # (m, 7)
            m = arr_rt.shape[0]

            # We create a dummy batch of 1 because estimate_initial_poses expects a batch of C cams
            length_one = jnp.array([m], dtype=jnp.int32)  # (1, M_max)
            rt_padded = geometry_2.pad_to_length(arr_rt, self._max_poses, axis=0, pad_value=0.0)  # (M_max, 7)
            rt_batch = rt_padded[jnp.newaxis, :, :]             # (1, M_max, 7)

            q_batch, t_batch = geometry_2.estimate_initial_poses(rt_batch, length_one)
            q_med, t_med = q_batch[0], t_batch[0]

            self._cams_world_medians[cam] = (q_med, t_med)

        # frame has now been processed, we can discard if from the frame pose buffer
        del self._frame_pose_buffer[frame_idx]

        # Once every camera has ≥ min_poses samples, we can compute the best guess
        if not self._estimated and all(len(dq) >= self._min_poses for dq in self._cams_world_poses.values()):
            self._compute_initial_extrinsics()

    def _compute_initial_extrinsics(self):

        # Each self._cams_world_poses[c] is a deque of (2, 3) JAX array, each with a diff length
        rt_samples = []
        lengths = []
        for c in range(self.nb_cameras):
            rt_flat = jnp.stack(self._cams_world_poses[c], axis=0)  # (m, 7)
            rt_samples.append(rt_flat)
            lengths.append(rt_flat.shape[0])

        lengths = jnp.array(lengths, dtype=jnp.int32)

        # Pad to (C, M_max, 7)
        M_max = int(jnp.max(lengths))

        rt_stack_flat = jnp.stack([
            geometry_2.pad_to_length(arr_c, M_max, axis=0, pad_value=0.0)
            for arr_c in rt_samples
        ], axis=0)  # (C, M_max, 7)

        # assume estimate_initial_poses_flat takes (C, M_max, 7) plus lengths
        q_cam_j, t_cam_j = geometry_2.estimate_initial_poses(rt_stack_flat, lengths)

        # Convert quaternions back to axis–angle vectors
        rvecs_cam_j = geometry_2.quaternion_to_axisangle_batched(q_cam_j)  # (C, 3)

        self._rvecs_cam2world_j = rvecs_cam_j
        self._tvecs_cam2world_j = t_cam_j

        self._estimated = True
        self._last_frame = np.full(self.nb_cameras, -1, dtype=int) # reset for Phase 2

    # -------------------------- PHASE 2 --------------------------

    def register_detection(self, cam_idx: int, detection: DetectionWithPosePayload):

        if not self._estimated:
            return

        frame_idx = detection.frame
        self._last_frame[cam_idx] = frame_idx

        # Store [board -> cam] (4, 4) + 2D points
        E_b2c = geometry_jax.extrinsics_matrix(
            jnp.asarray(detection.rvec),
            jnp.asarray(detection.tvec)
        )
        self._detection_buffer[cam_idx][frame_idx] = (E_b2c, detection.points2D, detection.pointsIDs)
        self._flush_frames()

    def _remap_phase1(self,
            E_board2cam_stack:  jnp.ndarray,     # (M, 4, 4)
            E_root2world:       jnp.ndarray,     # (4, 4)
            root_idx:           int  # which row in E_board2cam_stack is root
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        M = E_board2cam_stack.shape[0]

        # Expand E_root2world to (M, 4, 4)
        E_root2world_rep = jnp.repeat(E_root2world[None, ...], M, axis=0)  # (M, 4, 4)

        # Extract E_board2root = E_board2cam_stack[root_idx]  # (4, 4)
        E_board2root = E_board2cam_stack[root_idx]
        E_board2root_rep = jnp.repeat(E_board2root[None, ...], M, axis=0)  # (M, 4, 4)

        # [board -> world] = [root -> world] @ [board -> root]
        E_board2world = E_root2world_rep @ E_board2root_rep  # (M, 4, 4)

        # [cam -> board] = inv([board -> cam])
        E_cam2board = geometry_jax.invert_extrinsics_matrix(E_board2cam_stack)  # (M, 4, 4)

        # [cam -> world] = [board -> world] @ [cam -> board]
        E_cam2world = E_board2world @ E_cam2board  # (M, 4, 4)

        # Convert back to (rvec, tvec)
        return geometry_jax.extmat_to_rtvecs(E_cam2world)  # each (M, 3)

    def refine_all(self,
            camera_matrices:    np.ndarray,
            dist_coeffs:        np.ndarray,
            points3d_th:        jnp.ndarray,
            simple_focal:       bool = False,
            simple_distortion:  bool = False,
            complex_distortion: bool = False,
            shared:             bool = False,
            ) -> bool:
        """
        Once we have enough BA samples, run the bundle adjustment
        """

        if not self._estimated:
            return False

        P = len(self.ba_samples)
        if P < self._min_detections:
            return False

        C = self.nb_cameras
        N = points3d_th.shape[0]

        # Build pts2d_buf (C, P, N, 2) and vis_buf (C, P, N)
        pts2d_buf = np.full((C, P, N, 2), 0.0, dtype=np.float32)
        vis_buf = np.zeros((C, P, N), dtype=bool)

        # Build E_board2cam_all = identity, then we overwrite with valid entries
        E_board2cam_all = jnp.tile(
            jnp.eye(4, dtype=jnp.float32)[None, None, :, :],
            (P, C, 1, 1)
        )  # (P, C, 4, 4)

        root_idx_list = []
        for p_idx, sample in enumerate(self.ba_samples):
            cams_present = [ci for ci, (E_b2c, _, _) in enumerate(sample) if E_b2c is not None]
            for ci in cams_present:
                E_b2c, pts2d, pts_ids = sample[ci]
                pts2d_buf[ci, p_idx, pts_ids, :] = pts2d
                vis_buf[ci, p_idx, pts_ids] = True
                E_board2cam_all = E_board2cam_all.at[p_idx, ci].set(E_b2c)

            if self.origin_idx in cams_present:
                root_ci = self.origin_idx
            else:
                root_ci = cams_present[0]
            root_idx_list.append(root_ci)

        root_idx_arr = jnp.array(root_idx_list, dtype=jnp.int32)  # (P,)

        # Build E_cam2world_all once (C, 4, 4)
        E_cam2world_all = geometry_jax.extrinsics_matrix(
            self._rvecs_cam2world_j,    # (C, 3)
            self._tvecs_cam2world_j     # (C, 3)
        )

        # Choose one [cam -> world] per sample p
        E_cam2world_roots = E_cam2world_all[root_idx_arr, :, :]  # (P, 4, 4)

        # Choose one [board -> cam] per sample p
        p_indices = jnp.arange(P)
        E_board2cam_roots = E_board2cam_all[p_indices, root_idx_arr, :, :]  # (P, 4, 4)

        # Now compute [board -> world] in one batch
        E_board2world = jnp.einsum('pab,pbc->pac', E_cam2world_roots, E_board2cam_roots) # (P, 4, 4)
        
        # and back to (rvecs, tvecs)
        r_board_w, t_board_w = geometry_jax.extmat_to_rtvecs(E_board2world) # both (P, 3)

        # Finally call BA with exactly one 6‐DoF per cam and one per board‐sample
        K_opt, D_opt, cam_r_opt, cam_t_opt, board_r_opt, board_t_opt = \
            bundle_adjustment_2.run_bundle_adjustment(
                camera_matrices,    # (C, 3, 3)
                dist_coeffs,        # (C, ≤8)
                np.asarray(self._rvecs_cam2world_j),  # (C, 3)
                np.asarray(self._tvecs_cam2world_j),  # (C, 3)
                np.asarray(r_board_w),  # (P, 3)
                np.asarray(t_board_w),  # (P, 3)
                pts2d_buf,      # (C, P, N, 2)
                vis_buf,        # (C, P, N)
                points3d_th,    # (N, 3)
                simple_focal=simple_focal,
                simple_distortion=simple_distortion,
                complex_distortion=complex_distortion,
                shared=shared
            )

        self.ba_samples.clear()

        # Store the refined intrinsics and extrinsics
        self._cam_matrices_refined = K_opt
        self._dist_coeffs_refined = D_opt
        self._rvecs_refined = np.asarray(cam_r_opt)  # (C, 3)
        self._tvecs_refined = np.asarray(cam_t_opt)  # (C, 3)
        self._refined = True

        # Store the 2d points
        self._debug_points2d = pts2d_buf
        self._debug_pointsids = vis_buf

        return True
    
