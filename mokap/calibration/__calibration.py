from collections import defaultdict, deque
import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=5)
import cv2
import scipy.stats as stats
from scipy.spatial.distance import cdist
import jax.numpy as jnp
import jax
from typing import Optional, Iterable, Tuple, Union
from numpy.typing import ArrayLike

from mokap.utils import geometry
from mokap.utils.datatypes import ChessBoard, CharucoBoard
from mokap.calibration import monocular_2
from mokap.utils import geometry_jax


def _maybe_put(x):
    return jax.device_put(x) if x is not None else None


class ChessboardDetector:
    """
    Detects a standard chessboard. Returns 2D corner coords plus a row-major
    vector of points IDs (0 ... N-1).
    """

    def __init__(self, board_params: CharucoBoard,
                 downsample_size: int = 480):

        self._n_cols, self._n_rows = board_params.cols, board_params.rows

        if self._n_cols < 2 or self._n_rows < 2:
            raise ValueError("BoardParams must have at least 2x2 squares for a valid chessboard.")

        # Create 3D coordinates for board corners (in board-centric coordinates)
        self._board_points_3d = board_params.object_points()
        self._board_corners_3d = (
                np.array([[0, 0, 0],
                          [0, 1, 0],
                          [1, 1, 0],
                          [1, 0, 0]], dtype=np.float32) * [self._n_cols, self._n_rows, 0] * board_params.square_length)

        # Maximum number of board points and distances
        self._total_points: int = len(self._board_points_3d)
        self._total_distances: int = int((self._total_points * (self._total_points - 1)) / 2.0)

        # OpenCV expects this tuple
        self._n_inner_size: Tuple[int, int] = (self._n_cols - 1, self._n_rows - 1)

        # chessboard detections always returns either all or no points, so we fix the points_ids once
        self._points2d_ids = np.arange(np.prod(self._n_inner_size), dtype=np.int32)

        # classic chessboard detection is much slower than charuco so we kinda have to downsample
        self._downsample_size = downsample_size

        # Cache detection flags
        self._detection_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                                 cv2.CALIB_CB_NORMALIZE_IMAGE |
                                 cv2.CALIB_CB_FAST_CHECK |  # quickly dismisses frames with no board in view
                                 cv2.CALIB_CB_FILTER_QUADS) # pre‐filters candidate quads before full points grouping

        # Cache the criteria for subpixel refinement
        self._win_size:         Tuple[int, int] = (11, 11)
        self._zero_zone:        Tuple[int, int] = (-1, -1)
        self._subpix_criteria:  Tuple[int, int, float] = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                20,     # max iterations
                0.1     # epsilon is the minimum allowed movement (in pixels) of a point from one iteration to the next
            )

    @property
    def points3d(self) -> ArrayLike:
        """ Returns the coordinates of the chessboard points in 3D (in board-centric coordinates) """
        return self._board_points_3d

    @property
    def corners3d(self) -> ArrayLike:
        """ Returns the coordinates of the chessboard outer corners in 3D (in board-centric coordinates) """
        return self._board_corners_3d

    @property
    def total_points(self) -> int:
        return self._total_points

    @property
    def total_distances(self) -> int:
        return self._total_distances

    @property
    def board_dims(self) -> Tuple[int, int]:
        return self._n_cols, self._n_rows

    def detect(self,
               frame: ArrayLike,
               refine_points: bool = False
               ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Downsample image if bigger - otherwise chessboard detection is way too slow
        h_full, w_full = frame.shape[:2]
        max_dim = max(h_full, w_full)

        scale = self._downsample_size / float(max_dim) if max_dim > self._downsample_size else 1.0

        if scale < 1.0:
            new_w = int(w_full * scale)
            new_h = int(h_full * scale)
            frame_small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        found, chessboard_points = cv2.findChessboardCorners(
            frame_small if scale < 1.0 else frame,
            self._n_inner_size,
            flags=self._detection_flags
        )

        # If no points detected, abort
        if not found:
            return None, None

        chessboard_points = chessboard_points.astype(np.float32) / scale

        if refine_points:
            try:
                chessboard_points = cv2.cornerSubPix(frame, chessboard_points,
                                                     winSize=self._win_size,
                                                     zeroZone=self._zero_zone,
                                                     criteria=self._subpix_criteria)
            except cv2.error as e:
                print(e)

        # chessboard_points is (N, 1, 2), we want (N, 2)
        points2d_coords = chessboard_points.reshape(-1, 2).astype(np.float32)

        return points2d_coords, self._points2d_ids


class CharucoDetector(ChessboardDetector):
    def __init__(self, board_params: CharucoBoard):
        super().__init__(board_params)

        # We need to keep references to the OpenCV Charuco board object and detector parameters
        self.board = board_params.to_opencv()
        self._detector_parameters = cv2.aruco.DetectorParameters()
        self._detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(self.board.getDictionary(), detectorParams=self._detector_parameters)

        # Cache the criteria for subpixel refinement
        self._win_size:         Tuple[int, int] = (11, 11)
        self._zero_zone:        Tuple[int, int] = (-1, -1)
        self._subpix_criteria:  Tuple[int, int, float] = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                20,     # max iterations
                0.1     # epsilon is the minimum allowed movement (in pixels) of a point from one iteration to the next
            )

    def detect(self,
               frame:           ArrayLike,
               camera_matrix:   Optional[ArrayLike] = None,
               dist_coeffs:     Optional[ArrayLike] = None,
               refine_markers:  bool = True,
               refine_points:   bool = False
               ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        points2d_coords = None
        points2d_ids = None

        # Detect and refine aruco markers
        markers_coords, marker_ids, rejected = self._detector.detectMarkers(frame)

        if refine_markers:
            markers_coords, marker_ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(
                image=frame,
                board=self.board,
                detectedCorners=markers_coords,     # Input/Output /!\
                detectedIds=marker_ids,             # Input/Output /!\
                rejectedCorners=rejected,           # Input/Output /!\
                parameters=self._detector_parameters,
                # Known bug with refineDetectedMarkers, fixed in OpenCV 4.9: https://github.com/opencv/opencv/pull/24139
                cameraMatrix=camera_matrix if cv2.getVersionMajor() >= 4 and cv2.getVersionMinor() >= 9 else None,
                distCoeffs=dist_coeffs)

        # If no marker detected, abort
        if marker_ids is None:
            return points2d_coords, points2d_ids

        # If any marker has been detected, try to detect the chessboard corners
        else:
            nb_chessboard_points, chessboard_points, chessboard_points_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=markers_coords,
                markerIds=marker_ids,
                image=frame,
                board=self.board,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
                minMarkers=1)

            if refine_points and chessboard_points is not None:
                try:
                    # Refine the chessboard corners
                    chessboard_points = cv2.cornerSubPix(frame, chessboard_points,
                                                         winSize=self._win_size,
                                                         zeroZone=self._zero_zone,
                                                         criteria=self._subpix_criteria)
                except cv2.error as e:
                    print(e)

            if chessboard_points is not None and len(chessboard_points_ids[:, 0]) > 1:
                points2d_coords = chessboard_points[:, 0, :]
                points2d_ids = chessboard_points_ids[:, 0]

        return points2d_coords, points2d_ids


class MonocularCalibrationTool:
    """
    This object is stateful for the intrinsics *only*
    """
    def __init__(self,
                 board_params:  Union[ChessBoard, CharucoBoard],
                 imsize_hw:     Optional[Iterable[int]] = None,  # OpenCV order (height, width)
                 min_stack:     int = 15,
                 max_stack:     int = 100,
                 focal_mm:      Optional[int] = None,
                 sensor_size:   Optional[Union[Tuple[float], str]] = None):

        if type(board_params) is ChessBoard:
            self.dt: ChessboardDetector = ChessboardDetector(board_params)
        else:
            self.dt: CharucoDetector = CharucoDetector(board_params)

        # self._min_pts: int = 3   # SQPNP method needs at least 3 points
        # self._min_pts: int = 4   # ITERATIVE method needs at least 4 points
        self._min_pts: int = 6     # DLT algorithm needs at least 6 points for pose estimation

        self._grid_cells: int = 15
        self._cells_gamma: float = 2.0
        self._min_cells_weight: float = 0.25   # cells at centre get ~ min weight and cells at the edge get ~ 1.0

        # Defaults

        self.h, self.w = None, None
        self._sensor_size: Union[ArrayLike, None] = None

        self._points2d = None
        self._points_ids = None

        self._points3d = np.asarray(self.dt.points3d)
        self._points3d_j: jnp.ndarray = jnp.asarray(self._points3d)
        self._cornersd3d = np.asarray(self.dt.corners3d)
        self._cornersd3d_j: jnp.ndarray = jnp.asarray(self._cornersd3d)

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
            self.h, self.w = np.asarray(imsize_hw)[:2]
            self._update_imsize()
        # otherwise arrays will be initialised on the first detection

        # Process sensor size input
        if isinstance(sensor_size, str):
            self._sensor_size = monocular_2.SENSOR_SIZES.get(f'''{sensor_size.strip('"')}"''', None)
        elif isinstance(sensor_size, (tuple, list, set, np.ndarray)) and len(sensor_size) == 2:
            self._sensor_size = sensor_size

        # compute theoretical camera matrix if possible
        # (this allows to fix the fx/fy ratio and helps the first guess)
        if None not in (focal_mm, self._sensor_size, self.h, self.w):
            self._th_camera_matrix_j = _maybe_put(monocular_2.estimate_camera_matrix(
                focal_mm,
                self._sensor_size,
                (self.w, self.h))
            )

        # initialise intrinsics if possible
        if self._th_camera_matrix_j is not None:
            self._camera_matrix_j = _maybe_put(self._th_camera_matrix_j.copy())
            self._dist_coeffs_j = _maybe_put(np.zeros(5, dtype=np.float32))

        # Samples stack (to aggregate detections for calibration)
        self._min_stack: int = min_stack
        self._max_stack: int = max_stack
        self.stack_points2d: deque = deque(maxlen=self._max_stack)
        self.stack_points_ids: deque = deque(maxlen=self._max_stack)

        # Error metrics
        self._intrinsics_errors: ArrayLike = np.array([np.inf])
        self._pose_error: float = np.inf

        self.set_visualisation_scale(scale=1)

    def _update_imsize(self):

        self._frame_in = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # TODO - Error normalisation factor: we want to use image diagonal to normalise the errors
        #   np.sum(np.power(self.imsize, 2)) * 1e-6
        self._err_norm = 1

        self._grid_shape = (self._grid_cells, int(np.round((self.w / self.h) * self._grid_cells)))
        self._cumul_grid = np.zeros(self._grid_shape, dtype=bool)
        self._temp_grid = np.zeros(self._grid_shape, dtype=bool)

        # we want to weight the cells based on distance from the image centre (avoids oversampling the centre)
        grid_h, grid_w = self._grid_shape
        cell_h = self.h / grid_h
        cell_w = self.w / grid_w

        # cell centers
        xs = (np.arange(grid_w) + 0.5) * cell_w
        ys = (np.arange(grid_h) + 0.5) * cell_h
        grid_x, grid_y = np.meshgrid(xs, ys)

        # distance from center of the image
        center_x, center_y = self.w / 2, self.h / 2
        distances = np.sqrt((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2) # max dist is center -> one of the corners
        norm_dist = distances / max_distance

        # cells at centre (norm_dist ~ 0) get near min weight, and cells at the edge get near 1.0
        self._grid_weights = self._min_cells_weight + (1 - self._min_cells_weight) * (norm_dist ** self._cells_gamma)

    def set_visualisation_scale(self, scale=1):
        self.BIT_SHIFT = 4
        self.SCALE = scale
        self.shift_factor = 2 ** self.BIT_SHIFT
        self.draw_params = {'shift': self.BIT_SHIFT, 'lineType': cv2.LINE_AA}
        self.text_params = {'fontFace': cv2.FONT_HERSHEY_DUPLEX,
                            'fontScale': 0.8 * self.SCALE,
                            'color': (255, 255, 255),
                            'thickness': 1 * self.SCALE,
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
        return float((self._camera_matrix[np.diag_indices(2)] * (self._sensor_size / np.array([self.w, self.h]))).sum() / 2.0)

    def set_intrinsics(self, camera_matrix: ArrayLike, dist_coeffs: ArrayLike, errors: Optional[ArrayLike] = None):

        self._camera_matrix = np.asarray(camera_matrix)
        self._camera_matrix_j = _maybe_put(self._camera_matrix)

        dist_coeffs = np.asarray(dist_coeffs)
        if len(dist_coeffs) < 5:
            self._dist_coeffs = np.zeros(5, dtype=np.float32)
            self._dist_coeffs[:len(dist_coeffs)] = dist_coeffs
        self._dist_coeffs = dist_coeffs
        self._dist_coeffs_j = _maybe_put(self._dist_coeffs)

        if errors is not None:
            self._intrinsics_errors = np.asarray(errors)
        else:
            self._intrinsics_errors = np.array([np.inf])

    def clear_intrinsics(self):

        if self._th_camera_matrix_j is not None:
            self._camera_matrix_j = _maybe_put(self._th_camera_matrix_j.copy())
            self._dist_coeffs_j = _maybe_put(np.zeros(5, dtype=np.float32))
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
            # cv2.calibrateCamera will complain if the deques contain (N, 2) arrays, it expects (1, N, 2)
            self.stack_points2d.append(self._points2d[np.newaxis, :, :])
            self.stack_points_ids.append(self._points_ids[np.newaxis, :])

    def clear_stacks(self):
        self._cumul_grid.fill(False)
        self.stack_points2d.clear()
        self.stack_points_ids.clear()

    def _map_points_to_grid(self, points):
        nb_rows, nb_cols = self._grid_shape
        cell_height = self.h / nb_rows
        cell_width = self.w / nb_cols

        xs = points[:, 0]
        ys = points[:, 1]
        cols = (xs // cell_width).astype(int)
        rows = (ys // cell_height).astype(int)

        rows = np.clip(rows, 0, nb_rows - 1)
        cols = np.clip(cols, 0, nb_cols - 1)

        return np.column_stack((rows, cols))

    def compute_intrinsics(self, clear_stack=True, fix_aspect_ratio=True, simple_distortion=False, complex_distortion=False):

        if simple_distortion and complex_distortion:
            raise AttributeError("Can't enable simple and complex distortion modes at the same time!")

        # If there is fewer than 5 images (no matter the self._min_stack value), this will NOT be enough
        if len(self.stack_points2d) < 5:
            return  # Abort and keep the stacks

        if self._camera_matrix_j is None and fix_aspect_ratio:
            print('No current camera matrix guess, unfixing aspect ratio.')
            fix_aspect_ratio = False

        calib_flags = 0

        if fix_aspect_ratio:
            calib_flags |= cv2.CALIB_FIX_ASPECT_RATIO           # This locks the ratio of fx and fy
        if simple_distortion or not self.has_intrinsics:        # The first iteration will use the simple model - this helps
            calib_flags |= cv2.CALIB_FIX_K3
        if complex_distortion:
            calib_flags |= cv2.CALIB_RATIONAL_MODEL
        if self.has_intrinsics:
            calib_flags |= cv2.CALIB_USE_INTRINSIC_GUESS        # Important, otherwise it ignores the passed intrinsics

        calib_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

        # We need to copy to a new array, because OpenCV uses these as Input/Output buffers
        if self.has_intrinsics:
            current_camera_matrix = np.copy(self._camera_matrix)
            current_dist_coeffs = np.copy(self._dist_coeffs)
        else:
            current_camera_matrix = None
            current_dist_coeffs = None

        try:
            if type(self.dt) is ChessboardDetector:

                object_points_list = [self.dt.points3d.astype(np.float32)] * len(self.stack_points2d)

                global_intr_error, new_camera_matrix, new_dist_coeffs, stack_rvecs, stack_tvecs, std_intrinsics, std_extrinsics, stack_intr_errors = cv2.calibrateCameraExtended(
                    objectPoints=object_points_list,
                    imagePoints=self.stack_points2d,
                    imageSize=(self.w, self.h),
                    cameraMatrix=current_camera_matrix,  # Input/Output /!\
                    distCoeffs=current_dist_coeffs,      # Input/Output /!\
                    flags=calib_flags
                )

            else:

                calib_flags |= cv2.CALIB_USE_LU
                # calib_flags |= cv2.CALIB_USE_QR

                # Compute calibration using all the frames we selected
                global_intr_error, new_camera_matrix, new_dist_coeffs, stack_rvecs, stack_tvecs, std_intrinsics, std_extrinsics, stack_intr_errors = cv2.aruco.calibrateCameraCharucoExtended(
                    charucoCorners=self.stack_points2d,
                    charucoIds=self.stack_points_ids,
                    board=self.dt.board,
                    imageSize=(self.w, self.h),
                    cameraMatrix=current_camera_matrix,     # Input/Output /!\
                    distCoeffs=current_dist_coeffs,         # Input/Output /!\
                    flags=calib_flags)

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

            # The following things should not be possible ; if any happens, then we trash the stack and abort
            if (new_camera_matrix < 0).any() or (new_camera_matrix[:2, 2] >= np.array(self.w, self.h)).any():
                self.clear_stacks()
                return

            # store intrinsics if it is the very first estimation
            if not self.has_intrinsics or np.inf in self._intrinsics_errors:

                self._camera_matrix = new_camera_matrix
                self._dist_coeffs = new_dist_coeffs
                self._camera_matrix_j = _maybe_put(self._camera_matrix)
                self._dist_coeffs_j = _maybe_put(self._dist_coeffs)

                self._intrinsics_errors = stack_intr_errors

                print(f"---Computed intrinsics---")

            # or update them if this stack's errors are better
            elif self._check_new_errors(stack_intr_errors, self._intrinsics_errors):

                self._camera_matrix = new_camera_matrix
                self._dist_coeffs = new_dist_coeffs
                self._camera_matrix_j = _maybe_put(self._camera_matrix)
                self._dist_coeffs_j = _maybe_put(self._dist_coeffs)

                self._intrinsics_errors = stack_intr_errors

                print(f"---Updated intrinsics---")

        except cv2.error as e:
            print(e)

        if clear_stack:
            self.clear_stacks()

    def compute_extrinsics(self, refine=True):

        # We need a detection to get the extrinsics relative to it
        if not self.has_detection:
            self._rvec, self._tvec = None, None
            self._rvec_j, self._tvec_j = None, None
            self._pose_error = np.inf
            return

        # We also need intrinsics
        if not self.has_intrinsics:
            self._rvec, self._tvec = None, None
            self._rvec_j, self._tvec_j = None, None
            self._pose_error = np.inf
            return

        if type(self.dt) is CharucoDetector:
            # TODO: Check collinearity for classic chessboards too

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
            nb_solutions, rvecs, tvecs, solutions_errors = cv2.solvePnPGeneric(self._points3d[self._points_ids],
                                                         self._points2d,
                                                         self._camera_matrix,
                                                         self._dist_coeffs,
                                                         flags=pnp_flags)
        except cv2.error as e:
            self._rvec, self._tvec = None, None
            self._rvec_j, self._tvec_j = None, None
            self._pose_error = np.inf
            return

        # If no solution, or if multiple solutions were found, abort
        if nb_solutions != 1:
            self._rvec, self._tvec = None, None
            self._rvec_j, self._tvec_j = None, None
            self._pose_error = np.inf
            return

        # if only one solution, continue
        self._pose_error = float(solutions_errors.squeeze()) / self._err_norm  # TODO: Normalise error on image diagonal
        rvec, tvec = rvecs[0], tvecs[0]

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
                rvec=rvec,      # Input/Output /!\
                tvec=tvec,      # Input/Output /!\
                VVSlambda=1.0)

            # TODO - Test whether the Levenberg-Marquardt alternative solvePnPRefineLM() is better or not

        self._rvec, self._tvec = rvec.squeeze(), tvec.squeeze()
        self._rvec_j, self._tvec_j = _maybe_put(self._rvec), _maybe_put(self._tvec)

    def _compute_new_area(self) -> float:

        if not self.has_detection:
            return 0.0

        cells_indices = self._map_points_to_grid(self._points2d)
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
            self._update_imsize()

        # Load frame
        np.copyto(self._frame_in[:], frame)

        # Detect
        if type(self.dt) is ChessboardDetector:
            self._points2d, self._points_ids = self.dt.detect(self._frame_in,
                                                              refine_points=True)
        else:
            self._points2d, self._points_ids = self.dt.detect(self._frame_in,
                                                              camera_matrix=self._camera_matrix,
                                                              dist_coeffs=self._dist_coeffs,
                                                              refine_markers=True,
                                                              refine_points=True)

    def auto_register_area_based(self,
                                 area_threshold:        float = 0.2,
                                 nb_points_threshold:   int = 4
                                 ) -> bool:

        novel_area = self._compute_new_area()
        if novel_area >= area_threshold and self.nb_points >= nb_points_threshold:
            self.register_sample()
            return True
        else:
            return False

    def auto_compute_intrinsics(self,
                                coverage_threshold:     float = 80.0,
                                stack_length_threshold: int = 15,
                                simple_focal:           bool = False,
                                simple_distortion:      bool = False,
                                complex_distortion:     bool = False
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

    def draw_coverage_grid(self, img):
        # mask of covered cells at full resolution
        mask = cv2.resize(
            self._cumul_grid.astype(np.uint8),
            np.flip(img.shape[:2]),
            interpolation=cv2.INTER_NEAREST
        )

        # green overlay
        overlay = np.zeros_like(img)
        overlay[mask > 0] = (0, 255, 0)

        alpha = 0.3
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    def visualise(self, errors_mm=False):

        frame_out = np.copy(self._frame_in)

        if self.has_detection:
            # if corners have been found show them as red dots
            detected_points_int = (self._points2d * self.shift_factor).astype(np.int32)

            for xy in detected_points_int:
                frame_out = cv2.circle(frame_out, xy, 4 * self.SCALE, (0, 0, 255), 4 * self.SCALE, **self.draw_params)

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
            reproj_points_int = (reproj_points * self.shift_factor).astype(np.int32)

            for i, xy in enumerate(reproj_points_int):
                if i in self._points_ids:
                    frame_out = cv2.circle(frame_out, xy, 2 * self.SCALE, (0, 255, 255), 4 * self.SCALE, **self.draw_params)
                else:
                    frame_out = cv2.circle(frame_out, xy, 4 * self.SCALE, (255, 255, 255), 4 * self.SCALE, **self.draw_params)

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
                                                fontScale=0.3 * self.SCALE,
                                                color=(0, 255, 255),
                                                thickness=1 * self.SCALE,
                                                lineType=cv2.LINE_AA)

        # Draw grid-based coverage overlay
        frame_out = self.draw_coverage_grid(frame_out)

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
                    dist_coeffs=jnp.zeros(8, dtype=jnp.float32),
                    rvec=self._rvec_j,
                    tvec=self._tvec_j)

                pts_int = np.asarray((corners2d_j * self.shift_factor).astype(jnp.int32))

                frame_out = cv2.polylines(frame_out,
                                          [pts_int],
                                          True,
                                          (255, 0, 255),
                                          1 * self.SCALE,
                                          **self.draw_params)

        # Add information text to the visualisation image
        frame_out = cv2.putText(frame_out,
                                f"Points: {self.nb_points}/{self.dt.total_points}",
                                (30, 30 * self.SCALE),
                                **self.text_params)
        frame_out = cv2.putText(frame_out,
                                f"Area: {self.coverage:.2f}% ({len(self.stack_points2d)} snapshots)",
                                (30, 60 * self.SCALE),
                                **self.text_params)

        txt = f"{self._pose_error:.3f} px" if np.all(self._pose_error != np.inf) else '-'
        frame_out = cv2.putText(frame_out,
                                f"Current reprojection error: {txt}",
                                (30, 90 * self.SCALE),
                                **self.text_params)

        avg_intr_err = np.nanmean(self._intrinsics_errors)
        txt = f"{avg_intr_err:.3f} px" if np.all(avg_intr_err != np.inf) else '-'
        frame_out = cv2.putText(frame_out,
                                f"Best average reprojection error: {txt}",
                                (30, 120 * self.SCALE),
                                **self.text_params)

        f_mm = self.focal_mm
        txt = f"{f_mm:.2f} mm" if f_mm else '-'
        frame_out = cv2.putText(frame_out,
                                f"Estimated focal: {txt}",
                                (30, 150 * self.SCALE),
                                **self.text_params)
        return frame_out


class MultiviewCalibrationTool:
    """
    Class to aggregate multiple monocular detections into multi-view samples, compute, and refine cameras poses
    """

    def __init__(self, nb_cameras, origin_camera_idx=0, min_poses=15, max_poses=100, min_detections=15, max_detections=100):

        self.nb_cameras = nb_cameras
        self._origin_idx = origin_camera_idx
        self._min_poses = min_poses
        self._min_detections = min_detections

        self._is_refined = False
        self._intrinsics_records = np.zeros(self.nb_cameras, dtype=bool)

        self._multi_cam_mat = np.zeros((nb_cameras, 3, 3))
        self._multi_dist_coeffs = np.zeros((nb_cameras, 14))
        self._multi_cam_mat_refined = np.zeros((nb_cameras, 3, 3))
        self._multi_dist_coeffs_refined = np.zeros((nb_cameras, 14))

        self._detections_by_frame = defaultdict(dict)
        self._poses_by_frame = defaultdict(dict)

        self._detections_stack = deque(maxlen=max_detections)
        self._poses_stack = deque(maxlen=max_poses)

        self._poses_per_camera = {cam_idx: [] for cam_idx in range(nb_cameras)}

        self._multi_rvecs_estim: Optional[ArrayLike] = None     # when not None, shape is (self.nb_cams, 3)
        self._multi_tvecs_estim: Optional[ArrayLike] = None     # when not None, shape is (self.nb_cams, 3)
        self._multi_rvecs_refined: Optional[ArrayLike] = None   # when not None, shape is (self.nb_cams, 3)
        self._multi_tvecs_refined: Optional[ArrayLike] = None   # when not None, shape is (self.nb_cams, 3)

    @property
    def nb_detection_samples(self):
        return len(self._detections_stack)

    @property
    def nb_pose_samples(self):
        return len(self._poses_stack)

    @property
    def origin_camera(self):
        return self._origin_idx

    @origin_camera.setter
    def origin_camera(self, value: int):
        self._origin_idx = value
        self.clear_poses()
        print(f'[MultiviewCalibrationTool] Origin set to camera {self._origin_idx}')

    @property
    def has_extrinsics(self):
        rvecs, tvecs = self.extrinsics
        return rvecs is not None and tvecs is not None

    @property
    def has_intrinsics(self):
        return np.all(self._intrinsics_records)

    @property
    def is_refined(self):
        return self._is_refined

    @property
    def extrinsics(self):
        if self._is_refined:
            return self._multi_rvecs_refined, self._multi_tvecs_refined
        else:
            return self._multi_rvecs_estim, self._multi_tvecs_estim

    def register_intrinsics(self, cam_idx: int, camera_matrix: ArrayLike, dist_coeffs: ArrayLike):
        self._multi_cam_mat[cam_idx, :, :] = camera_matrix
        self._multi_dist_coeffs[cam_idx, :len(dist_coeffs)] = dist_coeffs
        self._intrinsics_records[cam_idx] = True

    def intrinsics(self):
        if self._is_refined:
            return self._multi_cam_mat_refined, self._multi_dist_coeffs_refined
        else:
            return self._multi_cam_mat, self._multi_dist_coeffs

    def register_pose(self, frame_idx: int, cam_idx: int, rvec: ArrayLike, tvec: ArrayLike):
        """
        This registers estimated monocular camera poses and stores them as complete pose samples
        """
        self._poses_by_frame[frame_idx][cam_idx] = (rvec, tvec)

        # Check if we have the reference camera and at least another one
        if self._origin_idx in self._poses_by_frame[frame_idx].keys() and len(self._poses_by_frame[frame_idx]) > 1:
            sample = self._poses_by_frame.pop(frame_idx)
            origin_rvec, origin_tvec = sample[self._origin_idx]

            # Remap the poses to a common origin (i.e. the reference camera)
            for cam, (rvec, tvec) in sample.items():
                remapped_rvec, remapped_tvec = geometry.remap_rtvecs(rvec, tvec, origin_rvec, origin_tvec)
                self._poses_per_camera[cam].append((remapped_rvec, remapped_tvec))

    def register_detection(self, frame_idx: int, cam_idx: int, points2d: ArrayLike, points_ids: ArrayLike):
        """
        This registers points detections from multiple cameras and stores them as a complete detection samples
        """
        self._detections_by_frame[frame_idx][cam_idx] = (points2d, points_ids)

        # Check if we have all cameras for that frame
        if len(self._detections_by_frame[frame_idx]) == self.nb_cameras:
            dbf = self._detections_by_frame.pop(frame_idx)
            # list of lists of tuples of arrays: M[N[(P_points, P_ids)]] because the number of points P is variable
            sample = [dbf[cidx] for cidx in range(self.nb_cameras)]
            self._detections_stack.append(sample)

            # Prepare data for triangulation
            points2d_list = [det[0] for det in sample]
            points2d_ids_list = [det[1] for det in sample]

            # Only triangulate if extrinsics are available
            if self.has_intrinsics and self.has_extrinsics:
                points3d, points3d_ids = multiview.triangulation(points2d_list, points2d_ids_list,
                                                                 self._multi_rvecs_estim, self._multi_tvecs_estim,
                                                                 self._multi_cam_mat, self._multi_dist_coeffs)
                if points3d is not None:
                    print("Triangulation succeeded, emitting 3D points.")
                else:
                    print("Triangulation failed or returned no points.")
            else:
                print("Extrinsics or Intrinsics not available yet; cannot triangulate.")

    def clear_intrinsics(self, clear_refined=True):
        self._intrinsics_records.fill(False)
        self._multi_cam_mat.fill(0)
        self._multi_dist_coeffs.fill(0)
        if clear_refined:
            self._multi_cam_mat_refined.fill(0)
            self._multi_dist_coeffs_refined.fill(0)

    def clear_extrinsics(self, clear_refined=True):
        self._multi_rvecs_estim = None
        self._multi_tvecs_estim = None
        if clear_refined:
            self._multi_rvecs_refined = None
            self._multi_tvecs_refined = None

    def clear_poses(self):
        self._poses_per_camera = {cam_idx: [] for cam_idx in range(self.nb_cameras)}
        self._poses_by_frame = defaultdict(dict)

    def clear_detections(self):
        self._detections_stack.clear()
        self._detections_by_frame = defaultdict(dict)

    def estimate_extrinsics(self, clear_poses_stack=True):
        """
        This uses the complete pose samples to compute a first estimate of the cameras arrangement
        """
        if not all(len(self._poses_per_camera[cam_idx]) >= self._min_poses for cam_idx in range(self.nb_cameras)):
            # print(f"Waiting for at least {self._min_poses} samples per camera; "
            #       f"current counts: {[len(self._poses_per_camera[cam]) for cam in range(self.nb_cameras)]}")
            return

        # Each camera’s list is converted to an array of shape (M, 3) where M varies
        n_m_rvecs = []
        n_m_tvecs = []
        for cam_idx in range(self.nb_cameras):
            samples = self._poses_per_camera.get(cam_idx, [])
            if not samples:
                # If no samples are available for this camera -> empty array
                n_m_rvecs.append(np.empty([0, 3]))
                n_m_tvecs.append(np.empty([0, 3]))
            else:
                samples_arr = np.array(samples)  # shape: (M, 2, 3) because each sample is (rvec, tvec)
                m_rvecs = samples_arr[:, 0, :]
                m_tvecs = samples_arr[:, 1, :]
                n_m_rvecs.append(m_rvecs)
                n_m_tvecs.append(m_tvecs)

        self._multi_rvecs_estim, self._multi_tvecs_estim = multiview.bestguess_rtvecs(n_m_rvecs, n_m_tvecs)

        if clear_poses_stack:
            self.clear_poses()
