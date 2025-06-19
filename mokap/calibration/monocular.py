import logging
from collections import deque
from typing import Union, Optional, Tuple, Sequence
import cv2
import numpy as np
import jax
from jax import numpy as jnp
from jax.typing import ArrayLike
from scipy import stats as stats
from mokap.calibration.common import solve_pnp_robust, calibrate_camera_robust
from mokap.calibration.detectors import ChessboardDetector, CharucoDetector
from mokap.utils.datatypes import ChessBoard, CharucoBoard, DistortionModel
from mokap.utils import SENSOR_SIZES, estimate_camera_matrix
from mokap.utils.geometry.projective import project_points

logger = logging.getLogger(__name__)


class MonocularCalibrationTool:
    """
    This object is stateful for the intrinsics *only*
    """

    def __init__(self,
                 calibration_board: Union[ChessBoard, CharucoBoard],
                 imsize_hw: Optional[Sequence[int]] = None,  # OpenCV order (height, width)
                 min_stack: int = 15,
                 max_stack: int = 100,
                 focal_mm: Optional[int] = None,
                 sensor_size: Optional[Union[Tuple[float], str]] = None
                 ):

        if calibration_board.type == 'chessboard':
            self.detector: ChessboardDetector = ChessboardDetector(calibration_board)
        else:
            self.detector: CharucoDetector = CharucoDetector(calibration_board)

        self.calibration_board = calibration_board

        # TODO: grid parameters should be configurable from the config file
        self._nb_grid_cells: int = 15
        self._cells_gamma: float = 2.0
        self._min_cells_weight: float = 0.25  # cells at centre get ~ min weight and cells at the edge get ~ 1.0

        self._min_pts = 6   # DLT algorithm in calibrateCamera fails if less than 6 points
        self._min_stack: int = min_stack
        self._max_stack: int = max_stack

        self._img_h, self._img_w = (imsize_hw[0], imsize_hw[1]) if imsize_hw else (0, 0)

        # TODO: Error normalisation factor: we want to use image diagonal to normalise the errors
        self._err_norm = 1

        self._points2d_np = None
        self._points_ids_np = None

        # Object points in 3D and their reprojection
        self._object_points_3d = jnp.concatenate([jnp.asarray(self.calibration_board.object_points),
                                                  jnp.asarray(self.calibration_board.corner_points)])
        self._reprojected_points = np.full((self._object_points_3d.shape[0], 2), np.nan, dtype=np.float32)

        # Samples stack
        self.stack_points2d: deque = deque(maxlen=self._max_stack)
        self.stack_points_ids: deque = deque(maxlen=self._max_stack)

        # Error metrics
        # The raw OpenCV errors for comparing future calibrations
        self._intrinsics_errors_opencv: ArrayLike = np.array([np.inf])
        # Also store the true RMS for comparison with other parts of the app
        self._intrinsics_errors_rms: ArrayLike = np.array([np.inf])

        self._pose_error: float = np.nan

        # Where to store the intrinsics
        self._camera_matrix: Union[jnp.ndarray, None] = None
        self._dist_coeffs: Union[jnp.ndarray, None] = None

        # Current estimated rvec and tvec (for a given frame)
        self._curr_rvec_b2c: Union[jnp.ndarray, None] = None
        self._curr_tvec_b2c: Union[jnp.ndarray, None] = None

        if imsize_hw is not None:
            self._update_grid(imsize_hw)
            # otherwise grid-related things will be initialised on the first detection

        # Process sensor size input
        self._cam_sensor_size = None

        if isinstance(sensor_size, str):
            self._cam_sensor_size = np.array(SENSOR_SIZES.get(f'''{sensor_size.strip('"')}"''', [0.0, 0.0]))

        elif isinstance(sensor_size, (tuple, list, set, ArrayLike)) and len(sensor_size) == 2:
            self._cam_sensor_size = np.array(sensor_size)

        # compute theoretical camera matrix if possible
        # (this helps the first estimation)
        self._zero_coeffs = jnp.zeros(8, dtype=jnp.float32)
        self._theoretical_cam_mat: Union[jnp.ndarray, None] = None

        if None not in (focal_mm, self._img_w, self._img_h) and self._cam_sensor_size is not None:
            theoretical_cam_mat = estimate_camera_matrix(
                f_mm=focal_mm,
                image_wh_px=(self._img_w, self._img_h),
                sensor_wh_mm=self._cam_sensor_size,
                pixel_pitch_um=None             # TODO: probably better to use this instead of sensor size
            )

            self._theoretical_cam_mat = jax.device_put(theoretical_cam_mat)
            self._camera_matrix = jax.device_put(self._theoretical_cam_mat.copy())
            self._dist_coeffs = jax.device_put(self._zero_coeffs)

    def _update_grid(self, imsize_hw):
        """ Internal method to set or update arrays related to image size """

        self._img_h, self._img_w = imsize_hw[:2]

        self._grid_shape = np.array([self._nb_grid_cells, int(np.round((self._img_w / self._img_h) * self._nb_grid_cells))],
                                    dtype=np.uint32)
        self._cumul_grid = np.zeros(self._grid_shape, dtype=bool)  # Keeps the total covered area
        self._temp_grid = np.zeros(self._grid_shape, dtype=bool)  # buffer reset at each new sample

        # we want to weight the cells based on distance from the image centre (to avoid oversampling the centre)
        grid_h, grid_w = self._grid_shape
        cell_h = self._img_h / grid_h
        cell_w = self._img_w / grid_w

        # cell centers
        xs = (np.arange(grid_w) + 0.5) * cell_w
        ys = (np.arange(grid_h) + 0.5) * cell_h
        grid_x, grid_y = np.meshgrid(xs, ys)

        # distance from centre of the image
        center_x, center_y = self._img_w / 2, self._img_h / 2
        distances = np.sqrt((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)  # max dist is from the centre to one of the corners
        norm_dist = distances / max_distance

        # cells near centre (norm_dist ~ 0) get near min weight, and cells at the edge get near 1.0
        self._grid_weights = self._min_cells_weight + (1 - self._min_cells_weight) * (norm_dist ** self._cells_gamma)

    @property
    def detection(self) -> Tuple[ArrayLike, ArrayLike]:
        return self._points2d_np, self._points_ids_np

    @property
    def intrinsics(self) -> Tuple[ArrayLike, ArrayLike]:
        return self._camera_matrix, self._dist_coeffs

    @property
    def extrinsics(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._curr_rvec_b2c is None or self._curr_tvec_b2c is None:
            return None, None
        # have to copy (the jax versions are read only)
        return np.asarray(self._curr_rvec_b2c).copy(), np.asarray(self._curr_tvec_b2c).copy()

    @property
    def has_detection(self) -> bool:
        return all(x is not None for x in self.detection) and len(self._points2d_np) >= self._min_pts

    @property
    def has_intrinsics(self) -> bool:
        return all(x is not None for x in self.intrinsics)

    @property
    def has_extrinsics(self) -> bool:
        return all(x is not None for x in self.extrinsics)

    @property
    def curr_nb_points(self) -> int:
        return self._points2d_np.shape[0] if self._points2d_np is not None else 0

    @property
    def curr_nb_samples(self) -> int:
        return len(self.stack_points2d)

    @property
    def grid(self) -> np.array:
        return self._cumul_grid

    @property
    def pct_coverage(self) -> float:
        return float(np.sum(self._cumul_grid) / self._cumul_grid.size) * 100

    @property
    def reprojected_points2d(self) -> ArrayLike:
        return self._reprojected_points

    @property
    def pose_error(self):
        return self._pose_error

    @property
    def intrinsics_errors(self) -> ArrayLike:
        return self._intrinsics_errors_rms

    @property
    def focal(self) -> float:

        if not self.has_intrinsics:
            return 0.0

        f_px = jnp.sum(self._camera_matrix[jnp.diag_indices(2)]) / 2.0
        return float(f_px)

    def set_intrinsics(self, camera_matrix: ArrayLike, dist_coeffs: ArrayLike, errors: Optional[ArrayLike] = None):

        self._camera_matrix = jax.device_put(camera_matrix)
        self._dist_coeffs = jax.device_put(jnp.pad(dist_coeffs, (0, max(0, 8 - len(dist_coeffs))), 'constant', constant_values=0.0))

        if errors is not None:
            # Store the raw OpenCV errors for internal comparison
            self._intrinsics_errors_opencv = np.asarray(errors)
            # Store the standardized RMS error for external reporting
            self._intrinsics_errors_rms = self._intrinsics_errors_opencv / np.sqrt(2)
        else:
            self._intrinsics_errors_opencv = np.array([np.inf])
            self._intrinsics_errors_rms = np.array([np.inf])

    def clear_intrinsics(self):

        if self._theoretical_cam_mat is not None:
            self._camera_matrix = jax.device_put(self._theoretical_cam_mat)
            self._dist_coeffs = jax.device_put(self._zero_coeffs)

        else:
            self._camera_matrix = None
            self._dist_coeffs = None

        self._intrinsics_errors_opencv = np.array([np.inf])
        self._intrinsics_errors_rms = np.array([np.inf])

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

    def _compute_new_area(self) -> float:

        if not self.has_detection:
            return 0.0

        cells_indices = np.fliplr(
            np.clip((self._points2d_np // ((self._img_h, self._img_w) / self._grid_shape)).astype(np.int32), [0, 0],
                    np.flip(self._grid_shape - 1)))

        rows, cols = cells_indices.T
        self._temp_grid[rows, cols] = True

        # Novel area = cells that are in the current grid but not in the cumulative grid
        novel_cells = self._temp_grid & (~self._cumul_grid)
        novel_weight = self._grid_weights[novel_cells].sum()
        total_weight = self._grid_weights.sum()

        # update cumulative coverage and clear current temporary grid
        self._cumul_grid |= self._temp_grid
        self._temp_grid.fill(False)

        return float(novel_weight / total_weight) * 100

    def detect(self, frame: np.ndarray):

        # TODO: Detector could be taken out completely from the monocular tool

        # initialise or update the internal arrays to match frame size if needed
        if self._img_h == 0 or self._img_w == 0 or self._img_h != frame.shape[0] or self._img_w != frame.shape[1]:
            self._update_grid(frame.shape)

        # Detect
        if type(self.detector) is ChessboardDetector:
            self._points2d_np, self._points_ids_np = self.detector.detect(
                frame,
                refine_points=True
            )

        else:
            self._points2d_np, self._points_ids_np = self.detector.detect(
                frame,
                camera_matrix=np.asarray(self._camera_matrix),
                dist_coeffs=np.asarray(self._dist_coeffs),
                refine_markers=True,
                refine_points=True
            )

    def register_sample(self, min_new_area: float = 0.2) -> bool:
        """ Registers a sample if the new area is above threshold """

        if not self.has_detection or self.curr_nb_points < self._min_pts:
            return False

        # if no threshold, or if the new area is above thrshold
        if min_new_area <= 0 or self._compute_new_area() > min_new_area:
            self.stack_points2d.append(self._points2d_np[np.newaxis, ...])
            self.stack_points_ids.append(self._points_ids_np[np.newaxis, ...])
            return True

        return False

    def compute_intrinsics(self,
                           fix_aspect_ratio:    bool = True,
                           distortion_model:    DistortionModel = 'standard',
                           keep_stacks:         bool = False
                           ) -> bool:
        """ Compute the camera intrinsics using the accumulated samples """

        if len(self.stack_points2d) < self._min_stack:
            return False

        current_camera_matrix, current_dist_coeffs = np.asarray(self._camera_matrix), np.asarray(self._dist_coeffs)

        calib_results = calibrate_camera_robust(
            board=self.calibration_board,
            image_points_stack=self.stack_points2d,
            image_ids_stack=self.stack_points_ids,
            image_size_wh=(self._img_w, self._img_h),
            initial_K=current_camera_matrix,
            initial_D=current_dist_coeffs,
            distortion_model=distortion_model,
            fix_aspect_ratio=fix_aspect_ratio and (current_camera_matrix is not None)
        )

        if not calib_results.success:
            return False

        # Get the raw per-view errors from OpenCV
        pve_opencv = calib_results.per_view_errors

        # We don't have intrinsics yet
        if not self.has_intrinsics or np.inf in self._intrinsics_errors_opencv:
            self.set_intrinsics(calib_results.K_new, calib_results.D_new, pve_opencv)
            logger.info(f"[MonocularCalibrationTool] Computed intrinsics.")

        # Decide whether to accept the new intrinsics or not by comparing the raw OpenCV errors
        elif self._check_new_errors(pve_opencv, self._intrinsics_errors_opencv):
            self.set_intrinsics(calib_results.K_new, calib_results.D_new, pve_opencv)
            logger.info(f"[MonocularCalibrationTool] Updated intrinsics.")

        # Default to clear on success only, unless asked not to
        if calib_results.success and not keep_stacks:
            self.clear_stacks()

        # if failure, keep stacks by default
        return True

    def compute_extrinsics(self, refine: bool = True) -> bool:
        """ Estimates (monocularly) the camera's pose relative to the board, in the current frame """

        # we need a detection and to have intrinsics to be able to compute extrinsics
        if not self.has_detection or not self.has_intrinsics:
            self._curr_rvec_b2c, self._curr_tvec_b2c = None, None
            self._pose_error = np.nan
            return False

        if self.calibration_board.type == 'charuco':
            # TODO: Check collinearity for classic chessboards too?

            # if the points are collinear, extrinsics estimation is garbage, so abort
            if cv2.aruco.testCharucoCornersCollinear(self.calibration_board.to_opencv(), self._points_ids_np):
                self._curr_rvec_b2c, self._curr_tvec_b2c = None, None
                self._pose_error = np.nan
                return False

        object_points_subset = self.calibration_board.object_points[self._points_ids_np]

        success, rvec_b2c, tvec_b2c, pose_errors = solve_pnp_robust(
            object_points=object_points_subset,
            image_points=self._points2d_np,
            camera_matrix=np.asarray(self._camera_matrix),
            dist_coeffs=np.asarray(self._dist_coeffs),
            refine_method='VVS' if refine else None
        )

        if not success:
            self._curr_rvec_b2c, self._curr_tvec_b2c = None, None
            self._pose_error = np.nan
            return False

        # if all good, push to GPU and store
        self._curr_rvec_b2c = jax.device_put(rvec_b2c.squeeze())
        self._curr_tvec_b2c = jax.device_put(tvec_b2c.squeeze())
        # and store the standard RMS error
        self._pose_error = pose_errors['rms']

        return True

    def reproject(self) -> Optional[ArrayLike]:
        """ Reprojects board points to the image plane for visualisation or error computation """

        if not self.has_intrinsics or not self.has_extrinsics:
            return None

        self._reprojected_points = project_points(
            self._object_points_3d, self._curr_rvec_b2c, self._curr_tvec_b2c, self._camera_matrix, self._dist_coeffs)

    def clear_grid(self):
        if self._cumul_grid is not None:
            self._cumul_grid.fill(False)
        if self._temp_grid is not None:
            self._temp_grid.fill(False)

    def clear_stacks(self):
        self.clear_grid()
        self.stack_points2d.clear()
        self.stack_points_ids.clear()