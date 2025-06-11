from collections import deque
from typing import Union, Optional, Iterable, Tuple
import cv2
import numpy as np
from jax import numpy as jnp
from numpy.typing import ArrayLike
from scipy import stats as stats
from scipy.spatial.distance import cdist
from mokap.calibration.detectors import ChessboardDetector, CharucoDetector
from mokap.utils import maybe_put
from mokap.utils.datatypes import ChessBoard, CharucoBoard
from mokap.utils.geometry.camera import SENSOR_SIZES, estimate_camera_matrix
from mokap.utils.geometry.projective import project_points


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
            self._sensor_size = SENSOR_SIZES.get(f'''{sensor_size.strip('"')}"''', None)
        elif isinstance(sensor_size, (tuple, list, set, np.ndarray)) and len(sensor_size) == 2:
            self._sensor_size = sensor_size

        # compute theoretical camera matrix if possible
        # (this allows to fix the fx/fy ratio and helps the first estimation)
        if None not in (focal_mm, self._sensor_size, self.h, self.w):
            self._th_camera_matrix_j = maybe_put(estimate_camera_matrix(
                focal_mm,
                self._sensor_size,
                (self.w, self.h))
            )

            self._camera_matrix_j = maybe_put(self._th_camera_matrix_j.copy())
            self._dist_coeffs_j = maybe_put(np.zeros(8, dtype=np.float32))

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
        self._camera_matrix_j = maybe_put(self._camera_matrix)

        dist_coeffs = np.asarray(dist_coeffs)
        dist_coeffs = dist_coeffs[:max(8, len(dist_coeffs))]
        dist_coeffs_pad = np.zeros(8, dtype=np.float32)
        dist_coeffs_pad[:len(dist_coeffs)] = dist_coeffs

        self._dist_coeffs = dist_coeffs_pad
        self._dist_coeffs_j = maybe_put(self._dist_coeffs)

        if errors is not None:
            self._intrinsics_errors = np.asarray(errors)
        else:
            self._intrinsics_errors = np.array([np.inf])

    def set_extrinsics(self, rvec: ArrayLike, tvec: ArrayLike):
        self._rvec = rvec
        self._tvec = tvec
        self._rvec_j = jnp.asarray(self._rvec)
        self._tvec_j = jnp.asarray(self._tvec)

    def clear_intrinsics(self):

        if self._th_camera_matrix_j is not None:
            self._camera_matrix_j = maybe_put(self._th_camera_matrix_j)
            self._dist_coeffs_j = maybe_put(self._zero_coeffs)
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
                self._camera_matrix_j = maybe_put(self._camera_matrix)
                self._dist_coeffs_j = maybe_put(self._dist_coeffs)

                self._intrinsics_errors = stack_intr_errors

                if self._verbose:
                    print(f"[INFO] [MonocularCalibrationTool] Computed intrinsics")

            # or update them if this stack's errors are better
            elif self._check_new_errors(stack_intr_errors, self._intrinsics_errors):

                self._camera_matrix = new_camera_matrix
                self._dist_coeffs = new_dist_coeffs
                self._camera_matrix_j = maybe_put(self._camera_matrix)
                self._dist_coeffs_j = maybe_put(self._dist_coeffs)

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
        self._rvec_j, self._tvec_j = maybe_put(self._rvec), maybe_put(self._tvec)

    def _compute_new_area(self) -> float:

        if not self.has_detection:
            return 0.0

        cells_indices = np.fliplr(
            np.clip((self._points2d // ((self.h, self.w) / self._grid_shape)).astype(np.int32), [0, 0],
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
            reproj_points_j = project_points(
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
                corners2d_j = project_points(
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
