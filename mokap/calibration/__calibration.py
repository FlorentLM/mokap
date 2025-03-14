from collections import defaultdict, deque
from pathlib import Path
import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=5)
import cv2
import toml
import scipy.stats as stats
from scipy.spatial.distance import cdist
from mokap.utils import geometry, generate_charuco
from mokap.calibration import monocular, multiview
from typing import List


class DetectionTool:
    def __init__(self, board_params):

        # Charuco board and detector parameters
        self.board = generate_charuco(
            board_rows=board_params['rows'],
            board_cols=board_params['cols'],
            square_length_mm=board_params['square_length'],
            marker_bits=board_params['markers_size']
        )
        aruco_dict = self.board.getDictionary()
        self.detector_parameters = cv2.aruco.DetectorParameters()
        self.detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, detectorParams=self.detector_parameters)

        # Maximum number of board points and distances
        self._total_points: int = len(self.board.getChessboardCorners())
        self._total_distances: int = int((self._total_points * (self._total_points - 1)) / 2.0)

        # Create 3D coordinates for board corners (in board-centric coordinates)
        self._n_cols, self._n_rows = self.board.getChessboardSize()
        self._board_points_3d = self.board.getChessboardCorners()

        self._board_corners_3d = (np.array([[0, 0, 0],
                                           [0, 1, 0],
                                           [1, 1, 0],
                                           [1, 0, 0]], dtype=np.float32) *
                                  [self._n_cols, self._n_rows, 0] * self.board.getSquareLength())

    @property
    def points3d(self):
        """ Returns the coordinates of the chessboard points in 3D (in board-centric coordinates) """
        return self._board_points_3d

    @property
    def corners3d(self):
        return self._board_corners_3d

    @property
    def total_points(self):
        return self._total_points

    @property
    def total_distances(self):
        return self._total_distances

    @property
    def board_dims(self):
        return self._n_cols, self._n_rows

    def detect(self, frame, camera_matrix=None, dist_coeffs=None, refine_markers=True, refine_points=False):

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        points2d_coords = None
        points2d_ids = None

        # Detect and refine aruco markers
        markers_coords, marker_ids, rejected = self.detector.detectMarkers(frame)

        if refine_markers:
            markers_coords, marker_ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(
                image=frame,
                board=self.board,
                detectedCorners=markers_coords,     # Input/Output /!\
                detectedIds=marker_ids,             # Input/Output /!\
                rejectedCorners=rejected,           # Input/Output /!\
                parameters=self.detector_parameters,
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
                    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 75, 0.01)

                    chessboard_points = cv2.cornerSubPix(frame, chessboard_points,
                                                         winSize=(11, 11),
                                                         zeroZone=(-1, -1),
                                                         criteria=crit)
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
    def __init__(self, board_params, imsize_hw=None, min_stack=15, max_stack=100, focal_mm=None, sensor_size=None):

        self.dt = DetectionTool(board_params=board_params)

        # self._min_pts = 3   # SQPNP method needs at least 3 points
        # self._min_pts = 4   # ITERATIVE method needs at least 4 points
        self._min_pts = 6     # DLT algorithm needs at least 6 points for pose estimation

        # Detection (2D points and IDs)
        self._points2d = None
        self._points_ids = None

        # Extrinsics
        self._rvec = None
        self._tvec = None

        # Image size and related arrays
        if imsize_hw is not None:
            self.imsize = np.asarray(imsize_hw)[:2]  # OpenCV format (height, width)
            self._update_imsize(self.imsize)    # Initialise the related arrays
        else:
            self.imsize = None      # If not known, can't initialise related arrays, they will be on the first detection

        # Process sensor size input
        if isinstance(sensor_size, str):
            sensor_size = monocular.SENSOR_SIZES.get(f'''{sensor_size.strip('"')}"''', None)
        elif isinstance(sensor_size, (tuple, list, set, np.ndarray)) and len(sensor_size) == 2:
            sensor_size = np.asarray(sensor_size)
        else:
            sensor_size = None
        self._sensor_size = sensor_size

        # Compute ideal camera matrix if possible - this allows to fix the fx/fy ration, and helps the first guess
        if all([v is not None for v in (focal_mm, self._sensor_size, self.imsize)]):
            self._ideal_camera_matrix = monocular.estimate_camera_matrix(focal_mm, sensor_size, image_wh_px=np.flip(self.imsize))
        else:
            self._ideal_camera_matrix = None

        # If theoretical camera matrix is available, initialise intrinsics with it (distortion coefficients as zeros)
        if self._ideal_camera_matrix is not None:
            self._camera_matrix = self._ideal_camera_matrix.copy()
            self._dist_coeffs = np.zeros(5, dtype=np.float32)
        else:
            self._camera_matrix = None
            self._dist_coeffs = None

        # Samples stack (to aggregate detections for calibration)
        self._min_stack = min_stack
        self._max_stack = max_stack
        self.stack_points2d = deque(maxlen=self._max_stack)
        self.stack_points_ids = deque(maxlen=self._max_stack)

        # Error metrics
        self.last_best_errors: List[float] = [np.inf]
        self._stack_error: float = np.inf      # This will be the mean of the last_best_errors
        self.curr_error: float = np.inf

        self.set_visualisation_scale(scale=1)

    def _update_imsize(self, new_size):

        new_size = np.asarray(new_size)[:2]

        self.imsize = new_size
        self._frame_in = np.zeros((*self.imsize, 3), dtype=np.uint8)
        self._err_norm = 1  # TODO - Error normalisation, disabled for now, use image diag np.sum(np.power(self.imsize, 2)) * 1e-6

        # Initialize coverage grid to track which cells have been covered
        nb_cells = 15
        self._coverage_grid_shape = (nb_cells, int(np.round((self.imsize[1] / self.imsize[0]) * nb_cells)))
        self._coverage_grid = np.zeros(self._coverage_grid_shape, dtype=bool)

        # Create a weight grid for the coverage cells
        self._coverage_weight_grid = self._create_weight_grid()

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
    def has_intrinsics(self):
        return self._camera_matrix is not None and self._dist_coeffs is not None

    @property
    def has_extrinsics(self):
        return self._rvec is not None and self._tvec is not None

    @property
    def has_detection(self):
        return self._points2d is not None and self._points_ids is not None and len(self._points2d) >= self._min_pts

    @property
    def detection(self):
        return self._points2d, self._points_ids

    @property
    def intrinsics(self):
        return self._camera_matrix, self._dist_coeffs

    @property
    def extrinsics(self):
        return self._rvec, self._tvec

    @property
    def nb_points(self):
        return self._points2d.shape[0] if self._points2d is not None else 0

    @property
    def nb_samples(self):
        return len(self.stack_points2d)

    @property
    def coverage(self):
        return (np.sum(self._coverage_grid) / self._coverage_grid.size) * 100

    @property
    def stackerror(self):
        return self._stack_error

    @property
    def error(self):
        return self.curr_error

    @property
    def focal(self):
        if self.has_intrinsics:
            return (self._camera_matrix[0, 0] + self._camera_matrix[1, 1])/2.0
        else:
            return None

    @property
    def focal_mm(self):
        if self._camera_matrix is None or self._sensor_size is None or self.imsize is None:
            return None

        sensor_w_mm, sensor_h_mm = self._sensor_size
        image_h_px, image_w_px  = self.imsize

        fx_px = self._camera_matrix[0, 0]
        fy_px = self._camera_matrix[1, 1]
        pixel_size_x = sensor_w_mm / image_w_px
        pixel_size_y = sensor_h_mm / image_h_px
        f_mm_x = fx_px * pixel_size_x
        f_mm_y = fy_px * pixel_size_y

        return (f_mm_x + f_mm_y) / 2.0

    def set_intrinsics(self, camera_matrix, dist_coeffs, errors=None):
        self._camera_matrix = np.asarray(camera_matrix)
        dist_coeffs = np.asarray(dist_coeffs)
        if len(dist_coeffs) < 4:
            self._dist_coeffs = np.zeros(5, dtype=np.float32)
            self._dist_coeffs[:len(dist_coeffs)] = dist_coeffs
        self._dist_coeffs = dist_coeffs
        if errors is not None:
            self.last_best_errors = errors

    def clear_intrinsics(self):
        if self._ideal_camera_matrix is not None:
            self._camera_matrix = self._ideal_camera_matrix.copy()
            self._dist_coeffs = np.zeros(5, dtype=np.float32)
        else:
            self._camera_matrix = None
            self._dist_coeffs = None
        self.last_best_errors = [np.inf]

    @staticmethod
    def _check_new_errors(errors_new, errors_prev, p_val=0.05, confidence_lvl=0.95):
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
            # calibrateCamera will complain if the deques contain Nx2 arrays, but not if they contain 1xNx2 arrays...
            self.stack_points2d.append(self._points2d[np.newaxis, :, :])
            self.stack_points_ids.append(self._points_ids[np.newaxis, :])

    def clear_stacks(self):
        # Clear grid and sample stacks
        self._coverage_grid.fill(False)
        self.stack_points2d.clear()
        self.stack_points_ids.clear()

    def _create_weight_grid(self, gamma=2, min_weight=0.5):
        """
            Creates a grid of weights for the coverage cells based on distance from the image center
        """
        h, w = self.imsize
        grid_h, grid_w = self._coverage_grid_shape
        cell_h = h / grid_h
        cell_w = w / grid_w

        # Coordinates for cell centers
        xs = (np.arange(grid_w) + 0.5) * cell_w
        ys = (np.arange(grid_h) + 0.5) * cell_h
        grid_x, grid_y = np.meshgrid(xs, ys)

        # Calculate distance from the center of the image
        center_x, center_y = w / 2, h / 2
        distances = np.sqrt((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)
        # Maximum distance is from center to one of the corners
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
        norm_dist = distances / max_distance

        # Weight: cells at the center (norm_dist near 0) get ~min_weight, and those at the edge get near 1.
        weights = min_weight + (1 - min_weight) * (norm_dist ** gamma)
        return weights

    def _map_point_to_grid(self, point):
        h, w = self.imsize
        nb_rows, nb_cols = self._coverage_grid_shape

        cell_height = h / nb_rows
        cell_width = w / nb_cols

        row = int(point[1] // cell_height)
        col = int(point[0] // cell_width)
        row = min(max(row, 0), nb_rows - 1)
        col = min(max(col, 0), nb_cols - 1)
        return row, col

    def compute_intrinsics(self, clear_stack=True, fix_aspect_ratio=True, simple_distortion=False, complex_distortion=False):

        if simple_distortion and complex_distortion:
            raise AttributeError("Can't enable simple and complex distortion modes at the same time!")

        # If there is fewer than 5 images (no matter the self._min_stack value), this will NOT be enough
        if len(self.stack_points2d) < 5:
            return  # Abort and keep the stacks

        if self._camera_matrix is None and fix_aspect_ratio:
            print('No current camera matrix guess, unfixing aspect ratio.')
            fix_aspect_ratio = False

        # calib_flags = cv2.CALIB_USE_LU
        # calib_flags = cv2.CALIB_USE_QR
        calib_flags = 0
        if fix_aspect_ratio:
            calib_flags |= cv2.CALIB_FIX_ASPECT_RATIO           # This locks the ratio of fx and fy
        if simple_distortion or not self.has_intrinsics:        # The first iteration will use the simple model - this helps
            calib_flags |= (cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
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
            # Compute calibration using all the frames we selected
            global_error, new_camera_matrix, new_dist_coeffs, stack_rvecs, stack_tvecs, std_intrinsics, std_extrinsics, stack_errors = cv2.aruco.calibrateCameraCharucoExtended(
                charucoCorners=self.stack_points2d,
                charucoIds=self.stack_points_ids,
                board=self.dt.board,
                imageSize=np.flip(self.imsize),
                cameraMatrix=current_camera_matrix,     # Input/Output /!\
                distCoeffs=current_dist_coeffs,         # Input/Output /!\
                flags=calib_flags)

            # std_cam_mat, std_dist_coeffs = np.split(std_intrinsics.squeeze(), [4])
            # std_rvecs, std_tvecs = std_extrinsics.reshape(2, -1, 3)
            # TODO - Use these std values in the plot - or to decide if the new round of calibrateCamera is good or not?

            new_dist_coeffs = new_dist_coeffs.squeeze()
            # stack_rvecs = np.stack(stack_rvecs).squeeze()       # Unused for now
            # stack_tvecs = np.stack(stack_tvecs).squeeze()       # Unused for now
            stack_errors = stack_errors.squeeze() / self._err_norm  # Normalise errors on image diagonal

            # Note:
            # ---------------
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
            # The global calibration error in calibrateCamera is:
            #       np.sqrt(np.sum([sq_diff for view in stack])) / np.sum([len(view) for view in stack]))
            #

            # The following things should not be possible ; if any happens, then we trash the stack and abort
            if (new_camera_matrix < 0).any() or (new_camera_matrix[:2, 2] >= np.flip(self.imsize)).any():
                self.clear_stacks()
                return

            # Update the intrinsics if this stack's errors are better (or if it is the very first stack computed)
            if not self.has_intrinsics or np.any(self.last_best_errors == np.inf):
                self._camera_matrix = new_camera_matrix
                self._dist_coeffs = new_dist_coeffs
                self.last_best_errors = stack_errors
                self._stack_error = np.mean(self.last_best_errors)
                print(f"---Computed intrinsics---")

            elif self._check_new_errors(stack_errors, self.last_best_errors):
                self._camera_matrix = new_camera_matrix
                self._dist_coeffs = new_dist_coeffs
                self.last_best_errors = stack_errors
                self._stack_error = np.mean(self.last_best_errors)
                print(f"---Updated intrinsics---")

        except cv2.error as e:
            print(e)

        if clear_stack:
            self.clear_stacks()

    def compute_extrinsics(self, refine=True):

        # We need a detection to get the extrinsics relative to it
        if not self.has_detection:
            self._rvec, self._tvec = None, None
            self.curr_error = np.inf
            return

        # We also need intrinsics
        if not self.has_intrinsics:
            self._rvec, self._tvec = None, None
            self.curr_error = np.inf
            return

        # If the points are collinear, extrinsics estimation is garbage, so abort
        if cv2.aruco.testCharucoCornersCollinear(self.dt.board, self._points_ids):
            self._rvec, self._tvec = None, None
            self.curr_error = np.inf
            return

        # pnp_flags = cv2.SOLVEPNP_ITERATIVE

        # SQPNP:
        # - "A Consistently Fast and Globally Optimal Solution to the Perspective-n-Point Problem", 2020,
        #   George Terzakis and Manolis Lourakis, 10.1007/978-3-030-58452-8_28
        pnp_flags = cv2.SOLVEPNP_SQPNP

        try:
            nb_solutions, rvecs, tvecs, errors = cv2.solvePnPGeneric(self.dt.points3d[self._points_ids],
                                                         self._points2d,
                                                         self._camera_matrix,
                                                         self._dist_coeffs,
                                                         flags=pnp_flags)
        except cv2.error as e:
            self._rvec, self._tvec = None, None
            self.curr_error = np.inf
            return

        # If no solution, or if multiple solutions were found, abort
        if nb_solutions != 1:
            self._rvec, self._tvec = None, None
            self.curr_error = np.inf
            return

        self.curr_error = errors[0].squeeze().item() / self._err_norm   # Normalise errors on image diagonal
        rvec, tvec = rvecs[0], tvecs[0]

        if refine:
            # Virtual Visual Servoing:
            # - "Visual servo control. I. Basic approaches", 2006,
            #   François Chaumette, Seth Hutchinson, 10.1109/MRA.2006.250573
            # - "Pose Estimation for Augmented Reality: A Hands-On Survey", 2015
            #   Eric Marchand, Hideaki Uchiyama, Fabien Spindler, 10.1109/TVCG.2015.2513408
            rvec, tvec = cv2.solvePnPRefineVVS(self.dt.points3d[self._points_ids],
                                                 self._points2d,
                                                 self._camera_matrix,
                                                 self._dist_coeffs,
                                                 rvec,      # Input/Output /!\
                                                 tvec,      # Input/Output /!\
                                                 VVSlambda=1.0)

            # TODO - Test whether the Levenberg-Marquardt alternative solvePnPRefineLM() is better or not

        self._rvec, self._tvec = rvec.squeeze(), tvec.squeeze()

    def _compute_new_area(self):
        # Create a temporary grid for the current detection
        current_grid = np.zeros(self._coverage_grid.shape, dtype=bool)
        if self.has_detection:
            for pt in self._points2d:
                row, col = self._map_point_to_grid(pt)
                current_grid[row, col] = True

            # Novel area: cells that are in current_grid but not in cumulative grid
            # novel_cells = current_grid & (~self._coverage_grid)
            # novel_area_pct = 100.0 * novel_cells.sum() / self._coverage_grid.size

            # Novel cells are those not previously covered.
            novel_cells = current_grid & (~self._coverage_grid)
            # Instead of counting cells, sum the weights of the novel cells.
            novel_weight = np.sum(self._coverage_weight_grid[novel_cells])
            total_weight = np.sum(self._coverage_weight_grid)
            novel_area_pct = 100.0 * novel_weight / total_weight


            # Update the cumulative coverage grid
            self._coverage_grid |= current_grid

            return novel_area_pct
        else:
            return 0.0

    def detect(self, frame):

        # Initialise or update framesize-related stuff
        new_size = np.asarray(frame.shape)[:2]
        if self.imsize is None or np.any(self.imsize != new_size):
            self._update_imsize(new_size)

        # Load frame
        np.copyto(self._frame_in[:], frame)

        # Detect
        self._points2d, self._points_ids = self.dt.detect(self._frame_in,
                                              camera_matrix=self._camera_matrix,
                                              dist_coeffs=self._dist_coeffs,
                                              refine_markers=True,
                                              refine_points=True)

    def auto_register_area_based(self, area_threshold=0.2, nb_points_threshold=4):

        # Compute grid-based coverage
        novel_area = self._compute_new_area()
        if novel_area >= area_threshold and self.nb_points >= nb_points_threshold:
            self.register_sample()
            return True
        else:
            return False

    def auto_compute_intrinsics(self, coverage_threshold=80, stack_length_threshold=15,
                                simple_focal=False, simple_distortion=False, complex_distortion=False):
        """
            Trigger computation if the percentage of grid cells marked as covered exceeds the threshold
             and there are enough samples
        """
        if self.coverage >= coverage_threshold and self.nb_samples > stack_length_threshold:
            self.compute_intrinsics(fix_aspect_ratio=simple_focal,
                                    simple_distortion=simple_distortion,
                                    complex_distortion=complex_distortion)
            return True
        else:
            return False

    def draw_coverage_grid(self, img):
        overlay = img.copy()
        height, width = img.shape[:2]
        grid_rows, grid_cols = self._coverage_grid_shape
        cell_h = height / grid_rows
        cell_w = width / grid_cols

        for i in range(grid_rows):
            for j in range(grid_cols):
                if self._coverage_grid[i, j]:
                    # Determine pixel boundaries for this cell
                    x1 = int(j * cell_w)
                    y1 = int(i * cell_h)
                    x2 = int((j + 1) * cell_w)
                    y2 = int((i + 1) * cell_h)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), thickness=-1)
        alpha = 0.3
        blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return blended

    def visualise(self, errors_mm=False):

        frame_out = np.copy(self._frame_in)

        if self.has_detection:
            # If corners have been found, show them as red dots
            detected_points_int = (self._points2d * self.shift_factor).astype(np.int32)

            for xy in detected_points_int:
                frame_out = cv2.circle(frame_out, xy, 4 * self.SCALE, (0, 0, 255), 4 * self.SCALE, **self.draw_params)

        if self.has_intrinsics and self.has_extrinsics:
            # Display reprojected points: currently detected corners as yellow dots, the others as white dots
            reproj_points = monocular.reprojection(self.dt.points3d,
                                   camera_matrix=self._camera_matrix,
                                   dist_coeffs=self._dist_coeffs,
                                   rvec=self._rvec,
                                   tvec=self._tvec)

            reproj_points_int = (reproj_points * self.shift_factor).astype(np.int32)

            for i, xy in enumerate(reproj_points_int):
                if i in self._points_ids:
                    frame_out = cv2.circle(frame_out, xy, 2 * self.SCALE, (0, 255, 255), 4 * self.SCALE, **self.draw_params)
                else:
                    frame_out = cv2.circle(frame_out, xy, 4 * self.SCALE, (255, 255, 255), 4 * self.SCALE, **self.draw_params)

            # Compute errors in mm for each point
            if errors_mm:
                # Get each detected point's distance to the camera, and its error in pixels
                cam_points_dists = cdist([self._tvec], self.dt.points3d[self._points_ids]).squeeze()
                per_point_error = np.nanmean(np.abs(self._points2d - reproj_points[self._points_ids]), axis=-1)

                # Horizontal field of view in pixels, and the pixel-angle (i.e. how many rads per pixels)
                f = self.focal
                if f is not None:
                    fov_h = 2 * np.arctan(self.imsize[1] / (2 * f))
                    pixel_angle = fov_h / self.imsize[1]

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
            optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self._camera_matrix,
                                                                       self._dist_coeffs,
                                                                       np.flip(self.imsize), 1.0, np.flip(self.imsize))
            frame_out = cv2.undistort(frame_out, self._camera_matrix, self._dist_coeffs, None, optimal_camera_matrix)

            # Display board perimeter in purple AFTER undistortion, so that the lines are straight
            if self.has_extrinsics:
                # we also need to undistort the corner points with the undistorted ("optimal") camera matrix
                # Note: NO dist coeffs, since they are "included" in the optimal camera matrix
                corners2d = monocular.reprojection(self.dt.corners3d,
                                       camera_matrix=optimal_camera_matrix,
                                       dist_coeffs=None,
                                       rvec=self._rvec, tvec=self._tvec)

                pts_int = (corners2d * self.shift_factor).astype(np.int32)
                frame_out = cv2.polylines(frame_out, [pts_int], True, (255, 0, 255), 1 * self.SCALE, **self.draw_params)

        # Add information text to the visualisation image
        frame_out = cv2.putText(frame_out, f"Points: {self.nb_points}/{self.dt.total_points}", (30, 30 * self.SCALE),
                                **self.text_params)
        frame_out = cv2.putText(frame_out,
                                f"Area: {self.coverage:.2f}% ({len(self.stack_points2d)} snapshots)",
                                (30, 60 * self.SCALE), **self.text_params)

        txt = f"{self.error:.3f} px" if self.error != np.inf else '-'
        frame_out = cv2.putText(frame_out, f"Current reprojection error: {txt}", (30, 90 * self.SCALE), **self.text_params)

        txt = f"{self.stackerror:.3f} px" if self.stackerror != np.inf else '-'
        frame_out = cv2.putText(frame_out, f"Best average reprojection error: {txt}", (30, 120 * self.SCALE), **self.text_params)

        f_mm = self.focal_mm
        txt = f"{f_mm:.2f} mm" if f_mm is not None else '-'
        frame_out = cv2.putText(frame_out, f"Estimated focal: {txt}", (30, 150 * self.SCALE),
                                **self.text_params)

        return frame_out


class MultiviewCalibrationTool:
    """
        Class to aggregate multiple monocular detections into multi-view samples, compute, and refine cameras poses
    """

    def __init__(self, nb_cameras, origin_camera=0, min_poses=15, max_poses=100, min_detections=15, max_detections=100):

        self.nb_cameras = nb_cameras

        self._multi_intrinsics = np.zeros((nb_cameras, 3, 3))
        self._multi_intrinsics_refined = np.zeros((nb_cameras, 3, 3))
        self._multi_dist_coeffs = np.zeros((nb_cameras, 14))
        self._multi_dist_coeffs_refined = np.zeros((nb_cameras, 14))

        self._origin_idx = origin_camera

        self._detections_by_frame = defaultdict(dict)
        self._poses_by_frame = defaultdict(dict)

        self._min_poses = min_poses
        self._min_detections = min_detections

        self._detections_stack = deque(maxlen=max_detections)
        # self._poses_stack = deque(maxlen=max_poses)
        self._poses_per_camera = {i: [] for i in range(nb_cameras)}

        self._optimised_rvecs = None
        self._optimised_tvecs = None
        self._refined_rvecs = None
        self._refined_tvecs = None
        self._refined = False

    @property
    def nb_detection_samples(self):
        return len(self._detections_stack)

    # @property
    # def nb_pose_samples(self):
    #     return len(self._poses_stack)

    @property
    def origin_camera(self):
        return self._origin_idx

    @origin_camera.setter
    def origin_camera(self, value: int):
        self._origin_idx = value
        self.clear_poses()

    @property
    def has_extrinsics(self):
        return self._optimised_rvecs is not None and self._optimised_tvecs is not None

    @property
    def has_refined_rig_pose(self):
        return self._refined_rvecs is not None and self._refined_tvecs is not None

    @property
    def is_refined(self):
        return self._refined

    @property
    def extrinsics(self):
        if self._refined:
            return self._refined_rvecs, self._refined_tvecs
        else:
            return self._optimised_rvecs, self._optimised_tvecs

    def register_intrinsics(self, cam_idx: int, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        self._multi_intrinsics[cam_idx, :, :] = camera_matrix
        self._multi_dist_coeffs[cam_idx, :len(dist_coeffs)] = dist_coeffs

    def intrinsics(self):
        if self._refined:
            return self._multi_intrinsics_refined
        else:
            return self._multi_intrinsics

    def register_extrinsics(self, frame_idx: int, cam_idx: int, rvec: np.ndarray, tvec: np.ndarray, similarity_threshold=10.0):
        """
            This registers estimated monocular camera poses and stores them as complete pose samples if all cameras have a pose
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

            # # TODO - the similarity threshold probably needs to be a bit more elaborate (geodesic distance and separate rotation and translation similarities?) ...but that will do for now
            #
            # # Compute the similarity between the new pose and the already stored ones
            # if len(self._poses_stack) == 0:
            #     deltas = np.ones(1) * np.inf
            # else:
            #     deltas = np.array([np.linalg.norm(remapped_poses - pose) for pose in self._poses_stack])
            #
            # # If the new pose is sufficiently different, add it
            # # if np.all(deltas > similarity_threshold):
            # if np.all(deltas > 0):  # 0 threshold for testing
            #     self._poses_stack.append(remapped_poses)

    def register_detection(self, frame_idx: int, cam_idx: int, points2d: np.ndarray, points_ids: np.ndarray):
        """
            This registers points detections from multiple cameras and
            stores them as a complete detection samples if all cameras have one
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
            if self._optimised_rvecs is not None and self._optimised_tvecs is not None:
                points3d, points3d_ids = multiview.triangulation(
                    points2d_list, points2d_ids_list,
                    self._optimised_rvecs, self._optimised_tvecs,
                    self._multi_intrinsics,  self._multi_dist_coeffs
                )
                if points3d is not None:
                    print("Triangulation succeeded, emitting 3D points.")
                else:
                    print("Triangulation failed or returned no points.")
            else:
                print("Extrinsics not available yet; cannot triangulate.")

    def clear_poses(self):
        self._poses_per_camera = {i: [] for i in range(self.nb_cameras)}
        self._poses_by_frame = defaultdict(dict)

    def clear_detections(self):
        self._detections_stack.clear()
        self._detections_by_frame = defaultdict(dict)

    def compute_estimation(self, clear_poses_stack=True):
        """
            This uses the complete pose samples to compute a first estimate of the cameras arrangement
        """

        if not all(len(self._poses_per_camera[cam]) >= self._min_poses for cam in range(self.nb_cameras)):
            # print(f"Waiting for at least {self._min_poses} samples per camera; "
            #       f"current counts: {[len(self._poses_per_camera[cam]) for cam in range(self.nb_cameras)]}")
            return

        # Each camera’s list is converted to an array of shape (M, 3) where M varies
        n_m_rvecs = []
        n_m_tvecs = []
        for cam in range(self.nb_cameras):
            samples = self._poses_per_camera.get(cam, [])
            if not samples:
                # If no samples are available for this camera -> empty array
                n_m_rvecs.append(np.empty((0, 3)))
                n_m_tvecs.append(np.empty((0, 3)))
            else:
                samples_arr = np.array(samples)  # shape: (M, 2, 3) because each sample is (rvec, tvec)
                m_rvecs = samples_arr[:, 0, :]
                m_tvecs = samples_arr[:, 1, :]
                n_m_rvecs.append(m_rvecs)
                n_m_tvecs.append(m_tvecs)

        self._optimised_rvecs, self._optimised_tvecs = multiview.bestguess_rtvecs(n_m_rvecs, n_m_tvecs)

        if clear_poses_stack:
            self.clear_poses()

    # def compute_refined(self, clear_detections_stack=True):
    #     """
    #         This uses the complete detections samples to refine the cameras poses and intrinsics with bundle adjustment
    #     """
    #
    #     if self.nb_detection_samples < self._min_detections:
    #         return
    #
    #     print(f"[MultiviewCalibrationTool] Refining cameras poses...")
    #
    #     this_m_points2d = [cam.loc[fr] for cam in test_points_2d]
    #     this_m_points2d_ids = [cam.loc[fr] for cam in test_points_2d_ids]
    #
    #     points3d_svd, points3d_ids = multicam.triangulation(this_m_points2d, this_m_points2d_ids,
    #                                                         n_optimised_rvecs, n_optimised_tvecs,
    #                                                         n_camera_matrices, n_dist_coeffs)
    #
    #     # if clear_detections_stack:
    #     #     self.clear_detections()

