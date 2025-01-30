from collections import defaultdict, deque
from pathlib import Path
import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=5)
import cv2
import toml
import scipy.stats as stats
from scipy.spatial.distance import cdist


class DetectionTool:
    def __init__(self, charuco_board):

        # Charuco board and detector parameters
        self.board = charuco_board
        aruco_dict = self.board.getDictionary()
        self.detector_parameters = cv2.aruco.DetectorParameters()
        self.detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, detectorParams=self.detector_parameters)

        # Maximum number of board points and distances
        self._total_points = len(self.board.getChessboardCorners())
        self._total_distances = int((self._total_points * (self._total_points - 1)) / 2.0)

        # Create 3D coordinates for board corners (in board-centric coordinates)
        self._n_cols, self._n_rows = self.board.getChessboardSize()
        self._board_points_3d = self.board.getChessboardCorners()

        self._board_corners_3d = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0]], dtype=np.float32) * [self._n_cols, self._n_rows, 0] * self.board.getSquareLength()

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
                # Refine the chessboard corners
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 0.0001)

                chessboard_points = cv2.cornerSubPix(frame, chessboard_points,
                                                     winSize=(11, 11),
                                                     zeroZone=(-1, -1),
                                                     criteria=crit)

            if chessboard_points is not None and len(chessboard_points_ids[:, 0]) > 1:
                points2d_coords = chessboard_points[:, 0, :]
                points2d_ids = chessboard_points_ids[:, 0]

        return points2d_coords, points2d_ids


class MonocularCalibrationTool:
    """
        This object is stateful for the intrinsics *only*
    """
    def __init__(self, detectiontool, imsize=None, min_stack=15, max_stack=100):

        self.dt = detectiontool

        self._min_pts = 3   # SQPNP method needs at least 3 points
        self._min_pts = 4   # ITERATIVE method needs at least 4 points

        # Detection
        self._points2d = None
        self._points_ids = None

        # Intrinsics
        self._camera_matrix = None
        self._dist_coeffs = None

        # Extrinsics
        self._rvec = None
        self._tvec = None

        # Coverage (We initialise these arrays later once we know the frame size)
        if imsize is not None:
            self.imsize = np.array(imsize[:2])
            self._ideal_central_point = np.flip(self.imsize) / 2.0
            # self._norm = np.sum(np.power(self.imsize, 2)) * 1e-6      # TODO test this more extensively
            self._norm = 1

            self._frame_in = np.zeros((*self.imsize, 3), dtype=np.uint8)
            self._current_area_mask = np.zeros(self.imsize, dtype=bool)
            self._cumul_coverage_mask = np.zeros(self.imsize, dtype=bool)
            self._current_area_px = np.zeros(self.imsize, dtype=np.uint8)
            self._cumul_coverage_px = np.zeros((*self.imsize, 3), dtype=np.uint8)

        else:
            self.imsize = None
            self._ideal_central_point = None

            self._frame_in = None
            self._current_area_mask = None
            self._cumul_coverage_mask = None
            self._current_area_px = None
            self._cumul_coverage_px = None


        # Samples stack
        self._min_stack = min_stack
        self._max_stack = max_stack
        self.stack_points2d = deque(maxlen=self._max_stack)
        self.stack_points_ids = deque(maxlen=self._max_stack)

        # Errors
        self.last_best_errors = np.inf
        self._stack_error = np.inf      # This will be the mean of the last_best_errors
        self.curr_error = np.inf

        self.set_visualisation_scale(scale=1)

    def set_visualisation_scale(self, scale=1):
        # Stuff for visualisation
        self.BIT_SHIFT = 4
        self.SCALE = scale
        self.shift_factor = 2 ** self.BIT_SHIFT
        self.draw_params = {'shift': self.BIT_SHIFT, 'lineType': cv2.LINE_AA}
        self.text_params = {'fontFace': cv2.FONT_HERSHEY_DUPLEX, 'fontScale': 0.8 * self.SCALE,
                            'color': (255, 255, 255), 'thickness': 1 * self.SCALE, 'lineType': cv2.LINE_AA}


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
        return self._cumul_coverage_mask.mean() * 100

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

    def load_intrinsics(self, camera_matrix, dist_coeffs):
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs

    def clear_intrinsics(self):
        self._camera_matrix = None
        self._dist_coeffs = None
        self.last_best_errors = np.inf

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

            # If the following is true, this is a manual snapshot and the area has not been computed
            if not self._current_area_mask.any():
                self._compute_new_area()

            # calibrateCamera will complain if the deques contain Nx2 arrays, but not if they contain 1xNx2 arrays...
            self.stack_points2d.append(self._points2d[np.newaxis, :, :])
            self.stack_points_ids.append(self._points_ids[np.newaxis, :])

            self._cumul_coverage_mask[self._current_area_mask] = True

            self._current_area_mask.fill(False)

    def clear_stacks(self):
        # Clear cumulative coverage and empty the stacks

        self._cumul_coverage_mask.fill(False)
        self._cumul_coverage_px.fill(0)

        self.stack_points2d.clear()
        self.stack_points_ids.clear()

    def compute_intrinsics(self, clear_stack=True, simple_focal=True, simple_distortion=False, complex_distortion=False):

        if simple_distortion and complex_distortion:
            raise AttributeError("Can't enable simple and complex distortion modes at the same time!")

        # If there is fewer than 5 images (no matter the self._min_stack value), this will NOT be enough
        # Also, OpenCV says DLT algorithm needs at least 6 points for pose estimation from 3D-2D point correspondences
        if len(self.stack_points2d) < 6:
            return  # Abort and keep the stacks

        calib_flags = cv2.CALIB_USE_LU + cv2.CALIB_SAME_FOCAL_LENGTH
        if simple_focal:
            calib_flags += cv2.CALIB_FIX_ASPECT_RATIO   # This is recommended (unless you use an anamorphic lens?)
        if simple_distortion or not self.has_intrinsics:   # The first iteration will use the simple model - this helps
            calib_flags += cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST
        if complex_distortion:
            calib_flags += cv2.CALIB_RATIONAL_MODEL
        if self.has_intrinsics:
            calib_flags += cv2.CALIB_USE_INTRINSIC_GUESS    # Important, otherwise it ignores the passed intrinsics

        # We need to copy to a new array, because OpenCV uses these as Input/Output buffers
        if self.has_intrinsics:
            current_camera_matrix = np.copy(self._camera_matrix)
            current_dist_coeffs = np.copy(self._dist_coeffs)
        else:
            current_camera_matrix = None
            current_dist_coeffs = None

        # Compute calibration using all the frames we selected
        global_error, new_camera_matrix, new_dist_coeffs, stack_rvecs, stack_tvecs, std_intrinsics, std_extrinsics, stack_errors = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=self.stack_points2d,
            charucoIds=self.stack_points_ids,
            board=self.dt.board,
            imageSize=np.flip(self.imsize),
            cameraMatrix=current_camera_matrix,     # Input/Output /!\
            distCoeffs=current_dist_coeffs,         # Input/Output /!\
            flags=calib_flags)

        std_cam_mat, std_dist_coeffs = np.split(std_intrinsics.squeeze(), [4])
        std_rvecs, std_tvecs = std_extrinsics.reshape(2, -1, 3)
        # TODO - Maybe use these std values to decide if the new round of calibrateCamera is good or not?

        new_dist_coeffs = new_dist_coeffs.squeeze()
        stack_rvecs = np.stack(stack_rvecs).squeeze()       # Unused for now
        stack_tvecs = np.stack(stack_tvecs).squeeze()       # Unused for now
        stack_errors = stack_errors.squeeze() / self._norm  # Normalise errors on image diagonal

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
        if not self.has_intrinsics or (self.last_best_errors == np.inf).any():
            self._camera_matrix = new_camera_matrix
            self._dist_coeffs = new_dist_coeffs
            self.last_best_errors = stack_errors
            self._stack_error = np.mean(self.last_best_errors)

            # For the first iteration, set the central point to the real image centre - that helps A LOT
            self._camera_matrix[:2, 2] = self._ideal_central_point

            print(f"---Computed intrinsics---")

        elif self._check_new_errors(stack_errors, self.last_best_errors):

            self._camera_matrix = new_camera_matrix
            self._dist_coeffs = new_dist_coeffs
            self.last_best_errors = stack_errors
            self._stack_error = np.mean(self.last_best_errors)

            print(f"---Updated intrinsics---")

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

        # SQPNP:
        # - "A Consistently Fast and Globally Optimal Solution to the Perspective-n-Point Problem", 2020,
        #   George Terzakis and Manolis Lourakis, 10.1007/978-3-030-58452-8_28
        pnp_flags = cv2.SOLVEPNP_SQPNP
        # pnp_flags = cv2.SOLVEPNP_ITERATIVE
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

        self.curr_error = errors[0].squeeze().item() / self._norm   # Normalise errors on image diagonal
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
                                                 VVSlambda=0.5)

            # TODO - Test whether the Levenberg-Marquardt alternative solvePnPRefineLM() is better or not

        self._rvec, self._tvec = rvec.squeeze(), tvec.squeeze()

    def _compute_new_area(self):
        self._current_area_px.fill(0)
        self._current_area_mask.fill(False)

        if self.has_detection:

            pts = cv2.convexHull(np.round(self._points2d).astype(np.int32))    # can't use bit shift here, so rounding it is
            np.copyto(self._current_area_mask, cv2.fillPoly(self._current_area_px, [pts], (255, 255, 255)).astype(bool))

            # Newly detected area is the union of the current detection and the inverse of the overlap with existing
            overlap_area = np.logical_and(self._current_area_mask, self._cumul_coverage_mask)
            novel_area = np.logical_and(self._current_area_mask, ~overlap_area)

            return overlap_area.mean() * 100, novel_area.mean() * 100
        else:
            return 0.0, 0.0

    def detect(self, frame):

        # Load frame
        if self._frame_in is None or frame.size != self._frame_in.size:
            self.imsize = np.array(frame.shape[:2])
            self._ideal_central_point = np.flip(self.imsize) / 2.0
            self._frame_in = np.zeros((*self.imsize, 3), dtype=np.uint8)
            self._current_area_mask = np.zeros(self.imsize, dtype=bool)
            self._cumul_coverage_mask = np.zeros(self.imsize, dtype=bool)
            self._current_area_px = np.zeros(self.imsize, dtype=np.uint8)
            self._cumul_coverage_px = np.zeros((*self.imsize, 3), dtype=np.uint8)

        np.copyto(self._frame_in[:], frame)

        # Detect
        self._points2d, self._points_ids = self.dt.detect(frame,
                                              camera_matrix=self._camera_matrix,
                                              dist_coeffs=self._dist_coeffs,
                                              refine_markers=True,
                                              refine_points=False)

    def auto_register_monocular(self, area_threshold=0.2, nb_points_threshold=4):

        # Compute image area with detection
        overlap_area, novel_area = self._compute_new_area()

        # Check if current sample can be added to the stack
        if novel_area >= area_threshold and self.nb_points >= nb_points_threshold:
            self.register_sample()
            return True
        else:
            return False

    def auto_compute_intrinsics(self, coverage_threshold=60, stack_length_threshold=15, simple_focal=True, simple_distortion=False, complex_distortion=False):
        # Check if stack is big enough and / or if cumul coverage is good enough
        if self.coverage >= coverage_threshold and self.nb_samples > stack_length_threshold:
            self.compute_intrinsics(simple_focal=simple_focal, simple_distortion=simple_distortion, complex_distortion=complex_distortion)
            return True
        else:
            return False

    def writefile(self, filepath):

        if self.has_intrinsics:

            filepath = Path(filepath)
            if not filepath.suffix == '.toml':
                filepath = filepath.parent / f'{filepath.stem}.toml'

            d = {'camera_matrix': self._camera_matrix.squeeze().tolist(), 'dist_coeffs': self._dist_coeffs.squeeze().tolist(),
                 'errors': np.array(self.last_best_errors).tolist()}

            with open(filepath, 'w') as f:
                # Remove trailing commas
                toml_str = toml.dumps(d).replace(',]', ' ]')
                # Add indents (yes this one-liner is atrocious)
                lines = [l.replace('], [', f'],\n{"".ljust(len(l.split("=")[0]) + 4)}[') for l in toml_str.splitlines()]
                toml_str_formatted = '\n'.join(lines)
                f.write(toml_str_formatted)

            print(f"Calibration saved: {filepath}")
        else:
            print(f"No intrinsics to save ({filepath.stem})")

    def readfile(self, filepath):

        filepath = Path(filepath)
        if not filepath.suffix == '.toml':
            filepath = filepath.parent / f'{filepath.stem}.toml'

        if filepath.exists():
            d = toml.load(filepath)

            self._camera_matrix = np.array(d['camera_matrix']).squeeze()
            self._dist_coeffs = np.array(d['dist_coeffs']).squeeze()
            self.last_best_errors = np.array(d.get('errors', np.inf)).squeeze()
            self._stack_error = np.mean(self.last_best_errors)

            print(f"Loaded intrinsics from {filepath}")

        else:
            print(f"File not found: {filepath}")

    def visualise(self, errors_mm=False):

        frame_out = np.copy(self._frame_in)

        if self.has_detection:
            # If corners have been found, show them as red dots
            detected_points_int = (self._points2d * self.shift_factor).astype(np.int32)

            for xy in detected_points_int:
                frame_out = cv2.circle(frame_out, xy, 4 * self.SCALE, (0, 0, 255), 4 * self.SCALE, **self.draw_params)

        if self.has_intrinsics and self.has_extrinsics:
            # Display reprojected points: currently detected corners as yellow dots, the others as white dots
            reproj_points, _ = cv2.projectPoints(self.dt.points3d,
                                                 self._rvec, self._tvec,
                                                 self._camera_matrix, self._dist_coeffs)
            reproj_points = reproj_points.squeeze()

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
                fov_h = 2 * np.arctan(self.imsize[1] / (2 * self.focal))
                pixel_angle = fov_h / self.imsize[1]

                # Determine the per-point error in millimeters
                error_arc = pixel_angle * per_point_error
                error_mm = error_arc * cam_points_dists

                for i, err in enumerate(error_mm):
                    frame_out = cv2.putText(frame_out, f"{err:.3f}", self._points2d[i].astype(int) + 6,
                                            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.3 * self.SCALE, color=(0, 255, 255),
                                            thickness=1 * self.SCALE, lineType=cv2.LINE_AA)

        # Add the coverage as a green overlay
        self._cumul_coverage_px[self._cumul_coverage_mask, 1] = 255
        frame_out = cv2.addWeighted(frame_out, 1.0, self._cumul_coverage_px, 0.25, 0)

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
                corners2d, _ = cv2.projectPoints(self.dt.corners3d,
                                                 self._rvec, self._tvec,
                                                 optimal_camera_matrix, None)

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

        # if self.curr_error != float('inf'):
        #     self.graph.update(self.curr_error)
        # overlay = np.zeros(self.imsize, dtype=np.uint8)
        # overlay[-5 - self.graph._h:-5, 5:5 + self.graph._w, :] = self.graph.render()
        # frame_out = cv2.addWeighted(frame_out, 1.0, overlay, 0.75, 1.0)

        return frame_out


