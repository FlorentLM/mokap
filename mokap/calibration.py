from collections import deque
from pathlib import Path
import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=5)
import cv2
import toml


class CalibrationTool:

    def __init__(self, charuco_board, min_samples=25, max_samples=100):

        # The image on which to detect
        self.frame_in = None

        self._has_markers = False
        self.has_detection = False
        self.has_intrinsics = False
        self.has_extrinsics = False

        # Charuco board and detector parameters
        self.board = charuco_board
        aruco_dict = self.board.getDictionary()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

        # Maximum number of markers and board points
        self.total_markers = len(self.board.getIds())
        self.total_points = len(self.board.getChessboardCorners())

        # Default attributes for markers and points coordinates and IDs
        self.markers_coords = np.array([])
        self.marker_ids = np.array([])
        self._nb_seen_markers = 0

        self.points_coords = np.array([])
        self.points_ids = np.array([])
        self._nb_seen_points = 0

        self.reprojected_points = np.array([])
        self.reprojected_corners = np.array([])

        # # Default attributes for extrinsics (i.e. camera pose in board-centric coordinates)
        self._rvec = np.array([])
        self._tvec = np.array([])

        # Create 3D coordinates for board corners (in board-centric coordinates)
        board_cols, board_rows = self.board.getChessboardSize()
        self._board_corners_3d = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0]], dtype=np.float32) * [board_cols, board_rows, 0] * self.board.getSquareLength()

        # This will be the area of the image that has been covered so far
        self._coverage = None

        # This will be the visualisation overlay
        self._coverage_overlay = None

        # These two deque() will store detected points for several samples
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.multi_samples_points_coords = deque(maxlen=self.max_samples)
        self.multi_samples_points_ids = deque(maxlen=self.max_samples)

        # These are the intrinsics we want to compute, we initialise them to None for 1st estimation without a prior
        self.camera_matrix = None
        self.dist_coeffs = None

        # Initialise reprojection errors to +inf
        self.best_error_px = float('inf')
        self.curr_error_px = float('inf')
        self.curr_error_mm = float('inf')

    @property
    def points3d(self):
        """ Returns the coordinates of the chessboard points in 3D (in board-centric coordinates) """
        return self.board.getChessboardCorners()

    @property
    def points2d_detect(self):
        """ Returns the coordinates of the detected chessboard points in 2D (in camera-centric coordinates) """
        if self.has_detection:
            return self._points_coords
        else:
            return np.array([])

    @property
    def ids_detect(self):
        """ Returns the IDs of the detected chessboard points in 2D (in camera-centric coordinates) """
        if self.has_detection:
            return self._points_ids
        else:
            return np.array([])

    @property
    def points2d_reproj(self):
        """ Returns the coordinates of the reprojected chessboard points in 2D (in camera-centric coordinates) """
        if self.has_intrinsics and self.has_detection and self.has_extrinsics:
            imgpoints, _ = cv2.projectPoints(self.points3d, self._rvec, self._tvec, self.camera_matrix, self.dist_coeffs)
            return imgpoints[:, 0, :]
        else:
            return np.array([])

    @property
    def corners3d(self):
        return self._board_corners_3d

    @property
    def corners2d(self):
        if self.has_intrinsics and self.has_detection:
            board_corners_2d, _ = cv2.projectPoints(self.corners3d, self._rvec, self._tvec, self.camera_matrix, self.dist_coeffs)
            return board_corners_2d[:, 0, :]
        else:
            return np.array([])

    @property
    def rvec(self):
        if self.has_extrinsics:
            return self._rvec
        else:
            return np.array([])

    @property
    def tvec(self):
        if self.has_extrinsics:
            return self._tvec
        else:
            return np.array([])

    @property
    def coverage(self):
        return self._coverage.mean() * 100

    def detect_markers(self, refine=True):

        # Detect and refine aruco markers
        markers_coords, marker_ids, rejected = self.detector.detectMarkers(self.frame_in)
        if refine:
            markers_coords, marker_ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(
                image=self.frame_in,
                board=self.board,
                detectedCorners=markers_coords,
                detectedIds=marker_ids,
                rejectedCorners=rejected,
                # Known bug with refineDetectedMarkers, fixed in OpenCV 4.9: https://github.com/opencv/opencv/pull/24139
                cameraMatrix=self.camera_matrix if cv2.getVersionMajor() >= 4 and cv2.getVersionMinor() >= 9 else None,
                distCoeffs=self.dist_coeffs)

        if marker_ids is not None:
            self._markers_coords = np.array(markers_coords)[:, 0, :, :]
            self._marker_ids = marker_ids[:, 0]
            self._nb_seen_markers = len(self._marker_ids)

            self._has_markers = True
        else:
            self._markers_coords = np.array([])
            self._marker_ids = np.array([])
            self._nb_seen_markers = 0

            self._has_markers = False
            self.has_detection = False

    def detect_corners(self, refine=True):

        # If any marker has been detected, try to detect the board corners
        if self._has_markers:
            nb_corners, charuco_coords, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=self._markers_coords,
                markerIds=self._marker_ids,
                image=self.frame_in,
                board=self.board,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                minMarkers=1)
            if refine:
                try:
                    # Refine the board corners
                    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
                    charuco_coords = cv2.cornerSubPix(self.frame_in, charuco_coords,
                                                       winSize=(20, 20),
                                                       zeroZone=(-1, -1),
                                                       criteria=crit)
                except:
                    pass

            if charuco_coords is not None:
                self._points_coords = charuco_coords[:, 0, :]
                self._points_ids = charuco_ids[:, 0]
                self._nb_seen_points = len(self._points_ids)

                self.has_detection = True
            else:
                self._points_coords = np.array([])
                self._points_ids = np.array([])
                self._nb_seen_points = 0

                self.has_detection = False
        else:
            self.has_detection = False

    def reproject(self):

        if self.has_intrinsics and self.has_detection and len(self._points_coords) > 6:
            _, rvec, tvec, error = cv2.solvePnPGeneric(self.points3d[self._points_ids],
                                                       self._points_coords,
                                                       self.camera_matrix,
                                                       self.dist_coeffs)
            self.curr_error_px = error[0][0]

            self.curr_error_mm = (error[0][0] * tvec[2] / (self.camera_matrix[0, 0] + self.camera_matrix[1, 1]) / 2.0).squeeze()

            self._rvec = rvec[0].squeeze()
            self._tvec = tvec[0].squeeze()

            self.has_extrinsics = True

        else:
            self._rvec = np.array([])
            self._tvec = np.array([])

            self.has_extrinsics = False

    def update_coverage(self):

        if self.has_detection:

            # Compute image area with detection
            detected_area = np.zeros(self.frame_in.shape[:2], dtype=np.uint8)
            pts = cv2.convexHull(np.round(self._points_coords[np.newaxis, ...]).astype(int))
            detected_area = cv2.fillPoly(detected_area, [pts], (255, 255, 255)).astype(bool)

            # Newly detected area is the union of the current detection and the inverse of the overlap with existing
            overlap = np.logical_and(detected_area, self._coverage)
            new_area = np.logical_and(detected_area, ~overlap)

            # If the new frame brings sufficient new coverage, add its data to the list
            if new_area.mean() * 100 >= 0.2 and len(self._points_coords) > 5:
                self.multi_samples_points_coords.append(self._points_coords[np.newaxis, ...])
                self.multi_samples_points_ids.append(self._points_ids[np.newaxis, ...])

                self._coverage[detected_area] = True

    def calibrate(self):

        nb_samples = len(self.multi_samples_points_ids)

        # Compute calibration using all the frames we selected
        calib_ret, camera_matrix_new, dist_coeffs_new, rvecs_new, tvecs_new = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=self.multi_samples_points_coords,
            charucoIds=self.multi_samples_points_ids,
            board=self.board,
            imageSize=self.frame_in.shape[:2],
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.dist_coeffs,
            flags=cv2.CALIB_USE_QR)

        multi_objpoints = [self.points3d[ids] for ids in self.multi_samples_points_ids]
        mean_error_px = 0.0
        for i in range(len(multi_objpoints)):
            _, rvec, tvec, error = cv2.solvePnPGeneric(multi_objpoints[i], self.multi_samples_points_coords[i], camera_matrix_new, dist_coeffs_new)
            mean_error_px += error[0][0]
        avg_err_px = mean_error_px / nb_samples

        if avg_err_px < self.best_error_px:
            self.camera_matrix = camera_matrix_new.squeeze()
            self.dist_coeffs = dist_coeffs_new.squeeze()
            self.best_error_px = avg_err_px

        self.has_intrinsics = True

        self.reset_samples()

    def reset_samples(self):

        self._coverage.fill(False)
        self._coverage_overlay.fill(0)

        self.multi_samples_points_coords.clear()
        self.multi_samples_points_ids.clear()

    def visualise(self):

        frame_out = np.copy(self.frame_in)

        # If corners have been found, show them as red dots
        detected_points = self.points2d_detect

        for xy in detected_points:
            frame_out = cv2.circle(frame_out, np.round(xy).astype(int), 2, (0, 0, 255), 2)

        # Display reprojected points: currently detected corners as yellow dots, the others as white dots
        reproj_points = self.points2d_reproj

        for i, xy in enumerate(reproj_points):
            if i in self._points_ids:
                frame_out = cv2.circle(frame_out, np.round(xy).astype(int), 2, (0, 255, 255), 2)
            else:
                frame_out = cv2.circle(frame_out, np.round(xy).astype(int), 2, (255, 255, 255), 2)

        # Display board corners in purple
        reproj_corners = self.corners2d

        # for xy in reproj_corners:
        #     frame_out = cv2.circle(frame_out, np.round(xy).astype(int), 4, (255, 0, 255), 4)

        # Display board perimeter in purple
        if len(reproj_corners) > 0:
            pts = np.round(reproj_corners).astype(int)
            frame_out = cv2.polylines(frame_out, [pts], True, (255, 0, 255), 2)

        # Add the coverage as a green overlay
        self._coverage_overlay[self._coverage, 1] = 255
        frame_out = cv2.addWeighted(frame_out, 0.85, self._coverage_overlay, 0.15, 0)

        # Undistort image
        if self.has_intrinsics:
            h, w = self.frame_in.shape[:2]
            optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 0, (w, h))
            frame_out = cv2.undistort(frame_out, self.camera_matrix, self.dist_coeffs, None, optimal_camera_matrix)

        # Add information text to the visualisation image
        frame_out = cv2.putText(frame_out,
                                     f"Aruco markers: {self._nb_seen_markers}/{self.total_markers}", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        frame_out = cv2.putText(frame_out,
                                     f"Corners: {self._nb_seen_points}/{self.total_points}", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        frame_out = cv2.putText(frame_out,
                                     f"Area: {self.coverage:.2f}% ({len(self.multi_samples_points_coords)} snapshots)", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        txt = f"{self.curr_error_px:.2f} px ({self.curr_error_mm:.3f} mm)" if (self.best_error_px != float('inf')) & (self.curr_error_mm != float('inf')) else '-'
        frame_out = cv2.putText(frame_out,
                                     f"Current reprojection error: {txt}",
                                     (30, 120),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        txt = f"{self.best_error_px:.2f} px" if self.best_error_px != float('inf') else '-'
        frame_out = cv2.putText(frame_out,
                                     f"Best average reprojection error: {txt}",
                                     (30, 150),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return frame_out

    def detect(self, frame):

        if frame.ndim == 3:
            self.frame_in = frame
        else:
            self.frame_in = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        if self._coverage is None or self._coverage_overlay is None or self._coverage.shape[0] != self.frame_in.shape[0] or self._coverage.shape[1] != self.frame_in.shape[1]:
            self._coverage = np.full(self.frame_in.shape[:2], False, dtype=bool)
            self._coverage_overlay = np.zeros((*self.frame_in.shape[:2], 3), dtype=np.uint8)

        self.detect_markers(refine=True)
        self.detect_corners(refine=True)

        self.update_coverage()

        self.reproject()

    def save(self, filepath):

        if self.has_intrinsics:

            filepath = Path(filepath)
            if not filepath.suffix == '.toml':
                filepath = filepath.parent / f'{filepath.stem}.toml'

            d = {'camera_matrix': self.camera_matrix.squeeze().tolist(), 'dist_coeffs': self.dist_coeffs.squeeze().tolist()}

            with open(filepath, 'w') as f:
                # Remove trailing commas
                toml_str = toml.dumps(d).replace(',]', ' ]')
                # Add indents (yes this one-liner is atrocious)
                lines = [l.replace('], [', f'],\n{"".ljust(len(l.split("=")[0]) + 4)}[') for l in toml_str.splitlines()]
                toml_str_formatted = '\n'.join(lines)
                f.write(toml_str_formatted)

    def load(self, filepath):

        filepath = Path(filepath)
        if not filepath.suffix == '.toml':
            filepath = filepath.parent / f'{filepath.stem}.toml'

        d = toml.load(filepath)

        self.camera_matrix = np.array(d['camera_matrix']).squeeze()
        self.dist_coeffs = np.array(d['dist_coeffs']).squeeze()

        self.has_intrinsics = True

