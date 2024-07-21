# from PyQt6.QtOpenGL import QOpenGLVersionProfile, QOpenGLTexture, QOpenGLVersionFunctionsFactory
# from PyQt6.QtOpenGLWidgets import QOpenGLWidget

# QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)
# QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

# class VideoGLWidget(QOpenGLWidget):
#     TEX_SLOTS = None
#
#     def __init__(self, height, width, idx, parent=None, *args, **kwargs):
#         super(VideoGLWidget, self).__init__(parent, *args, **kwargs)
#         self.idx = idx
#         self.gl = None
#         self.texture = None
#         self.ratio = width/height
#         self.arraybuffer = np.zeros((height, width, 3), dtype=np.uint8)
#
#     def initializeGL(self):
#         version_profile = QOpenGLVersionProfile()
#         version_profile.setVersion(2, 0)
#         self.gl = QOpenGLVersionFunctionsFactory.get(version_profile, self.context())
#         self.gl.initializeOpenGLFunctions()
#         # self.gl.glClearColor(0.5, 0.8, 0.7, 1.0)
#
#         self.gl.glEnable(self.gl.GL_TEXTURE_2D)
#
#         if VideoGLWidget.TEX_SLOTS is None:
#             VideoGLWidget.TEX_SLOTS = self.gl.glGenTextures(5)
#
#         self.texture = VideoGLWidget.TEX_SLOTS[self.idx]
#
#         self._gen_texture()
#
#     def resizeGL(self, width, height):
#         side = min(width, height)
#         x = int((width - side) / 2)
#         y = int((height - side) / 2)
#         self.gl.glViewport(x, y, side, side)
#
#     def paintGL(self):
#         self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT)
#         if self.texture:
#
#             self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.texture)
#
#             self.gl.glBegin(self.gl.GL_QUADS)
#             self.gl.glTexCoord2f(0, 0)
#             self.gl.glVertex2f(-1, -1)
#             self.gl.glTexCoord2f(1, 0)
#             self.gl.glVertex2f(1, -1)
#             self.gl.glTexCoord2f(1, 1)
#             self.gl.glVertex2f(1, 1)
#             self.gl.glTexCoord2f(0, 1)
#             self.gl.glVertex2f(-1, 1)
#             self.gl.glEnd()
#
#             self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, 0)
#
#     def _gen_texture(self):
#
#         self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.texture)
#
#         self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MIN_FILTER, self.gl.GL_LINEAR)
#         self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MAG_FILTER, self.gl.GL_LINEAR)
#
#         self.gl.glTexImage2D(self.gl.GL_TEXTURE_2D,
#                      0,
#                      self.gl.GL_RGB,
#                      self.arraybuffer.shape[1],
#                      self.arraybuffer.shape[0],
#                      0,
#                      self.gl.GL_RGB,
#                      self.gl.GL_UNSIGNED_BYTE,
#                      self.arraybuffer.tobytes())
#
#         self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, 0)
#
#     def updatedata(self, imagedata):
#         flipped = np.flip(imagedata, axis=0)
#         self.arraybuffer[:, :, 0] = flipped
#         self.arraybuffer[:, :, 1] = flipped
#         self.arraybuffer[:, :, 2] = flipped
#         self._gen_texture()
#         self.update()




# class VideoWindowCalib(VideoWindowBase):
#
#     def __init__(self, parent, idx):
#         super().__init__(parent, idx)
#
#         self._total_coverage_area = np.zeros((*self._source_shape, 3), dtype=np.uint8)
#         self._current_coverage_area = np.zeros(self._source_shape, dtype=np.uint8)
#
#         ## ChAruco board variables
#         BOARD_COLS = 7              # Total rows in the board (chessboard)
#         BOARD_ROWS = 10             # Total cols in the board
#         SQUARE_LENGTH = 3.2         # Length of one chessboard square in real life units (i.e. mm)
#         MARKER_LENGTH = 2.4
#         MARKER_BITS = 4
#         DICT_SIZE = 1000
#         # TODO - Load these from the config file
#
#         self._aruco_dict, self._charuco_board = utils.generate_charuco(BOARD_ROWS, BOARD_COLS,
#                                                                        square_length=SQUARE_LENGTH,
#                                                                        marker_length=MARKER_LENGTH,
#                                                                        marker_bits=MARKER_BITS,
#                                                                        dict_size=DICT_SIZE,
#                                                                        save_svg=False)
#
#         detector_params = cv2.aruco.DetectorParameters()
#         self.detector = cv2.aruco.ArucoDetector(self._aruco_dict, detector_params)
#
#         self._max_frames = 150
#         self._recommended_coverage_pct_high = 80
#         self._recommended_coverage_pct_mid = 50
#         self._recommended_coverage_pct_low = 25
#
#         self.current_charuco_corners = None                                 # Currently visible corners
#         self.current_charuco_ids = None                                     # Corresponding aruco ids
#         self.detected_charuco_corners = deque(maxlen=self._max_frames)      # All corners seen so far
#         self.detected_charuco_ids = deque(maxlen=self._max_frames)          # All corresponding aruco ids
#
#         self.camera_matrix = None
#         self.dist_coeffs = None
#
#         self._coverage_pct = 0
#
#         self._manual_snapshot = False
#
#         self._init_common_ui()
#         self._init_specific_ui()
#
#
#     def _init_specific_ui(self):
#
#         ## Centre Frame: Calibration controls
#         self.CENTRE_FRAME.config(text="Calibration")
#
#         f_snapshots = tk.Frame(self.CENTRE_FRAME)
#         f_snapshots.pack(side="top", fill="both", expand=True)
#
#         self.snap_button = tk.Button(f_snapshots, text="Take Snapshot",
#                                      font=self.parent.font_regular,
#                                      command=self._toggle_snapshot)
#         self.snap_button.pack(padx=(5, 0), side="left", fill="both", expand=False)
#
#         rf = tk.Frame(f_snapshots)
#         rf.pack(padx=5, side="left", fill="both", expand=True)
#
#         self.autosnap_var = tk.IntVar(value=0)
#         autosnap_button = tk.Checkbutton(rf, text="Auto snapshot", variable=self.autosnap_var, anchor='w',
#                                          font=self.parent.font_regular)
#         autosnap_button.pack(side="top", fill="both", expand=True)
#
#         self.reset_coverage_button = tk.Button(rf, text="Clear snapshots",
#                                                font=self.parent.font_regular,
#                                      command=self._reset_coverage)
#         self.reset_coverage_button.pack(side="top", fill="both", expand=False)
#
#         f_calibrate = tk.Frame(self.CENTRE_FRAME)
#         f_calibrate.pack(side="top", fill="both", expand=True)
#
#         separator = ttk.Separator(f_calibrate, orient='horizontal')
#         separator.pack(ipadx=5, side="top", fill="x", expand=True)
#
#         self.calibrate_button = tk.Button(f_calibrate, text="Calibrate",
#                                          highlightthickness=2, highlightbackground=self.parent.col_red,
#                                          font=self.parent.font_bold,
#                                          command=self._perform_calibration)
#
#         self.calibrate_button.pack(padx=(5, 0), pady=(0, 5), side="left", fill="both", expand=False)
#
#         f_saveload = tk.Frame(f_calibrate)
#         f_saveload.pack(padx=(5, 5), pady=(0, 5), side="left", fill="both", expand=True)
#
#         f_saveload_buttons = tk.Frame(f_saveload)
#         f_saveload_buttons.pack(side="top", fill="both", expand=True)
#
#         self.load_button = tk.Button(f_saveload_buttons, text="Load", font=self.parent.font_regular, command=self.load_calibration)
#         self.load_button.pack(padx=(0, 3), side="left", fill="both", expand=False)
#
#         self.save_button = tk.Button(f_saveload_buttons, text="Save", font=self.parent.font_regular, command=self.save_calibration)
#         self.save_button.pack(side="left", fill="both", expand=False)
#
#         self.saved_label = tk.Label(f_saveload, text='', anchor='w', justify='left', font=self.parent.font_regular)
#         self.saved_label.pack(side='bottom', fill='x')
#
#     def _toggle_snapshot(self):
#         self._manual_snapshot = True
#
#     def _detect(self) -> Image:
#
#         img_arr = np.frombuffer(self._frame_buffer, dtype=np.uint8).reshape(self._source_shape)
#         img_col = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
#
#         # Detect aruco markers
#         marker_corners, marker_ids, rejected = self.detector.detectMarkers(img_arr)
#
#         marker_corners, marker_ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(
#             image=img_arr,
#             board=self._charuco_board,
#             detectedCorners=marker_corners,
#             detectedIds=marker_ids,
#             rejectedCorners=rejected)
#
#         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
#
#         if marker_ids is not None and len(marker_ids) > 5:
#             img_col = cv2.aruco.drawDetectedMarkers(img_col, marker_corners, marker_ids)
#             charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(markerCorners=marker_corners,
#                                                                                                markerIds=marker_ids,
#                                                                                                image=img_arr,
#                                                                                                board=self._charuco_board,
#                                                                                                cameraMatrix=self.camera_matrix,
#                                                                                                distCoeffs=self.dist_coeffs,
#                                                                                                minMarkers=0)
#
#             try:
#                 charuco_corners = cv2.cornerSubPix(img_arr, charuco_corners,
#                                  winSize=(20, 20),
#                                  zeroZone=(-1, -1),
#                                  criteria=criteria)
#             except:
#                 pass
#
#             # Keep copy for visualisation in case of resetting
#             self.current_charuco_ids = charuco_ids
#             self.current_charuco_corners = charuco_corners
#
#             if charuco_retval > 4:
#
#                 img_col = cv2.aruco.drawDetectedCornersCharuco(
#                     image=img_col,
#                     charucoCorners=charuco_corners,
#                     charucoIds=charuco_ids)
#
#                 hull = cv2.convexHull(charuco_corners)
#
#                 self._current_coverage_area.fill(0)
#                 current = cv2.drawContours(self._current_coverage_area,
#                                            [hull.astype(int)], 0,
#                                            self.parent.col_white_rgb, -1).astype(bool)
#                 img_col = cv2.drawContours(img_col,
#                                            [hull.astype(int)], 0,
#                                            self.parent.col_green_rgb, 2)
#
#                 current_total = self._total_coverage_area[:, :, 1].astype(bool)     # Total 'seen' area
#
#                 overlap = (current_total & current)     # Overlap between current detection and everything seen so far
#                 new = (current & ~overlap)              # Area that is new in current detection
#                 # missing_area = ~current_total          # Area that is still missing
#
#                 self._coverage_pct = current_total.sum()/np.prod(self._source_shape) * 100   # Percentage covered so far
#
#                 # auto_snapshot = bool(self.autosnap_var.get()) & ((new & missing_area).sum() > new.sum() * 0.75)
#                 auto_snapshot = bool(self.autosnap_var.get()) & (new.sum() > current.sum() * 0.2)
#                 if auto_snapshot or self._manual_snapshot:
#                     self._total_coverage_area[new] += (np.array(self.parent.col_green_rgb) * 0.25).astype(np.uint8)
#
#                     self.detected_charuco_corners.append(charuco_corners)
#                     self.detected_charuco_ids.append(charuco_ids)
#                     self._manual_snapshot = False
#
#         return img_col
#
#     def _perform_calibration(self):
#
#         self.VIDEO_PANEL.configure(text='Calibrating...', fg='white')
#         self._refresh_videofeed(Image.fromarray(np.zeros_like(self._frame_buffer), mode='RGB'))
#
#         retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(charucoCorners=self.detected_charuco_corners,
#                                                                                             charucoIds=self.detected_charuco_ids,
#                                                                                             board=self._charuco_board,
#                                                                                             imageSize=self._source_shape[:2],
#                                                                                             cameraMatrix=self.camera_matrix,
#                                                                                             distCoeffs=self.dist_coeffs,
#                                                                                             flags=cv2.CALIB_USE_QR)
#
#         self._calib_error = retval
#         self.camera_matrix = camera_matrix
#         self.dist_coeffs = dist_coeffs
#
#         self.VIDEO_PANEL.configure(text='')
#
#         self._reset_coverage()
#         self.saved_label.config(text=f'')
#
#     def _reset_coverage(self):
#         self._total_coverage_area = np.zeros((*self._source_shape, 3), dtype=np.uint8)
#         self._current_coverage_area = np.zeros(self._source_shape, dtype=np.uint8)
#
#         self.detected_charuco_corners = deque(maxlen=self._max_frames)  # Corners seen so far
#         self.detected_charuco_ids = deque(maxlen=self._max_frames)  # Corresponding aruco ids
#
#         self._coverage_pct = 0
#
#     def save_calibration(self):
#         cam_name = self._camera.name.lower()
#
#         save_folder = self.parent.mgr.full_path.parent / 'calibrations' / self.parent.mgr.full_path.name / cam_name.lower()
#         save_folder.mkdir(exist_ok=True, parents=True)
#
#         np.save(save_folder / 'camera_matrix.npy', self.camera_matrix)
#         np.save(save_folder / 'dist_coeffs.npy', self.dist_coeffs)
#
#         if (save_folder / 'camera_matrix.npy').exists() and (save_folder / 'dist_coeffs.npy').exists():
#             self.saved_label.config(text=f'Saved.')
#
#     def load_calibration(self, load_path=None):
#
#         if load_path is None:
#             load_path = askdirectory()
#         load_path = Path(load_path)
#
#         if load_path.is_file():
#             load_path = load_path.parent
#
#         cam_name = self._camera.name.lower()
#
#         if cam_name not in load_path.name and (load_path / cam_name).exists():
#             load_path = load_path / f'cam{self.idx}'
#
#         if cam_name in load_path.name:
#             self.camera_matrix = np.load(load_path / 'camera_matrix.npy')
#             self.dist_coeffs = np.load(load_path / 'dist_coeffs.npy')
#             self.saved_label.config(text=f'Loaded.')
#         else:
#             self.saved_label.config(text=f'No calibration loaded.')
#
#     # def detect_pose(self):
#     #
#     #     img_arr = np.frombuffer(self._frame_buffer, dtype=np.uint8).reshape(self._source_shape)
#     #
#     #     # Undistort the image
#     #     undistorted_image = cv2.undistort(img_arr, self.camera_matrix, self.dist_coeffs)
#     #     img_col = cv2.cvtColor(undistorted_image, cv2.COLOR_GRAY2BGR)
#     #
#     #     # Detect markers in the undistorted image
#     #     marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(undistorted_image,
#     #                                                             self.aruco_dict,
#     #                                                             parameters=self.detector_parameters)
#     #
#     #     # If at least one marker is detected
#     #     if marker_ids is not None and len(marker_ids) > 0:
#     #         # Interpolate CharUco corners
#     #         charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners,
#     #                                                                                            marker_ids,
#     #                                                                                            undistorted_image,
#     #                                                                                            self._charuco_board)
#     #
#     #         # If enough corners are found, estimate the pose
#     #         if charuco_retval > 4:
#     #             retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids,
#     #                                                                     self._charuco_board,
#     #                                                                     self.camera_matrix, self.dist_coeffs,
#     #                                                                     None, None)
#     #
#     #             # If pose estimation is successful, draw the axis
#     #             if retval:
#     #                 img_col = cv2.drawFrameAxes(img_col, self.camera_matrix, self.dist_coeffs, rvec, tvec,
#     #                                   length=1,
#     #                                   thickness=3)
#     #     return img_col
#
#     def _full_frame_processing(self) -> Image:
#         return self._detect()
#
#     def _update_visualisations(self):
#
#         if self._coverage_pct >= self._recommended_coverage_pct_high:
#             self.calibrate_button.configure(highlightbackground=self.parent.col_green)
#             pct_color = self.parent.col_green_rgb
#         elif self._recommended_coverage_pct_high > self._coverage_pct >= self._recommended_coverage_pct_mid:
#             self.calibrate_button.configure(highlightbackground=self.parent.col_yelgreen)
#             pct_color = self.parent.col_yelgreen_rgb
#         elif self._recommended_coverage_pct_mid > self._coverage_pct >= self._recommended_coverage_pct_low:
#             self.calibrate_button.configure(highlightbackground=self.parent.col_orange)
#             pct_color = self.parent.col_orange_rgb
#         else:
#             self.calibrate_button.configure(highlightbackground=self.parent.col_red)
#             pct_color = self.parent.col_red_rgb
#
#         image = self._full_frame_processing()
#
#         image = cv2.addWeighted(image, 1.0, self._total_coverage_area, 0.8, 0.0)
#
#         if self.camera_matrix is not None:
#             image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
#
#             if self.current_charuco_corners is not None:
#                 valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners=self.current_charuco_corners,
#                                                                        charucoIds=self.current_charuco_ids,
#                                                                        board=self._charuco_board,
#                                                                        cameraMatrix=self.camera_matrix,
#                                                                        distCoeffs=self.dist_coeffs,
#                                                                        rvec=None, tvec=None)
#                 if valid:
#                     cv2.drawFrameAxes(image=image,
#                                       cameraMatrix=self.camera_matrix,
#                                       distCoeffs=self.dist_coeffs,
#                                       rvec=rvec, tvec=tvec, length=5)
#
#         image = cv2.putText(image, f'Snapshots coverage:',
#                                 (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                                 1, (255, 255, 255), 2, cv2.LINE_AA)
#
#         image = cv2.putText(image,
#                                 f'{self._coverage_pct:.2f}% ({len(self.detected_charuco_corners)} images)',
#                                 (400, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                                 1, pct_color, 2, cv2.LINE_AA)
#
#         image = cv2.putText(image, f'Calibration:',
#                                 (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
#                                 1, (255, 255, 255), 2, cv2.LINE_AA)
#
#         calib_col = self.parent.col_green_rgb if self.camera_matrix is not None else self.parent.col_white_rgb
#         image = cv2.putText(image, f'{"Applied" if self.camera_matrix is not None else "-"}',
#                                 (250, 100), cv2.FONT_HERSHEY_SIMPLEX,
#                                 1, calib_col, 2, cv2.LINE_AA)
#
#         img_pil = Image.fromarray(image, mode='RGB')
#         resized = self._resize_videofeed_image(img_pil)
#
#         return resized


