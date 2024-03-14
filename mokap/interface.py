import sys
import cv2
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
import tkinter.font as font
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
from datetime import datetime
from threading import Thread, Event
import screeninfo
from scipy import ndimage
import platform
from pathlib import Path
import os
import subprocess
from mokap import utils
from functools import partial
from collections import deque
import warnings
np.set_printoptions(precision=4, suppress=True)


def whxy(what):
    if isinstance(what, tk.Toplevel):
        dims, x, y = what.geometry().split('+')
        w, h = dims.split('x')
    elif isinstance(what, GUI):
        dims, x, y = what.root.geometry().split('+')
        w, h = dims.split('x')
    elif isinstance(what, VideoWindowMain):
        dims, x, y = what.window.geometry().split('+')
        w, h = dims.split('x')
    elif isinstance(what, screeninfo.Monitor):
        w, h, x, y = what.width, what.height, what.x, what.y
    else:
        w, h, x, y = 0, 0, 0, 0
    return int(w), int(h), int(x), int(y)


def compute_windows_size(source_dims, screen_dims):
    screenh, screenw = screen_dims[:2]
    sourceh, sourcew = source_dims[:2].max(axis=1)

    # For 2x3 screen grid
    max_window_w = screenw // 3
    max_window_h = screenh // 2 - VideoWindowMain.INFO_PANEL_FIXED_H

    # Scale it up or down, so it fits half the screen height
    if sourceh / max_window_h >= sourcew / max_window_w:
        scale = max_window_h / sourceh
    else:
        scale = max_window_w / sourcew

    computed_dims = np.floor(np.array([sourceh, sourcew]) * scale).astype(int)
    computed_dims[0] += VideoWindowMain.INFO_PANEL_FIXED_H

    return computed_dims


class VideoWindowBase:
    mainvideowindows_ids = []

    if 'Windows' in platform.system():
        INFO_PANEL_FIXED_H = 250  # in pixels
        INFO_PANEL_FIXED_W = 650  # in pixels
    else:
        INFO_PANEL_FIXED_H = 220  # in pixels
        INFO_PANEL_FIXED_W = 630  # in pixels

    def __init__(self, parent, idx=None):

        self.window = tk.Toplevel()
        self.window.minsize(self.INFO_PANEL_FIXED_W, self.INFO_PANEL_FIXED_H + 50)  # 50 px video haha

        if 'Darwin' in platform.system():
            # Trick to force macOS to open a window and not a tab
            self.window.resizable(False, False)
            self._macos_trick = True
        else:
            self._macos_trick = False
        self.parent = parent

        if idx is None:
            self.idx = len(VideoWindowBase.mainvideowindows_ids)
            VideoWindowBase.mainvideowindows_ids.append(self.idx)
        else:
            self.idx = idx

        self._source_shape = (self.parent.mgr.cameras[self.idx].height, self.parent.mgr.cameras[self.idx].width)
        self._cam_name = self.parent.mgr.cameras[self.idx].name

        self._bg_colour = f'#{self.parent.mgr.colours[self._cam_name].lstrip("#")}'
        self._fg_colour = self.parent.col_white if utils.hex_to_hls(self._bg_colour)[1] < 60 else self.parent.col_black
        self._window_bg_colour = self.window.cget('background')

        h, w = self._source_shape
        self.window.geometry(f"{w}x{h + self.INFO_PANEL_FIXED_H}")
        self.window.protocol("WM_DELETE_WINDOW", self.toggle_visibility)

        # Init state
        self._counter = 0
        self._counter_start = 0
        self._clock = datetime.now()
        self._fps = 0

        # Set thread-safe events
        self.visible = Event()
        self.visible.set()

        self._imgfnt = ImageFont.load_default()

        # Initialise text vars
        self.txtvar_warning = tk.StringVar()

        self.txtvar_resolution = tk.StringVar()
        self.txtvar_exposure = tk.StringVar()
        self.txtvar_capture_fps = tk.StringVar()
        self.txtvar_brightness = tk.StringVar()
        self.txtvar_display_fps = tk.StringVar()

        self.txtvar_camera_name = tk.StringVar()
        self.txtvar_camera_name.set(f'{self._cam_name.title()} camera')
        self.window.title(self.txtvar_camera_name.get())

        ## Initialise main video frame

        # Where the video will be displayed
        self._videofeed = tk.Label(self.window, bg='black', compound='center')
        self._videofeed.pack(fill='both', expand=True)

        # Where the full frame data will be stored
        self._frame_buffer = np.zeros(self._source_shape, dtype='<u1')

        # Resize window and apply the image
        self.auto_size()
        self._refresh_videofeed(Image.fromarray(self._frame_buffer))

    def _create_controls(self):
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._cam_name

    @property
    def colour(self) -> str:
        return f'#{self._bg_colour.lstrip("#")}'

    @property
    def colour_2(self) -> str:
        return f'#{self._fg_colour.lstrip("#")}'

    def color(self) -> str:
        return self.colour

    @property
    def color_2(self) -> str:
        return self.colour_2

    @property
    def source_shape(self):
        return self._source_shape

    @property
    def source_half_shape(self):
        return self._source_shape[0] // 2, self._source_shape[1] // 2

    @property
    def aspect_ratio(self):
        return self._source_shape[1] / self._source_shape[0]

    @property
    def videofeed_shape(self):
        h, w = self.window.winfo_height() - self.INFO_PANEL_FIXED_H, self.window.winfo_width()
        if h <= 1 or w <= 1:
            return self.source_shape
        return h, w

    def auto_size(self, apply=True):
        arbitrary_taskbar_h = 60
        if self.parent.selected_monitor.height < self.parent.selected_monitor.width:
            h = self.parent.selected_monitor.height // 2 - arbitrary_taskbar_h
            w = int(self.aspect_ratio * (h - self.INFO_PANEL_FIXED_H))
        else:
            w = self.parent.selected_monitor.width // 2
            h = int(w / self.aspect_ratio) + self.INFO_PANEL_FIXED_H

        if w < self.INFO_PANEL_FIXED_W:
            w = self.INFO_PANEL_FIXED_W
        if h < self.INFO_PANEL_FIXED_H:
            h = self.INFO_PANEL_FIXED_H
        if apply:
            self.window.geometry(f'{w}x{h}')
        return w, h

    def auto_move(self):
        # First corners and monitor centre, and then edge centres
        positions = np.hstack((self.positions.ravel()[::2], self.positions.ravel()[1::2]))
        nb_positions = len(positions)

        if self.idx <= nb_positions:
            pos = positions[self.idx]
        else:  # Start over to first position
            pos = positions[self.idx % nb_positions]

        self.move_to(pos)

    def move_to(self, pos):

        w_scr, h_scr, x_scr, y_scr = whxy(self.parent.selected_monitor)
        arbitrary_taskbar_h = 80

        w, h = self.auto_size(apply=False)

        if pos == 'nw':
            self.window.geometry(f"{w}x{h}+{x_scr}+{y_scr}")
        elif pos == 'n':
            self.window.geometry(f"{w}x{h}+{x_scr + w_scr // 2 - w // 2}+{y_scr}")
        elif pos == 'ne':
            self.window.geometry(f"{w}x{h}+{x_scr + w_scr - w - 1}+{y_scr}")

        elif pos == 'w':
            self.window.geometry(f"{w}x{h}+{x_scr}+{y_scr + h_scr // 2 - h // 2}")
        elif pos == 'c':
            self.window.geometry(f"{w}x{h}+{x_scr + w_scr // 2 - w // 2}+{y_scr + h_scr // 2 - h // 2}")
        elif pos == 'e':
            self.window.geometry(f"{w}x{h}+{x_scr + w_scr - w - 1}+{y_scr + h_scr // 2 - h // 2}")

        elif pos == 'sw':
            self.window.geometry(f"{w}x{h}+{x_scr}+{y_scr + h_scr - h - arbitrary_taskbar_h}")
        elif pos == 's':
            self.window.geometry(f"{w}x{h}+{x_scr + w_scr // 2 - w // 2}+{y_scr + h_scr - h - arbitrary_taskbar_h}")
        elif pos == 'se':
            self.window.geometry(f"{w}x{h}+{x_scr + w_scr - w - 1}+{y_scr + h_scr - h - arbitrary_taskbar_h}")

    def _refresh_framebuffer(self):
        self._frame_buffer.fill(0)

        if self.parent.mgr.acquiring and self.parent.current_buffers is not None:
            buf = self.parent.current_buffers[self.idx]
            if buf is not None:
                self._frame_buffer[:] = np.frombuffer(buf, dtype=np.uint8).reshape(self._source_shape)

    def _refresh_videofeed(self, image):
        imagetk = ImageTk.PhotoImage(image=image)
        try:
            self._videofeed.configure(image=imagetk)
            self.imagetk = imagetk
        except Exception:
            # If new image is garbage collected too early, do nothing - this prevents the image from flashing
            pass

    def _update_visualisation(self):

        # Get window size and set new videofeed size, preserving aspect ratio
        h, w = self.videofeed_shape

        if w / h > self.aspect_ratio:
            w = int(h * self.aspect_ratio)
        else:
            h = int(w / self.aspect_ratio)

        img_pillow = Image.fromarray(self._frame_buffer, mode='L').convert('RGBA')
        img_pillow = img_pillow.resize((w, h))

        self._refresh_videofeed(img_pillow)

    def _update_txtvars(self):
        pass

    def update(self):
        if self._macos_trick:
            self.window.resizable(True, True)

        while True:
            if self.visible.is_set():
                # Update display fps counter
                now = datetime.now()
                dt = (now - self._clock).total_seconds()
                self._fps = (self._counter - self._counter_start) / dt
                self._refresh_framebuffer()
                self._update_visualisation()
                self._update_txtvars()
                self._counter += 1
            else:
                self.visible.wait()

    def toggle_visibility(self, tf=None):
        if tf is None:
            tf = not self.visible.is_set()

        if self.visible.is_set() and tf is False:
            self.visible.clear()
            self.parent.vis_checkboxes[self.idx].set(0)
            self.window.withdraw()

        elif not self.visible.is_set() and tf is True:
            if 'Darwin' in platform.system():
                # Trick to force macOS to open a window and not a tab
                self.window.resizable(False, False)
                self._macos_trick = True
            self.visible.set()
            self.parent.vis_checkboxes[self.idx].set(1)
            self.window.deiconify()
            if self._macos_trick:
                self.window.resizable(True, True)
        else:
            pass


class VideoWindowCalib(VideoWindowBase):
    def __init__(self, parent, idx):
        super().__init__(parent, idx)

        self._total_coverage_area = np.zeros((*self.source_shape, 3), dtype=np.uint8)
        self._current_coverage_area = np.zeros(self.source_shape, dtype=np.uint8)

        # ChAruco board variables
        BOARD_ROWS = 5  # Total rows in the board (chessboard)
        BOARD_COLS = 4  # Total cols in the board
        ARUCO_SQ_L = 4  # Side length of each individual aruco markers
        PHYSICAL_L = 15  # Length of the small side of the board in real life units (i.e. mm)

        self._aruco_dict, self._charuco_board = utils.generate_board(BOARD_ROWS, BOARD_COLS, ARUCO_SQ_L, PHYSICAL_L)

        detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self._aruco_dict, detector_params)

        self._max_frames = 150
        self._recommended_coverage_pct_high = 80
        self._recommended_coverage_pct_mid = 50
        self._recommended_coverage_pct_low = 25

        self.current_charuco_corners = None                                 # Currently visible corners
        self.current_charuco_ids = None                                     # Corresponding aruco ids
        self.detected_charuco_corners = deque(maxlen=self._max_frames)      # All corners seen so far
        self.detected_charuco_ids = deque(maxlen=self._max_frames)          # All corresponding aruco ids

        self.camera_matrix = None
        self.dist_coeffs = None

        self._coverage_pct = 0

        # Set initial values
        h, w = self.parent.mgr.cameras[self.idx].height, self.parent.mgr.cameras[self.idx].width
        self.txtvar_resolution.set(f"{w}×{h} px")
        self.txtvar_exposure.set(f"{self.parent.mgr.cameras[self.idx].exposure} µs")

        self._manual_snapshot = False

        self._create_controls()

    def _create_controls(self):

        ## Bottom panel
        INFO_PANEL_FRAME = tk.Frame(self.window, height=self.INFO_PANEL_FIXED_H, )
        INFO_PANEL_FRAME.pack(side='top', fill='x', expand=False)

        # Camera name bar
        name_bar = tk.Label(INFO_PANEL_FRAME, textvariable=self.txtvar_camera_name,
                            anchor='center', justify='center', height=2,
                            fg=self.colour_2, bg=self.colour, font=self.parent.font_bold)
        name_bar.pack(side='top', fill='x')

        ## Information block
        f_information = tk.LabelFrame(INFO_PANEL_FRAME, text="Information",
                                      height=self.INFO_PANEL_FIXED_H,
                                      width=200)
        f_information.pack(ipadx=10, ipady=3, padx=5, pady=5, side='left', fill='y', expand=False)

        for label, var in zip(['Resolution', 'Capture', 'Exposure', 'Brightness', 'Display'],
                              [self.txtvar_resolution,
                               self.txtvar_capture_fps,
                               self.txtvar_exposure,
                               self.txtvar_brightness,
                               self.txtvar_display_fps]):
            f = tk.Frame(f_information)
            f.pack(side='top', fill='x', expand=True)

            l = tk.Label(f, text=f"{label} :",
                         anchor='e', justify='right', width=13,
                         font=self.parent.font_regular)
            l.pack(side='left', fill='y')

            v = tk.Label(f, textvariable=var,
                         anchor='w', justify='left',
                         font=self.parent.font_regular)
            v.pack(side='left', fill='y')

        ## Save/Load block

        f_information = tk.LabelFrame(INFO_PANEL_FRAME, text="Calibration",
                                      height=self.INFO_PANEL_FIXED_H,
                                      width=300)
        f_information.pack(ipadx=10, ipady=3, padx=5, pady=5, side='left', fill='y', expand=False)

        f_snapshots = tk.Frame(f_information)
        f_snapshots.pack(side="top", fill="x", expand=True)

        self.snap_button = tk.Button(f_snapshots, text="Snapshot", font=self.parent.font_bold,
                                     command=self._toggle_snapshot)
        self.snap_button.pack(padx=5, pady=5, side="left", fill="both", expand=False)

        self.autosnap_var = tk.IntVar(value=0)
        autosnap_button = tk.Checkbutton(f_snapshots, text="Auto snapshot", variable=self.autosnap_var, font=self.parent.font_regular)
        autosnap_button.pack(padx=5, pady=5, side="top", fill="both", expand=False)

        self.reset_coverage_button = tk.Button(f_snapshots, text="Clear snapshots", font=self.parent.font_regular,
                                     command=self._reset_coverage)

        self.reset_coverage_button.pack(padx=5, pady=5, side="top", fill="both", expand=False)

        self.calibrate_button = tk.Button(f_information, text="Compute calibration",
                                         highlightthickness=2, highlightbackground=self.parent.col_red,
                                         font=self.parent.font_bold,
                                         command=self._perform_calibration)

        self.calibrate_button.pack(padx=5, pady=5, side="left", fill="both", expand=False)

        f_saveload = tk.Frame(f_information)
        f_saveload.pack(padx=5, pady=5, side="top", fill="x", expand=True)

        self.load_button = tk.Button(f_saveload, text="Load", font=self.parent.font_regular, command=self.load_calibration)
        self.load_button.pack(side="left", fill="both", expand=False)

        self.save_button = tk.Button(f_saveload, text="Save", font=self.parent.font_regular, command=self.save_calibration)
        self.save_button.pack(side="left", fill="both", expand=False)

        self.saved_label = tk.Label(f_information, text='', anchor='w', justify='left', font=self.parent.font_regular)
        self.saved_label.pack(side='left', fill='y')

        ## View controls block
        view_info_frame = tk.LabelFrame(INFO_PANEL_FRAME, text="View",
                                        height=self.INFO_PANEL_FIXED_H,
                                        width=200)
        view_info_frame.pack(ipadx=10, ipady=3, padx=5, pady=5, side='left', fill='y', expand=False)

        f_windowsnap = tk.Frame(view_info_frame)
        f_windowsnap.pack(side='top', fill='y')

        l_windowsnap = tk.Label(f_windowsnap, text=f"Snap window to:",
                                anchor='w', justify='left',
                                font=self.parent.font_regular,
                                padx=5, pady=10)
        l_windowsnap.pack(side='left', fill='y')

        f_buttons_windowsnap = tk.Frame(f_windowsnap, padx=5, pady=10)
        f_buttons_windowsnap.pack(side='left', fill='both', expand=True)

        self.positions = np.array([['nw', 'n', 'ne'],
                                   ['w', 'c', 'e'],
                                   ['sw', 's', 'se']])

        self._pixel = tk.PhotoImage(width=1, height=1)
        for r in range(3):
            for c in range(3):
                b = tk.Button(f_buttons_windowsnap,
                              image=self._pixel, compound="center",
                              width=6, height=6,
                              command=partial(self.move_to, self.positions[r, c]))
                b.grid(row=r, column=c)

        f_buttons_controls = tk.Frame(view_info_frame, padx=5)
        f_buttons_controls.pack(ipadx=2, side='top', fill='y', expand=False)

    def _toggle_snapshot(self):
        self._manual_snapshot = True

    def _detect(self):

        img_arr = np.frombuffer(self._frame_buffer, dtype=np.uint8).reshape(self.source_shape)
        img_col = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)

        # Detect aruco markers
        marker_corners, marker_ids, rejected = self.detector.detectMarkers(img_arr)

        marker_corners, marker_ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(
            image=img_arr,
            board=self._charuco_board,
            detectedCorners=marker_corners,
            detectedIds=marker_ids,
            rejectedCorners=rejected)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        if marker_ids is not None and len(marker_ids) > 5:
            img_col = cv2.aruco.drawDetectedMarkers(img_col, marker_corners, marker_ids)
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(markerCorners=marker_corners,
                                                                                               markerIds=marker_ids,
                                                                                               image=img_arr,
                                                                                               board=self._charuco_board,
                                                                                               cameraMatrix=self.camera_matrix,
                                                                                               distCoeffs=self.dist_coeffs,
                                                                                               minMarkers=0)

            charuco_corners = cv2.cornerSubPix(img_arr, charuco_corners,
                             winSize=(20, 20),
                             zeroZone=(-1, -1),
                             criteria=criteria)

            # Keep copy for visualisation in case of resetting
            self.current_charuco_ids = charuco_ids
            self.current_charuco_corners = charuco_corners

            if charuco_retval > 4:

                img_col = cv2.aruco.drawDetectedCornersCharuco(
                    image=img_col,
                    charucoCorners=charuco_corners,
                    charucoIds=charuco_ids)

                hull = cv2.convexHull(charuco_corners)

                self._current_coverage_area.fill(0)
                current = cv2.drawContours(self._current_coverage_area,
                                           [hull.astype(int)], 0,
                                           self.parent.col_white_rgb, -1).astype(bool)
                img_col = cv2.drawContours(img_col,
                                           [hull.astype(int)], 0,
                                           self.parent.col_green_rgb, 2)

                current_total = self._total_coverage_area[:, :, 1].astype(bool)     # Total 'seen' area

                overlap = (current_total & current)     # Overlap between current detection and everything seen so far
                new = (current & ~overlap)              # Area that is new in current detection
                # missing_area = ~current_total          # Area that is still missing

                self._coverage_pct = current_total.sum()/np.prod(self.source_shape) * 100   # Percentage covered so far

                # auto_snapshot = bool(self.autosnap_var.get()) & ((new & missing_area).sum() > new.sum() * 0.75)
                auto_snapshot = bool(self.autosnap_var.get()) & (new.sum() > current.sum() * 0.2)
                if auto_snapshot or self._manual_snapshot:
                    self._total_coverage_area[new] += (np.array(self.parent.col_green_rgb) * 0.25).astype(np.uint8)

                    self.detected_charuco_corners.append(charuco_corners)
                    self.detected_charuco_ids.append(charuco_ids)
                    self._manual_snapshot = False

        return img_col


    def _perform_calibration(self):

        self._videofeed.configure(text='Calibrating...', fg='white')
        self._refresh_videofeed(Image.fromarray(np.zeros_like(self._frame_buffer), mode='L'))

        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(charucoCorners=self.detected_charuco_corners,
                                                                                            charucoIds=self.detected_charuco_ids,
                                                                                            board=self._charuco_board,
                                                                                            imageSize=self._source_shape[:2],
                                                                                            cameraMatrix=self.camera_matrix,
                                                                                            distCoeffs=self.dist_coeffs,
                                                                                            flags=cv2.CALIB_USE_QR)

        self._calib_error = retval
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        self._videofeed.configure(text='')

        self._reset_coverage()
        self.saved_label.config(text=f'')

    def _reset_coverage(self):
        self._total_coverage_area = np.zeros((*self.source_shape, 3), dtype=np.uint8)
        self._current_coverage_area = np.zeros(self.source_shape, dtype=np.uint8)

        self.detected_charuco_corners = deque(maxlen=self._max_frames)  # Corners seen so far
        self.detected_charuco_ids = deque(maxlen=self._max_frames)  # Corresponding aruco ids

        self._coverage_pct = 0

    def save_calibration(self):
        cam_name = self._cam_name.lower()

        save_folder = self.parent.mgr.full_path.parent / 'calibrations' / self.parent.mgr.full_path.name / cam_name.lower()
        save_folder.mkdir(exist_ok=True, parents=True)

        np.save(save_folder / 'camera_matrix.npy', self.camera_matrix)
        np.save(save_folder / 'dist_coeffs.npy', self.dist_coeffs)

        if (save_folder / 'camera_matrix.npy').exists() and (save_folder / 'dist_coeffs.npy').exists():
            self.saved_label.config(text=f'Saved.')

    def load_calibration(self, load_path=None):

        if load_path is None:
            load_path = askdirectory()
        load_path = Path(load_path)

        if load_path.is_file():
            load_path = load_path.parent

        cam_name = self._cam_name.lower()

        if cam_name not in load_path.name and (load_path / cam_name).exists():
            load_path = load_path / f'cam{self.idx}'

        if cam_name in load_path.name:
            self.camera_matrix = np.load(load_path / 'camera_matrix.npy')
            self.dist_coeffs = np.load(load_path / 'dist_coeffs.npy')
            self.saved_label.config(text=f'Loaded.')
        else:
            print('Calibration files not found.')

    # def detect_pose(self):
    #
    #     img_arr = np.frombuffer(self._frame_buffer, dtype=np.uint8).reshape(self.source_shape)
    #
    #     # Undistort the image
    #     undistorted_image = cv2.undistort(img_arr, self.camera_matrix, self.dist_coeffs)
    #     img_col = cv2.cvtColor(undistorted_image, cv2.COLOR_GRAY2BGR)
    #
    #     # Detect markers in the undistorted image
    #     marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(undistorted_image,
    #                                                             self.aruco_dict,
    #                                                             parameters=self.detector_parameters)
    #
    #     # If at least one marker is detected
    #     if marker_ids is not None and len(marker_ids) > 0:
    #         # Interpolate CharUco corners
    #         charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners,
    #                                                                                            marker_ids,
    #                                                                                            undistorted_image,
    #                                                                                            self._charuco_board)
    #
    #         # If enough corners are found, estimate the pose
    #         if charuco_retval > 4:
    #             retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids,
    #                                                                     self._charuco_board,
    #                                                                     self.camera_matrix, self.dist_coeffs,
    #                                                                     None, None)
    #
    #             # If pose estimation is successful, draw the axis
    #             if retval:
    #                 img_col = cv2.drawFrameAxes(img_col, self.camera_matrix, self.dist_coeffs, rvec, tvec,
    #                                   length=1,
    #                                   thickness=3)
    #     return img_col

    def _update_txtvars(self):

        if self.parent.mgr.acquiring:
            cap_fps = self.parent.capture_fps[self.idx]

            if 0 < cap_fps < 1000:  # only makes sense to display real values
                self.txtvar_capture_fps.set(f"{cap_fps:.2f} fps")
            else:
                self.txtvar_capture_fps.set("-")

            brightness = np.round(self._frame_buffer.mean() / 255 * 100, decimals=2)
            self.txtvar_brightness.set(f"{brightness:.2f}%")
        else:
            self.txtvar_capture_fps.set("Off")
            self.txtvar_brightness.set("-")

        self.txtvar_display_fps.set(f"{self._fps:.2f} fps")

    def _update_visualisation(self):

        if self._coverage_pct >= self._recommended_coverage_pct_high:
            self.calibrate_button.configure(highlightbackground=self.parent.col_green)
            pct_color = self.parent.col_green_rgb
        elif self._recommended_coverage_pct_high > self._coverage_pct >= self._recommended_coverage_pct_mid:
            self.calibrate_button.configure(highlightbackground=self.parent.col_yelgreen)
            pct_color = self.parent.col_yelgreen_rgb
        elif self._recommended_coverage_pct_mid > self._coverage_pct >= self._recommended_coverage_pct_low:
            self.calibrate_button.configure(highlightbackground=self.parent.col_orange)
            pct_color = self.parent.col_orange_rgb
        else:
            self.calibrate_button.configure(highlightbackground=self.parent.col_red)
            pct_color = self.parent.col_red_rgb

        image_viz = self._detect()

        image_viz = cv2.addWeighted(image_viz, 1.0, self._total_coverage_area, 0.8, 0.0)

        if self.camera_matrix is not None:
            image_viz = cv2.undistort(image_viz, self.camera_matrix, self.dist_coeffs)

            if self.current_charuco_corners is not None:
                valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners=self.current_charuco_corners,
                                                                       charucoIds=self.current_charuco_ids,
                                                                       board=self._charuco_board,
                                                                       cameraMatrix=self.camera_matrix,
                                                                       distCoeffs=self.dist_coeffs,
                                                                       rvec=None, tvec=None)
                if valid:
                    cv2.drawFrameAxes(image=image_viz,
                                      cameraMatrix=self.camera_matrix,
                                      distCoeffs=self.dist_coeffs,
                                      rvec=rvec, tvec=tvec, length=5)

        image_viz = cv2.putText(image_viz, f'Snapshots coverage:',
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2, cv2.LINE_AA)

        image_viz = cv2.putText(image_viz,
                                f'{self._coverage_pct:.2f}% ({len(self.detected_charuco_corners)} images)',
                                (400, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, pct_color, 2, cv2.LINE_AA)

        image_viz = cv2.putText(image_viz, f'Calibration:',
                                (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2, cv2.LINE_AA)

        calib_col = self.parent.col_green_rgb if self.camera_matrix is not None else self.parent.col_white_rgb
        image_viz = cv2.putText(image_viz, f'{"Applied" if self.camera_matrix is not None else "-"}',
                                (250, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                1, calib_col, 2, cv2.LINE_AA)

        # Get window size and set new videofeed size, preserving aspect ratio
        h, w = self.videofeed_shape

        if w / h > self.aspect_ratio:
            w = int(h * self.aspect_ratio)
        else:
            h = int(w / self.aspect_ratio)

        img_pillow = Image.fromarray(image_viz)
        img_pillow = img_pillow.resize((w, h))

        self._refresh_videofeed(img_pillow)


    def update(self):
        if self.window.resizable() == (0, 0):
            # Reenable resizing on macOS (see trick at winow creation)
            self.window.resizable(True, True)

        while self.parent._is_calibrating.is_set():

            if self.visible.is_set():
                # Update display fps counter
                now = datetime.now()
                dt = (now - self._clock).total_seconds()
                self._fps = (self._counter - self._counter_start) / dt
                self._refresh_framebuffer()
                self._update_txtvars()
                self._update_visualisation()

                self._counter += 1
            else:
                self.visible.wait()

        self.window.destroy()


class VideoWindowMain(VideoWindowBase):

    def __init__(self, parent):
        super().__init__(parent)

        self._show_focus = Event()
        self._show_focus.clear()

        self._magnification = Event()
        self._magnification.set()

        self._warning = Event()
        self._warning.clear()

        ## Magnification parameters
        magn_portion = 12
        self.magn_zoom = tk.DoubleVar()
        self.magn_zoom.set(2)

        self.magn_size = self.source_shape[0] // magn_portion, self.source_shape[1] // magn_portion
        magn_half_size = self.magn_size[0] // 2, self.magn_size[1] // 2

        self.magn_A = self.source_half_shape[0] - magn_half_size[0], self.source_half_shape[1] - magn_half_size[1]
        self.magn_B = self.source_half_shape[0] + magn_half_size[0], self.source_half_shape[1] + magn_half_size[1]

        # Initialise text vars
        self.txtvar_warning = tk.StringVar()

        self.txtvar_resolution = tk.StringVar()
        self.txtvar_exposure = tk.StringVar()
        self.txtvar_capture_fps = tk.StringVar()
        self.txtvar_brightness = tk.StringVar()
        self.txtvar_display_fps = tk.StringVar()

        # Kernel to use for focus detection
        self._kernel = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]], dtype=np.uint8)

        # Set initial values
        h, w = self.parent.mgr.cameras[self.idx].height, self.parent.mgr.cameras[self.idx].width
        self.txtvar_resolution.set(f"{w}×{h} px")
        self.txtvar_exposure.set(f"{self.parent.mgr.cameras[self.idx].exposure} µs")
        self._applied_fps = self.parent.mgr.cameras[self.idx].framerate

        self._create_controls()

    def _create_controls(self):

        ## Bottom panel
        INFO_PANEL_FRAME = tk.Frame(self.window, height=self.INFO_PANEL_FIXED_H, )
        INFO_PANEL_FRAME.pack(side='top', fill='x', expand=False)

        # Camera name bar
        name_bar = tk.Label(INFO_PANEL_FRAME, textvariable=self.txtvar_camera_name,
                            anchor='center', justify='center', height=2,
                            fg=self.colour_2, bg=self.colour, font=self.parent.font_bold)
        name_bar.pack(side='top', fill='x')

        ## Information block
        f_information = tk.LabelFrame(INFO_PANEL_FRAME, text="Information",
                                      height=self.INFO_PANEL_FIXED_H,
                                      width=200)
        f_information.pack(ipadx=10, ipady=3, padx=5, pady=5, side='left', fill='y', expand=False)

        for label, var in zip(['Resolution', 'Capture', 'Exposure', 'Brightness', 'Display'],
                              [self.txtvar_resolution,
                               self.txtvar_capture_fps,
                               self.txtvar_exposure,
                               self.txtvar_brightness,
                               self.txtvar_display_fps]):
            f = tk.Frame(f_information)
            f.pack(side='top', fill='x', expand=True)

            l = tk.Label(f, text=f"{label} :",
                         anchor='e', justify='right', width=13,
                         font=self.parent.font_regular)
            l.pack(side='left', fill='y')

            v = tk.Label(f, textvariable=var,
                         anchor='w', justify='left',
                         font=self.parent.font_regular)
            v.pack(side='left', fill='y')

        ## Camera controls block
        f_camera_controls = tk.LabelFrame(INFO_PANEL_FRAME, text="Control",
                                          height=self.INFO_PANEL_FIXED_H,
                                          width=200)
        f_camera_controls.pack(ipadx=10, ipady=3, padx=5, pady=5, side='left', fill='y', expand=False)

        lf = tk.Frame(f_camera_controls)
        lf.pack(ipady=10, side='left', fill='y', expand=True)

        self.camera_controls_sliders = {}
        self._apply_all_vars = {}

        rf = tk.LabelFrame(f_camera_controls, text='Sync',
                           width=1, font=self.parent.font_mini)
        rf.pack(side='right', fill='y', expand=True)

        for label, val, func, slider_params in zip(['Framerate', 'Exposure', 'Blacks', 'Gain', 'Gamma'],
                                                   [self.parent.mgr.cameras[self.idx].framerate,
                                                    self.parent.mgr.cameras[self.idx].exposure,
                                                    self.parent.mgr.cameras[self.idx].blacks,
                                                    self.parent.mgr.cameras[self.idx].gain,
                                                    self.parent.mgr.cameras[self.idx].gamma],
                                                   [self._update_fps_all_cams,
                                                    self._update_exp_all_cams,
                                                    self._update_blacks_all_cams,
                                                    self._update_gain_all_cams,
                                                    self._update_gamma_all_cams],
                                                   [(1, 220, 1, 1),
                                                    (21, 1e5, 5, 1),  # in microseconds - 1e5 ~ 10 fps
                                                    (0.0, 32.0, 0.5, 3),
                                                    (0.0, 36.0, 0.5, 3),
                                                    (0.0, 3.99, 0.05, 3),
                                                    ]):
            f = tk.Frame(lf)
            f.pack(side='top', fill='y', expand=True)

            v = tk.IntVar()
            b = tk.Checkbutton(rf, variable=v,
                               state='normal')
            v.set(1)
            b.pack(side="top", fill="y", expand=True)
            self._apply_all_vars[label.lower()] = v

            s = tk.Scale(f,
                         from_=slider_params[0], to=slider_params[1], resolution=slider_params[2],
                         digits=slider_params[3],
                         orient='horizontal', width=6, sliderlength=10, length=80)
            s.set(val)
            s.bind("<ButtonRelease-1>", func)
            s.pack(padx=3, side='right', fill='y', expand=True)

            key = label.split('(')[0].strip().lower()  # Get the simplified label (= wihtout spaces and units)
            self.camera_controls_sliders[key] = s

            l = tk.Label(f, text=f'{label} :',
                         anchor='se', justify='right', width=12,
                         font=self.parent.font_regular)
            l.pack(side='right', fill='both', expand=True)

        ## View controls block
        view_info_frame = tk.LabelFrame(INFO_PANEL_FRAME, text="View",
                                        height=self.INFO_PANEL_FIXED_H,
                                        width=200)
        view_info_frame.pack(ipadx=10, ipady=3, padx=5, pady=5, side='left', fill='y', expand=False)

        f_windowsnap = tk.Frame(view_info_frame)
        f_windowsnap.pack(side='top', fill='y')

        l_windowsnap = tk.Label(f_windowsnap, text=f"Snap window to:",
                                anchor='w', justify='left',
                                font=self.parent.font_regular,
                                padx=5, pady=10)
        l_windowsnap.pack(side='left', fill='y')

        f_buttons_windowsnap = tk.Frame(f_windowsnap, padx=5, pady=10)
        f_buttons_windowsnap.pack(side='left', fill='both', expand=True)

        self.positions = np.array([['nw', 'n', 'ne'],
                                   ['w', 'c', 'e'],
                                   ['sw', 's', 'se']])

        self._pixel = tk.PhotoImage(width=1, height=1)
        for r in range(3):
            for c in range(3):
                b = tk.Button(f_buttons_windowsnap,
                              image=self._pixel, compound="center",
                              width=6, height=6,
                              command=partial(self.move_to, self.positions[r, c]))
                b.grid(row=r, column=c)

        f_buttons_controls = tk.Frame(view_info_frame, padx=5)
        f_buttons_controls.pack(ipadx=2, side='top', fill='y', expand=False)

        self.show_focus_button = tk.Button(f_buttons_controls, text="Focus",
                                           width=8,
                                           highlightthickness=2, highlightbackground=self._window_bg_colour,
                                           font=self.parent.font_regular,
                                           command=self._toggle_focus_display)
        self.show_focus_button.grid(row=0, column=0)

        self.show_mag_button = tk.Button(f_buttons_controls, text="Magnifier",
                                         width=10,
                                         highlightthickness=2, highlightbackground=self.parent.col_yellow,
                                         font=self.parent.font_regular,
                                         command=self._toggle_mag_display)
        self.show_mag_button.grid(row=0, column=1)

        self.slider_magn = tk.Scale(f_buttons_controls, variable=self.magn_zoom,
                                    from_=1, to=5, resolution=0.1, orient='horizontal',
                                    width=10, sliderlength=10, length=80)
        self.slider_magn.grid(row=1, column=1, padx=(0, 0))

    # === TODO - merge these functions below ===
    def update_framerate(self, event=None):
        slider = self.camera_controls_sliders['framerate']
        new_val = slider.get()
        self.parent.mgr.cameras[self.idx].framerate = new_val

        # The actual maximum framerate depends on the exposure, so it might not be what the user requested
        # Thus we need to update the slider value to the actual framerate
        applied_fps = self.parent.mgr.cameras[self.idx].framerate
        slider.set(applied_fps)

        # Keep a local copy to warn user if read framerate is too different from wanted fps
        self._applied_fps = applied_fps

        # Refresh fps counters for the UI
        self.parent._capture_clock = datetime.now()
        self.parent.start_indices[:] = self.parent.mgr.indices

    def update_exposure(self, event=None):
        slider = self.camera_controls_sliders['exposure']
        new_val = slider.get()
        self.parent.mgr.cameras[self.idx].exposure = new_val

        # And update the slider to the actual new value (can be different than the one requested)
        slider.set(self.parent.mgr.cameras[self.idx].exposure)

        #
        #
        # We also need to update the framerate slider to current resulting fps after exposure change
        slider_framerate = self.camera_controls_sliders['framerate']
        slider_framerate.set(self.parent.mgr.cameras[self.idx].framerate)

        self.txtvar_exposure.set(f"{self.parent.mgr.cameras[self.idx].exposure} µs")

        # And callback to the update framerate function because the new exposure time might cap the framerate out
        self.update_framerate(event)

        #
        #

    def update_blacks(self, event=None):
        slider = self.camera_controls_sliders['blacks']
        new_val = slider.get()
        self.parent.mgr.cameras[self.idx].blacks = new_val

        # And update the slider to the actual new value (can be different than the one requested)
        slider.set(self.parent.mgr.cameras[self.idx].blacks)

    def update_gain(self, event=None):
        slider = self.camera_controls_sliders['gain']
        new_val = slider.get()
        self.parent.mgr.cameras[self.idx].gain = new_val

        # And update the slider to the actual new value (can be different than the one requested)
        slider.set(self.parent.mgr.cameras[self.idx].gain)

    def update_gamma(self, event=None):
        slider = self.camera_controls_sliders['gamma']
        new_val = slider.get()
        self.parent.mgr.cameras[self.idx].gamma = new_val

        # And update the slider to the actual new value (can be different than the one requested)
        slider.set(self.parent.mgr.cameras[self.idx].gamma)

    def _update_fps_all_cams(self, event=None):
        self.update_framerate()
        for window in self.parent.video_windows:
            if window is not self and bool(window._apply_all_vars['framerate'].get()):
                slider = self.camera_controls_sliders['framerate']
                new_val = slider.get()
                window.camera_controls_sliders['framerate'].set(new_val)
                window.update_framerate()

    def _update_exp_all_cams(self, event=None):
        self.update_exposure()
        for window in self.parent.video_windows:
            if window is not self and bool(window._apply_all_vars['exposure'].get()):
                slider = self.camera_controls_sliders['exposure']
                new_val = slider.get()
                window.camera_controls_sliders['exposure'].set(new_val)
                window.update_exposure()

    def _update_blacks_all_cams(self, event=None):
        self.update_blacks()
        for window in self.parent.video_windows:
            if window is not self and bool(window._apply_all_vars['blacks'].get()):
                slider = self.camera_controls_sliders['blacks']
                new_val = slider.get()
                window.camera_controls_sliders['blacks'].set(new_val)
                window.update_blacks()

    def _update_gain_all_cams(self, event=None):
        self.update_gain()
        for window in self.parent.video_windows:
            if window is not self and bool(window._apply_all_vars['gain'].get()):
                slider = self.camera_controls_sliders['gain']
                new_val = slider.get()
                window.camera_controls_sliders['gain'].set(new_val)
                window.update_gain()

    def _update_gamma_all_cams(self, event=None):
        self.update_gamma()
        for window in self.parent.video_windows:
            if window is not self and bool(window._apply_all_vars['gamma'].get()):
                slider = self.camera_controls_sliders['gamma']
                new_val = slider.get()
                window.camera_controls_sliders['gamma'].set(new_val)
                window.update_gamma()

    def _toggle_focus_display(self):

        if self._show_focus.is_set():
            self._show_focus.clear()
            self.show_focus_button.configure(highlightbackground=self._window_bg_colour)
        else:
            self._show_focus.set()
            self.show_focus_button.configure(highlightbackground=self.parent.col_green)

    def _toggle_mag_display(self):

        if self._magnification.is_set():
            self._magnification.clear()
            self.show_mag_button.configure(highlightbackground=self._window_bg_colour)
            self.slider_magn.config(state='disabled')
        else:
            self._magnification.set()
            self.show_mag_button.configure(highlightbackground=self.parent.col_yellow)
            self.slider_magn.config(state='active')

    # === end TODO ===

    def _update_txtvars(self):

        if self.parent.mgr.acquiring:
            cap_fps = self.parent.capture_fps[self.idx]

            if 0 < cap_fps < 1000:  # only makes sense to display real values
                if abs(cap_fps - self._applied_fps) > 10:
                    self.txtvar_warning.set('[ WARNING: Framerate ]')
                    self._warning.set()
                else:
                    self._warning.clear()
                self.txtvar_capture_fps.set(f"{cap_fps:.2f} fps")
            else:
                self.txtvar_capture_fps.set("-")

            brightness = np.round(self._frame_buffer.mean() / 255 * 100, decimals=2)
            self.txtvar_brightness.set(f"{brightness:.2f}%")
        else:
            self.txtvar_capture_fps.set("Off")
            self.txtvar_brightness.set("-")

        self.txtvar_display_fps.set(f"{self._fps:.2f} fps")

    def _update_visualisation(self):

        # Get window size and set new videofeed size, preserving aspect ratio
        h, w = self.videofeed_shape

        if w / h > self.aspect_ratio:
            w = int(h * self.aspect_ratio)
        else:
            h = int(w / self.aspect_ratio)

        # Get new coordinates
        x_centre, y_centre = w // 2, h // 2
        x_north, y_north = w // 2, 0
        x_south, y_south = w // 2, h
        x_east, y_east = w, h // 2
        x_west, y_west = 0, h // 2

        img_pillow = Image.fromarray(self._frame_buffer, mode='L').convert('RGBA')
        img_pillow = img_pillow.resize((w, h))

        d = ImageDraw.Draw(img_pillow)
        # Draw crosshair
        d.line((x_west, y_west, x_east, y_east), fill=self.parent.col_white_rgb, width=1)  # Horizontal
        d.line((x_north, y_north, x_south, y_south), fill=self.parent.col_white_rgb, width=1)  # Vertical

        # Position the 'Recording' indicator
        d.text((x_centre, (y_south - y_centre / 2)), self.parent.txtvar_recording.get(),
               anchor="ms", font=self._imgfnt,
               fill=self.parent.col_red)

        if self._warning.is_set():
            d.text((x_centre, y_centre / 2), self.txtvar_warning.get(),
                   anchor="ms", font=self._imgfnt,
                   fill=self.parent.col_orange)

        if self._magnification.is_set():

            col = self.parent.col_yellow_rgb

            # Slice directly from the framebuffer and make a (then zoomed) image
            magn_img = Image.fromarray(self._frame_buffer[self.magn_A[0]:self.magn_B[0],
                                       self.magn_A[1]:self.magn_B[1]]).convert('RGB')
            magn_img = magn_img.resize(
                (int(magn_img.width * self.magn_zoom.get()), int(magn_img.height * self.magn_zoom.get())))

            # Position of the top left corner of magnification window in the whole videofeed
            magn_pos = w - magn_img.width - 10, h - magn_img.height - 10  # small margin of 10 px

            # Add frame around the magnified area
            d.rectangle([(x_centre - self.magn_size[1] // 2, y_centre - self.magn_size[0] // 2),
                         (x_centre + self.magn_size[1] // 2, y_centre + self.magn_size[0] // 2)],
                        outline=col, width=1)

            img_pillow.paste(magn_img, magn_pos)

            # Add frame around the magnification
            d.rectangle([magn_pos, (w - 10, h - 10)], outline=col, width=1)

            # Add a small + in the centre
            c = magn_pos[0] + magn_img.width // 2, magn_pos[1] + magn_img.height // 2
            d.line((c[0] - 5, c[1], c[0] + 5, c[1]), fill=col, width=1)  # Horizontal
            d.line((c[0], c[1] - 5, c[0], c[1] + 5), fill=col, width=1)  # Vertical

        if self._show_focus.is_set():
            lapl = ndimage.gaussian_laplace(np.array(img_pillow)[:, :, 0].astype(np.uint8), sigma=1)
            blur = ndimage.gaussian_filter(lapl, 5)

            # threshold
            lim = 90
            blur[blur < lim] = 0
            blur[blur >= lim] = 255

            ones = np.ones_like(blur)
            r = Image.fromarray(ones * 98, mode='L')
            g = Image.fromarray(ones * 203, mode='L')
            b = Image.fromarray(ones * 90, mode='L')
            a = Image.fromarray(blur * 127, mode='L')

            overlay = Image.merge('RGBA', (r, g, b, a))
            img_pillow = Image.alpha_composite(img_pillow, overlay)

        self._refresh_videofeed(img_pillow)


class GUI:
    CONTROLS_MIN_WIDTH = 600
    if 'Windows' in platform.system():
        CONTROLS_MIN_HEIGHT = 320
    else:
        CONTROLS_MIN_HEIGHT = 280

    def __init__(self, mgr):

        # Detect monitors and pick the default one
        self.selected_monitor = None
        self._monitors = screeninfo.get_monitors()
        self.set_monitor()

        # Set up root window
        self.root = tk.Tk()

        resources_path = [p for p in Path().cwd().glob('**/*') if p.is_dir() and p.name == 'icons'][0]

        ico = ImageTk.PhotoImage(Image.open(resources_path / "mokap.png"))
        self.root.wm_iconphoto(True, ico)

        # self.root.wait_visibility(self.root)
        self.root.title("Controls")
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.bind('<KeyPress>', self._handle_keypress)

        self.icon_capture = tk.PhotoImage(file=resources_path / 'capture.png')
        self.icon_capture_bw = tk.PhotoImage(file=resources_path / 'capture_bw.png')
        self.icon_snapshot = tk.PhotoImage(file=resources_path / 'snapshot.png')
        self.icon_snapshot_bw = tk.PhotoImage(file=resources_path / 'snapshot_bw.png')
        self.icon_rec_on = tk.PhotoImage(file=resources_path / 'rec.png')
        self.icon_rec_bw = tk.PhotoImage(file=resources_path / 'rec_bw.png')

        # Colours
        self.col_white = "#ffffff"
        self.col_black = "#000000"
        self.col_lightgray = "#e3e3e3"
        self.col_midgray = "#c0c0c0"
        self.col_darkgray = "#515151"
        self.col_red = "#FF3C3C"
        self.col_orange = "#FF9B32"
        self.col_yellow = "#FFEB1E"
        self.col_yelgreen = "#A5EB14"
        self.col_green = "#00E655"
        self.col_blue = "#5ac3f5"
        self.col_purple = "#c887ff"

        self.col_white_rgb = utils.hex_to_rgb(self.col_white)
        self.col_black_rgb = utils.hex_to_rgb(self.col_black)
        self.col_lightgray_rgb = utils.hex_to_rgb(self.col_lightgray)
        self.col_midgray_rgb = utils.hex_to_rgb(self.col_midgray)
        self.col_darkgray_rgb = utils.hex_to_rgb(self.col_darkgray)
        self.col_red_rgb = utils.hex_to_rgb(self.col_red)
        self.col_orange_rgb = utils.hex_to_rgb(self.col_orange)
        self.col_yellow_rgb = utils.hex_to_rgb(self.col_yellow)
        self.col_yelgreen_rgb = utils.hex_to_rgb(self.col_green)
        self.col_green_rgb = utils.hex_to_rgb(self.col_green)
        self.col_blue_rgb = utils.hex_to_rgb(self.col_blue)
        self.col_purple_rgb = utils.hex_to_rgb(self.col_purple)

        # Set up fonts
        self.font_bold = font.Font(weight='bold', size=10)
        self.font_regular = font.Font(size=9)
        self.font_mini = font.Font(size=7)

        # Init default things
        self.mgr = mgr
        self.editing_disabled = True

        self._capture_clock = datetime.now()
        self._capture_fps = np.zeros(self.mgr.nb_cameras, dtype=np.uint32)
        self._now_indices = np.zeros(self.mgr.nb_cameras, dtype=np.uint32)
        self.start_indices = np.zeros(self.mgr.nb_cameras, dtype=np.uint32)
        self._saved_frames = np.zeros(self.mgr.nb_cameras, dtype=np.uint32)

        self._counter = 0

        self.txtvar_recording = tk.StringVar()
        self.txtvar_userentry = tk.StringVar()
        self.txtvar_applied_name = tk.StringVar()
        self.txtvar_frames_saved = tk.StringVar()

        self.txtvar_recording.set('')
        self.txtvar_userentry.set('')
        self.txtvar_applied_name.set('')
        self.txtvar_frames_saved.set('')

        # Compute optimal video windows sizes
        self._frame_sizes_bytes = np.prod(self.source_dims, axis=0)

        self._reference = None
        self._current_buffers = None

        # Create video windows
        self.video_windows = []
        self.windows_threads = []
        for i in range(self.mgr.nb_cameras):
            vw = VideoWindowMain(parent=self)
            self.video_windows.append(vw)

            t = Thread(target=vw.update, args=(), daemon=False)
            t.start()
            self.windows_threads.append(t)

        # Create list to store calibration windows
        self._is_calibrating = Event()
        self._is_calibrating.clear()
        self.calib_windows = []
        self.calib_windows_threads = []

        # x = self.selected_monitor.x + self.selected_monitor.width // 2 - self.CONTROLS_MIN_HEIGHT // 2
        # y = self.selected_monitor.y + self.selected_monitor.height // 2 - self.CONTROLS_MIN_WIDTH // 2
        #
        self.root.geometry(f"{self.CONTROLS_MIN_WIDTH}x{self.CONTROLS_MIN_HEIGHT}")

        ##

        toolbar = tk.Frame(self.root, height=40)
        # statusbar = tk.Frame(self.root, background="#e3e3e3", height=20)
        maincontent = tk.PanedWindow(self.root)

        toolbar.pack(side="top", fill="x")
        # statusbar.pack(side="bottom", fill="x")
        maincontent.pack(padx=3, pady=3, side="top", fill="both", expand=True)

        # Creating Menubar

        # Mode switch
        self.mode_var = tk.StringVar()
        modes_choices = {'Recording', 'Calibration'}
        self.mode_var.set('Recording')
        mode_label = tk.Label(toolbar, text='Mode: ', anchor=tk.W)
        mode_label.pack(side="left", fill="y", expand=False)
        mode_button = tk.OptionMenu(toolbar, self.mode_var, *modes_choices)
        mode_button.config(anchor=tk.CENTER)
        mode_button.pack(ipady=2, side="left", fill="y", expand=False)
        self.mode_var.trace('w', self._toggle_calibrate)

        # TOOLBAR
        if 'Darwin' in platform.system():
            self.button_exit = tk.Button(toolbar, text="Exit (Esc)", anchor=tk.CENTER, font=self.font_bold,
                                         fg=self.col_red,
                                         command=self.quit)
        else:
            self.button_exit = tk.Button(toolbar, text="Exit (Esc)", anchor=tk.CENTER, font=self.font_bold,
                                         bg=self.col_red, fg=self.col_white,
                                         command=self.quit)
        self.button_exit.pack(pady=2, side="right", fill="y", expand=False)

        left_pane = tk.LabelFrame(maincontent, text="Acquisition")
        right_pane = tk.LabelFrame(maincontent, text="Display")

        maincontent.add(left_pane)
        maincontent.add(right_pane)
        maincontent.paneconfig(left_pane, width=400)
        maincontent.paneconfig(right_pane, width=200)

        # LEFT HALF

        name_frame = tk.Frame(left_pane)
        name_frame.pack(side="top", fill="x", expand=True)

        editable_name_frame = tk.Frame(name_frame)
        editable_name_frame.pack(side="top", fill="x", expand=True)

        pathname_label = tk.Label(editable_name_frame, text='Name: ', anchor=tk.W)
        pathname_label.pack(side="left", fill="y", expand=False)

        self.pathname_textbox = tk.Entry(editable_name_frame, bg=self.col_white, fg=self.col_black,
                                         textvariable=self.txtvar_userentry,
                                         font=self.font_regular, state='disabled')
        self.pathname_textbox.pack(side="left", fill="both", expand=True)

        self.pathname_button = tk.Button(editable_name_frame,
                                         text="Edit", font=self.font_regular, command=self._toggle_text_editing)
        self.pathname_button.pack(side="right", fill="both", expand=False)

        info_name_frame = tk.Frame(name_frame)
        info_name_frame.pack(side="top", fill="x", expand=True)

        save_dir_label = tk.Label(info_name_frame, text='Saves to:', anchor=tk.W)
        save_dir_label.pack(side="top", fill="both", expand=False)

        save_dir_current = tk.Label(info_name_frame, textvariable=self.txtvar_applied_name, anchor=tk.W)
        save_dir_current.pack(side="left", fill="y", expand=True)

        gothere_button = tk.Button(info_name_frame, text="Go", font=self.font_regular, command=self.open_session_folder)
        gothere_button.pack(side="right", fill="y", expand=False)

        #
        f_buttons = tk.Frame(left_pane)
        f_buttons.pack(padx=5, side="top", fill="both", expand=True)

        self.button_acquisition = tk.Button(f_buttons,
                                            image=self.icon_capture_bw,
                                            compound='left', text="Acquisition off", anchor='center',
                                            width=150,
                                            font=self.font_regular,
                                            command=self._toggle_acquisition,
                                            state='normal')
        self.button_acquisition.grid(padx=2, pady=2, row=0, column=0, sticky="news")

        self.button_snapshot = tk.Button(f_buttons,
                                         image=self.icon_snapshot,
                                         compound='left', text="Snapshot", anchor='center',
                                         width=150,
                                         font=self.font_regular,
                                         command=self._take_snapshot,
                                         state='disabled')
        self.button_snapshot.grid(padx=2, pady=2, row=0, column=1, sticky="news")

        self.button_recpause = tk.Button(f_buttons,
                                         image=self.icon_rec_bw,
                                         compound='left', text="Not recording (Space to toggle)", anchor='center',
                                         font=self.font_bold,
                                         command=self._toggle_recording,
                                         state='disabled')
        self.button_recpause.grid(padx=2, pady=2, row=1, column=0, columnspan=2, sticky="news")

        frames_saved_label = tk.Label(left_pane, textvariable=self.txtvar_frames_saved, anchor=tk.E)
        frames_saved_label.pack(side="bottom", fill="x", expand=True)

        # RIGHT HALF

        windows_visibility_frame = tk.Frame(right_pane)
        windows_visibility_frame.pack(side="top", fill="x", expand=True)

        visibility_label = tk.Label(windows_visibility_frame, text='Show previews:', anchor=tk.W)
        visibility_label.pack(side="top", fill="x", expand=False)

        windows_list_frame = tk.Frame(windows_visibility_frame)
        windows_list_frame.pack(side="top", fill="both", expand=True)

        self.vis_checkboxes = []
        for window in self.video_windows:
            vis_var = tk.IntVar()
            vis_checkbox = tk.Checkbutton(windows_list_frame, text=f" {window.name.title()} camera", anchor=tk.W,
                                          font=self.font_bold,
                                          fg=window.colour_2,
                                          bg=window.colour,
                                          selectcolor=window.colour,
                                          activebackground=window.colour,
                                          activeforeground=window.colour,
                                          variable=vis_var,
                                          command=window.toggle_visibility,
                                          state='normal')
            vis_var.set(int(window.visible.is_set()))
            vis_checkbox.pack(side="top", fill="x", expand=True)
            self.vis_checkboxes.append(vis_var)

        monitors_frame = tk.Frame(right_pane)
        monitors_frame.pack(side="top", fill="both", expand=True)

        monitors_label = tk.Label(monitors_frame, text='Active monitor:', anchor=tk.W)
        monitors_label.pack(side="top", fill="x", expand=False)

        m_canvas_y_size = max([m.y + m.height for m in self._monitors]) // 40 + 2 * 10

        self.monitors_buttons = tk.Canvas(monitors_frame, height=m_canvas_y_size)
        self.update_monitors_buttons()

        for i, m in enumerate(self._monitors):
            self.monitors_buttons.tag_bind(f'screen_{i}', '<Button-1>', lambda _, val=i: self.screen_update(val))
        self.monitors_buttons.pack(side="top", fill="x", expand=False)

        self.autotile_button = tk.Button(monitors_frame,
                                         text="Auto-tile windows", font=self.font_regular,
                                         command=self.autotile_windows)
        self.autotile_button.pack(side="top", fill="both", expand=False)

        ##

        self.update()  # Called once to init

        self.root.attributes("-topmost", True)
        self.root.mainloop()

    @property
    def current_buffers(self):
        return self._current_buffers

    @property
    def capture_fps(self):
        return self._capture_fps

    @property
    def count(self):
        return self._counter

    @property
    def source_dims(self):
        return np.array([(cam.height, cam.width) for cam in self.mgr.cameras], dtype=np.uint32).T

    @property
    def screen_dims(self):
        monitor = self.selected_monitor
        return np.array([monitor.height, monitor.width, monitor.x, monitor.y], dtype=np.uint32)

    def set_monitor(self, idx=None):
        if len(self._monitors) > 1 and idx is None:
            self.selected_monitor = next(m for m in self._monitors if m.is_primary)
        elif len(self._monitors) > 1 and idx is not None:
            self.selected_monitor = self._monitors[idx]
        else:
            self.selected_monitor = self._monitors[0]

    def open_session_folder(self):
        path = Path(self.txtvar_applied_name.get()).resolve()

        if 'Linux' in platform.system():
            subprocess.Popen(['xdg-open', path])
        elif 'Windows' in platform.system():
            os.startfile(path)
        elif 'Darwin' in platform.system():
            subprocess.Popen(['open', path])
        else:
            pass

    def nothing(self):
        print('Nothing')
        pass

    def update_monitors_buttons(self):
        self.monitors_buttons.delete("all")

        for i, m in enumerate(self._monitors):
            w, h, x, y = m.width // 40, m.height // 40, m.x // 40, m.y // 40
            if m.name == self.selected_monitor.name:
                col = self.col_darkgray
            else:
                col = self.col_midgray

            rect_x = x + 10
            rect_y = y + 10
            self.monitors_buttons.create_rectangle(rect_x, rect_y, rect_x + w - 2, rect_y + h - 2,
                                                   fill=col, outline='',
                                                   tag=f'screen_{i}')

    def visible_windows(self, include_main=False):
        windows = [w for w in self.video_windows if w.visible.is_set()]
        windows += [w for w in self.calib_windows if w.visible.is_set()]
        if include_main:
            windows += [self]
        return windows

    def screen_update(self, val):

        old_monitor_x, old_monitor_y = self.selected_monitor.x, self.selected_monitor.y

        old_root_x = self.root.winfo_rootx()
        old_root_y = self.root.winfo_rooty()
        rel_mouse_x = self.root.winfo_pointerx() - old_root_x
        rel_mouse_y = self.root.winfo_pointery() - old_root_y

        self.set_monitor(val)
        self.update_monitors_buttons()

        for window_to_move in self.visible_windows(include_main=True):
            w, h, x, y = whxy(window_to_move)

            d_x = x - old_monitor_x
            d_y = y - old_monitor_y

            new_x = self.selected_monitor.x + d_x
            new_y = self.selected_monitor.y + d_y

            if new_x <= self.selected_monitor.x:
                new_x = self.selected_monitor.x

            if new_y <= self.selected_monitor.y:
                new_y = self.selected_monitor.y

            if new_x + w >= self.selected_monitor.width + self.selected_monitor.x:
                new_x = self.selected_monitor.width + self.selected_monitor.x - w

            if new_y + h >= self.selected_monitor.height + self.selected_monitor.y:
                new_y = self.selected_monitor.height + self.selected_monitor.y - h

            if window_to_move is self:
                self.root.geometry(f'{w}x{h}+{new_x}+{new_y}')
            else:
                window_to_move.window.geometry(f'{w}x{h}+{new_x}+{new_y}')

        movement_x = self.root.winfo_rootx() - old_root_x
        movement_y = self.root.winfo_rooty() - old_root_y

        self.root.event_generate('<Motion>', warp=True, x=movement_x + rel_mouse_x, y=movement_y + rel_mouse_y)

    def autotile_windows(self):
        for w in self.visible_windows(include_main=False):
            w.auto_size()
            w.auto_move()

    def _handle_keypress(self, event):
        if self.editing_disabled:
            # if event.keycode == 9:  # Esc
            if event.keycode == 27:  # Esc
                self.quit()
            # elif event.keycode == 65:  # Space
            elif event.keycode == 32:  # Space
                self._toggle_recording()
            else:
                pass
        else:
            if event.keycode == 13:  # Enter
                self._toggle_text_editing(True)

    def _take_snapshot(self):

        dims = self.source_dims.T
        ext = self.mgr.saving_ext
        now = datetime.now().strftime('%y%m%d-%H%M')

        if self.mgr.acquiring:
            arrays = [np.frombuffer(c, dtype=np.uint8) for c in self._current_buffers]

            for a, arr in enumerate(arrays):
                img = Image.fromarray(arr.reshape(dims[a]))
                img.save(self.mgr.full_path.resolve() / f"snapshot_{now}_{self.mgr.cameras[a].name}.{ext}")

    def _toggle_text_editing(self, tf=None):
        if tf is None:
            tf = not self.editing_disabled

        if self.editing_disabled and tf is False:
            self.pathname_textbox.config(state='normal')
            self.pathname_button.config(text='Set')
            self.editing_disabled = False

        elif not self.editing_disabled and tf is True:
            self.pathname_textbox.config(state='disabled')
            self.pathname_button.config(text='Edit')
            self.mgr.session_name = self.txtvar_userentry.get()
            self.editing_disabled = True
            self.txtvar_applied_name.set(f'{self.mgr.full_path.resolve()}')
        else:
            pass

    def _toggle_recording(self, tf=None):
        if self.mgr.acquiring:
            if tf is None:
                tf = not self.mgr.recording

            if self.mgr.recording and tf is False:
                self.mgr.pause()
                self.txtvar_recording.set('')
                self.button_recpause.config(text="Not recording (Space to toggle)", image=self.icon_rec_bw)

            elif not self.mgr.recording and tf is True:
                self.mgr.record()
                self.txtvar_recording.set('[ Recording... ]')
                self.button_recpause.config(text="Recording... (Space to toggle)", image=self.icon_rec_on)

            else:
                pass

    def _toggle_calibrate(self, *events):
        mode = self.mode_var.get()

        if self._is_calibrating.is_set() and mode == 'Recording':
            self._is_calibrating.clear()

            if self.mgr.acquiring:
                self.button_snapshot.config(state="normal")
                self.button_recpause.config(state="normal")

            for window in self.calib_windows:
                self.video_windows[window.idx].toggle_visibility(True)

        elif not self._is_calibrating.is_set() and mode == 'Calibration':

            self._is_calibrating.set()

            self.button_recpause.config(state="disabled")

            for window in self.video_windows:
                w, h, x, y = whxy(window)
                window.toggle_visibility(False)

                c = VideoWindowCalib(parent=self, idx=window.idx)
                self.calib_windows.append(c)

                c.window.geometry(f'{w}x{h}+{x}+{y}')

                t = Thread(target=c.update, args=(), daemon=False)
                t.start()
                self.calib_windows_threads.append(t)
        else:
            pass

    def _toggle_acquisition(self):

        if self.mgr.acquiring:
            self._toggle_recording(False)
            self.mgr.off()

            self._capture_fps = np.zeros(self.mgr.nb_cameras, dtype=np.uintc)

            self.txtvar_userentry.set('')
            self.txtvar_applied_name.set('')

            self.button_acquisition.config(text="Acquisition off", image=self.icon_capture_bw)
            self.button_snapshot.config(state="disabled")
            self.button_recpause.config(state="disabled")

        else:
            self.mgr.on()

            self.txtvar_applied_name.set(f'{self.mgr.full_path.resolve()}')

            self._capture_clock = datetime.now()
            self.start_indices[:] = self.mgr.indices

            self.button_acquisition.config(text="Acquiring", image=self.icon_capture)
            self.button_snapshot.config(state="normal")
            if not self._is_calibrating:
                self.button_recpause.config(state="normal")

    def quit(self):
        for vw in self.video_windows:
            vw.visible.clear()

        if self.mgr.acquiring:
            self.mgr.off()

        self.mgr.disconnect()

        self.root.quit()
        self.root.destroy()
        sys.exit()

    def update(self):

        if self.mgr.acquiring:
            now = datetime.now()
            capture_dt = (now - self._capture_clock).total_seconds()

            self._now_indices[:] = self.mgr.indices

            with warnings.catch_warnings():
                try:
                    self._capture_fps[:] = (self._now_indices - self.start_indices) / capture_dt
                except Warning:
                    self._capture_fps.fill(0)

            self._current_buffers = self.mgr.get_current_framebuffer()

            self._saved_frames = self.mgr.saved
            self.txtvar_frames_saved.set(
                f'Saved {sum(self._saved_frames)} frames total ({utils.pretty_size(sum(self._frame_sizes_bytes * self._saved_frames))})')

        self._counter += 1
        self.root.after(1, self.update)
