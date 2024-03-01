import sys
import tkinter as tk
import tkinter.font as font
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
from datetime import datetime
from threading import Thread, Event
import screeninfo
import colorsys
from scipy import ndimage
import platform
from pathlib import Path
import os
import subprocess
from mokap import utils
from functools import partial

def hex_to_hls(hex_str: str):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 3:
        hex_str = ''.join([c + c for c in hex_str])
    return colorsys.rgb_to_hls(*tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4)))


def hls_to_hex(h, l, s):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    new_hex = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
    return new_hex


def whxy(what):
    if isinstance(what, tk.Toplevel):
        dims, x, y = what.geometry().split('+')
        w, h = dims.split('x')
    elif isinstance(what, GUI):
        dims, x, y = what.root.geometry().split('+')
        w, h = dims.split('x')
    elif isinstance(what, VideoWindow):
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
    max_window_h = screenh // 2 - VideoWindow.INFO_PANEL_MIN_H

    # Scale it up or down, so it fits half the screen height
    if sourceh / max_window_h >= sourcew / max_window_w:
        scale = max_window_h / sourceh
    else:
        scale = max_window_w / sourcew

    computed_dims = np.floor(np.array([sourceh, sourcew]) * scale).astype(int)
    computed_dims[0] += VideoWindow.INFO_PANEL_MIN_H

    return computed_dims


class VideoWindow:
    videowindows_ids = []

    INFO_PANEL_MIN_H = 180  # in pixels

    def __init__(self, parent):

        self.window = tk.Toplevel()
        self.parent = parent

        self.idx = len(VideoWindow.videowindows_ids)
        VideoWindow.videowindows_ids.append(self.idx)

        self._source_shape = (self.parent.mgr.cameras[self.idx].height, self.parent.mgr.cameras[self.idx].width)
        self._cam_name = self.parent.mgr.cameras[self.idx].name
        self._bg_color = self.parent.mgr.cameras[self.idx].color

        hls = hex_to_hls(self._bg_color)

        if hls[1] < 150:
            self._fg_color = '#ffffff'
        else:
            self._fg_color = '#000000'

        h, w = parent.max_videowindows_dims
        self.window.geometry(f"{w}x{h}")
        self.window.wm_aspect(w, h, w, h)
        self.window.protocol("WM_DELETE_WINDOW", self.toggle_visibility)

        # Init state
        self._counter = 0
        self._clock = datetime.now()
        self._fps = 0
        self._brightness_var = 0

        self.visible = Event()
        self.visible.set()

        self._display_focus = Event()
        self._display_focus.clear()

        self._display_mag = Event()
        self._display_mag.set()

        self._kernel = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]], dtype=np.uint8)

        if 'Linux' in platform.system():
            try:
                self._imgfnt = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
            except OSError:
                self._imgfnt = ImageFont.load_default()
        elif 'Windows' in platform.system():
            try:
                self._imgfnt = ImageFont.truetype("arial-bold.ttf", 30)
            except OSError:
                self._imgfnt = ImageFont.load_default()
        elif 'Darwin' in platform.system():
            try:
                self._imgfnt = ImageFont.truetype("KeyboardBold.ttf", 30)
            except OSError:
                self._imgfnt = ImageFont.load_default()
        else:
            self._imgfnt = ImageFont.load_default()

        # Initialise text vars
        self.txtvar_camera_name = tk.StringVar()

        self.txtvar_resolution = tk.StringVar()
        self.txtvar_exposure = tk.StringVar()
        self.txtvar_capture_fps = tk.StringVar()
        self.txtvar_brightness = tk.StringVar()
        self.txtvar_display_fps = tk.StringVar()

        ## Main video frame
        imagetk = ImageTk.PhotoImage(image=Image.fromarray(np.zeros(self.video_dims, dtype='<u1')))

        self.video_canvas = tk.Label(self.window, bg="black", compound='center')
        self.video_canvas.pack(fill='both', expand=True)
        self.video_canvas.imgtk = imagetk
        self.video_canvas.configure(image=imagetk)

        ## Camera name bar
        self.txtvar_camera_name.set(f'{self._cam_name.title()} camera')

        name_bar = tk.Label(self.window, textvariable=self.txtvar_camera_name,
                            anchor='n', justify='center',
                            fg=self.color_2, bg=self.color, font=parent.bold)
        name_bar.pack(fill='both')

        # Also set the window title while we're at it
        self.window.title(self.txtvar_camera_name.get())

        ## Bottom panel

        # Set initial values
        h, w = self.parent.mgr.cameras[self.idx].height, self.parent.mgr.cameras[self.idx].width
        self.txtvar_resolution.set(f"{h}×{w} px")
        self.txtvar_exposure.set(f"{self.parent.mgr.cameras[self.idx].exposure} µs")

        ## Information block
        f_information = tk.LabelFrame(self.window, text="Information",
                                      height=self.INFO_PANEL_MIN_H)
        f_information.pack(padx=5, pady=5, side='left', fill='both', expand=True)

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
                         font=parent.regular)
            l.pack(side='left', fill='y')

            v = tk.Label(f, textvariable=var,
                         anchor='w', justify='left',
                         font=parent.regular)
            v.pack(side='left', fill='y')

        ## View controls block
        view_info_frame = tk.LabelFrame(self.window, text="View",
                                        height=self.INFO_PANEL_MIN_H, width=100)
        view_info_frame.pack(padx=5, pady=5, side='left', fill='y', expand=True)

        f_windowsnap = tk.Frame(view_info_frame)
        f_windowsnap.pack(side='top', fill='y')

        l_windowsnap = tk.Label(f_windowsnap, text=f"Snap window to:", bg='red',
                     anchor='e', justify='right',
                     font=parent.regular)
        l_windowsnap.pack(side='left', fill='y')

        f_buttons_windowsnap = tk.Frame(f_windowsnap, bg='green',)
        f_buttons_windowsnap.pack(padx=2, pady=2, side='left', fill='both', expand=True)

        positions = np.array([['nw', 'n', 'ne'],
                              ['w', 'c', 'e'],
                              ['sw', 's', 'se']])
        self._pixel = tk.PhotoImage(width=1, height=1)
        for r in range(3):
            for c in range(3):
                pos = positions[r, c]
                b = tk.Button(f_buttons_windowsnap,
                              image=self._pixel, compound="center", padx=0, pady=0,
                              width=6, height=6,
                              command=partial(self.move_to, pos))
                b.grid(row=r, column=c)

        f_buttons_controls = tk.Frame(view_info_frame)
        f_buttons_controls.pack(padx=2, pady=2, side='top', fill='both', expand=True)

        self.show_focus_button = tk.Button(f_buttons_controls, text="Focus",
                                           width=10,
                                           highlightthickness=0, highlightbackground="#62CB5A",
                                           font=self.parent.regular,
                                           command=self._toggle_focus_display)
        self.show_focus_button.pack(side='left', fill='x')

        self.show_mag_button = tk.Button(f_buttons_controls, text="Magnification",
                                         width=10,
                                         highlightthickness=2, highlightbackground="#FFED30",
                                         font=self.parent.regular,
                                         command=self._toggle_mag_display)
        self.show_mag_button.pack(side='left', fill='x')

        ## Camera controls block
        f_camera_controls = tk.LabelFrame(self.window, text="Control",
                                          height=self.INFO_PANEL_MIN_H)
        f_camera_controls.pack(padx=5, pady=5, side='left', fill='both', expand=True)

        self.camera_controls_sliders = {}
        for label, val, func, func_all, slider_params in zip(['Framerate (fps)', 'Exposure (µs)', 'Gain', 'Gamma'],
                                              [self.parent.mgr.cameras[self.idx].framerate,
                                               self.parent.mgr.cameras[self.idx].exposure,
                                               self.parent.mgr.cameras[self.idx].gain,
                                               self.parent.mgr.cameras[self.idx].gamma],
                                              [self.update_framerate,
                                               self.update_exposure,
                                               self.update_gain,
                                               self.update_gamma],
                                              [self._update_fps_all_cams,
                                               self._update_exp_all_cams,
                                               self._update_gain_all_cams,
                                               self._update_gamma_all_cams],
                                                             [(1, 220, 1, 1),
                                                              (4300, 25000, 5, 1),
                                                              (0.0, 24.0, 0.5, 3),
                                                              (0.0, 4.0, 0.05, 3),
                                                              ]):
            f = tk.Frame(f_camera_controls)
            f.pack(side='top', fill='both', expand=True)

            l = tk.Label(f, text=f'{label} :',
                         anchor='se', justify='right', width=18,
                         font=parent.regular)
            l.pack(side='left', fill='both', expand=True)

            s = tk.Scale(f,
                         from_=slider_params[0], to=slider_params[1], resolution=slider_params[2], digits=slider_params[3],
                         orient='horizontal', width=8, sliderlength=16)
            s.set(val)
            s.bind("<ButtonRelease-1>", func)
            s.pack(side='left', fill='both', expand=True)

            key = label.split('(')[0].strip().lower()   # Get the simplified label (= wihtout spaces and units)
            self.camera_controls_sliders[key] = s

            b = tk.Button(f, text="Apply all",
                          font=self.parent.regular,
                          command=func_all)
            b.pack(padx=2, side='right', fill='y')

    def move_to(self, pos):
        w, h, x, y = whxy(self)
        w_scr, h_scr, x_scr, y_scr = whxy(self.parent.selected_monitor)

        os_taskbar_size = 30
        if pos == 'nw':
            self.window.geometry(f"{w}x{h}+{x_scr}+{y_scr}")
        elif pos == 'n':
            self.window.geometry(f"{w}x{h}+{x_scr + w_scr//2 - w//2}+{y_scr}")
        elif pos == 'ne':
            self.window.geometry(f"{w}x{h}+{x_scr + w_scr - w}+{y_scr}")

        elif pos == 'w':
            self.window.geometry(f"{w}x{h}+{x_scr}+{y_scr + h_scr//2 - h//2}")
        elif pos == 'c':
            self.window.geometry(f"{w}x{h}+{x_scr + w_scr//2 - w//2}+{y_scr + h_scr//2 - h//2}")
        elif pos == 'e':
            self.window.geometry(f"{w}x{h}+{x_scr + w_scr - w}+{y_scr + h_scr//2 - h//2}")

        elif pos == 'sw':
            self.window.geometry(f"{w}x{h}+{x_scr}+{y_scr + h_scr - h - os_taskbar_size}")
        elif pos == 's':
            self.window.geometry(f"{w}x{h}+{x_scr + w_scr//2 - w//2}+{y_scr + h_scr - h - os_taskbar_size}")
        elif pos == 'se':
            self.window.geometry(f"{w}x{h}+{x_scr + w_scr - w}+{y_scr + h_scr - h - os_taskbar_size}")

    def update_framerate(self, event=None):
        slider = self.camera_controls_sliders['framerate']
        new_val = slider.get()
        self.parent.mgr.cameras[self.idx].framerate = new_val

        # The actual maximum framerate depends on the exposure, so it might not be what the user requested
        # Thus we need to update the slider value to the actual framerate
        slider.set(self.parent.mgr.cameras[self.idx].framerate)

        # Refresh fps counters for the UI
        self.parent._capture_clock = datetime.now()
        self.parent.start_indices[:] = self.parent.mgr.indices

    def update_exposure(self, event=None):
        slider = self.camera_controls_sliders['exposure']
        new_val = slider.get()
        self.parent.mgr.cameras[self.idx].exposure = new_val

        # We also need to update the framerate slider to current resulting fps after exposure change
        slider_framerate = self.camera_controls_sliders['framerate']
        slider_framerate.set(self.parent.mgr.cameras[self.idx].framerate)

        self.txtvar_exposure.set(f"{self.parent.mgr.cameras[self.idx].exposure} µs")

        # And callback to the update framerate function because the new exposure time might cap the framerate out
        self.update_framerate(event)

    def update_gain(self, event=None):
        slider = self.camera_controls_sliders['gain']
        new_val = slider.get()
        self.parent.mgr.cameras[self.idx].gain = new_val

    def update_gamma(self, event=None):
        slider = self.camera_controls_sliders['gamma']
        new_val = slider.get()
        self.parent.mgr.cameras[self.idx].gamma = new_val

    def _update_fps_all_cams(self):
        for window in self.parent.video_windows:
            if window is not self:
                slider = self.camera_controls_sliders['framerate']
                new_val = slider.get()
                window.camera_controls_sliders['framerate'].set(new_val)
                window.update_framerate()

    def _update_exp_all_cams(self):
        for window in self.parent.video_windows:
            if window is not self:
                slider = self.camera_controls_sliders['exposure']
                new_val = slider.get()
                window.camera_controls_sliders['exposure'].set(new_val)
                window.update_exposure()

    def _update_gain_all_cams(self):
        for window in self.parent.video_windows:
            if window is not self:
                slider = self.camera_controls_sliders['gain']
                new_val = slider.get()
                window.camera_controls_sliders['gain'].set(new_val)
                window.update_gain()

    def _update_gamma_all_cams(self):
        for window in self.parent.video_windows:
            if window is not self:
                slider = self.camera_controls_sliders['gamma']
                new_val = slider.get()
                window.camera_controls_sliders['gamma'].set(new_val)
                window.update_gamma()

    def _toggle_focus_display(self):
        if self._display_focus.is_set():
            self._display_focus.clear()
            self.show_focus_button.configure(highlightthickness=0)
        else:
            self._display_focus.set()
            self.show_focus_button.configure(highlightthickness=2)

    def _toggle_mag_display(self):
        if self._display_mag.is_set():
            self._display_mag.clear()
            self.show_mag_button.configure(highlightthickness=0)
        else:
            self._display_mag.set()
            self.show_mag_button.configure(highlightthickness=2)

    def _update_txtvars(self):

        if self.parent.mgr.acquiring:
            cap_fps = self.parent.capture_fps[self.idx]

            if 0 < cap_fps < 1000:  # only makes sense to display real values
                self.txtvar_capture_fps.set(f"{cap_fps:.2f} fps")
            else:
                self.txtvar_capture_fps.set("-")

            self.txtvar_brightness.set(f"{self._brightness_var:.2f}%")
        else:
            self.txtvar_capture_fps.set("Off")
            self.txtvar_brightness.set("-")

        self.txtvar_display_fps.set(f"{self._fps:.2f} fps")

    def _update_video(self):
        frame = np.random.randint(0, 255, self.video_dims, dtype='<u1')

        if self.parent.mgr.acquiring:
            if self.parent.current_buffers is not None:
                buf = self.parent.current_buffers[self.idx]
                if buf is not None:
                    frame = np.frombuffer(buf, dtype=np.uint8).reshape(self._source_shape)
        else:
            frame = np.random.randint(0, 255, self.video_dims, dtype='<u1')

        self._brightness_var = np.round(frame.mean() / 255 * 100, decimals=2)

        if self._display_focus.is_set():
            overlay = ndimage.gaussian_laplace(frame, sigma=1).astype(np.int32)
            overlay = ndimage.gaussian_filter(overlay, 5).astype(np.int32)
            lim = 90
            overlay[overlay < lim] = 0
            overlay[overlay >= lim] = 100

            fo = np.clip(frame.astype(np.int32) + overlay, 0, 255).astype(np.uint8)
            frame = np.stack([frame, fo, frame]).T.swapaxes(0, 1)

        y_displ, x_displ = self.video_dims
        x2_displ, y2_displ = x_displ // 2, y_displ // 2

        img_pillow = Image.fromarray(frame).convert('RGBA')
        img_pillow = img_pillow.resize((x_displ, y_displ))

        d = ImageDraw.Draw(img_pillow)
        d.line((0, y2_displ, x_displ, y2_displ), fill=(255, 255, 255, 200), width=1)
        d.line((x2_displ, 0, x2_displ, y_displ), fill=(255, 255, 255, 200), width=1)

        d.text((x2_displ - 100, y_displ - 40), self.parent.recording_var.get(), font=self._imgfnt, fill='red')

        if self._display_mag.is_set():
            y_source, x_source = self._source_shape
            x2_source, y2_source = x_source // 2, y_source // 2

            mini_size_ratio = 12
            mini_size = x2_source // mini_size_ratio, y2_source // mini_size_ratio

            mini_x0 = x2_source - mini_size[0]
            mini_y0 = y2_source - mini_size[1]
            mini_x1 = x2_source + mini_size[0]
            mini_y1 = y2_source + mini_size[1]
            mini = Image.fromarray(frame[mini_y0:mini_y1, mini_x0:mini_x1]).convert('RGBA')


            d.rectangle([(x2_displ - x2_displ // mini_size_ratio, y2_displ - y2_displ // mini_size_ratio),
                             (x2_displ + x2_displ // mini_size_ratio, y2_displ + y2_displ // mini_size_ratio)],
                             outline=(255, 255, 20, 200), width=1)

            mag = 4
            d.rectangle([(x_displ - (mini_size[0] * mag + 11), y_displ - (mini_size[1] * mag + 11)),
                         (x_displ - 10, y_displ - 10)],
                        outline=(255, 255, 20, 200), width=1)

            mini = mini.resize((mini_size[0] * mag, mini_size[1] * mag))
            img_pillow.paste(mini, (x_displ - (mini_size[0] * mag + 10), y_displ - (mini_size[1] * mag + 10)))

        imgtk = ImageTk.PhotoImage(image=img_pillow)
        try:
            self.video_canvas.configure(image=imgtk)
            self.imagetk = imgtk
        except Exception:
            # If new image is garbage collected too early, do nothing - this prevents the image from flashing
            pass

    def update(self):

        while True:
            # Update display fps counter
            now = datetime.now()
            dt = (now - self._clock).total_seconds()
            self._fps = self._counter / dt

            self._counter += 1

            self._update_video()
            self._update_txtvars()

            self.visible.wait()

    def toggle_visibility(self):
        if self.visible.is_set():
            self.visible.clear()
            self.parent._vis_checkboxes[self.idx].set(0)
            self.window.withdraw()
        else:
            self.visible.set()
            self.parent._vis_checkboxes[self.idx].set(1)
            self.window.deiconify()

    @property
    def name(self):
        return self._cam_name

    @property
    def color(self):
        return self._bg_color

    @property
    def color_2(self):
        return self._fg_color

    @property
    def count(self):
        return self._counter

    @property
    def display_fps(self):
        return self._fps

    @property
    def video_dims(self):
        dims = self.current_dims - (self.INFO_PANEL_MIN_H, 0)
        if any(dims <= 1):
            return np.array([*self._source_shape])
        else:
            return dims

    @property
    def current_dims(self):
        return np.array([self.window.winfo_height(), self.window.winfo_width()])

class GUI:
    # Values below are in pixels
    PADDING = 0
    CONTROLS_WIDTH = 550
    CONTROLS_HEIGHT = 300

    def __init__(self, mgr):

        # Detect monitors and pick the default one
        self.selected_monitor = None
        self._monitors = screeninfo.get_monitors()
        self.set_monitor()

        # Set up root window
        self.root = tk.Tk()
        self.root.wait_visibility(self.root)
        self.root.title("Controls")
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.bind('<KeyPress>', self._handle_keypress)

        self.icon_capture_on = tk.PhotoImage(file='./mokap/icons/capture_on.png')
        self.icon_capture_off = tk.PhotoImage(file='./mokap/icons/capture_off_bw.png')
        self.icon_rec_on = tk.PhotoImage(file='./mokap/icons/rec.png')
        self.icon_rec_off = tk.PhotoImage(file='./mokap/icons/rec_bw.png')

        # Set up fonts
        self.bold = font.Font(weight='bold', size=10)
        self.regular = font.Font(size=9)

        # Init default things
        self.mgr = mgr
        self.editing_disabled = True

        self._capture_clock = datetime.now()
        self._capture_fps = np.zeros(self.mgr.nb_cameras, dtype=np.uint32)
        self._now_indices = np.zeros(self.mgr.nb_cameras, dtype=np.uint32)
        self.start_indices = np.zeros(self.mgr.nb_cameras, dtype=np.uint32)
        self._saved_frames = np.zeros(self.mgr.nb_cameras, dtype=np.uint32)

        self._counter = 0

        self.recording_var = tk.StringVar()
        self.userentry_var = tk.StringVar()
        self.applied_name_var = tk.StringVar()
        self.frames_saved_var = tk.StringVar()

        self.recording_var.set('')
        self.userentry_var.set('')
        self.applied_name_var.set('')
        self.frames_saved_var.set('')

        # Compute optimal video windows sizes
        self._max_videowindows_dims = compute_windows_size(self.source_dims, self.screen_dims)
        self._frame_sizes_bytes = np.prod(self.source_dims, axis=0)

        self._reference = None
        self._current_buffers = None

        # Create video windows
        self.video_windows = []
        self.windows_threads = []
        for i in range(self.mgr.nb_cameras):
            vw = VideoWindow(parent=self)
            self.video_windows.append(vw)

            t = Thread(target=vw.update, args=(), daemon=False)
            t.start()
            self.windows_threads.append(t)

        x = self.selected_monitor.x + self.selected_monitor.width // 2 - self.CONTROLS_HEIGHT // 2
        y = self.selected_monitor.y + self.selected_monitor.height // 2 - self.CONTROLS_WIDTH // 2

        self.root.geometry(f"{self.CONTROLS_WIDTH}x{self.CONTROLS_HEIGHT}+{x}+{y}")

        # Create control window
        self._create_controls()

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
    def max_videowindows_dims(self):
        return self._max_videowindows_dims

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

    def open_save_folder(self):
        path = Path(self.applied_name_var.get()).resolve()
        if path != Path(os.getcwd()).resolve():
            try:
                os.startfile(path)
            except AttributeError:
                subprocess.Popen(['xdg-open', path])

    def nothing(self):
        print('Nothing')
        pass

    def _create_controls(self):

        toolbar = tk.Frame(self.root, background="#E8E8E8", height=40)
        # statusbar = tk.Frame(self.root, background="#e3e3e3", height=20)
        maincontent = tk.PanedWindow(self.root)

        toolbar.pack(side="top", fill="x")
        # statusbar.pack(side="bottom", fill="x")
        maincontent.pack(padx=3, pady=3, side="top", fill="both", expand=True)

        # TOOLBAR
        self.button_exit = tk.Button(toolbar, text="Exit (Esc)", anchor=tk.CENTER, font=self.bold,
                                     bg='#fd5754', fg='white',
                                     command=self.quit)
        self.button_exit.pack(side="left", fill="y", expand=False)

        left_pane = tk.LabelFrame(maincontent, text="Acquisition")
        right_pane = tk.LabelFrame(maincontent, text="Display")

        maincontent.add(left_pane)
        maincontent.add(right_pane)
        maincontent.paneconfig(left_pane, width=300)
        maincontent.paneconfig(right_pane, width=200)

        # LEFT HALF

        name_frame = tk.Frame(left_pane)
        name_frame.pack(side="top", fill="x", expand=True)

        editable_name_frame = tk.Frame(name_frame)
        editable_name_frame.pack(side="top", fill="x", expand=True)

        pathname_label = tk.Label(editable_name_frame, text='Name: ', anchor=tk.W)
        pathname_label.pack(side="left", fill="y", expand=False)

        self.pathname_textbox = tk.Entry(editable_name_frame, bg='white', fg='black', textvariable=self.userentry_var,
                                         font=self.regular, state='disabled')
        self.pathname_textbox.pack(side="left", fill="both", expand=True)

        self.pathname_button = tk.Button(editable_name_frame,
                                         text="Edit", font=self.regular, command=self.gui_toggle_set_name)
        self.pathname_button.pack(side="right", fill="both", expand=False)

        info_name_frame = tk.Frame(name_frame)
        info_name_frame.pack(side="top", fill="x", expand=True)

        save_dir_label = tk.Label(info_name_frame, text='Saves to:', anchor=tk.W)
        save_dir_label.pack(side="top", fill="both", expand=False)

        save_dir_current = tk.Label(info_name_frame, textvariable=self.applied_name_var, anchor=tk.W)
        save_dir_current.pack(side="left", fill="y", expand=True)

        # gothere_button = tk.Button(info_name_frame, text="Go", font=self.regular, command=self.open_save_folder)
        gothere_button = tk.Button(info_name_frame, text="Go", font=self.regular, command=self.nothing)
        gothere_button.pack(side="right", fill="y", expand=False)

        #

        self.button_acquisition = tk.Button(left_pane,
                                            image=self.icon_capture_off,
                                            compound='left', text="Acquisition off", anchor='center', font=self.regular,
                                            command=self.gui_toggle_acquisition,
                                            state='normal')
        self.button_acquisition.pack(padx=5, pady=5, side="top", fill="both", expand=True)

        self.button_recpause = tk.Button(left_pane,
                                         image=self.icon_rec_off,
                                         compound='left',
                                         text="Not recording\n\n(Space to toggle)", anchor='center', font=self.bold,
                                         command=self.gui_toggle_recording,
                                         state='disabled')
        self.button_recpause.pack(padx=5, pady=5, side="top", fill="both", expand=True)

        frames_saved_label = tk.Label(left_pane, textvariable=self.frames_saved_var, anchor=tk.E)
        frames_saved_label.pack(side="bottom", fill="x", expand=True)

        # RIGHT HALF

        windows_visibility_frame = tk.Frame(right_pane)
        windows_visibility_frame.pack(side="top", fill="x", expand=True)

        visibility_label = tk.Label(windows_visibility_frame, text='Show previews:', anchor=tk.W)
        visibility_label.pack(side="top", fill="x", expand=False)

        windows_list_frame = tk.Frame(windows_visibility_frame)
        windows_list_frame.pack(side="top", fill="both", expand=True)

        self._vis_checkboxes = []
        for window in self.video_windows:
            vis_var = tk.IntVar()
            vis_checkbox = tk.Checkbutton(windows_list_frame, text=f" {window.name.title()} camera", anchor=tk.W,
                                          font=self.bold,
                                          fg=window.color_2,
                                          bg=window.color,
                                          selectcolor=window.color,
                                          activebackground=window.color,
                                          activeforeground=window.color,
                                          variable=vis_var,
                                          command=window.toggle_visibility,
                                          state='normal')
            vis_var.set(int(window.visible.is_set()))
            vis_checkbox.pack(side="top", fill="x", expand=True)
            self._vis_checkboxes.append(vis_var)

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
                                         text="Auto-tile windows", font=self.regular, command=self.autotile_windows)
        self.autotile_button.pack(side="top", fill="both", expand=False)

    def update_monitors_buttons(self):
        self.monitors_buttons.delete("all")

        for i, m in enumerate(self._monitors):
            w, h, x, y = m.width // 40, m.height // 40, m.x // 40, m.y // 40
            if m.name == self.selected_monitor.name:
                col = '#515151'
            else:
                col = '#c0c0c0'

            rect_x = x + 10
            rect_y = y + 10
            self.monitors_buttons.create_rectangle(rect_x, rect_y, rect_x + w - 2, rect_y + h - 2,
                                                   fill=col, outline='',
                                                   tag=f'screen_{i}')

    def screen_update(self, val):

        old_monitor_x, old_monitor_y = self.selected_monitor.x, self.selected_monitor.y

        self.set_monitor(val)
        self.update_monitors_buttons()

        for window_to_move in [self, *self.video_windows]:
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

    def autotile_windows(self):
        new_x = self.selected_monitor.x
        new_y = self.selected_monitor.y

        for window_to_move in self.video_windows:
            w, h, x, y = whxy(window_to_move)
            window_to_move.window.geometry(f'{w}x{h}+{new_x}+{new_y}')
            new_x += w
            if new_x >= self.selected_monitor.x + self.selected_monitor.width:
                new_x = self.selected_monitor.x
                new_y += h

    def _handle_keypress(self, event):
        match event.keycode:
            case 9:  # Esc
                self.quit()
            case 65:  # Space
                self.gui_toggle_recording()
            case _:
                pass

    def gui_set_name(self):
        self.mgr.savepath = self.userentry_var.get()
        self.applied_name_var.set(f'{Path(self.mgr.savepath).resolve()}')

    def gui_toggle_set_name(self):
        if self.editing_disabled:
            self.pathname_textbox.config(state='normal')
            self.pathname_button.config(text='Set')
            self.editing_disabled = False
        else:
            self.gui_set_name()
            self.userentry_var.set('')
            self.pathname_textbox.config(state='disabled')
            self.pathname_button.config(text='Edit')
            self.editing_disabled = True

    def gui_recording(self):
        if not self.mgr.recording:
            self.mgr.record()

            self.recording_var.set('[ Recording... ]')
            self.button_recpause.config(text="Recording...\n\n(Space to toggle)", image=self.icon_rec_on)

    def gui_pause(self):
        if self.mgr.recording:
            self.mgr.pause()

            self.recording_var.set('')
            self.button_recpause.config(text="Not recording\n\n(Space to toggle)", image=self.icon_rec_off)

    def gui_toggle_recording(self):
        if self.mgr.acquiring:
            if not self.mgr.recording:
                self.gui_recording()
            else:
                self.gui_pause()

    def gui_toggle_acquisition(self):

        if self.mgr.savepath is None:
            self.gui_set_name()

        if not self.mgr.acquiring:
            self.mgr.on()

            self._capture_clock = datetime.now()
            self.start_indices[:] = self.mgr.indices

            self.button_acquisition.config(text="Acquiring", image=self.icon_capture_on)
            self.button_recpause.config(state="normal")
        else:
            self.gui_pause()
            self.mgr.off()

            self._capture_fps = np.zeros(self.mgr.nb_cameras, dtype=np.uintc)

            self.userentry_var.set('')
            self.applied_name_var.set('')

            self.button_acquisition.config(text="Acquisition off", image=self.icon_capture_off)
            self.button_recpause.config(state="disabled")

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
            self._capture_fps[:] = (self._now_indices - self.start_indices) / capture_dt + 0.00001

            self._current_buffers = self.mgr.get_current_framebuffer()

            self._saved_frames = self.mgr.saved
            self.frames_saved_var.set(f'Saved {sum(self._saved_frames)} frames total ({utils.pretty_size(sum(self._frame_sizes_bytes * self._saved_frames))})')

        self._counter += 1

        self.root.after(50, self.update)
