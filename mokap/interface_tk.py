import sys
import cv2
import tkinter as tk
from scipy import ndimage
from tkinter.filedialog import askopenfilename, askdirectory
import tkinter.font as font
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
from datetime import datetime
from threading import Thread, Event
import screeninfo
import platform
from pathlib import Path
import os
import subprocess
from mokap import utils
from functools import partial
from collections import deque
np.set_printoptions(precision=4, suppress=True)
import psutil

if 'Windows' in platform.system():
    import win32api


class GUILogger:
    def __init__(self):
        self.text_area = None
        self._temp_output = ''
        sys.stdout = self
        sys.stderr = self

    def register_text_area(self, text_area):
        self.text_area = text_area
        self.text_area.configure(state="normal")
        self.text_area.insert("end", self._temp_output)
        self.text_area.see("end")
        self.text_area.configure(state="disabled")

    def write(self, text):
        if self.text_area is None:
            # Temporarily capture console output to display later in the log widget
            self._temp_output += f'{text}'
        else:
            self.text_area.configure(state="normal")
            self.text_area.insert("end", text)
            self.text_area.see("end")
            self.text_area.configure(state="disabled")

    def flush(self):
        pass


# Create this immediately to capture everything
# gui_logger = GUILogger()
gui_logger = False


def whxy(what):
    if isinstance(what, tk.Toplevel):
        dims, x, y = what.geometry().split('+')
        w, h = dims.split('x')
    elif isinstance(what, MainWindow):
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


class VideoWindowBase:

    INFO_PANEL_H = 200
    WINDOW_MIN_W = 630
    TASKBAR_H = 80

    def __init__(self, parent, idx):

        self.parent = parent
        self.idx = idx
        self.window = tk.Toplevel()

        if 'Darwin' in platform.system():
            # Trick to force macOS to open a window and not a tab
            self.window.resizable(False, False)
            self._macOS_trick = True
        else:
            self._macOS_trick = False

        self._camera = self.parent.mc.cameras[self.idx]
        self._source_shape = self._camera.shape

        self.window.minsize(VideoWindowBase.WINDOW_MIN_W, VideoWindowBase.INFO_PANEL_H)
        self.window.protocol("WM_DELETE_WINDOW", self.toggle_visibility)

        self._bg_colour = f'#{self.parent.mc.colours[self._camera.name].lstrip("#")}'
        self._fg_colour = self.parent.col_white if utils.hex_to_hls(self._bg_colour)[1] < 60 else self.parent.col_black
        self._window_bg_colour = self.window.cget('background')

        # Where the frame data from the camera will be stored
        self._frame_buffer = np.zeros((*self._source_shape[:2], 3), dtype=np.uint8)

        # Init clock and counter
        self._clock = datetime.now()
        self._fps = deque(maxlen=10)
        self._capture_fps = deque(maxlen=10)
        self._last_capture_count = 0

        # Init states
        self.visible = True
        self.should_stop = False
        self._warning = False

        self.var_warning = tk.StringVar()

        # Some other stuff
        self._wanted_fps = self._camera.framerate
        self.window.title(f'{self._camera.name.title()} camera')
        self.positions = np.array([['nw', 'n', 'ne'],
                                   ['w', 'c', 'e'],
                                   ['sw', 's', 'se']])

        self.auto_size()

    def _init_common_ui(self):

        # Initialise main video frame
        panels = tk.PanedWindow(self.window, orient='vertical', opaqueresize=False)
        panels.pack(side="top", fill="both", expand=True)

        # VIDEO PANEL is the live video
        self.VIDEO_PANEL = tk.Label(panels, compound='center', bg=self.parent.col_black)

        # INFO PANEL is the bottom area
        self.INFO_PANEL = tk.Frame(panels)
        panels.add(self.VIDEO_PANEL, stretch='never')
        panels.add(self.INFO_PANEL)
        panels.paneconfig(self.VIDEO_PANEL, height=1000)
        panels.paneconfig(self.INFO_PANEL, height=self.INFO_PANEL_H, minsize=self.INFO_PANEL_H)

        # Place the image in the resized video panel
        image = Image.fromarray(self._frame_buffer, mode='RGB')
        image.thumbnail((self.VIDEO_PANEL.winfo_width(), self.VIDEO_PANEL.winfo_height()))
        self.imagetk = ImageTk.PhotoImage(image)
        self.VIDEO_PANEL.configure(image=self.imagetk)

        # Camera name bar
        name_bar = tk.Label(self.INFO_PANEL, text=f'{self._camera.name.title()} camera',
                            anchor='center', justify='center', height=1,
                            fg=self.colour_2, bg=self.colour, font=self.parent.font_bold)
        name_bar.pack(side='top', fill='x')

        self.LEFT_FRAME = tk.LabelFrame(self.INFO_PANEL, text="Left")
        self.LEFT_FRAME.pack(padx=(3, 0), pady=3, side='left', fill='both', expand=True)

        self.CENTRE_FRAME = tk.LabelFrame(self.INFO_PANEL, text="Centre")
        self.CENTRE_FRAME.pack(padx=(3, 3), pady=3, side='left', fill='both', expand=True)

        self.RIGHT_FRAME = tk.LabelFrame(self.INFO_PANEL, text="Right")
        self.RIGHT_FRAME.pack(padx=(0, 3), pady=3, side='left', fill='both', expand=True)

        # LEFT FRAME: Live information
        self.LEFT_FRAME.config(text='Information')

        f_labels = tk.Frame(self.LEFT_FRAME)
        f_labels.pack(side='left', fill='y', expand=False)

        f_values = tk.Frame(self.LEFT_FRAME)
        f_values.pack(side='left', fill='both', expand=True)

        self.var_resolution = tk.StringVar()
        self.var_exposure = tk.StringVar()
        self.var_capture_fps = tk.StringVar()
        self.var_brightness = tk.StringVar()
        # self.var_display_fps = tk.StringVar()
        self.var_temperature = tk.StringVar()

        # for label, var in zip(['Resolution', 'Capture', 'Exposure', 'Brightness', 'Display', 'Temperature'],
        for label, var in zip(['Resolution', 'Capture', 'Exposure', 'Brightness', 'Temperature'],
                              [self.var_resolution,
                               self.var_capture_fps,
                               self.var_exposure,
                               self.var_brightness,
                               # self.var_display_fps,
                               self.var_temperature]):
            l = tk.Label(f_labels, text=f"{label} :",
                         anchor='ne', justify='right',
                         fg=self.parent.col_darkgray,
                         font=self.parent.font_bold)
            l.pack(side='top', fill='both', expand=True)

            v = tk.Label(f_values, textvariable=var,
                         anchor='nw', justify='left',
                         font=self.parent.font_regular)
            v.pack(pady=(1, 0), side='top', fill='both', expand=True)

            if label == 'Temperature':
                self.label_temperature = l

        self.var_resolution.set(f"{self._camera.width}×{self._camera.height} px")
        self.var_exposure.set(f"{self._camera.exposure} µs")
        self.var_capture_fps.set(f"-")
        self.var_brightness.set(f"-")
        # self.var_display_fps.set(f"-")
        self.var_temperature.set(f"{self._camera.temperature}°C" if self._camera.temperature is not None else '-')

        # RIGHT FRAME: View control
        self.RIGHT_FRAME.config(text='View')

        f_windowsnap = tk.Frame(self.RIGHT_FRAME)
        f_windowsnap.pack(side='top', fill='both', expand=True)

        l_windowsnap = tk.Label(f_windowsnap, text=f"Window snap : ",
                                anchor='e', justify='right',
                                font=self.parent.font_bold,
                                fg=self.parent.col_darkgray)
        l_windowsnap.pack(side='left', fill='y')

        f_buttons_windowsnap = tk.Frame(f_windowsnap)
        f_buttons_windowsnap.pack(side='left', fill='x', expand=True)

        self._pixel = tk.PhotoImage(width=1, height=1)
        for r in range(3):
            for c in range(3):
                b = tk.Button(f_buttons_windowsnap,
                              image=self._pixel, compound="center",
                              width=6, height=6,
                              command=partial(self.move_to, self.positions[r, c]))
                b.grid(row=r, column=c)

    def _init_specific_ui(self):
        pass

    @property
    def name(self) -> str:
        return self._camera.name

    @property
    def colour(self) -> str:
        return f'#{self._bg_colour.lstrip("#")}'
    color = colour

    @property
    def colour_2(self) -> str:
        return f'#{self._fg_colour.lstrip("#")}'
    color_2 = colour_2

    @property
    def aspect_ratio(self):
        return self._source_shape[1] / self._source_shape[0]

    def auto_size(self):

        # If landscape screen
        if self.parent.selected_monitor.height < self.parent.selected_monitor.width:
            h = self.parent.selected_monitor.height // 2 - VideoWindowBase.TASKBAR_H
            w = int(self.aspect_ratio * (h - VideoWindowBase.INFO_PANEL_H))

        # If portrait screen
        else:
            w = self.parent.selected_monitor.width // 2
            h = int(w / self.aspect_ratio) + VideoWindowBase.INFO_PANEL_H

        self.window.geometry(f'{w}x{h}')

    def auto_move(self):
        if self.parent.selected_monitor.height < self.parent.selected_monitor.width:
            # First corners, then left right, then top and bottom,  and finally centre
            positions = ['nw', 'sw', 'ne', 'se', 'n', 's', 'w', 'e', 'c']
        else:
            # First corners, then top and bottom, then left right, and finally centre
            positions = ['nw', 'sw', 'ne', 'se', 'w', 'e', 'n', 's', 'c']

        nb_positions = len(positions)

        if self.idx <= nb_positions:
            pos = positions[self.idx]
        else:  # Start over to first position
            pos = positions[self.idx % nb_positions]

        self.move_to(pos)

    def move_to(self, pos):

        monitor = self.parent.selected_monitor
        w = self.window.winfo_width()
        h = self.window.winfo_height()

        match pos:
            case 'nw':
                self.window.geometry(f"{w}x{h}+{monitor.x}+{monitor.y}")
            case 'n':
                self.window.geometry(f"{w}x{h}+{monitor.x + monitor.width // 2 - w // 2}+{monitor.y}")
            case 'ne':
                self.window.geometry(f"{w}x{h}+{monitor.x + monitor.width - w - 1}+{monitor.y}")
            case 'w':
                self.window.geometry(f"{w}x{h}+{monitor.x}+{monitor.y + monitor.height // 2 - h // 2}")
            case 'c':
                self.window.geometry(f"{w}x{h}+{monitor.x + monitor.width // 2 - w // 2}+{monitor.y + monitor.height // 2 - h // 2}")
            case 'e':
                self.window.geometry(f"{w}x{h}+{monitor.x + monitor.width - w - 1}+{monitor.y + monitor.height // 2 - h // 2}")
            case 'sw':
                self.window.geometry(f"{w}x{h}+{monitor.x}+{monitor.y + monitor.height - h - VideoWindowBase.TASKBAR_H}")
            case 's':
                self.window.geometry(f"{w}x{h}+{monitor.x + monitor.width // 2 - w // 2}+{monitor.y + monitor.height - h - VideoWindowBase.TASKBAR_H}")
            case 'se':
                self.window.geometry(f"{w}x{h}+{monitor.x + monitor.width - w - 1}+{monitor.y + monitor.height - h - VideoWindowBase.TASKBAR_H}")

    def toggle_visibility(self, event=None):

        if self.visible:
            self.parent.child_windows_visibility_vars[self.idx].set(0)
            self.visible = False
            self.window.withdraw()

        elif not self.visible:
            self.parent.child_windows_visibility_vars[self.idx].set(1)
            self.visible = True
            self.window.deiconify()

    def _refresh_framebuffer(self):
        if self.parent.mc.acquiring and self.parent.current_buffers is not None:
            frame = self.parent.current_buffers[self.idx]
            if frame is not None:
                if len(self._source_shape) == 2:
                    np.copyto(self._frame_buffer[:, :, 0], frame)
                    np.copyto(self._frame_buffer[:, :, 1], frame)
                    np.copyto(self._frame_buffer[:, :, 2], frame)
                else:
                    np.copyto(self._frame_buffer, frame)
        else:
            self._frame_buffer.fill(0)

    def _process_full_size(self, image):
        return image

    def _process_resized(self, image_resized):
        return image_resized

    def _update_video(self):

        raw_image = Image.fromarray(self._frame_buffer, mode='RGB')

        # Do something on the image at full resolution (i.e. stuff like motion detection)
        image = self._process_full_size(raw_image)

        # Resize the image
        image.thumbnail((self.VIDEO_PANEL.winfo_width(), self.VIDEO_PANEL.winfo_height()))

        # Do something on the image after resize (i.e. add GUI elements that don't get scaled)
        image_processed_resized = self._process_resized(image)

        if self.imagetk.width() != image_processed_resized.width or self.imagetk.height() != image_processed_resized.height:
            self.imagetk = ImageTk.PhotoImage(image_processed_resized)
            self.VIDEO_PANEL.configure(image=self.imagetk)
        else:
            self.imagetk.paste(image_processed_resized)

    def _update_txtvars(self):

        fps = np.mean(list(self._fps)) if self._fps else 0

        if self.parent.mc.acquiring:

            cap_fps = np.mean(list(self._capture_fps))

            if 0 < cap_fps < 1000:  # only makes sense to display real values
                if abs(cap_fps - self._wanted_fps) > 10:
                    self.var_warning.set('[ WARNING: Framerate ]')
                    self._warning = True
                else:
                    self._warning = False
                self.var_capture_fps.set(f"{cap_fps:.2f} fps")
            else:
                self.var_capture_fps.set("-")

            brightness = np.round(self._frame_buffer.mean() / 255 * 100, decimals=2)
            self.var_brightness.set(f"{brightness:.2f}%")
        else:
            self.var_capture_fps.set("Off")
            self.var_brightness.set("-")

        # self.var_display_fps.set(f"{fps:.2f} fps")

        if self._camera.temperature is not None:
            self.var_temperature.set(f'{self._camera.temperature:.1f}°C')
        if self._camera.temperature_state == 'Ok':
            self.label_temperature.config(fg="green")
        elif self._camera.temperature_state == 'Critical':
            self.label_temperature.config(fg="orange")
        elif self._camera.temperature_state == 'Error':
            self.label_temperature.config(fg="red")
        else:
            self.label_temperature.config(fg="yellow")

    def update(self):

        while not self.should_stop:
            # Disable the trick if it's on
            if self._macOS_trick:
                self.window.resizable(True, True)
                self._macOS_trick = False
                Event().wait(0.2)   # This is needed otherwise macOS freaks out and SIGKILLs the thread...

            if self.visible:

                self._update_txtvars()

                self._refresh_framebuffer()
                self._update_video()

                now = datetime.now()

                # Update display fps
                dt = (now - self._clock).total_seconds()
                ind = int(self.parent.mc.indices[self.idx])
                if dt > 0:
                    self._fps.append(1.0 / dt)
                    self._capture_fps.append((ind - self._last_capture_count) / dt)

                self._clock = now
                self._last_capture_count = ind

            else:
                Event().wait(0.1)


class VideoWindowMain(VideoWindowBase):

    def __init__(self, parent, idx):
        super().__init__(parent, idx)

        self._show_focus = False
        self._magnification = False

        # Magnification parameters
        self.magn_zoom = tk.DoubleVar()
        self.magn_zoom.set(1)

        self.magn_window_w = 100
        self.magn_window_h = 100
        self.magn_window_x = 10 + self.magn_window_w//2     # Initialise in the corner
        self.magn_window_y = 10 + self.magn_window_h//2

        self.magn_target_cx = self._source_shape[1] // 2
        self.magn_target_cy = self._source_shape[0] // 2

        # Focus view parameters
        # Kernel to use for focus detection
        self._kernel = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]], dtype=np.uint8)

        # Other specific stuff
        try:
            self._imgfnt = ImageFont.load_default(30)
        except TypeError:
            print('[INFO] Mokap works better with Pillow version 10.1.0 or more!')
            self._imgfnt = ImageFont.load_default()

        self.col_default = None

        self._init_common_ui()
        self._init_specific_ui()

        self.VIDEO_PANEL.bind('<Button-3>', self._get_magn_pos)
        self.VIDEO_PANEL.bind("<B3-Motion>", self._get_magn_pos)
        self.VIDEO_PANEL.bind('<Button-1>', self._get_magn_target)
        self.VIDEO_PANEL.bind("<B1-Motion>", self._get_magn_target)

    def _init_specific_ui(self):

        # CENTRE FRAME: Camera controls block
        self.CENTRE_FRAME.config(text='Controls')

        self.camera_controls_sliders = {}
        self._val_in_sync = {}

        lf = tk.Frame(self.CENTRE_FRAME)
        lf.pack(pady=(15, 0), side='left', fill='both', expand=True)

        rf = tk.LabelFrame(self.CENTRE_FRAME, text='Sync', font=self.parent.font_mini)
        rf.pack(padx=3, pady=(0, 3), side='right', fill='y', expand=True)

        f_labels = tk.Frame(lf)
        f_labels.pack(side='left', fill='y', expand=True)

        f_values = tk.Frame(lf)
        f_values.pack(side='left', fill='both', expand=True)

        for label, slider_params in zip(['framerate', 'exposure', 'blacks', 'gain', 'gamma'],
                                        # type,   from,   to,     resolution,     digits
                                         [(int, 1, self.parent.mc.cameras[self.idx].max_framerate, 1, 1),
                                          (int,   21,     1e5,    5,              1),  # in microseconds - 1e5 ~ 10 fps
                                          (float, 0.0,    32.0,   0.5,            3),
                                          (float, 0.0,    36.0,   0.5,            3),
                                          (float, 0.0,    3.99,   0.05,           3)
                                          ]):

            param_value = getattr(self.parent.mc.cameras[self.idx], label)

            v = tk.IntVar()
            b = tk.Checkbutton(rf, variable=v, state='normal')
            v.set(1)
            b.pack(side="top", fill="x", expand=True)
            self._val_in_sync[label] = v

            scale_frame = tk.Frame(f_values)
            scale_frame.pack(side='top', fill='x', expand=True)

            if slider_params[0] == int:
                var = tk.IntVar()
            else:
                var = tk.DoubleVar()

            s = tk.Scale(scale_frame,
                         from_=slider_params[1], to=slider_params[2], resolution=slider_params[3],
                         digits=slider_params[4],
                         font=self.parent.font_mini,
                         variable=var,
                         showvalue=False,
                         orient='horizontal', width=15, sliderlength=10)
            s.set(param_value)
            s.bind("<ButtonRelease-1>", partial(self._update_param_all, label))
            s.pack(side='left', anchor='w', fill='both', expand=True)

            if self.col_default is None:
                self.col_default = s.cget('trough')

            scale_val = tk.Label(scale_frame, textvariable=var,
                       anchor='w', justify='left', width=6,
                       font=self.parent.font_regular)
            scale_val.pack(side='left', anchor='w', fill='both', expand=False)

            self.camera_controls_sliders[label] = s

            l = tk.Label(f_labels, text=f'{label.title()} :',
                         anchor='e', justify='right',
                         font=self.parent.font_bold,
                         fg=self.parent.col_darkgray)
            l.pack(side='top', fill='x', expand=True)

        ## Right Frame: Specific buttons
        f_buttons_controls = tk.LabelFrame(self.RIGHT_FRAME, text='Visualisations')
        f_buttons_controls.pack(padx=3, pady=3, side='top', fill='both', expand=True)

        f = tk.Frame(f_buttons_controls)
        f.pack(pady=(5, 0), side='left', fill='y', expand=True)

        self.show_focus_button = tk.Button(f, text="Focus zone",
                                           highlightthickness=2, highlightbackground=self._window_bg_colour,
                                           font=self.parent.font_regular,
                                           command=self._toggle_focus_display)
        self.show_focus_button.pack(side='top', fill='both', expand=False)

        f = tk.Frame(f_buttons_controls)
        f.pack(pady=(5, 0), side='left', fill='y', expand=True)

        self.show_mag_button = tk.Button(f, text="Magnifier",
                                         highlightthickness=2, highlightbackground=self.parent.col_yellow,
                                         font=self.parent.font_regular,
                                         command=self._toggle_mag_display)
        self.show_mag_button.pack(side='top', fill='x', expand=False)

        self.slider_magn = tk.Scale(f, variable=self.magn_zoom,
                                    from_=1, to=5, resolution=0.1, orient='horizontal',
                                    width=10, sliderlength=10)
        self.slider_magn.pack(side='top', fill='x', expand=False)

    def _get_magn_pos(self, event):
        if self._magnification:
            margin_horiz = (self.VIDEO_PANEL.winfo_width() - self.imagetk.width()) // 2
            margin_vert = (self.VIDEO_PANEL.winfo_height() - self.imagetk.height()) // 2

            self.magn_window_x = event.x - margin_horiz - self.magn_window_w // 2
            self.magn_window_y = event.y - margin_vert - self.magn_window_h // 2

    def _get_magn_target(self, event):
        if self._magnification:
            margin_horiz = (self.VIDEO_PANEL.winfo_width() - self.imagetk.width()) // 2
            margin_vert = (self.VIDEO_PANEL.winfo_height() - self.imagetk.height()) // 2

            self.magn_target_cx = int(round((event.x - margin_horiz) * self._source_shape[1] / self.imagetk.width()))
            self.magn_target_cy = int(round((event.y - margin_vert) * self._source_shape[0] / self.imagetk.height()))

    def update_param(self, param):

        if param == 'framerate' and self.parent.mc.triggered and self.parent.mc.acquiring:
            return

        slider = self.camera_controls_sliders[param]
        new_val = slider.get()

        setattr(self.parent.mc.cameras[self.idx], param, new_val)

        # And update the slider to the actual new value (can be different from the one requested)
        slider.set(getattr(self.parent.mc.cameras[self.idx], param))

        if param == 'exposure':
            # Refresh exposure value for UI display
            self.var_exposure.set(f"{self.parent.mc.cameras[self.idx].exposure} µs")

            # We also need to update the framerate slider to current resulting fps after exposure change
            self.update_param('framerate')

        elif param == 'framerate':
            # Keep a local copy to warn user if actual framerate is too different from requested fps
            self._wanted_fps = self.camera_controls_sliders[param].get()

            if self.parent.mc.triggered:
                self.parent.mc.framerate = self._wanted_fps
            else:
                self.parent.mc.cameras[self.idx].framerate = self._wanted_fps

            self.camera_controls_sliders['framerate'].config(to=self.parent.mc.cameras[self.idx].max_framerate)

    def _update_param_all(self, param, event=None):
        self.update_param(param)
        should_apply = bool(self._val_in_sync[param].get())
        if should_apply:
            for window in self.parent.secondary_windows:
                if window is not self and bool(window._val_in_sync[param].get()):
                    slider = self.camera_controls_sliders[param]
                    new_val = slider.get()
                    window.camera_controls_sliders[param].set(new_val)
                    window.update_param(param)

    def _toggle_focus_display(self):
        if self._show_focus:
            self.show_focus_button.configure(highlightbackground=self._window_bg_colour)
            self._show_focus = False
        else:
            self.show_focus_button.configure(highlightbackground=self.parent.col_green)
            self._show_focus = True

    def _toggle_mag_display(self):
        if self._magnification:
            self.show_mag_button.configure(highlightbackground=self._window_bg_colour)
            self.slider_magn.config(state='disabled')
            self._magnification = False
        else:
            self.show_mag_button.configure(highlightbackground=self.parent.col_yellow)
            self.slider_magn.config(state='active')
            self._magnification = True

    def _process_resized(self, image):

        # Get new coordinates
        w, h = image.size

        x_centre, y_centre = w // 2, h // 2
        x_north, y_north = w // 2, 0
        x_south, y_south = w // 2, h
        x_east, y_east = w, h // 2
        x_west, y_west = 0, h // 2

        d = ImageDraw.Draw(image)

        # Draw crosshair
        d.line((x_west, y_west, x_east, y_east), fill=self.parent.col_white_rgb, width=1)       # Horizontal
        d.line((x_north, y_north, x_south, y_south), fill=self.parent.col_white_rgb, width=1)   # Vertical

        # Position the 'Recording' indicator
        d.text((x_centre, y_south - y_centre/2.0), self.parent.var_recording.get(),
               anchor="ms", font=self._imgfnt,
               fill=self.parent.col_red)

        if self._warning:
            d.text((x_centre, y_north + y_centre/2.0), self.var_warning.get(),
                   anchor="ms", font=self._imgfnt,
                   fill=self.parent.col_orange)

        if self._magnification:

            col = self.parent.col_yellow_rgb

            ratio_w = w / self._source_shape[1]
            ratio_h = h / self._source_shape[0]

            # Size of the slice to extract from the source
            slice_w = self.magn_window_w
            slice_h = self.magn_window_h

            # Position of the slice in source pixels coordinates
            slice_cx = self.magn_target_cx
            slice_cy = self.magn_target_cy

            slice_x1 = max(0, slice_cx - slice_w//2)
            slice_y1 = max(0, slice_cy - slice_h//2)
            slice_x2 = slice_x1 + slice_w
            slice_y2 = slice_y1 + slice_h

            if slice_x2 > self._source_shape[1]:
                slice_x1 = self._source_shape[1] - slice_w
                slice_x2 = self._source_shape[1]

            if slice_y2 > self._source_shape[0]:
                slice_y1 = self._source_shape[0] - slice_h
                slice_y2 = self._source_shape[0]

            # Slice directly from the framebuffer and make a (then zoomed) image
            magn_img = Image.fromarray(self._frame_buffer[slice_y1:slice_y2, slice_x1:slice_x2], mode='RGB')
            magn_img = magn_img.resize(
                (int(magn_img.width * self.magn_zoom.get()), int(magn_img.height * self.magn_zoom.get())))

            image.paste(magn_img, (self.magn_window_x, self.magn_window_y))

            # Add frame around the magnified area
            tgt_x1 = int(slice_x1 * ratio_w)
            tgt_x2 = int(slice_x2 * ratio_w)
            tgt_y1 = int(slice_y1 * ratio_h)
            tgt_y2 = int(slice_y2 * ratio_h)
            d.rectangle([(tgt_x1, tgt_y1), (tgt_x2, tgt_y2)],
                        outline=col, width=1)

            # Add frame around the magnification
            d.rectangle([(self.magn_window_x, self.magn_window_y),
                         (self.magn_window_x + magn_img.width,
                          self.magn_window_y + magn_img.height)], outline=col, width=1)

            # Add a small + in the centre
            c = self.magn_window_x + magn_img.width // 2, self.magn_window_y + magn_img.height // 2
            d.line((c[0] - 5, c[1], c[0] + 5, c[1]), fill=col, width=1)  # Horizontal
            d.line((c[0], c[1] - 5, c[0], c[1] + 5), fill=col, width=1)  # Vertical

        if self._show_focus:
            lapl = ndimage.gaussian_laplace(np.array(image)[:, :, 0].astype(np.uint8), sigma=1)
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
            image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')

        return image


class MainWindow:
    MIN_WIDTH = 600
    MIN_HEIGHT = 370

    # Colours
    col_white = "#ffffff"
    col_white_rgb = utils.hex_to_rgb(col_white)
    col_black = "#000000"
    col_black_rgb = utils.hex_to_rgb(col_black)
    col_lightgray = "#e3e3e3"
    col_lightgray_rgb = utils.hex_to_rgb(col_lightgray)
    col_midgray = "#c0c0c0"
    col_midgray_rgb = utils.hex_to_rgb(col_midgray)
    col_darkgray = "#515151"
    col_darkgray_rgb = utils.hex_to_rgb(col_darkgray)
    col_red = "#FF3C3C"
    col_red_rgb = utils.hex_to_rgb(col_red)
    col_orange = "#FF9B32"
    col_orange_rgb = utils.hex_to_rgb(col_orange)
    col_yellow = "#FFEB1E"
    col_yellow_rgb = utils.hex_to_rgb(col_yellow)
    col_yelgreen = "#A5EB14"
    col_yelgreen_rgb = utils.hex_to_rgb(col_yelgreen)
    col_green = "#00E655"
    col_green_rgb = utils.hex_to_rgb(col_green)
    col_blue = "#5ac3f5"
    col_blue_rgb = utils.hex_to_rgb(col_blue)
    col_purple = "#c887ff"
    col_purple_rgb = utils.hex_to_rgb(col_purple)

    def __init__(self, mc):

        # Set up root window
        self.root = tk.Tk()
        self.root.minsize(MainWindow.MIN_WIDTH, MainWindow.MIN_HEIGHT)
        self.root.geometry(f"{MainWindow.MIN_WIDTH}x{MainWindow.MIN_HEIGHT}")

        # Identify monitors
        self.selected_monitor = None
        self._monitors = screeninfo.get_monitors()
        self.set_monitor()

        # Icons
        resources_path = [p for p in Path().cwd().glob('**/*') if p.is_dir() and p.name == 'icons'][0]

        ico = ImageTk.PhotoImage(Image.open(resources_path / "mokap.png"))
        self.root.wm_iconphoto(True, ico)

        self.root.title("Controls")
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.bind('<KeyPress>', self._handle_keypress)

        self.icon_capture = tk.PhotoImage(file=resources_path / 'capture.png')
        self.icon_capture_bw = tk.PhotoImage(file=resources_path / 'capture_bw.png')
        self.icon_snapshot = tk.PhotoImage(file=resources_path / 'snapshot.png')
        self.icon_snapshot_bw = tk.PhotoImage(file=resources_path / 'snapshot_bw.png')
        self.icon_rec_on = tk.PhotoImage(file=resources_path / 'rec.png')
        self.icon_rec_bw = tk.PhotoImage(file=resources_path / 'rec_bw.png')

        # Fonts
        self.font_bold = font.Font(weight='bold', size=10)
        self.font_regular = font.Font(size=9)
        self.font_mini = font.Font(size=8)

        self.mc = mc

        self.var_recording = tk.StringVar()
        self.var_recording.set('')

        # States
        self.editing_disabled = True
        self._is_calibrating = False

        # Clock
        self._clock = datetime.now()

        # Other things to init
        self._current_buffers = None
        self._mem_baseline = None
        self._mem_pressure = 0.0

        #  Refs for the secondary windows and their threads
        self.child_windows = []
        self.child_threads = []

        # Build the gui
        self._init_main_ui()

        # Start the secondary windows
        self._start_child_windows()

        # Start the main thread
        self.update_main()
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    def _init_main_ui(self):

        # Create toolbar, main content panel, and statusbar
        toolbar = tk.Frame(self.root, height=38)
        statusbar = tk.Frame(self.root, background="#e3e3e3")
        content_panels = tk.PanedWindow(self.root, orient='vertical', relief=tk.GROOVE)

        toolbar.pack(side="top", fill="x")
        statusbar.pack(side="bottom", fill="both", pady=2)
        content_panels.pack(padx=2, pady=2, side="top", fill="both", expand=True)

        # --- TOOLBAR
        # Mode switcher
        self.mode_var = tk.StringVar()
        modes_choices = ['Recording', 'Calibration']
        self.mode_var.set('Recording')
        mode_label = tk.Label(toolbar, text='Mode: ', anchor=tk.W)
        mode_label.pack(side="left", fill="y", expand=False)
        mode_button = tk.OptionMenu(toolbar, self.mode_var, *modes_choices)
        mode_button.config(anchor=tk.CENTER)
        mode_button.pack(padx=2, pady=2, ipady=2, side="left", fill="y", expand=False)
        self.mode_var.trace('w', self._toggle_calibrate)

        # Exit button
        if 'Darwin' in platform.system():
            self.button_exit = tk.Button(toolbar, text="Exit (Esc)", anchor=tk.CENTER, font=self.font_bold,
                                         fg=self.col_red,
                                         command=self.quit)
        else:
            self.button_exit = tk.Button(toolbar, text="Exit (Esc)", anchor=tk.CENTER, font=self.font_bold,
                                         bg=self.col_red, fg=self.col_white,
                                         command=self.quit)
        self.button_exit.pack(padx=3, pady=4, side="right", fill="y", expand=False)
        # ---

        # --- STATUSBAR
        self.var_frames_saved = tk.StringVar()
        self.var_frames_saved.set(f'Saved frames: {self.mc.saved} (0 bytes)')

        mem_pressure_label = tk.Label(statusbar, text='Memory pressure: ', anchor=tk.NW)
        mem_pressure_label.pack(side="left", expand=False)
        self._mem_pressure_bar = ttk.Progressbar(statusbar, length=20, maximum=0.9)
        self._mem_pressure_bar.pack(side="left", fill="both", expand=True)

        frames_saved_label = tk.Label(statusbar, textvariable=self.var_frames_saved, anchor=tk.NE)
        frames_saved_label.pack(side="right", fill="both", expand=True)
        # ---

        # --- MAIN CONTENT
        maincontent = tk.Frame(content_panels)
        left_pane = tk.LabelFrame(maincontent, text="Acquisition")
        right_pane = tk.LabelFrame(maincontent, text="Display")

        left_pane.pack(padx=3, pady=3, side="left", fill="both", expand=True)
        right_pane.pack(padx=3, pady=3, side="left", fill="both", expand=True)

        content_panels.add(maincontent)

        # --- LEFT HALF
        name_frame = tk.Frame(left_pane)
        name_frame.pack(padx=3, side="top", fill="x", expand=True)

        editable_name_frame = tk.Frame(name_frame)
        editable_name_frame.pack(side="top", fill="x", expand=True)

        pathname_label = tk.Label(editable_name_frame, text='Name: ', anchor=tk.W)
        pathname_label.pack(side="left", fill="y", expand=False)

        self.var_userentry = tk.StringVar()
        self.var_userentry.set('')

        self.pathname_textbox = tk.Entry(editable_name_frame, bg=self.col_white, fg=self.col_black,
                                         textvariable=self.var_userentry,
                                         font=self.font_regular, state='disabled')
        self.pathname_textbox.pack(side="left", fill="both", expand=True)

        self.pathname_button = tk.Button(editable_name_frame,
                                         text="Edit", font=self.font_regular, command=self._toggle_text_editing)
        self.pathname_button.pack(padx=3, side="right", fill="both", expand=False)

        info_name_frame = tk.Frame(name_frame)
        info_name_frame.pack(side="top", fill="x", expand=True)

        save_dir_label = tk.Label(info_name_frame, text='Saves to: ', anchor=tk.W)
        save_dir_label.pack(side="top", fill="both", expand=False)

        self.var_save_dir_current = tk.StringVar()
        self.var_save_dir_current.set(self.mc.full_path.resolve())

        save_dir_current = tk.Label(info_name_frame, textvariable=self.var_save_dir_current, anchor=tk.W, fg=self.col_darkgray)
        save_dir_current.pack(side="left", fill="both", expand=True)

        gothere_button = tk.Button(info_name_frame, text="Open", font=self.font_regular, command=self.open_session_folder)
        gothere_button.pack(padx=3, pady=4, side="right", fill="y", expand=False)


        f_buttons = tk.Frame(left_pane)
        f_buttons.pack(padx=3, pady=3, side="top", fill="both", expand=True)

        f_buttons_row_1 = tk.Frame(f_buttons)
        f_buttons_row_1.pack(pady=3, side="top", fill="both", expand=True)

        self.button_acquisition = tk.Button(f_buttons_row_1,
                                            image=self.icon_capture_bw,
                                            compound='left', text="Acquisition off", anchor='center',
                                            width=150,
                                            font=self.font_regular,
                                            command=self._toggle_acquisition,
                                            state='normal')
        self.button_acquisition.pack(padx=3, side="left", fill="both", expand=True)

        self.button_snapshot = tk.Button(f_buttons_row_1,
                                         image=self.icon_snapshot,
                                         compound='left', text="Snapshot", anchor='center',
                                         width=150,
                                         font=self.font_regular,
                                         command=self._take_snapshot,
                                         state='disabled')
        self.button_snapshot.pack(padx=3, side="left", fill="both", expand=True)

        self.button_recpause = tk.Button(f_buttons,
                                         image=self.icon_rec_bw,
                                         compound='left', text="Not recording (Space to toggle)", anchor='center',
                                         font=self.font_bold,
                                         command=self._toggle_recording,
                                         state='disabled')
        self.button_recpause.pack(padx=3, pady=3, side="top", fill="both", expand=True)

        # RIGHT HALF
        windows_visibility_frame = tk.Frame(right_pane)
        windows_visibility_frame.pack(side="top", fill="x", expand=True)

        visibility_label = tk.Label(windows_visibility_frame, text='Show previews:', anchor=tk.W)
        visibility_label.pack(side="top", fill="x", expand=False)

        windows_list_frame = tk.Frame(windows_visibility_frame)
        windows_list_frame.pack(side="top", fill="both", expand=True)

        self.child_windows_visibility_vars = []
        self.child_windows_visibility_buttons = []

        for i in range(self.mc.nb_cameras):
            vis_var = tk.IntVar()
            vis_checkbox = tk.Checkbutton(windows_list_frame, text=f"", anchor=tk.W,
                                          font=self.font_bold,
                                          variable=vis_var,
                                          command=self.nothing,
                                          state='normal')
            vis_var.set(0)
            vis_checkbox.pack(side="top", fill="x", expand=True)

            self.child_windows_visibility_buttons.append(vis_checkbox)
            self.child_windows_visibility_vars.append(vis_var)

        monitors_frame = tk.Frame(right_pane)
        monitors_frame.pack(side="top", fill="both", expand=True)

        monitors_label = tk.Label(monitors_frame, text='Active monitor:', anchor=tk.W)
        monitors_label.pack(side="top", fill="x", expand=False)

        m_canvas_y_size = max([m.y + m.height for m in self._monitors]) // 70 + 2 * 10

        self.monitors_buttons = tk.Canvas(monitors_frame, height=m_canvas_y_size)
        self.update_monitors_buttons()

        for i, m in enumerate(self._monitors):
            self.monitors_buttons.tag_bind(f'screen_{i}', '<Button-1>', lambda _, val=i: self.screen_update(val))
        self.monitors_buttons.pack(side="top", fill="x", expand=False)

        self.autotile_button = tk.Button(monitors_frame,
                                         text="Auto-tile windows", font=self.font_regular,
                                         command=self.autotile_windows)
        self.autotile_button.pack(padx=6, pady=6, side="bottom", fill="both", expand=True)

        # # LOG PANEL
        # if gui_logger:
        #     log_label_frame = tk.Frame(content_panels)
        #     log_label = tk.Label(log_label_frame, text='↓ pull for log ↓', anchor=tk.S, font=('Arial', 6))
        #     log_label.pack(side="top", fill="x", expand=True)
        #
        #     log_frame = tk.Frame(content_panels)
        #     log_text_area = tk.Text(log_frame, font=("consolas", "9", "normal"))
        #     log_text_area.pack(side="top", fill="both", expand=True)
        #     content_panels.add(log_label_frame)
        #     content_panels.add(log_frame)
        #
        #     gui_logger.register_text_area(log_text_area)

    def _update_child_windows_list(self):

        for w, window in enumerate(self.child_windows):

            self.child_windows_visibility_vars[w].set(int(window.visible))
            self.child_windows_visibility_buttons[w].config(text=f" {window.name.title()} camera",
                                                            fg=window.colour_2,
                                                            bg=window.colour,
                                                            selectcolor=window.colour,
                                                            activebackground=window.colour,
                                                            activeforeground=window.colour,
                                                            command=window.toggle_visibility)
    def _start_child_windows(self):
        for c in self.mc.cameras:

            if self._is_calibrating:
                # w = VideoWindowCalib(parent=self, idx=c.idx)
                # self.child_windows.append(w)
                # t = Thread(target=w.update, args=(), daemon=True)
                # t.start()
                # self.child_threads.append(t)
                pass
            else:
                w = VideoWindowMain(parent=self, idx=c.idx)
                self.child_windows.append(w)
                t = Thread(target=w.update, args=(), daemon=True)
                t.start()
                self.child_threads.append(t)

        self._update_child_windows_list()

    def _stop_child_windows(self):
        for w in self.child_windows:
            w.should_stop = True

        for w in self.child_windows:
            try:
                w.window.destroy()
            except:
                pass

        self.child_windows = []
        self.child_threads = []

    @property
    def current_buffers(self):
        return self._current_buffers

    def set_monitor(self, idx=None):
        if len(self._monitors) > 1 and idx is None:
            self.selected_monitor = next(m for m in self._monitors if m.is_primary)
        elif len(self._monitors) > 1 and idx is not None:
            self.selected_monitor = self._monitors[idx]
        else:
            self.selected_monitor = self._monitors[0]

    def open_session_folder(self):
        path = Path(self.var_save_dir_current.get()).resolve()

        if self.var_save_dir_current.get() == "":
            path = self.mc.full_path

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
        windows = [w for w in self.child_windows if w.visible]
        if include_main:
            windows += [self]
        return windows

    def screen_update(self, val):

        # Get current monitor coordinates - this is the origin
        prev_monitor_x, prev_monitor_y = self.selected_monitor.x, self.selected_monitor.y

        # Get current window position in relation to origin
        prev_root_x = self.root.winfo_rootx()
        prev_root_y = self.root.winfo_rooty()

        # Get current mouse cursor position in relation to origin
        rel_mouse_x = self.root.winfo_pointerx() - prev_root_x
        rel_mouse_y = self.root.winfo_pointery() - prev_root_y

        # Set new monitor
        self.set_monitor(val)
        self.update_monitors_buttons()

        # Move windows by the difference
        for window_to_move in self.visible_windows(include_main=True):
            w, h, x, y = whxy(window_to_move)

            d_x = x - prev_monitor_x
            d_y = y - prev_monitor_y

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

                # Move cursor with the root window
                if 'Windows' in platform.system():
                    win32api.SetCursorPos((new_x + rel_mouse_x, new_y + rel_mouse_y))
                elif 'Linux' in platform.system():
                    self.root.event_generate('<Motion>', warp=True, x=0, y=0)
                # TODO - macOS
            else:
                window_to_move.window.geometry(f'{w}x{h}+{new_x}+{new_y}')

    def auto_size(self):
        pass

    def move_to(self, pos):

        w_scr, h_scr, x_scr, y_scr = whxy(self.selected_monitor)
        arbitrary_taskbar_h = 80

        w, h, _, _ = whxy(self)

        if pos == 'nw':
            self.root.geometry(f"{w}x{h}+{x_scr}+{y_scr}")
        elif pos == 'n':
            self.root.geometry(f"{w}x{h}+{x_scr + w_scr // 2 - w // 2}+{y_scr}")
        elif pos == 'ne':
            self.root.geometry(f"{w}x{h}+{x_scr + w_scr - w - 1}+{y_scr}")

        elif pos == 'w':
            self.root.geometry(f"{w}x{h}+{x_scr}+{y_scr + h_scr // 2 - h // 2}")
        elif pos == 'c':
            self.root.geometry(f"{w}x{h}+{x_scr + w_scr // 2 - w // 2}+{y_scr + h_scr // 2 - h // 2}")
        elif pos == 'e':
            self.root.geometry(f"{w}x{h}+{x_scr + w_scr - w - 1}+{y_scr + h_scr // 2 - h // 2}")

        elif pos == 'sw':
            self.root.geometry(f"{w}x{h}+{x_scr}+{y_scr + h_scr - h - arbitrary_taskbar_h}")
        elif pos == 's':
            self.root.geometry(f"{w}x{h}+{x_scr + w_scr // 2 - w // 2}+{y_scr + h_scr - h - arbitrary_taskbar_h}")
        elif pos == 'se':
            self.root.geometry(f"{w}x{h}+{x_scr + w_scr - w - 1}+{y_scr + h_scr - h - arbitrary_taskbar_h}")

    def auto_move(self):
        if self.selected_monitor.height < self.selected_monitor.width:
            # First corners, then left right, then top and bottom,  and finally centre
            positions = ['nw', 'sw', 'ne', 'se', 'n', 's', 'w', 'e', 'c']
        else:
            # First corners, then top and bottom, then left right, and finally centre
            positions = ['nw', 'sw', 'ne', 'se', 'w', 'e', 'n', 's', 'c']

        nb_positions = len(positions)

        idx = len(self.child_windows)
        if idx <= nb_positions:
            pos = positions[idx]
        else:  # Start over to first position
            pos = positions[idx % nb_positions]

        self.move_to(pos)

    def autotile_windows(self):
        """
            Automatically arranges and resizes the windows
        """
        for w in self.visible_windows(include_main=True):
            w.auto_size()
            w.auto_move()

    def _handle_keypress(self, event):
        if self.editing_disabled:
            if event.keycode == 27:         # Esc
                self.quit()
            elif event.keycode == 32:       # Space
                self._toggle_recording()
            else:
                pass
        else:
            if event.keycode == 13:         # Enter
                self._toggle_text_editing(True)

    def _take_snapshot(self):
        """
            Takes an instantaneous snapshot from all cameras
        """

        dims = np.array([(cam.height, cam.width) for cam in self.mc.cameras], dtype=np.uint32).T
        ext = self.mc.saving_ext
        now = datetime.now().strftime('%y%m%d-%H%M')

        if self.mc.acquiring:
            arrays = [np.frombuffer(c, dtype=np.uint8) for c in self._current_buffers]

            for a, arr in enumerate(arrays):
                img = Image.fromarray(arr.reshape(dims[a]))
                img.save(self.mc.full_path.resolve() / f"snapshot_{now}_{self.mc.cameras[a].name}.{ext}")

    def _toggle_text_editing(self, override=None):

        if override is None:
            override = not self.mc.recording

        if self.editing_disabled and override is False:
            self.pathname_textbox.config(state='normal')
            self.pathname_button.config(text='Set')
            self.editing_disabled = False

        elif not self.editing_disabled and override is True:
            self.pathname_textbox.config(state='disabled')
            self.pathname_button.config(text='Edit')
            self.mc.session_name = self.var_userentry.get()
            self.editing_disabled = True

            self.var_save_dir_current.set(f'{self.mc.full_path.resolve()}')

    def _toggle_recording(self, override=None):

        if override is None:
            override = not self.mc.recording

        # If we're currently recording, then we should stop
        if self.mc.acquiring:

            if self.mc.recording and override is False:
                self.mc.pause()
                self.var_recording.set('')
                self.button_recpause.config(text="Not recording (Space to toggle)", image=self.icon_rec_bw)
            elif not self.mc.recording and override is True:
                self._mem_baseline = psutil.virtual_memory().percent
                self.mc.record()
                self.var_recording.set('[ Recording... ]')
                self.button_recpause.config(text="Recording... (Space to toggle)", image=self.icon_rec_on)

    def _toggle_calibrate(self, *events):
        mode = self.mode_var.get()

        if self._is_calibrating and mode == 'Recording':
            self._stop_child_windows()

            self._is_calibrating = False

            if self.mc.acquiring:
                self.button_snapshot.config(state="normal")
                self.button_recpause.config(state="normal")

            self._start_child_windows()

        elif not self._is_calibrating and mode == 'Calibration':
            self._stop_child_windows()

            self._is_calibrating = True
            self.button_recpause.config(state="disabled")

            self._start_child_windows()

        else:
            pass

    def _toggle_acquisition(self, override=None):

        if override is None:
            override = not self.mc.acquiring

        # If we're currently acquiring, then we should stop
        if self.mc.acquiring and override is False:

            self._toggle_recording(False)
            self.mc.off()

            # Reset Acquisition folder name
            self.var_userentry.set('')
            self.var_save_dir_current.set('')

            self.button_acquisition.config(text="Acquisition off", image=self.icon_capture_bw)
            self.button_snapshot.config(state="disabled")
            self.button_recpause.config(state="disabled")

            # Re-enable the framerate sliders (only in case of hardware-triggered cameras)
            if self.mc.triggered:
                for w in self.child_windows:
                    w.camera_controls_sliders['framerate'].config(state='normal', troughcolor=w.col_default)

        elif not self.mc.acquiring and override is True:
            self.mc.on()

            if self.mc.triggered:
                for w in self.child_windows:
                    w.camera_controls_sliders['framerate'].config(state='disabled', troughcolor=self.col_lightgray)

            self.var_save_dir_current.set(f'{self.mc.full_path.resolve()}')

            self.button_acquisition.config(text="Acquiring", image=self.icon_capture)
            self.button_snapshot.config(state="normal")
            if not self._is_calibrating:
                self.button_recpause.config(state="normal")

    def quit(self):

        # Close the child windows and stop their threads
        for w in self.child_windows:
            w.should_stop = True

        # Stop camera acquisition
        if self.mc.acquiring:
            self.mc.off()

        self.mc.disconnect()

        # Close main window and exit Python program
        self.root.quit()
        self.root.destroy()
        exit()

    def update_main(self):
        # Update real time counter and determine display fps
        now = datetime.now()

        if self._mem_baseline is None:
            self._mem_baseline = psutil.virtual_memory().percent

        if self.mc.acquiring:

            # Grab the latest frames for displaying
            self._current_buffers = self.mc.get_current_framebuffer()

            # Get an estimation of the saved data size
            if self.mc._estim_file_size is None:
                self.var_frames_saved.set(f'Saved frames: {self.mc.saved} (0 bytes)')

            elif self.mc._estim_file_size == -1:
                size = sum(sum(os.path.getsize(os.path.join(res[0], element)) for element in res[2]) for res in
                           os.walk(self.mc.full_path))
                self.var_frames_saved.set(f'Saved frames: {self.mc.saved} ({utils.pretty_size(size)})')
            else:
                saved = self.mc.saved
                size = sum(self.mc._estim_file_size * saved)
                self.var_frames_saved.set(f'Saved frames: {saved} ({utils.pretty_size(size)})')

        # Update memory pressure estimation
        self._mem_pressure += (psutil.virtual_memory().percent - self._mem_baseline) / self._mem_baseline
        self._mem_pressure_bar.config(value=self._mem_pressure)
        self._mem_baseline = psutil.virtual_memory().percent

        self._clock = now
        self.root.after(200, self.update_main)

