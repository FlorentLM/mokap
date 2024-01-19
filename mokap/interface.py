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


def hex_to_hls(hex_str: str):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 3:
        hex_str = ''.join([c + c for c in hex_str])
    return colorsys.rgb_to_hls(*tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4)))


def hls_to_hex(h, l, s):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    new_hex = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
    return new_hex


##


def compute_windows_size(source_dims, screen_dims):
    screenh, screenw = screen_dims[:2]
    sourceh, sourcew = source_dims[:2].max(axis=1)

    # For 2x3 screen grid
    max_window_w = screenw // 3 - GUI.WBORDERS_W
    max_window_h = (screenh - GUI.TASKBAR_H) // 2 - GUI.WBORDERS_H - VideoWindow.INFO_PANEL_MIN_H

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
        self.window.protocol("WM_DELETE_WINDOW", self.parent.quit)

        # Init state
        self._counter = 0
        self._clock = datetime.now()
        self._fps = 0
        self._brightness_var = 0

        self.visible = Event()
        self.visible.set()

        self._display_focus = Event()
        self._display_focus.clear()

        self._kernel = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]], dtype=np.uint8)

        self._imgfnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMonoBold.ttf", 30)

        # Variable texts
        self.title_var = tk.StringVar()
        self.resolution_var = tk.StringVar()
        self.exposure_var = tk.StringVar()
        self.capture_fps_var = tk.StringVar()
        self.display_fps_var = tk.StringVar()
        self.display_brightness_var = tk.StringVar()

        # Create the video image frame
        imagetk = ImageTk.PhotoImage(image=Image.fromarray(np.zeros(self.video_dims, dtype='<u1')))

        self.video_canvas = tk.Label(self.window, bg="black", compound='center')
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        self.video_canvas.imgtk = imagetk
        self.video_canvas.configure(image=imagetk)

        title = tk.Label(self.window, fg=self.color_2, bg=self.color, anchor=tk.N, justify='center',
                         font=parent.bold, textvariable=self.title_var)
        title.pack(fill=tk.BOTH)

        # Create info frame
        infoframe = tk.Frame(self.window, height=self.INFO_PANEL_MIN_H, width=w)
        infoframe.pack(padx=8, pady=8, side=tk.LEFT, fill=tk.BOTH, expand=True)

        resolution_label = tk.Label(infoframe, fg="black", anchor=tk.W, justify='left',
                                    font=parent.regular, textvariable=self.resolution_var)
        resolution_label.pack(fill=tk.BOTH)

        exposure_label = tk.Label(infoframe, fg="black", anchor=tk.W, justify='left',
                                  font=parent.regular, textvariable=self.exposure_var)
        exposure_label.pack(fill=tk.BOTH)

        capture_fps_label = tk.Label(infoframe, fg="black", anchor=tk.W, justify='left',
                                     font=parent.regular, textvariable=self.capture_fps_var)
        capture_fps_label.pack(fill=tk.BOTH)

        display_fps_label = tk.Label(infoframe, fg="black", anchor=tk.W, justify='left',
                                     font=parent.regular, textvariable=self.display_fps_var)
        display_fps_label.pack(fill=tk.BOTH)

        display_brightness_label = tk.Label(infoframe, fg="black", anchor=tk.W, justify='left',
                                            font=parent.regular, textvariable=self.display_brightness_var)
        display_brightness_label.pack(fill=tk.BOTH)

        controls_layout_frame = tk.Frame(self.window, height=self.INFO_PANEL_MIN_H, width=w)
        controls_layout_frame.pack(padx=8, pady=8, side=tk.LEFT, fill=tk.BOTH, expand=True)

        framerate_frame = tk.Frame(controls_layout_frame)
        framerate_frame.pack(side=tk.TOP, fill=tk.X, expand=True)
        framerate_slider_label = tk.Label(framerate_frame, fg="black", anchor=tk.SE, justify='right',
                                          font=parent.regular, text='Framerate (fps):')
        framerate_slider_label.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        self.framerate_slider = tk.Scale(framerate_frame, from_=1, to=220, orient=tk.HORIZONTAL,
                                         width=9, sliderlength=9)
        self.framerate_slider.set(self.parent.mgr.cameras[self.idx].framerate)
        self.framerate_slider.bind("<ButtonRelease-1>", self.update_framerate)
        self.framerate_slider.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        apply_fps_all_button = tk.Button(framerate_frame, text="Apply to all", font=self.parent.regular,
                                         command=self._update_fps_all_cams)
        apply_fps_all_button.pack(padx=2, fill=tk.BOTH, side=tk.LEFT)

        exposure_frame = tk.Frame(controls_layout_frame)
        exposure_frame.pack(side=tk.TOP, fill=tk.X, expand=True)
        exposure_slider_label = tk.Label(exposure_frame, fg="black", anchor=tk.SE, justify='right',
                                         font=parent.regular, text='Exposure (µs)  :')

        exposure_slider_label.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        self.exposure_slider = tk.Scale(exposure_frame, from_=4300, to=25000,
                                        orient=tk.HORIZONTAL, width=9, sliderlength=9)
        self.exposure_slider.set(self.parent.mgr.cameras[self.idx].exposure)
        self.exposure_slider.bind("<ButtonRelease-1>", self.update_exposure)
        self.exposure_slider.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        apply_exp_all_button = tk.Button(exposure_frame, text="Apply to all", font=self.parent.regular,
                                         command=self._update_exp_all_cams)
        apply_exp_all_button.pack(padx=2, fill=tk.BOTH, side=tk.LEFT)

        # Set static vars
        self.title_var.set(f'{self._cam_name.title()} camera')
        self.window.title(self.title_var.get())
        self.resolution_var.set(
            f"Resolution : {self.parent.mgr.cameras[self.idx].height}×{self.parent.mgr.cameras[self.idx].width} px")
        self.exposure_var.set(f"Exposure   : {self.parent.mgr.cameras[self.idx].exposure} µs")

        show_focus_button = tk.Button(controls_layout_frame, text="Show focus", font=self.parent.regular,
                                      command=self._toggle_focus_display)
        show_focus_button.pack(padx=2, fill=tk.BOTH, side=tk.LEFT)

    @property
    def whxy(self):
        dims = self.window.geometry()
        wh, x, y = dims.split('+')
        w, h = wh.split('x')
        return [int(s) for s in [w, h, x, y]]

    def update_framerate(self, event=None):
        new_fps = self.framerate_slider.get()
        self.parent.mgr.cameras[self.idx].framerate = new_fps
        self.framerate_slider.set(self.parent.mgr.cameras[self.idx].framerate)

        self.parent._capture_clock = datetime.now()
        self.parent.start_indices[:] = self.parent.mgr.indices

    def update_exposure(self, event=None):
        new_exp = self.exposure_slider.get()
        self.parent.mgr.cameras[self.idx].exposure = new_exp
        self.framerate_slider.set(self.parent.mgr.cameras[self.idx].framerate)
        self.exposure_var.set(f"Exposure   : {self.parent.mgr.cameras[self.idx].exposure} µs")
        self.update_framerate(event)

    def _update_fps_all_cams(self):
        for window in self.parent.video_windows:
            if window is not self:
                new_fps = self.framerate_slider.get()
                window.framerate_slider.set(new_fps)
                window.update_framerate()

    def _update_exp_all_cams(self):
        for window in self.parent.video_windows:
            if window is not self:
                new_exp = self.exposure_slider.get()
                window.exposure_slider.set(new_exp)
                window.update_exposure()

    def _toggle_focus_display(self):
        if self._display_focus.is_set():
            self._display_focus.clear()
        else:
            self._display_focus.set()

    def _update_fps_vars(self):

        if self.parent.mgr.acquiring:
            cap_fps = self.parent.capture_fps[self.idx]

            if cap_fps > 1000 or cap_fps <= 0:
                self.capture_fps_var.set(f"Acquisition: ... fps")
            else:
                self.capture_fps_var.set(f"Acquisition: {cap_fps:.2f} fps")

            self.display_brightness_var.set(
                f"Brightness : {self._brightness_var:.2f}%")
        else:
            self.capture_fps_var.set(f"Acquisition: Off")
            self.display_brightness_var.set(f"Brightness : ∅")

        self.display_fps_var.set(f"Display    : {self._fps:.2f} fps")

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
            frame = np.stack([fo, fo, frame]).T.swapaxes(0, 1)

        y, x = self.video_dims
        x2, y2 = x // 2, y // 2

        img_pillow = Image.fromarray(frame).convert('RGB')
        img_pillow = img_pillow.resize((x, y))

        d = ImageDraw.Draw(img_pillow)
        d.line((0, y2, x, y2), fill='white')
        d.line((x2, 0, x2, y), fill='white')

        d.text((x2 - 100, y - 40), self.parent.recording_var.get(), font=self._imgfnt, fill='red')

        imgtk = ImageTk.PhotoImage(image=img_pillow)
        try:
            self.video_canvas.configure(image=imgtk)
            self.imagetk = imgtk
        except Exception as e:
            print(e)
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
            self._update_fps_vars()

            self.visible.wait()

    def toggle_visibility(self):
        if self.visible.is_set():
            self.visible.clear()
            self.window.withdraw()
        else:
            self.visible.set()
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

    # These depend on the OS and the user's theme...
    WBORDER_TH = 4  # [KDE System Settings > Appearance > Window Decorations > Window border size]
    WTITLEBAR_H = 31  # [KDE System Settings > Appearance > Window Decorations]
    TASKBAR_H = 44  # [KDE Plasma 'Task Manager' widget > Panel height]

    WBORDERS_W = WBORDER_TH * 2
    WBORDERS_H = WTITLEBAR_H + WBORDER_TH * 2

    def __init__(self, mgr):

        # Detect monitors and pick the default one
        self._selected_monitor = None
        self._monitors = screeninfo.get_monitors()
        self.set_monitor()

        # Set up root window
        self.root = tk.Tk()
        self.root.wait_visibility(self.root)
        self.root.title("Controls")
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.bind('<KeyPress>', self._handle_keypress)

        # Set up fonts
        self.bold = font.Font(weight='bold', size=10)
        self.regular = font.Font(size=9)

        # Init default things
        self.mgr = mgr
        self.editing_disabled = True

        self._capture_fps = np.zeros(self.mgr.nb_cameras, dtype=np.uint32)
        self._capture_clock = datetime.now()
        self._now_indices = np.zeros(self.mgr.nb_cameras, dtype=np.uint32)

        self.start_indices = np.zeros(self.mgr.nb_cameras, dtype=np.uint32)

        self._counter = 0

        self.recording_var = tk.StringVar()
        self.userentry_var = tk.StringVar()
        self.applied_name_var = tk.StringVar()

        self.recording_var.set('')
        self.userentry_var.set('')
        self.applied_name_var.set('')

        # Compute optimal video windows sizes
        self._max_videowindows_dims = compute_windows_size(self.source_dims, self.screen_dims)

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

        # Deduce control window size and position
        w = self.max_videowindows_dims[1]
        h = int(self.max_videowindows_dims[0] // 2.1)
        x = (self.screen_dims[1] // 2 - w // 2) + self.screen_dims[2]
        y = (self.max_videowindows_dims[0] * 2) + self.screen_dims[3]  # TODO - improve positioning
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        # Create control window
        self._create_controls()

        self.update()  # Called once to init
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
    def controlwindow_dims(self):
        return np.array([self.root.winfo_height(), self.root.winfo_width()])

    @property
    def source_dims(self):
        return np.array([(cam.height, cam.width) for cam in self.mgr.cameras]).T

    @property
    def screen_dims(self):
        monitor = self._selected_monitor
        return np.array([monitor.height, monitor.width, monitor.x, monitor.y], dtype=np.uint32)

    def set_monitor(self, idx=None):
        if len(self._monitors) > 1 and idx is None:
            self._selected_monitor = next(m for m in self._monitors if m.is_primary)
        elif len(self._monitors) > 1 and idx is not None:
            self._selected_monitor = self._monitors[idx]
        else:
            self._selected_monitor = self._monitors[0]

    @property
    def whxy(self):
        dims = self.root.geometry()
        wh, x, y = dims.split('+')
        w, h = wh.split('x')
        return [int(s) for s in [w, h, x, y]]

    def nothing(self):
        pass

    def _create_controls(self):

        ipadx = 2
        ipady = 2
        padx = 2
        pady = 2

        left_half = tk.Frame(self.root)
        left_half.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

        right_half = tk.Frame(self.root)
        right_half.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

        # LEFT HALF

        acquisition_label = tk.Label(left_half, text='Acquisition:', anchor=tk.W)
        acquisition_label.pack(ipadx=ipadx, ipady=ipady, padx=padx, pady=pady, fill=tk.X)

        startstop_frame = tk.Frame(left_half, height=50)
        startstop_frame.pack(ipadx=ipadx, ipady=ipady, padx=padx, pady=pady, fill=tk.X)

        self.button_start = tk.Button(startstop_frame, text="Start", anchor=tk.CENTER,
                                      font=self.regular,
                                      command=self.gui_acquire,
                                      state='normal')
        self.button_start.pack(ipadx=ipadx, ipady=ipady, padx=padx, fill=tk.BOTH, side=tk.LEFT, expand=True)

        self.button_stop = tk.Button(startstop_frame, text="Stop", anchor=tk.CENTER,
                                     font=self.regular,
                                     command=self.gui_snow,
                                     state='disabled')
        self.button_stop.pack(ipadx=ipadx, ipady=ipady, padx=padx, fill=tk.BOTH, side=tk.LEFT, expand=True)

        #

        pathname_frame = tk.Frame(left_half, height=60)
        pathname_frame.pack(ipadx=ipadx, ipady=ipady, padx=padx, pady=pady, fill=tk.X)

        pathname_label = tk.Label(pathname_frame, text='Name: ', anchor=tk.W)
        pathname_label.pack(ipadx=ipadx, fill=tk.X)

        self.pathname_textbox = tk.Entry(pathname_frame, bg='white', fg='black', textvariable=self.userentry_var,
                                         font=self.regular, state='disabled')
        self.pathname_textbox.pack(padx=padx, fill=tk.BOTH, side=tk.LEFT, expand=True)

        self.pathname_button = tk.Button(pathname_frame,
                                         text="Edit", font=self.regular, command=self.gui_toggle_set_name)
        self.pathname_button.pack(ipadx=ipadx, ipady=ipady, padx=padx, fill=tk.BOTH, side=tk.LEFT)

        current_name = tk.Label(left_half, textvariable=self.applied_name_var, anchor=tk.W)
        current_name.pack(ipadx=ipadx, fill=tk.BOTH, expand=True)

        #

        self.button_recpause = tk.Button(left_half, text="● Record (Space)", anchor=tk.CENTER, font=self.bold,
                                         command=self.gui_toggle_recording,
                                         state='disabled')
        self.button_recpause.pack(ipadx=ipadx, ipady=ipady, padx=padx, pady=pady * 2, fill=tk.X, expand=True)

        self.button_exit = tk.Button(left_half, text="Exit (Esc)", anchor=tk.CENTER, font=self.bold,
                                     bg='#EE4457', fg='White',
                                     command=self.quit)
        self.button_exit.pack(ipadx=ipadx, ipady=ipady, padx=padx, pady=pady, anchor=tk.SE)

        # RIGHT HALF

        visibility_label = tk.Label(right_half, text='Show previews:', anchor=tk.W)
        visibility_label.pack()

        windows_list_frame = tk.Frame(right_half)
        windows_list_frame.pack()

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
            vis_var.set(1)
            vis_checkbox.pack(ipadx=ipadx, ipady=ipady, padx=padx, pady=pady, fill=tk.X, expand=True)
            self._vis_checkboxes.append(vis_var)

        monitors_label = tk.Label(right_half, text='Active monitor:', anchor=tk.W)
        monitors_label.pack()

        self.monitors_buttons = tk.Canvas(right_half, width=120, height=60)
        self.update_monitors_buttons()
        for i, m in enumerate(self._monitors):
            self.monitors_buttons.tag_bind(f'screen_{i}', '<Button-1>', lambda _, val=i: self.screen_update(val))
        self.monitors_buttons.pack(ipadx=ipadx, ipady=ipady, padx=padx, pady=pady)

    def update_monitors_buttons(self):
        self.monitors_buttons.delete("all")

        for i, m in enumerate(self._monitors):
            w, h, x, y = m.width // 80, m.height // 80, m.x // 80, m.y // 80
            if m.name == self._selected_monitor.name:
                a = '#515151'
            else:
                a = '#c0c0c0'
            self.monitors_buttons.create_rectangle(x + 2, y + 2, x + 2 + w, y + 2 + h, fill=a, outline='',
                                                   tag=f'screen_{i}')

    def screen_update(self, val):
        old_monitor_x, old_monitor_y = self._selected_monitor.x, self._selected_monitor.y

        self.set_monitor(val)
        self.update_monitors_buttons()

        w, h, x, y = self.whxy
        new_x = x + self._selected_monitor.x - old_monitor_x
        new_y = y + self._selected_monitor.y - old_monitor_y

        self.root.geometry(f'{w}x{h}+{new_x}+{new_y}')
        for window in self.video_windows:
            w, h, x, y = window.whxy
            new_x = x + self._selected_monitor.x - old_monitor_x
            new_y = y + self._selected_monitor.y - old_monitor_y

            window.window.geometry(f'{w}x{h}+{new_x}+{new_y}')

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
        self.applied_name_var.set(f'(will save to ./{self.mgr.savepath})')

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

            self.recording_var.set('[Recording]')
            self.button_recpause.config(text="■ Stop (Space)")

    def gui_pause(self):
        if self.mgr.recording:
            self.mgr.pause()

            self.recording_var.set('')
            self.button_recpause.config(text="● Record (Space)")

    def gui_toggle_recording(self):
        if self.mgr.acquiring:
            if not self.mgr.recording:
                self.gui_recording()
            else:
                self.gui_pause()

    def gui_acquire(self):

        if self.mgr.savepath is None:
            self.gui_set_name()

        if not self.mgr.acquiring:
            self.mgr.on()

            self._capture_clock = datetime.now()
            self.start_indices[:] = self.mgr.indices

            self.button_start.config(state="disabled")
            self.button_stop.config(state="normal")
            self.button_recpause.config(state="normal")

    def gui_snow(self):

        if self.mgr.acquiring:
            self.gui_pause()
            self.mgr.off()

            self._capture_fps = np.zeros(self.mgr.nb_cameras, dtype=np.uintc)

            self.button_start.config(state="normal")
            self.button_stop.config(state="disabled")

            self.button_recpause.config(state="disabled")

            self.userentry_var.set('')
            self.applied_name_var.set('')

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
            self._capture_fps = (self._now_indices - self.start_indices) / capture_dt

            self._current_buffers = self.mgr.get_current_framebuffer()

        self._counter += 1

        self.root.after(100, self.update)
