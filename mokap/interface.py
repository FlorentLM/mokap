import sys
import time
import tkinter as tk
import tkinter.font as font
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime
from threading import Thread, Event
import sv_ttk
from matplotlib import style as mplstyle
mplstyle.use('ggplot')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

THEME = 'light'


class GraphWidget:
    def __init__(self, canvas, master):

        self.fig = Figure(figsize=(12, 8), dpi=100)
        if 'light' in THEME.lower():
            self.fig.patch.set_facecolor('#fafafa')
        elif 'dark' in THEME.lower():
            self.fig.patch.set_facecolor('#000000')
        self.axes = self.fig.subplots(1, 3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        plt.tight_layout()
        self.master = master

        self.tail_len = 50

        self.reset()

    def reset(self):

        self.x = np.arange(self.tail_len)

        self.disp_fps = np.zeros(self.tail_len)
        self.capt_fps = np.zeros(self.tail_len)
        self.abs_dif_values = np.zeros(self.tail_len)

        self.line0, = self.axes[0].plot(self.x, self.disp_fps, color='#4cdccf', alpha=0.75, lw=2)
        self.axes[0].set_xlabel('Display FPS', fontsize=9)
        self.axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes[0].yaxis.set_ticklabels([])
        self.axes[0].xaxis.set_ticklabels([])
        self.axes[0].xaxis.set_ticks_position('none')
        self.axes[0].yaxis.set_ticks_position('none')

        self.line1, = self.axes[1].plot(self.x, self.capt_fps, color='#f3a0f2', alpha=0.75, lw=2)
        self.axes[1].set_xlabel('Capture FPS', fontsize=9)
        self.axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes[1].yaxis.set_ticklabels([])
        self.axes[1].xaxis.set_ticklabels([])
        self.axes[1].xaxis.set_ticks_position('none')
        self.axes[1].yaxis.set_ticks_position('none')

        self.line2, = self.axes[2].plot(self.x, self.abs_dif_values, color='#f5b14c', alpha=0.75, lw=2)
        self.axes[2].set_xlabel('Ant detection', fontsize=9)
        self.axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
        self.axes[2].yaxis.set_ticklabels([])
        self.axes[2].xaxis.set_ticklabels([])
        self.axes[2].xaxis.set_ticks_position('none')
        self.axes[2].yaxis.set_ticks_position('none')

        self._count = 0

    def update(self):

        if self._count % 5 == 0:
            if self._count < self.tail_len:
                self.x[self._count] = self.master.count

                self.disp_fps[:] = np.mean([w.display_fps for w in self.master.video_windows])
                self.capt_fps[:] = np.mean(self.master.capture_fps)
                self.abs_dif_values[:] = np.mean(self.master.absdif)

            else:
                self.x = np.roll(self.x, -1)
                self.x[-1] = self.master.count

                self.disp_fps = np.roll(self.disp_fps, -1)
                self.capt_fps = np.roll(self.capt_fps, -1)
                self.abs_dif_values = np.roll(self.abs_dif_values, -1)

                self.disp_fps[-1] = self.master.video_windows[0].display_fps
                self.capt_fps[-1] = np.mean(self.master.capture_fps)
                self.abs_dif_values[-1] = np.mean(self.master.absdif)

            xmin, xmax = self.x.min(), self.x.max()

            self.line0.set_ydata(self.disp_fps)
            self.line0.set_xdata(self.x)
            self.axes[0].set(xlim=(xmin, xmax))
            ymax = self.disp_fps.max() * 2
            self.axes[0].set(ylim=(0, ymax if ymax > 0 else 1))
            [txt.remove() for txt in self.axes[0].texts]
            self.axes[0].text(xmax - 100, self.disp_fps[-1] + 10, f"{self.disp_fps[-1]:.2f}", color='#4cdccf', alpha=0.75)

            if self.master.mgr.acquiring:
                self.line1.set_ydata(self.capt_fps)
                self.line1.set_xdata(self.x)
                self.axes[1].set(xlim=(xmin, xmax))
                ymax = self.capt_fps.max() * 2
                self.axes[1].set(ylim=(0, ymax if ymax > 0 else 1))
                [txt.remove() for txt in self.axes[1].texts]
                self.axes[1].text(xmax - 100, self.capt_fps[-1] + 10, f"{self.capt_fps[-1]:.2f}", color='#f3a0f2', alpha=0.75)

            if self.master._autodetection_enabled.is_set():
                self.line2.set_ydata(self.abs_dif_values)
                self.axes[2].set(xlim=(xmin, xmax))
                self.line2.set_xdata(self.x)
                ymax = self.abs_dif_values.max() * 2
                self.axes[2].set(ylim=(0, ymax if ymax > 0 else 1))
                [txt.remove() for txt in self.axes[2].texts]
                self.axes[2].text(xmax - 100, self.abs_dif_values[-1] + 1, f"{self.abs_dif_values[-1]:.2f}", color='#f5b14c',
                                  alpha=0.75)

            self.canvas.draw()
            self.canvas.flush_events()

        self._count += 1

def compute_windows_size(source_dims, screen_dims):

    screenh, screenw = screen_dims
    sourceh, sourcew = source_dims.max(axis=1)

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

    INFO_PANEL_MIN_H = 150   # in pixels

    def __init__(self, parent):

        self.window = tk.Toplevel()

        self.parent = parent

        self.idx = len(VideoWindow.videowindows_ids)
        VideoWindow.videowindows_ids.append(self.idx)

        self._source_shape = (self.parent.mgr.cameras[self.idx].height, self.parent.mgr.cameras[self.idx].width)
        self._pos = self.parent.mgr.cameras[self.idx].pos
        
        match self._pos:
            case 'south-west' | 'virtual_4':
                x = 0
                y = 0
                col_fg = '#515151'
                col_bg = '#ffffff'
            case 'south-east' | 'virtual_3':
                x = 0
                y = parent.screen_dims[0] - parent.max_videowindows_dims[0]
                col_fg = '#000000'
                col_bg = '#fff153'
            case 'north-west' | 'virtual_2':
                x = parent.screen_dims[1] - parent.max_videowindows_dims[1]
                y = 0
                col_fg = '#ffffff'
                col_bg = '#fb95b5'
            case 'north-east' | 'virtual_1':
                x = parent.screen_dims[1] - parent.max_videowindows_dims[1]
                y = parent.screen_dims[0] - parent.max_videowindows_dims[0]
                col_fg = '#ffffff'
                col_bg = '#989898'
            case 'top' | 'virtual_0':
                x = parent.screen_dims[1] // 2 - parent.max_videowindows_dims[1] // 2
                y = parent.screen_dims[0] // 2 - parent.max_videowindows_dims[0] // 2 - GUI.WTITLEBAR_H // 2
                col_fg = '#ffffff'
                col_bg = '#b91025'
            case _:
                x = parent.screen_dims[1] // 2 - parent.max_videowindows_dims[1] // 2
                y = parent.screen_dims[0] // 2 - parent.max_videowindows_dims[0] // 2 - GUI.WTITLEBAR_H // 2
                col_fg = '#ffffff'
                col_bg = '#F0A108'

        h, w = parent.max_videowindows_dims
        self.window.geometry(f"{w}x{h}+{x}+{y}")
        self.window.wm_aspect(w, h, w, h)
        self.window.protocol("WM_DELETE_WINDOW", self.parent.quit)

        # Variable texts
        self.title_var = tk.StringVar()
        self.resolution_var = tk.StringVar()
        self.exposure_var = tk.StringVar()
        self.capture_fps_var = tk.StringVar()
        self.display_fps_var = tk.StringVar()

        # Create the video image frame
        self.imagecontainer = tk.Label(self.window, fg='#FF3C21', bg="black",
                                       textvariable=parent.recording_var, font=parent.bold,
                                       compound='center')
        self.imagecontainer.pack(fill=tk.BOTH, expand=True)

        # Create info frame
        infoframe = tk.Frame(self.window, height=self.INFO_PANEL_MIN_H)
        infoframe.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(infoframe, fg=col_fg, bg=col_bg, anchor=tk.N, justify='center',
                         font=parent.bold, textvariable=self.title_var)
        title.pack(fill=tk.BOTH)

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

        # Init state
        self._counter = 0
        self._clock = datetime.now()
        self._fps = 0
        self.visible = Event()
        self.visible.set()
        
    def _update_vars(self):
        self.title_var.set(f'{self._pos.title()} camera')
        self.window.title(self.title_var.get())
        self.resolution_var.set(f"Resolution: {self.parent.mgr.cameras[self.idx].height}??{self.parent.mgr.cameras[self.idx].width} px")
        self.exposure_var.set(f"Exposure: {self.parent.mgr.cameras[self.idx].exposure} ms")
        if self.parent.mgr.acquiring:
            self.capture_fps_var.set(f"Acquisition: {self.parent.capture_fps[self.idx]:.2f} fps")
        else:
            self.capture_fps_var.set(f"Acquisition: Off")
        self.display_fps_var.set(f"Display: {self._fps:.2f} fps")

    def _update_video(self):
        frame = None

        if self.parent.mgr.acquiring and self.parent.current_buffers is not None:
            if 'top' in self._pos and self.parent.show_diff:
                frame = np.frombuffer(self.parent.absdif, dtype=np.uint8).reshape(self._source_shape)
            else:
                frame = np.frombuffer(self.parent.current_buffers[self.idx], dtype=np.uint8).reshape(self._source_shape)

        if frame is not None:
            imgdata = Image.fromarray(frame)
            img = imgdata.resize((self.video_dims[1], self.video_dims[0]))
        else:
            img = Image.fromarray(np.random.randint(0, 255, self.video_dims, dtype='<u1'))
            # img = Image.fromarray(np.zeros(shape=self.video_dims, dtype='<u1'))

        imgtk = ImageTk.PhotoImage(image=img)
        self.imagecontainer.imgtk = imgtk
        self.imagecontainer.configure(image=imgtk)

    def update(self):

        while self.visible.is_set():

            # Update display fps counter
            now = datetime.now()
            dt = (now - self._clock).total_seconds()

            if dt >= 0.5:
                self._fps = self._counter / dt

            self._counter += 1

            self._update_video()
            self._update_vars()

        print('Exited thread')

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
    WBORDER_TH = 4      # [KDE System Settings > Appearance > Window Decorations > Window border size]
    WTITLEBAR_H = 31    # [KDE System Settings > Appearance > Window Decorations]
    TASKBAR_H = 44      # [KDE Plasma 'Task Manager' widget > Panel height]

    WBORDERS_W = WBORDER_TH * 2
    WBORDERS_H = WTITLEBAR_H + WBORDER_TH * 2

    def __init__(self, mgr):

        # Set up root window
        self.root = tk.Tk()
        self.root.wait_visibility(self.root)
        self.root.title("Controls")
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.bind('<KeyPress>', self._handle_keypress)

        # Set up fonts
        self.bold = font.Font(weight='bold', size=10)
        self.regular = font.Font(size=9)
        # print(list(font.families()).sort())

        # Init default things
        self.mgr = mgr
        self.editing_disabled = True
        self._capture_fps = np.zeros(self.mgr.nb_cameras, dtype=np.uintc)
        self._capture_clock = datetime.now()
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
        self._autodetection_thread = None
        # self._graph_thread = None
        # self.graph_shown = Event()
        self._autodetection_enabled = Event()
        self._show_diff = Event()

        # Create video windows
        self.video_windows = []
        self.windows_threads = []
        for i in range(self.mgr.nb_cameras):
            vw = VideoWindow(parent=self)
            self.video_windows.append(vw)

            t = Thread(target=vw.update, args=(), daemon=False)
            t.start()
            self.windows_threads.append(t)

        self._absdif = bytearray(b'\0' * self.mgr.cameras['top'].height * self.mgr.cameras['top'].width)

        # Deduce control window size and position
        w = self.max_videowindows_dims[1]
        h = self.max_videowindows_dims[0] // 2
        x = self.screen_dims[1] // 2 - w // 2
        y = 0
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        # Create control window
        self._create_controls()

        sv_ttk.set_theme(THEME.lower())

        self.update()               # Called once to init
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
        return np.array([self.root.winfo_screenheight(), self.root.winfo_screenwidth()])

    def _create_controls(self):

        ipadx = 2
        ipady = 2
        padx = 2
        pady = 2

        left_half = tk.Frame(self.root)
        left_half.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

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

        self.button_recpause = tk.Button(left_half, text="??? Record (Space)", anchor=tk.CENTER, font=self.bold,
                                         command=self.gui_toggle_recording,
                                         state='disabled')
        self.button_recpause.pack(ipadx=ipadx, ipady=ipady, padx=padx, pady=pady * 2, fill=tk.X, expand=True)

        #
        reference_layout_frame = tk.Frame(left_half, height=50)
        reference_layout_frame.pack(ipadx=ipadx, ipady=ipady, padx=padx, pady=pady, fill=tk.X)

        reference_buttons_frame = tk.Frame(reference_layout_frame)
        reference_buttons_frame.pack(fill=tk.X)

        self.button_acqref = tk.Button(reference_buttons_frame, text="Acquire ref.", anchor=tk.CENTER,
                                       font=self.regular,
                                       command=self.gui_acquire_ref,
                                       state='disabled')
        self.button_acqref.pack(ipadx=ipadx, ipady=ipady, padx=padx, fill=tk.X, side=tk.LEFT, expand=True)

        self.button_clearref = tk.Button(reference_buttons_frame, text="Clear ref.", anchor=tk.CENTER, font=self.regular,
                                       command=self.gui_clear_ref,
                                       state='disabled')
        self.button_clearref.pack(ipadx=ipadx, ipady=ipady, padx=padx, fill=tk.X, side=tk.LEFT, expand=True)

        self.button_show_diff = tk.Button(reference_buttons_frame, text="Show diff.", anchor=tk.CENTER,
                                       font=self.regular,
                                       command=self.gui_toggle_show_diff,
                                       state='disabled')

        self.button_show_diff.pack(ipadx=ipadx, ipady=ipady, padx=padx, fill=tk.X, side=tk.LEFT, expand=True)

        # reference_slider_frame = tk.Frame(reference_layout_frame)
        # reference_slider_frame.pack(fill=tk.X)
        #
        # self.reference_slider = tk.Scale(reference_slider_frame, from_=0, to=255, orient='horizontal')
        # self.reference_slider.pack(fill=tk.X, expand=True)
        #
        # self.reference_allow_var = tk.BooleanVar()
        #
        # self.reference_slider_toggle = tk.Checkbutton(reference_buttons_frame, onvalue=1, offvalue=0, text="Allow recording", anchor=tk.CENTER,
        #                                font=self.regular, variable=self.reference_allow_var)
        # self.reference_slider_toggle.pack(ipadx=ipadx, ipady=ipady, padx=padx, fill=tk.X, side=tk.LEFT, expand=True)

        #

        # self.button_graph = tk.Button(left_half, text="Graph", anchor=tk.CENTER, font=self.regular,
        #                              command=self.gui_toggle_graph)
        # self.button_graph.pack(ipadx=ipadx, ipady=ipady, padx=padx, pady=pady, anchor=tk.SW)

        self.button_exit = tk.Button(left_half, text="Exit (Esc)", anchor=tk.CENTER, font=self.regular,
                                     bg='#EE4457', fg='White',
                                     command=self.quit)
        self.button_exit.pack(ipadx=ipadx, ipady=ipady, padx=padx, pady=pady, anchor=tk.SE)

        #

        # self.right_half = tk.Frame(self.root)
        #
        # graph_frame = tk.Frame(self.right_half, height=60)
        # graph_frame.pack(ipadx=ipadx, ipady=ipady, padx=padx, pady=pady, fill=tk.X)
        #
        # self.graph = GraphWidget(canvas=graph_frame, master=self)

    def _handle_keypress(self, event):
        match event.keycode:
            case 9:     # Esc
                self.quit()
            case 65:    # Space
                self.gui_toggle_recording()
            case _:
                pass

    # def gui_toggle_graph(self):
    #
    #     if self.graph_shown.is_set():
    #         self.right_half.pack_forget()
    #         self.graph_shown.clear()
    #         self.graph.reset()
    #         # self._graph_thread = None
    #     else:
    #         self.graph_shown.set()
    #         # self._graph_thread = Thread(target=self.graph.update, daemon=False)
    #         # self._graph_thread.start()
    #         self.right_half.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

    @property
    def show_diff(self):
        return self._show_diff.is_set()

    @property
    def absdif(self):
        return self._absdif

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

            self.recording_var.set('??? Recording')
            self.button_recpause.config(text="Pause (Space)")

    def gui_pause(self):
        if self.mgr.recording:
            self.mgr.pause()

            self.recording_var.set('')
            self.button_recpause.config(text="??? Record (Space)")

    def gui_toggle_recording(self):
        if self.mgr.acquiring:
            if not self.mgr.recording:
                self.gui_recording()
            else:
                self.gui_pause()

    def gui_acquire_ref(self):
        self.mgr.acquire_reference()

        while self.mgr.reference is None:
            time.sleep(0.1)
        assert self.mgr.reference is not None

        self._reference = np.copy(self.mgr.reference)

        self._autodetection_enabled.set()
        self._autodetection_thread = Thread(target=self._ant_detection, daemon=False)
        self._autodetection_thread.start()

        self.button_acqref.config(state="disabled")
        self.button_clearref.config(state="normal")
        self.button_show_diff.config(state="normal")
        self.button_recpause.config(state="disabled")

    def gui_clear_ref(self):

        self._autodetection_enabled.clear()
        self._autodetection_thread = None

        self.mgr.clear_reference()
        self._reference = None

        self._absdif[:] = b'\0' * len(self._absdif)

        self.gui_hide_diff()

        self.button_acqref.config(state="normal")
        self.button_clearref.config(state="disabled")
        self.button_show_diff.config(state="disabled")
        self.button_recpause.config(state="normal")

    def gui_show_diff(self):
        if not self._show_diff.is_set():
            self.button_show_diff.config(text='Hide diff.')
            self._show_diff.set()

    def gui_hide_diff(self):
        if self._show_diff.is_set():
            self.button_show_diff.config(text='Show diff.')
            self._show_diff.clear()

    def gui_toggle_show_diff(self):
        if self._show_diff.is_set():
            self.gui_hide_diff()
        else:
            self.gui_show_diff()

    def gui_acquire(self):

        if self.mgr.savepath is None:
            self.gui_set_name()

        if not self.mgr.acquiring:
            self.mgr.on()

            self._capture_clock = datetime.now()

            self.button_start.config(state="disabled")
            self.button_stop.config(state="normal")
            self.button_acqref.config(state="normal")
            self.button_recpause.config(state="normal")

    def gui_snow(self):

        self._autodetection_enabled.clear()
        self._autodetection_thread = None
        self.gui_hide_diff()

        if self.mgr.acquiring:

            self.gui_pause()
            self.mgr.off()

            self._capture_fps = np.zeros(self.mgr.nb_cameras, dtype=np.uintc)

            self.button_start.config(state="normal")
            self.button_stop.config(state="disabled")
            self.button_acqref.config(state="disabled")
            self.button_clearref.config(state="disabled")
            self.button_show_diff.config(state="disabled")
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

    def _ant_detection(self):

        while self._autodetection_enabled.is_set():

            # thresh = self.reference_slider.get()

            frame = np.frombuffer(self.current_buffers[0], dtype=np.uint8).reshape(self._reference.shape)

            if frame is not None and self._reference is not None:
                normalized = (frame / frame.max()) * 255
                absdif = np.abs(normalized - self._reference).astype('<u1')
                # gauss = scipy.ndimage.gaussian_filter(absdif, 0.25)
                # thresh = np.clip(np.round(absdif).astype('<u1'), thrmin, thrmax)

                spread = np.max(absdif) - np.min(absdif)

                if spread >= 100:   # TODO - make sure this value is good
                    self.gui_recording()
                else:
                    self.gui_pause()

                self._absdif[:] = absdif.data.tobytes()

    def update(self):

        if self.mgr.acquiring:
            now = datetime.now()
            capture_dt = (now - self._capture_clock).total_seconds()
            self._capture_fps = self.mgr.indices / capture_dt

            self._current_buffers = self.mgr.get_current_framebuffer()

        self._counter += 1

        # if self.graph_shown.is_set():
        #     self.graph.update()

        self.root.after(1, self.update)
