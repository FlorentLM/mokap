from scipy import ndimage
import cv2
from threading import Event, Lock
from multiprocessing import Process
from pathlib import Path
import time
import numpy as np
import pypylon.pylon as py
import mokap.fileio as files_op
from collections import deque
from mokap.hardware import SSHTrigger, BaslerCamera, setup_ulimit, enumerate_basler_devices
from mokap.utils import ensure_list
import platform


class MotionDetector:

    def __init__(self, cam_id=0, learning_rate=-100, thresh=5, lag=0, preview=False, framerate=30, silent=True):
        self._silent = silent
        self._learning_rate = learning_rate

        self._id = cam_id

        self._fgbg = cv2.createBackgroundSubtractorMOG2()

        self._kernel = np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]],
                                np.uint8)
        self._running = Event()
        self._movement = Event()

        log_path = Path(
            files_op.data_folder / f'detection_cam_{self._id}_{time.strftime("%y%m%d-%H%M%S", time.localtime())}.log')

        self._worker = Process(target=self._worker_func, args=(cam_id, thresh, lag, framerate, preview, log_path))

    def start(self):

        self._running.set()
        self._worker.start()
        time.sleep(0.1)
        if not self._silent:
            print('[INFO] Started movement detection...')

    def stop(self):
        self._running.clear()
        time.sleep(0.1)
        if not self._silent:
            print('[INFO] Stopped movement detection.')

    def _worker_func(self, cam_id, thresh, lag, framerate, preview, log_path):

        log_path.touch()

        cap = cv2.VideoCapture(cam_id)

        if not cap.isOpened():
            print("[ERROR] Camera is not open... Try again?")
            return

        success, first_frame = cap.read()
        if not success:
            print("[ERROR] Camera is not ready... Try again?")
            return

        shape = first_frame.shape

        detection_start = time.time()
        tick = time.time()

        loop_duration = 1.0 / float(framerate)
        log_every_n_seconds = 60

        initialised = False
        last_log_time = time.time()
        values_list = []

        if preview:
            cv2.namedWindow(f'Preview (Cam {cam_id})', cv2.WINDOW_NORMAL)

        while self._running.is_set():
            now = time.time()

            time_since_last_loop = now - tick
            time_since_last_log = now - last_log_time

            ret, frame = cap.read()

            if time_since_last_loop >= loop_duration:
                tick = time.time()

                detection = self.process(frame)
                value = detection.sum() / (shape[0] * shape[1]) * 10
                values_list.append(value)

                if not initialised:
                    self._movement.clear()
                    text = 'Initialising...'

                    if value < thresh and tick - detection_start >= 5:
                        initialised = True
                        text = f'{value:.2f}'
                else:
                    text = f'{value:.2f}'

                    if value >= thresh:
                        self._movement.set()
                        detection_start = time.time()
                    else:
                        if tick - detection_start >= lag:
                            self._movement.clear()

                if preview:
                    if self._movement.is_set():
                        text += ' - [ACTIVE]'

                    detection = cv2.putText(frame, text, (30, 30),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=1,
                                            color=(255, 0, 255),
                                            thickness=3)

                    cv2.imshow(f'Preview (Cam {cam_id})', detection)
                    cv2.waitKey(1)

            if time_since_last_log >= log_every_n_seconds:
                log_value = round(sum(values_list) / len(values_list), 3)
                to_log = f'{time.strftime("%d/%m/%y %H:%M:%S -", time.localtime())} {log_value}\n'

                with open(log_path, "a") as log_file:
                    log_file.write(to_log)

                values_list = []
                last_log_time = time.time()

    def process(self, frame):
        motion_mask = self._fgbg.apply(frame, self._learning_rate)

        se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        detection = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, se1)
        detection = cv2.morphologyEx(detection, cv2.MORPH_OPEN, se2)

        filtered = ndimage.gaussian_filter(detection, 1)
        _, filtered = cv2.threshold(filtered, 50, 255, cv2.THRESH_BINARY)

        return filtered

    @property
    def moves(self):
        return self._movement.is_set()
