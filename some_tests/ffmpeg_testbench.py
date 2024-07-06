import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread, get_ident
from collections import deque
import numpy as np
from typing import List, Union
from subprocess import Popen, PIPE, STDOUT
import shlex
import time
import cv2
import platform
import sys

h = 1080
w = 1440
framerate = 200

folder = Path("./test_output")
folder.mkdir(exist_ok=True, parents=True)

event_acquiring = Event()
event_recording = Event()

executor = ThreadPoolExecutor()

nb_streams = 5

image_queues: List[deque] = []
finished_saving: List[Event] = []
videowriters: List[Union[False, subprocess.Popen]] = []

for i in range(nb_streams):
    image_queues.append(deque())
    finished_saving.append(Event())
    videowriters.append(False)

vid_length = 20

ftw = vid_length * framerate


##

class ImageGenerator(Thread):
    def __init__(self, event, framerate):
        Thread.__init__(self)
        self.allowed = event
        self.framerate = framerate

    def run(self):
        interval = 1 / self.framerate
        counts = [0] * nb_streams
        while self.allowed.wait(interval):
            for i in range(nb_streams):
                img = np.random.randint(0, 255, (h, w), dtype='<u1')
                img = cv2.putText(img, f'Frame {counts[i]}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                  2.5, (255, 255, 255), 3, cv2.LINE_AA)
                image_queues[i].append(img)
                counts[i] += 1


def init_videowriter(i: int):
    if not videowriters[i]:
        dummy_frame = np.zeros((h, w), dtype=np.uint8)
        filepath = folder / f"stream{i}.mp4"

        # TODO - Get available hardware-accelerated encoders on user's system and choose the best one automatically
        # TODO - Write a good software-based encoder command for users without GPUs
        # TODO - h265 only for now, x264 would be nice too

        if 'Linux' in platform.system():
            command = f'ffmpeg -threads 1 -y -s {w}x{h} -f rawvideo -framerate {fps} -pix_fmt gray8 -i pipe:0 -an -c:v hevc_nvenc -preset llhp -zerolatency 1 -2pass 0 -rc cbr_ld_hq -pix_fmt yuv420p -r:v {fps} {filepath.as_posix()}'
        elif 'Windows' in platform.system():
            command = f'ffmpeg -threads 1 -y -s {w}x{h} -f rawvideo -framerate {fps} -pix_fmt gray8 -i pipe:0 -an -c:v hevc_nvenc -preset llhp -zerolatency 1 -2pass 0 -rc cbr_ld_hq -pix_fmt yuv420p -r:v {fps} {filepath.as_posix()}'
        elif 'Darwin' in platform.system():
            command = f'ffmpeg -threads 1 -y -s {w}x{h} -f rawvideo -framerate {fps} -pix_fmt gray8 -i pipe:0 -an -c:v hevc_videotoolbox -realtime 1 -q:v 100 -tag:v hvc1 -pix_fmt yuv420p -r:v {fps} {filepath.as_posix()}'
            # command = f'ffmpeg -threads 1 -y -s {w}x{h} -f rawvideo -framerate {fps} -pix_fmt gray8 -i pipe:0 -an -c:v h264_videotoolbox -realtime 1 -q:v 100 -pix_fmt yuv420p -r:v {fps} {filepath.as_posix()}'
        else:
            raise SystemExit('[ERROR] Unsupported platform')

        ON_POSIX = 'posix' in sys.builtin_module_names
        # p = Popen(shlex.split(command), stdin=PIPE, close_fds=ON_POSIX)       # Debug mode (stderr/stdout on)
        p = Popen(shlex.split(command), stdin=PIPE, stdout=False, stderr=False, close_fds=ON_POSIX)
        p.stdin.write(dummy_frame.tobytes())
        videowriters[i] = p


def close_videowriter(i: int):
    if videowriters[i]:
        videowriters[i].stdin.close()
        videowriters[i].wait()
    videowriters[i] = False


def writer_test(i: int):
    fr = 0
    started_saving = False
    while event_acquiring.is_set():
        if event_recording.is_set():
            init_videowriter(i)

            if not started_saving:
                started_saving = True

            if image_queues[i]:
                frame = image_queues[i].popleft()
                videowriters[i].stdin.write(frame.tobytes())

            else:
                time.sleep(0.01)
        else:
            if started_saving:
                if image_queues[i]:
                    frame = image_queues[i].popleft()
                    videowriters[i].stdin.write(frame.tobytes())
                else:
                    close_videowriter(i)
                    finished_saving[i].set()
                    started_saving = False
            else:
                event_recording.wait()

    print(f"Thread {get_ident()}: {fr}/{ftw}")


##

event_acquiring.set()

generator = ImageGenerator(event_acquiring, framerate)
generator.start()

for i in range(5):
    executor.submit(writer_test, i)

time.sleep(0.5)

event_recording.set()
time.sleep(vid_length)
event_recording.clear()

[e.wait() for e in finished_saving]

event_acquiring.clear()

##

# Check video has been written fine
cap = cv2.VideoCapture((folder / f"stream{0}.mp4").as_posix())
nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

cap.set(cv2.CAP_PROP_POS_FRAMES, nb_frames - 1)
r, frametest = cap.read()

print(f"{nb_frames} frames in video")