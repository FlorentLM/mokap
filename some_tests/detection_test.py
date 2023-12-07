from mokap.core import Manager, MotionDetector
from sys import platform
import subprocess
import cv2

def get_webcam_devices():

    if platform == "linux" or platform == "linux2":
        result = subprocess.run(["ls", "/dev/"],
                                 stdout=subprocess.PIPE,
                                 text=True)
        devices = [int(v.replace('video', '')) for v in result.stdout.split() if 'video' in v]
    elif platform == "darwin":
        print('macOS: TODO')
    elif platform == "win32":
        print('macOS: TODO')
    else:
        raise OSError('Unsupported OS')

    working_ports = []

    prev_log_level = cv2.setLogLevel(0)

    for dev in devices:

        cap = cv2.VideoCapture(dev)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                working_ports.append(dev)

    cv2.setLogLevel(prev_log_level)
    return working_ports


##

md = MotionDetector()

use_webcam = True

if use_webcam:
    available_cameras = get_webcam_devices()
    cap = cv2.VideoCapture(available_cameras[0])
else:
    mgr = Manager()
    mgr.connect()
    mgr.exposure = 15000
    mgr.framerate = 10
    mgr.binning = 2
    mgr.on()

cv2.namedWindow('Video')
cv2.namedWindow('Detection')


dims = None
while True:

    if use_webcam:
        _, frame = cap.read()
    else:
        frame = mgr.get_current_framearray(0)

    if dims is None:
        dims = frame.shape[0] * frame.shape[1]

    detection = md.process(frame)
    value = (detection.astype(bool).sum() / dims) * 100

    detection = cv2.putText(detection, f'Moving pixels: {value:.2f}%',
                            (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2, cv2.LINE_AA)

    cv2.imshow('Video', frame)
    cv2.imshow('Detection', detection)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cv2. destroyAllWindows()

if use_webcam:
    cap.release()
else:
    mgr.off()
