from mokap.core import Manager, MotionDetector, cv2

md = MotionDetector()

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
mgr.off()