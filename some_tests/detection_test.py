from mokap.core import MotionDetector


threshold = 1
preview = True
lag = 2

md = MotionDetector(cam_id=0, thresh=threshold, lag=lag, preview=preview)
md2 = MotionDetector(cam_id=2, thresh=threshold, lag=lag, preview=preview)
md3 = MotionDetector(cam_id=4, thresh=threshold, lag=lag, preview=preview)

md.start()
md2.start()
md3.start()

