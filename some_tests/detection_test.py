from mokap.core import MotionDetector

##

prv = True
trh = 5
lrt = -100

##

md = MotionDetector(cam_id=0, learning_rate=lrt, thresh=trh, preview=prv)
md2 = MotionDetector(cam_id=2, learning_rate=lrt, thresh=trh, preview=prv)
md3 = MotionDetector(cam_id=4, learning_rate=-lrt, thresh=trh, preview=prv)

md.start()
md2.start()
md3.start()

##

