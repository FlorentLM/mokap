from mokap.core import MotionDetector

##

md = MotionDetector(cam_id=0, learning_rate=-100, thresh=5)
md2 = MotionDetector(cam_id=2, learning_rate=-100, thresh=5)
md3 = MotionDetector(cam_id=4, learning_rate=-100, thresh=5)

md.start()
md2.start()
md3.start()


##

