import os

IMG_WIDTH = 600
IMG_HEIGHT = 600

TRACKER_NAME = "csrt"
VIDEO_PATH = "data/input_video.mp4"
DATA_SAVE_PATH = "data/images/"
TFRECORD_PATH = "data/tfrecords/"
ANNOTATIONS_PATH = "data/annotations/"

LABELMAP_PATH = ANNOTATIONS_PATH + "labelmap.pbtxt"

def load_pbtxt(pbtxt_path: str) -> list:
	labels = []
	if os.path.isfile(pbtxt_path):
		with open(pbtxt_path, 'r') as lm:
			labels = [line.split("'")[1] for line in lm.readlines() if "'" in line] # kind of crappy but should work :)
	return labels