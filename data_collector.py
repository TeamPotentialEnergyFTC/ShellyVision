import cv2, imutils, os

from utils import load_pbtxt, IMG_WIDTH, IMG_HEIGHT, TRACKER_NAME, VIDEO_PATH, DATA_SAVE_PATH, ANNOTATIONS_PATH, LABELMAP_PATH

labels = load_pbtxt(LABELMAP_PATH)

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "kcf": cv2.legacy.TrackerKCF_create,
    "boosting": cv2.legacy.TrackerBoosting_create,
    "mil": cv2.legacy.TrackerMIL_create,
    "tld": cv2.legacy.TrackerTLD_create,
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
    "mosse": cv2.legacy.TrackerMOSSE_create
}

trackers = cv2.legacy.MultiTracker_create()
cap = cv2.VideoCapture(VIDEO_PATH)

csv_writer = open(ANNOTATIONS_PATH + "annotations.csv", "w+")
csv_writer.write("filename,xmin,xmax,ymin,ymax,class\n")

def select_objs(frame, trackers):
	global labels
	print("---\nSelecting objects, press esc to save and exit\n---") # just to make it more clear than what is already there

	box = cv2.selectROIs("Frame", frame, fromCenter=False, showCrosshair=True)
	box = tuple(map(tuple, box))

	for bb in box:
		tracker = OPENCV_OBJECT_TRACKERS[TRACKER_NAME]()
		trackers.add(tracker, frame, bb)

	if not labels:
		while not len(labels) == len(box): 
			labels = [name for name in input("Enter all object labels in order > ").split()]
			print(labels, len(box))
	return labels

first_frame = True
frame_num = 0
while cap.isOpened():
	ret, frame = cap.read()

	if frame is None:
		print("El fin")
		break

	frame = imutils.resize(frame, width=IMG_WIDTH, height=IMG_HEIGHT)
	frame_name = str(frame_num) + ".jpg"
	cv2.imwrite(DATA_SAVE_PATH + frame_name, frame)
	frame_num += 1

	if first_frame:
		select_objs(frame, trackers)
		first_frame = False

	(success, boxes) = trackers.update(frame)

	# loop over the bounding boxes and draw them on the frame
	for box_n in range(len(boxes)):
		print(labels, box_n)
		(x, y, w, h) = [int(v) for v in boxes[box_n]]
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		csv_writer.write(f"{frame_name},{x},{x + w},{y},{y + h},{labels[box_n]}\n")

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord('a'): # add bounding boxes with a
		labels = select_objs(frame, trackers)
	elif key == ord('r'): # reset bounding box with r
		trackers.clear()
		trackers = cv2.legacy.MultiTracker_create()

		labels = select_objs(frame, trackers)
	elif key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()