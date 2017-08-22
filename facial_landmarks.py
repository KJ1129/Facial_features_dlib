# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from collections import OrderedDict

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = image
# detect faces in the grayscale image
rects = detector(gray, 1)

image_shape = image.shape

output = np.zeros((image_shape[0],image_shape[1],image_shape[2]), np.uint8)
mask = np.zeros((image_shape[0],image_shape[1],image_shape[2]), np.uint8)

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	#cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the face number
	#cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image

	#for (x, y) in shape:
		#cv2.circle(image, (x, y), 1, (0, 0, 255), -1)


	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		if name == 'right_eye' or name == 'left_eye' or name == 'left_eyebrow' or name == 'right_eyebrow': 
			rect = cv2.minAreaRect(np.array([shape[i:j]]))
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			cv2.drawContours(mask,[box],0,(1,1,1),-1)
		if name == 'nose': 
			pts = np.array([shape[27],shape[28],shape[29],shape[30],shape[33],shape[32],shape[31]]) #left nose
			cv2.fillConvexPoly(mask,pts,(1,1,1))
			pts = np.array([shape[27],shape[28],shape[29],shape[30],shape[33],shape[34],shape[35]]) #right nose
			cv2.fillConvexPoly(mask,pts,(1,1,1))
		if name == 'mouth':
			ellipse = cv2.fitEllipse(np.array([shape[i:j]]))
			cv2.ellipse(mask,ellipse,(1,1,1),-1)
		###rect
		#(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
		#roi = image[y:y + h, x:x + w]
		#output[y:y + h, x:x + w,:] = image[y:y + h, x:x + w]
		#roi = imutils.resize(roi, width=20, inter=cv2.INTER_CUBIC)
		#cv2.imshow("ROI", roi)
		#cv2.waitKey(0)

		###circle
		# (x,y),radius = cv2.minEnclosingCircle(np.array([shape[i:j]]))
		# center = (int(x),int(y))
		# radius = int(radius)
		# cv2.circle(mask,center,radius,(1,1,1),-1,2)

		###rotate circle
		#ellipse = cv2.fitEllipse(np.array([shape[i:j]]))
		#cv2.ellipse(mask,ellipse,(1,1,1),2)

		###rotate rect
		#rect = cv2.minAreaRect(np.array([shape[i:j]]))
		#box = cv2.boxPoints(rect)
		#box = np.int0(box)
		#cv2.drawContours(mask,[box],0,(1,1,1),-1)
# show the output image with the face detections + facial landmarks

# cv2.imshow("Output", image)
# cv2.waitKey(0)


output = image * mask
cv2.imshow("ROI", output)

cv2.imwrite("ROI.jpg", output)

cv2.waitKey(0)