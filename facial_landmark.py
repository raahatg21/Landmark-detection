# Face Landmark Detection
# to run, eg: python facial_landmark.py --i images/ex3.jpeg --o ex3
# 					 file name			input image arg		output file arg

import argparse
import numpy as np
import os
import cv2
import dlib
import imutils
from imutils import face_utils


colours = [(0, 255, 0),		# bounding box and text
		   (0, 0, 255),		# landmark points
		   (0, 0, 255)]		# jawline curve


# Convert a bounding box to the format (x, y, w, h)
def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - ys
	return (x, y, w, h)
	

# Convert the coordinates to numpy array
def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)	
	return coords
	

# Initialise dlib's face detector (HOG-based) and create the facial landmark predictor
shape_predictor = "shape_predictor_68_face_landmarks.dat"  
detector = dlib.get_frontal_face_detector()  		# to detect the faces
predictor = dlib.shape_predictor(shape_predictor)  	# to define the landmark points


# Construct the argument parser and parse the image argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-o", "--output", required=True, help="path to output csv file")
args = vars(ap.parse_args())


# Load the input image and preprocess it
image = cv2.imread(args["image"])
try:
	image = imutils.resize(image, width=500)
except AttributeError:
	print("Image not found. Please check the path.")
	quit()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)  # it detects the faces in the image


# Path of output csv file
file_name = str(args["output"])


# Plot the landmarks over all the faces in the image
try:
	rect = rects[0]  # dummy try condition to check if 'rects' contains any coordinates or not
	all_shapes = np.zeros(shape=(len(rects), 68, 2), dtype = np.uint8)  # array to contain the coordinates of all the faces
	
	for (i, rect) in enumerate(rects):  		# for total number of faces in the image
		shape = predictor(gray, rect)  			# defines the landmark points
		shape = face_utils.shape_to_np(shape)  	# converts landmark points to numpy array
		all_shapes[i, :, :] = shape  
	
		(x, y, w, h) = face_utils.rect_to_bb(rect) 		# converts the 'rectangle' to 'bounding box'
		cv2.rectangle(image, (x, y), (x + w, y + h), colours[0], 2)
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colours[0], 2)

		for (x, y) in shape:  	# for all the landmark coordinates
			cv2.circle(image, (x, y), 1, colours[1], -1)
	
		pts = shape[0:17]  		# points containing the jawline
		for l in range(1, len(pts)):
			ptA = tuple(pts[l - 1])
			ptB = tuple(pts[l])
			cv2.line(image, ptA, ptB, colours[2], 2)

	with open(file_name + '.csv', 'w') as outfile:	# write the data onto a file 
		for slice in all_shapes:
			np.savetxt(outfile, slice, delimiter=',', fmt = '%d')
			
except IndexError:
	print("Couldn't find a face in the image.")


# Show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
