# Facial Landmark Detection on Faces
Python program which inputs an image of a face, and detects 68 different kinds of landmark points around the eyes, nose, mouth and jawline.

## Overview
The problem is divided into 2 parts:

- localising the face in the image (finding the bounding box).
- detecting the key facial structures and landmark points.

We have used a pre-trained Histogram of Oriented Gradients (HOG) + Linear SVM object detector [1] for the first task, and the facial landmark detector proposed by Kazemi and Sullivan (which uses ensemble of regression trees) [2] for the second task. Optimal implementation of both are available in the dlib library. The models were pretrained on the iBUG 300-W dataset [3].


## How to run

### Prerequisties
- Python 3
- Numpy
- OpenCV
- Dlib
- Imutils

### Steps
- Download the model from dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and put it in the same directory as `facial_landmark.py`.
- Run `facial_landmark.py` by specifying 2 arguments (first the input image `-i`, followed by the location of the output file `-o`). For eg: `python facial_landmark.py -i images/ex1.jpeg -o output_files/ex1`
 - Thus the directories of the images and files can be chanfed outside of the program.
 
 
 ## Summary of results
 #### Image 1
 ![Input](/images/ex2.jpeg){:height="50%" width="50%"}
 ![Output](/images/output-img/output-2.PNG)
 
 ####Image 2
 ![Input](/images/ex3.jpeg){:height="50%" width="50%"}
 ![Output](/images/output-img/output-3.PNG)
 
 
 ## Sources
 - [1] http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
 - [2] https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6909637
 - [3] https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
