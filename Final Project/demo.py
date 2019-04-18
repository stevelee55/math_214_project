# import pdb; pdb.set_trace()

import os
import matplotlib.pyplot as plt
from skimage import transform
import cv2
import numpy
import math
import imageio
from PIL import Image
import smtplib
from skimage import img_as_ubyte

# Custom Modules.
from Stablizing import Stablizing

# Input and output paths.
videoFramesPath = "./data1" #"./Crosswalk"
resultOutputPath = "."

numberOfFramesToUse = 100


###############################################################

# Getting the list of all of the file names at the specified path.
fileNames = os.listdir(videoFramesPath + "/")

# Filtering out only the .jpg files, aka the video frames.
videoFrameNames = []
for fileName in fileNames:
	if fileName.endswith('.jpg'):
		videoFrameNames.append(fileName)

# Sorting the videoFrameNames, because time to time,
# the frames get read in randomly.
videoFrameNames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

# Getting the total number of video frames.
numberOfVideoFrames = len(videoFrameNames)

# Deciding how many frames to actually use based on number of total video frames.
if (numberOfFramesToUse > numberOfVideoFrames):
	numberOfFramesToUse = numberOfVideoFrames

# Getting the width and height of the video using the
# first frame of the video.
# Returned value is numpy array type.
firstFrame = cv2.imread(videoFramesPath + "/" + videoFrameNames[0])
imageDimension = firstFrame.shape
height = imageDimension[0]
width = imageDimension[1]

# Calculating the multiplier for the video frames'width and height
desiredWidth = 427.0
multiplier = 1.0
if (width > desiredWidth):
	multiplier = desiredWidth / width
newHeight = math.floor(float(height * multiplier))
newWidth = math.floor(float(width * multiplier))
print("Original Width", width)
print("Original Height", height)
print("New Width", newWidth)
print("New Height", newHeight)

# Initializing the 4-D Matrix with 0s.
# The MovMat is a 4-D array/Matrix that has the dimension of
# newHeight * newWidth * 3 * # of frames to use.
RGBDimension = 3
MovMat = numpy.zeros((newHeight, newWidth, RGBDimension, numberOfFramesToUse), numpy.uint8)

# Goes through every single frame and sets each to the MovMat.
for i in range(numberOfFramesToUse):
	frame = cv2.imread(videoFramesPath + "/" + videoFrameNames[i])
	MovMat[:,:,:,i] = cv2.resize(frame, (newWidth, newHeight))

# Homography transformation
stablizingInstance = Stablizing()
stablizingInstance.Stablizing_Main(MovMat)



