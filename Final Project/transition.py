import numpy
import struct
from skimage.color import rgb2gray
import cv2
import matplotlib.transforms
import matplotlib.pyplot as plt
import time
import math
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

# Transformation functions.

# Higher the speed, slower it changes.
def fade(img_0, img_1, img_2, img_3, speed):
	for n in range(1, speed):
		big = 1 - (n/speed)
		small = (n/speed)
		cv2.imshow("test", (big*img_0) + (small*img_1))
		cv2.waitKey(5)
	for n in range(1, speed):
		big = 1 - (n/speed)
		small = (n/speed)
		cv2.imshow("test", (big*img_1) + (small*img_2))
		cv2.waitKey(5)
	for n in range(1, speed):
		big = 1 - (n/speed)
		small = (n/speed)
		cv2.imshow("test", (big*img_2) + (small*img_3))
		cv2.waitKey(5)
	for n in range(1, speed):
		big = 1 - (n/speed)
		small = (n/speed)
		cv2.imshow("test", (big*img_3) + (small*img_0))
		cv2.waitKey(5)


def rotation(img_0, img_1, img_2, img_3, max_rotation_angle, speed):
	for n in range(1, max_rotation_angle):
		rows = img_0.shape[0]
		cols = img_0.shape[1]
		# Image 0
		M_0 = cv2.getRotationMatrix2D((0,0),n,1)
		img_trans_0 = cv2.warpAffine(img_0, M_0, (cols,rows))
		# Image 1
		M_1 = cv2.getRotationMatrix2D((0,0),n + 90,1)
		img_trans_1 = cv2.warpAffine(img_1, M_1, (cols,rows))
		# Image 2
		M_2 = cv2.getRotationMatrix2D((0,0),n + 180,1)
		img_trans_2 = cv2.warpAffine(img_2, M_2, (cols,rows))
		# Image 3
		M_3 = cv2.getRotationMatrix2D((0,0),n + 270,1)
		img_trans_3 = cv2.warpAffine(img_3, M_3, (cols,rows))

		# The key idea here is that images are rotated and they are added to combined them together.
		cv2.imshow("test", img_trans_0 + img_trans_1 + img_trans_2 + img_trans_3)
		cv2.waitKey(speed)

def scaling(img_0, img_1, img_2, img_3):
	for n in range(1, 180):
		rows = img_0.shape[0]
		cols = img_0.shape[1]

def fadeRotation(img_0, img_1, img_2, img_3, max_rotation_angle, speed):
	for n in range(1, speed):
		big = 1 - (n/speed)
		small = (n/speed)
		cv2.imshow("test", (big*img_0) + (small*img_1))
		cv2.waitKey(5)
	for n in range(1, speed):
		big = 1 - (n/speed)
		small = (n/speed)
		cv2.imshow("test", (big*img_1) + (small*img_2))
		cv2.waitKey(5)
	for n in range(1, speed):
		big = 1 - (n/speed)
		small = (n/speed)
		cv2.imshow("test", (big*img_2) + (small*img_3))
		cv2.waitKey(5)
	for n in range(1, speed):
		big = 1 - (n/speed)
		small = (n/speed)
		cv2.imshow("test", (big*img_3) + (small*img_0))
		cv2.waitKey(5)
	for n in range(1, max_rotation_angle):
		rows = img_0.shape[0]
		cols = img_0.shape[1]
		# Image 0
		M_0 = cv2.getRotationMatrix2D((0,0),n,1)
		img_trans_0 = cv2.warpAffine(img_0, M_0, (cols,rows))
		# Image 1
		M_1 = cv2.getRotationMatrix2D((0,0),n + 90,1)
		img_trans_1 = cv2.warpAffine(img_1, M_1, (cols,rows))
		# Image 2
		M_2 = cv2.getRotationMatrix2D((0,0),n + 180,1)
		img_trans_2 = cv2.warpAffine(img_2, M_2, (cols,rows))
		# Image 3
		M_3 = cv2.getRotationMatrix2D((0,0),n + 270,1)
		img_trans_3 = cv2.warpAffine(img_3, M_3, (cols,rows))

		# The key idea here is that images are rotated and they are added to combined them together.
		cv2.imshow("test", img_trans_0 + img_trans_1 + img_trans_2 + img_trans_3)
		cv2.waitKey(speed)

# def flip(img_0, img_1, img_2, img_3):
# 	width = img_0.shape[0]
# 	height = img_0.shape[1]
# 	for n in range(1, width):
# 		newimg = cv2.resize(img_1, (height, width - 100))
# 		cv2.imshow("test", newimg)
# 		cv2.waitKey(50)


# Importing images.
img_0 = numpy.array(cv2.imread("0.jpg"))
img_1 = numpy.array(cv2.imread("1.jpg"))
img_2 = numpy.array(cv2.imread("2.jpg"))
img_3 = numpy.array(cv2.imread("3.jpg"))

# First image width and height.
width = img_0.shape[0]
height = img_0.shape[1]
img_1 = cv2.resize(img_1, (height, width))
img_2 = cv2.resize(img_2, (height, width))
img_3 = cv2.resize(img_3, (height, width))
# Changing the pixel values to 0 - 1 range so the pixels can be manipulated.
img_0 = cv2.normalize(img_0.astype(float), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, dst=None)
img_1 = cv2.normalize(img_1.astype(float), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, dst=None)
img_2 = cv2.normalize(img_2.astype(float), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, dst=None)
img_3 = cv2.normalize(img_3.astype(float), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, dst=None)


#fade(img_0, img_1, img_2, img_3, 50)
rotation(img_0, img_1, img_2, img_3, 360, 10)
#flip(img_0, img_1, img_2, img_3)

