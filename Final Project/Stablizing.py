import numpy
import struct
from skimage.color import rgb2gray
import cv2
import matplotlib.transforms
import matplotlib.pyplot as plt
import time
import math

class Stablizing(object):

	# Initializer for the this class' instance.
	def __init__(self):
		print("Stablizing is created!")

	# Optimized by calcualting the transformation matrcies with grayscale and applying the transformation matrices to
	# the RGB frames (original frames).
	def Stablizing_Main(self, movmat):
		# Getting the number of frames.
		numOfFrames = movmat.shape[3]
		# Creating array to hold each of the transformation matrices.
		tformObject = numpy.eye(3)
		tforms = numpy.asarray([tformObject for x in range(numOfFrames)])

		# Surf only works with this for some reason.
		imgB = cv2.cvtColor(movmat[:,:,:,0], cv2.COLOR_BGR2GRAY)
		# Extract features points in the first frame.
		surf = cv2.xfeatures2d.SURF_create()
		mainPoints, mainFeatures = surf.detectAndCompute(imgB, None)

		imgB = movmat[:,:,:,0]
		imgB = cv2.normalize(imgB.astype(float), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, dst=None)
		cv2.imshow("test", imgB)
		cv2.waitKey(2000)

		for n in range(1, numOfFrames):

			pointsPrevious = mainPoints
			featuresPrevious = mainFeatures
			# Doing this for every frame because SURF only works for grayscale.
			imgB = cv2.cvtColor(movmat[:,:,:,n], cv2.COLOR_BGR2GRAY)
			# Getting the points and the freatures for the new frame.
			points, features = surf.detectAndCompute(imgB, None)
			# Doing freature matching using the current and previous points.
			bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=True)
			indexPairs = bf.match(features, featuresPrevious)

			# queryIdx is features, and trainIdx is featuresPrevious.
			# indexPairs[i].queryIdx gives index of points that were matched.
			matchedPoints = numpy.asarray([points[indexPairs[i].queryIdx] for i in range(len(indexPairs))])
			numpyArrayMatchedPoints = numpy.asarray([matchedPoints[i].pt for i in range(len(matchedPoints))])
			# matchedPoints have the KeyPoint objects, which an be accesesd by index and .pt.
			# print(matchedPoints[0].pt)
			matchedPointsPrev = numpy.asarray([pointsPrevious[indexPairs[i].trainIdx] for i in range(len(indexPairs))])
			numpyArrayMatchedPointsPrev = numpy.asarray([matchedPointsPrev[i].pt for i in range(len(matchedPointsPrev))])

			test = cv2.estimateRigidTransform(numpyArrayMatchedPoints, numpyArrayMatchedPointsPrev, fullAffine=True)
			imgB = movmat[:,:,:,n]
			# imgB = cv2.normalize(imgB.astype(float), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, dst=None)
			# cv2.imshow("test", imgB)
			# cv2.waitKey(10000)
			M = numpy.float64(test)
			# Perspective transforming it. This needs to be used to apply the transfomration matrix to the whole image.
			height = movmat.shape[0] * 2
			width = movmat.shape[1] * 2
			warpedImage = cv2.warpAffine(imgB, M, (width, height))
			warpedImage = cv2.normalize(warpedImage.astype(float), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, dst=None)
			cv2.imshow("test", warpedImage)
			cv2.waitKey(100)

		# Repeating the above for every frame.
		# Starting from the index 1 because the first frame has already been processed.
		# for n in range(1, numOfFrames):
		# 	pointsPrevious = points
		# 	featuresPrevious = features
		# 	# Doing this for every frame because SURF only works for grayscale.
		# 	imgB = cv2.cvtColor(movmat[:,:,:,n], cv2.COLOR_BGR2GRAY)
		# 	# Getting the points and the freatures for the new frame.
		# 	points, features = surf.detectAndCompute(imgB, None)
		# 	# Doing freature matching using the current and previous points.
		# 	bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=True)
		# 	indexPairs = bf.match(features, featuresPrevious)

		# 	# queryIdx is features, and trainIdx is featuresPrevious.
		# 	# indexPairs[i].queryIdx gives index of points that were matched.
		# 	matchedPoints = numpy.asarray([points[indexPairs[i].queryIdx] for i in range(len(indexPairs))])
		# 	numpyArrayMatchedPoints = numpy.asarray([matchedPoints[i].pt for i in range(len(matchedPoints))])
		# 	# matchedPoints have the KeyPoint objects, which an be accesesd by index and .pt.
		# 	# print(matchedPoints[0].pt)
		# 	matchedPointsPrev = numpy.asarray([pointsPrevious[indexPairs[i].trainIdx] for i in range(len(indexPairs))])
		# 	numpyArrayMatchedPointsPrev = numpy.asarray([matchedPointsPrev[i].pt for i in range(len(matchedPointsPrev))])

		# 	test = cv2.estimateRigidTransform(numpyArrayMatchedPointsPrev, numpyArrayMatchedPoints, fullAffine=True)
		# 	imgB = movmat[:,:,:,n - 1]
		# 	# imgB = cv2.normalize(imgB.astype(float), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, dst=None)
		# 	# cv2.imshow("test", imgB)
		# 	# cv2.waitKey(10000)
		# 	M = numpy.float64(test)
		# 	# Perspective transforming it. This needs to be used to apply the transfomration matrix to the whole image.
		# 	height = movmat.shape[0]
		# 	width = movmat.shape[1]
		# 	warpedImage = cv2.warpAffine(imgB, M, (width, height))
		# 	warpedImage = cv2.normalize(warpedImage.astype(float), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, dst=None)
		# 	cv2.imshow("test", warpedImage)
		# 	cv2.waitKey(400)

			# Estimating geometric transform, aka getting the transformation matrices.
			# estimatePair = cv2.getAffineTransform(numpy.float32([[P[0][0],P[0][1]],[P[1][0],P[1][1]],[P[2][0],P[2][1]]]), numpy.float32([[P[0][2],P[0][3]],[P[1][2],P[1][3]],[P[2][2],P[2][3]]]))#,cv2.RANSAC,5.0) #cv2.estimateRigidTransform(numpyArrayMatchedPoints, numpyArrayMatchedPointsPrev, True)
			# print(estimatePair)
			# #print(estimatePair)
			# #print(status)
			# # first = [estimatePair[0][0], estimatePair[1][0], 91.59]
			# # second = [estimatePair[0][1], estimatePair[1][1], -0.63]
			# # third = [estimatePair[0][2], estimatePair[1][2], 0.9896]
			# tforms[n] = estimatePair #[estimatePair[0], estimatePair[1], numpy.array([0,0,1])]
			# # Estimate transform from frame A to frame B by transforming previous transform matrix to the new perspective (?).
			# # This is in a way setting an anchor point for the other frames to perspective transform from.
			# # Anchor starts from the first frame and each frame after it depends on the previous frame, which creates like a paranoaram effect.
			# print(numpy.matmul(tforms[n],tforms[n-1]))
			# print(numpy.matmul(tforms[n-1],tforms[n]))
			# tforms[n] = numpy.matmul(tforms[n-1], tforms[n])

		# # Setting the output image frame size.
		# height = movmat.shape[0] * 2
		# width = movmat.shape[1] * 2

		# for i in range(len(tforms)):
			
		# 	imgB = movmat[:,:,:,i]
		# 	M = numpy.float64(tforms[i])
		# 	# Perspective transforming it. This needs to be used to apply the transfomration matrix to the whole image.
		# 	warpedImage = cv2.warpAffine(imgB, M, (width, height))
		# 	# Converting the image to float 64, an image with pixel values that vary from 0.0 to 1.0.
		# 	warpedImage = cv2.normalize(warpedImage.astype(float), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, dst=None)
		# 	cv2.imshow("test", warpedImage)
		# 	cv2.waitKey(130)

		# 	print("Homography frame #: ", i)

		print("Homography Complete")
