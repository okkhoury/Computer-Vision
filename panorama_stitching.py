import cv2
import numpy as np 
import random
import math
import skimage
import skimage.transform
import matplotlib.pyplot as plt

# Read in the left and right images. dtype = uint8
imgA = cv2.imread("inputA.jpg")
imgB = cv2.imread("inputB.jpg")

# Flip the color channels
imgA = imgA[:,:,::-1]
imgB = imgB[:,:,::-1]

#Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(imgA, None)
kp2, des2 = sift.detectAndCompute(imgB, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:  
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
#img3 = cv2.drawMatchesKnn(imgA,kp1,imgB,kp2,good,flags=2, outImg=None)

#plt.imshow(img3),plt.show()

def applyHomography(hMatrix, xB, yB):
	a = hMatrix[(0, 0)]
	b = hMatrix[(0, 1)]
	c = hMatrix[(0, 2)]

	d = hMatrix[(1, 0)]
	e = hMatrix[(1, 1)]
	f = hMatrix[(1, 2)]

	g = hMatrix[(2, 0)]
	h = hMatrix[(2, 1)]

	xA = (a*xB + b*yB + c) / (g*xB + h*yB + 1)
	yA = (d*xB + e*yB + f) / (g*xB + h*yB + 1)

	return xA, yA

# aPoints and bPoints are both numpy arrays with 4 2D points
# Returns a 3x3 homography matrix
def fitHomography(aPoints, bPoints):
	# Make 8x8 matrix to hold the 8 equations, each with 8 variables
	equations = np.zeros((8, 8))
	b = np.zeros((8, 1))

	# Iterate through the 4 points and fill in the two numpy arrays 
	index = -1
	for i in range(4):
		xA1, yA1 = aPoints[i][0], aPoints[i][1]
		xB1, yB1 = bPoints[i][0], bPoints[i][1]

		index += 1

		# Order of variabes -> a, b, c, d, e, f, g, h
		# Fill in values for first equation
		b[(index, 0)] = xA1
		equations[(index,0)] = -xB1
		equations[(index,1)] = -yB1
		equations[(index,2)] = -1
		equations[(index,3)] = 0
		equations[(index,4)] = 0
		equations[(index,5)] = 0
		equations[(index,6)] = xA1*xB1
		equations[(index,7)] = xA1*yB1

		index += 1

		# Fill in values for second equation
		b[(index, 0)] = yA1
		equations[(index,0)] = 0
		equations[(index,1)] = 0
		equations[(index,2)] = 0
		equations[(index,3)] = -xB1  
		equations[(index,4)] = -yB1
		equations[(index,5)] = -1
		equations[(index,6)] = yA1*xB1
		equations[(index,7)] = yA1*yB1

	solveSystem = np.multiply(np.linalg.solve(equations, b), -1)

	homography = np.zeros((3,3))
	homography[(0,0)] = solveSystem[(0,0)] # a
	homography[(0,1)] = solveSystem[(1,0)] # b
	homography[(0,2)] = solveSystem[(2,0)] # c
	homography[(1,0)] = solveSystem[(3,0)] # d
	homography[(1,1)] = solveSystem[(4,0)] # e
	homography[(1,2)] = solveSystem[(5,0)] # f
	homography[(2,0)] = solveSystem[(6,0)] # g
	homography[(2,1)] = solveSystem[(7,0)] # h
	homography[(2,2)] = 1                  # 1

	return homography

# A = [(0, 0), (1, 0), (0, 1), (1, 1)]
# B = [(1, 2), (3, 2), (1, 4), (3, 4)]
# print(fitHomography(A, B))

def RANSAC(matches, kp1, kp2, iterations):

	BestHomography = None
	largestInlier = 0

	# Run the steps a fixed number of times
	for i in range(iterations):
		# Randomly select 4 matched points and fit a homography to them
		randMatches = random.sample(matches, 4)
		aPoints = []
		bPoints = []

		for mat in randMatches:
			# Get the matching keypoints for each of the images
			img1_idx = mat[0].queryIdx
			img2_idx = mat[0].trainIdx
			# Get the coordinates
			(x1,y1) = kp1[img1_idx].pt
			(x2,y2) = kp2[img2_idx].pt
			# Append to each list
			aPoints.append((x1, y1))
			bPoints.append((x2, y2))

		homography = fitHomography(aPoints, bPoints)
		
		# Loop through all matched pairs and determine all of the inliers
		thresh = .5
		inliers = []
		for mat in matches:
			# Get the coordinates
			(xA, yA) = kp1[mat[0].queryIdx].pt 
			(xB, yB) = kp2[mat[0].trainIdx].pt

			# Apply homography to (xB, yB)
			xANEW, yANEW = applyHomography(homography, xB, yB)

			# Find the magnitude between the new A point and the old one
			dist = math.sqrt((xANEW - xA) ** 2 + (yANEW - yA) ** 2)

			#If the dist is less than threshold add the match to inliers
			if dist <= thresh:
				inliers.append(mat)

			# If this set of inliers is the largest, update the best homography and largestInlier
			if len(inliers) > largestInlier:
				largestInlier = len(inliers)
				BestHomography = homography
	
	return BestHomography

h = RANSAC(good, kp1, kp2, 100)

def composite_warped(a, b, H):
	"Warp images a and b to a's coordinate system using the homography H which maps b coordinates to a a coordinate"
	out_shape = (a.shape[0], 2*a.shape[1])                               # Output image (height, width)
	p = skimage.transform.ProjectiveTransform(np.linalg.inv(H))       # Inverse of homography (used for inverse warping)
	bwarp = skimage.transform.warp(b, p, output_shape=out_shape)         # Inverse warp b to a coords
	bvalid = np.zeros(b.shape, 'uint8')                               # Establish a region of interior pixels in b
	bvalid[1:-1,1:-1,:] = 255
	bmask = skimage.transform.warp(bvalid, p, output_shape=out_shape)    # Inverse warp interior pixel region to a coords
	apad = np.hstack((skimage.img_as_float(a), np.zeros(a.shape))) # Pad a with black pixels on the right
	return skimage.img_as_ubyte(np.where(bmask==1.0, bwarp, apad))    # Select either bwarp or apad based on mask


final_img = composite_warped(imgA, imgB, h)

plt.imshow(final_img)
plt.show()






















