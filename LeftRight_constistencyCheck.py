import skimage
import skimage.filters
from skimage import io 
import cv2
import numpy as np 
import scipy.misc
import matplotlib.pyplot as plt

# Load in the left and right camera images
left_image = cv2.imread("right_image.png")
right_image = cv2.imread("left_image.png")

#Convert the two image arrays two floats
left_image = skimage.img_as_float(left_image)
right_image = skimage.img_as_float(right_image)

max_disparity = left_image.shape[0] / 3

#Right shift the left image by different amounts and figure out the sum of squared errors between the shifted left image and the right image
# Do this for several different shift values and store the smallest cost for each pixel

# 3d matrix to store the DSI
DSI = np.zeros((left_image.shape[0], left_image.shape[1], int(max_disparity)))

DSI = DSI.astype(np.float32, copy=False)

print(DSI.shape)

def InBounds(col, d, max_col):
	if (col - d) <= 0:
		return False
	else:
		return True

def sumSquareError(row, col, d):
	if not InBounds(col, d, left_image.shape[1]):
		return 50000
	# Right shift the left image
	error = 0
	for i in range(3):
		Rpoint = (row, col, i)
		Lpoint = (row, col - d, i)
		error += np.square(right_image[Rpoint] - left_image[Lpoint])

	return error

print(DSI.dtype)

# Calculate the sum of squared errors at different disparities
for d in range(int(max_disparity)):
	for row in range(left_image.shape[0]):
		for col in range(left_image.shape[1]):
			val = sumSquareError(row, col, d)
			DSI[(row, col, d)] = val
	# Perfrom bilateral filtering at every stage
	DSI[:, :, d] = cv2.bilateralFilter(DSI[:, :, d], 5, 75, 75)

# Perfrom gaussian filtering at every disparity
#DSI = skimage.filters.gaussian(DSI, sigma=1)


# Determine the smallest cost at every pixel
# Start by making every value very large
smallest_DSI = np.zeros((left_image.shape[0], left_image.shape[1]))
print(smallest_DSI.shape)

# store the depth at the lowest cost. not the actual cost
for row in range(left_image.shape[0]):
	for col in range(left_image.shape[1]):
		min_density = 0
		count = 0
		for d in range(int(max_disparity)):
			if DSI[(row, col, d)] < smallest_DSI[(row, col)] or smallest_DSI[(row,col)] == 0:
				# print(row, col, min_density)
				# print(DSI[row,col,d], smallest_DSI[(row,col)])
				# print()
				smallest_DSI[(row, col)] = DSI[(row,col,d)]
				min_density = d	
			count += 1
		smallest_DSI[(row, col)] = min_density + 1

smallest_DSI = smallest_DSI[:, :270]
print(smallest_DSI.shape)


# plt.imshow(smallest_DSI)
# plt.show()










"""Now do right to left"""

# Load in the left and right camera images
left_image = cv2.imread("right_image.png")
right_image = cv2.imread("left_image.png")

#Convert the two image arrays two floats
left_image = skimage.img_as_float(left_image)
right_image = skimage.img_as_float(right_image)

max_disparity = left_image.shape[0] / 3


# 3d matrix to store the DSI
DSI = np.zeros((left_image.shape[0], left_image.shape[1], int(max_disparity)))

DSI = DSI.astype(np.float32, copy=False)

print(DSI.shape)

def InBoundsReverse(col, d, max_col):
	if (col + d) >= max_col:
		return False
	else:
		return True

def sumSquareErrorReverse(row, col, d):
	if not InBoundsReverse(col, d, left_image.shape[1]):
		return 50000
	# Right shift the left image
	error = 0
	for i in range(3):
		Rpoint = (row, col + d, i)
		Lpoint = (row, col, i)
		error += np.square(left_image[Lpoint] - right_image[Rpoint])

	return error

print(DSI.dtype)

# Calculate the sum of squared errors at different disparities
for d in range(int(max_disparity)):
	for row in range(left_image.shape[0]):
		for col in range(left_image.shape[1]):
			val = sumSquareErrorReverse(row, col, d)
			DSI[(row, col, d)] = val
	# Perfrom bilateral filtering at every stage
	DSI[:, :, d] = cv2.bilateralFilter(DSI[:, :, d], 5, 75, 75)

# Perfrom gaussian filtering at every disparity
#DSI = skimage.filters.gaussian(DSI, sigma=1)


# Determine the smallest cost at every pixel
# Start by making every value very large
smallest_DSI2 = np.zeros((left_image.shape[0], left_image.shape[1]))

# store the depth at the lowest cost. not the actual cost
for row in range(left_image.shape[0]):
	for col in range(left_image.shape[1]):
		min_density = 0
		count = 0
		for d in range(int(max_disparity)):
			if DSI[(row, col, d)] < smallest_DSI2[(row, col)] or smallest_DSI2[(row,col)] == 0:
				# print(row, col, min_density)
				# print(DSI[row,col,d], smallest_DSI[(row,col)])
				# print()
				smallest_DSI2[(row, col)] = DSI[(row,col,d)]
				min_density = d	
			count += 1
		smallest_DSI2[(row, col)] = min_density + 1

smallest_DSI2 = smallest_DSI2[:, :270]

# plt.imshow(smallest_DSI2)
# plt.show()






"""Left right consistency check"""
thresh = 15
for row in range(smallest_DSI.shape[0]):
	for col in range(smallest_DSI.shape[1]):
		if abs(smallest_DSI[(row, col)] - smallest_DSI2[(row, col)]) >= thresh:
			smallest_DSI2[(row, col)] = 1

plt.imshow(smallest_DSI2)
plt.show()