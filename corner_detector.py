#Owen Khoury (okk6nb)

import skimage
from skimage import io 
from scipy.ndimage.filters import gaussian_filter
import numpy as np 
import math
from itertools import product, starmap
from scipy import signal

# Read in image and convert it from uint8 to float64
file = input("Enter the name of the file ")

image = io.imread(file)
image = skimage.img_as_float(image)
#Convert it to grey scale
grey_image = skimage.color.rgb2grey(image)

# Smooth image by convolving it with 7x7 gaussian kernel
gaussian_kernel = np.array([[0.003765,	0.015019,	0.023792,	0.015019,	0.003765],
[0.015019,	0.059912,	0.094907,	0.059912,	0.015019],
[0.023792,	0.094907,	0.150342,	0.094907,	0.023792],
[0.015019,	0.059912,	0.094907,	0.059912,	0.015019],
[0.003765,	0.015019,	0.023792,	0.015019,	0.003765]], dtype=np.float)
filtered_image = signal.convolve2d(grey_image, gaussian_kernel)

# Find the gradient of the smoothed image
x_gradient, y_gradient = np.gradient(filtered_image)

# Compute the covariance matrix for each pixel
Fx2 = x_gradient**2
Fy2 = y_gradient**2
FxFy = x_gradient*y_gradient

M = 4
corner_thresh = 2
corner_pixels = []
divisor = np.divide(1, 81)
for y in range(image.shape[0]):
	for x in range(image.shape[1]):
		covariance_matrix = np.zeros(shape=(2,2))
		for row in range(y-M, y+M + 1):
			for col in range(x-M, x+M + 1):
				covariance_matrix[(0,0)] += np.divide(Fx2[(row, col)], divisor)
				covariance_matrix[(0,1)] += np.divide(FxFy[(row, col)], divisor)
				covariance_matrix[(1,0)] += np.divide(FxFy[(row, col)], divisor)
				covariance_matrix[(1,1)] += np.divide(Fy2[(row, col)], divisor)
		smallest_eigen_value = min(np.linalg.eig(covariance_matrix)[0])
		if (smallest_eigen_value > corner_thresh):
			point = (y, x)
			corner_pixels.append((point, smallest_eigen_value))

corner_pixels = sorted(corner_pixels,key=lambda x: x[1], reverse=True)

points = []
for item in corner_pixels:
	points.append(item[0])


def in_bounds(x, y):
	lower_bound = 0
	upper_x_bound = image.shape[0]
	upper_y_bound = image.shape[1]
	if (x < 0 or y < 0 or x >= upper_x_bound or y >= upper_y_bound):
		return False
	else:
		return True

for p in points:
	if in_bounds(p[0]-7, p[1]-7) and in_bounds(p[0]+7, p[1]+7):
		for x in range(p[0]-7, p[0]+7 +1):
			for y in range(p[1]-7, p[1]+7 + 1):
				coor = (x,y)
				if coor in points and coor != p:
					points.remove(coor)


for p in points:
	if in_bounds(p[0]-2, p[1]-2) and in_bounds(p[0]+2, p[1]+2):
		for x in range(p[0]-2, p[0]+2 +1):
			coor = (x, p[1])
			image[coor] = 0
		for y in range(p[1]-2, p[1]+2 +1):
			coor = (p[0], y)
			image[coor] = 0	


import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()















